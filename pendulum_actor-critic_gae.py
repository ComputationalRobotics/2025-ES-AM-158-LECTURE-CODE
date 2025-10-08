# On-Policy Actor–Critic with GAE(λ) on Pendulum-v1 — minibatch + multi-critic steps
# -----------------------------------------------------------------------------------
# pip install gymnasium torch matplotlib  (or: pip install gym torch matplotlib)

import os, random, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- Gym import with fallback (Gymnasium preferred) ---
try:
    import gymnasium as gym
    GYMN = True
except Exception:
    import gym
    GYMN = False

from torch.distributions import Normal
import torch.nn.functional as F

# ---- Reproducibility ----
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def make_env(seed=SEED):
    env = gym.make("Pendulum-v1")
    try:
        env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return env

# ---- Policy Network (tanh-squashed Gaussian for continuous actions) ----
class PolicyNet(nn.Module):
    """
    Outputs a tanh-squashed Gaussian policy scaled to env bounds.
    Provides act(), act_deterministic(), and exact log_prob(obs, act_env).
    """
    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden=128, log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)
        self.log_std_min, self.log_std_max = log_std_bounds

        # Precompute scaling from (-1,1) -> [low, high]
        act_low = torch.as_tensor(act_low, dtype=torch.float32)
        act_high = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("act_low", act_low)
        self.register_buffer("act_high", act_high)
        self.register_buffer("act_scale", (act_high - act_low) / 2.0)
        self.register_buffer("act_bias", (act_high + act_low) / 2.0)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    @torch.no_grad()
    def act(self, obs):
        """Sample env-space action (numpy) and return (action, logp_torch0d)."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        u = Normal(mu, std).sample()           # pre-tanh
        a = torch.tanh(u)                      # (-1,1)
        a_env = a * self.act_scale + self.act_bias
        # exact log prob in env space
        logp = self.log_prob(obs, a_env)
        return a_env.squeeze(0).cpu().numpy(), logp.squeeze()

    @torch.no_grad()
    def act_deterministic(self, obs):
        mu, _ = self.forward(obs)
        a = torch.tanh(mu)
        a_env = a * self.act_scale + self.act_bias
        return a_env.squeeze(0).cpu().numpy()

    def _atanh(self, x, eps=1e-6):
        x = x.clamp(min=-1+eps, max=1-eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def log_prob(self, obs, act_env):
        """
        Exact log-prob for tanh-squashed scaled Gaussian.
        obs: (B, obs_dim)
        act_env: (B, act_dim) in env action space
        returns: (B,)
        """
        mu, log_std = self.forward(obs)
        std = log_std.exp()

        # map env action -> (-1,1) -> pre-tanh
        a = (act_env - self.act_bias) / (self.act_scale + 1e-8)
        u = self._atanh(a)

        base = Normal(mu, std)
        logp_u = base.log_prob(u).sum(dim=-1)

        # log|det(d tanh(u)/du)| (stable SAC-style)
        log_det_tanh = (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(dim=-1)

        # scaling from (-1,1) -> [low, high]
        log_det_scale = torch.log(self.act_scale.abs() + 1e-8).sum(dim=-1)

        return logp_u - log_det_tanh - log_det_scale

# ---- Value Network ----
class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,)

# ---- Rollout (collect transitions for on-policy updates) ----
def rollout_episode(env, policy, render=False, device="cpu"):
    """
    Collect one on-policy episode and return per-step transitions:
      states, actions, rewards, next_states, dones, logps, ep_return, steps
    Actions are continuous (float arrays); logps are exact env-space log-probs.
    """
    if GYMN:
        obs, info = env.reset()
    else:
        obs = env.reset()
    done = False

    states, actions, rewards, next_states, dones, logps = [], [], [], [], [], []
    ep_ret, steps = 0.0, 0

    while not done:
        if render: env.render()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a_env, logp = policy.act(obs_t)

        if GYMN:
            nobs, r, terminated, truncated, info = env.step(a_env)
            done = terminated or truncated
        else:
            nobs, r, done, info = env.step(a_env)

        states.append(np.array(obs, dtype=np.float32))
        actions.append(np.array(a_env, dtype=np.float32))
        rewards.append(float(r))
        next_states.append(np.array(nobs, dtype=np.float32))
        dones.append(float(done))
        logps.append(logp.detach())

        ep_ret += float(r)
        obs = nobs
        steps += 1

    return states, rewards, next_states, dones, logps, actions, ep_ret, steps

# ---- GAE(λ) helper ----
@torch.no_grad()
def compute_gae(R, V, V_next, D, gamma=0.99, lam=0.95):
    """
    Inputs are 1D tensors of same length T:
      R: rewards_t, V: V(s_t), V_next: V(s_{t+1}), D: done flags {0.,1.}
    Returns: adv, v_targ, delta  (each shape: (T,))
    """
    m = 1.0 - D  # mask to stop at terminals
    delta = R + gamma * m * V_next - V
    T = R.shape[0]
    adv = torch.zeros_like(R)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        gae = delta[t] + gamma * lam * m[t] * gae
        adv[t] = gae
    v_targ = adv + V
    return adv, v_targ, delta

# ---- Plotting helper ----
def plot_training_curve(returns_hist, window=50, out_path="pendulum_returns.png"):
    """Plot per-episode returns and a moving average."""
    x = np.arange(1, len(returns_hist) + 1)
    rets = np.array(returns_hist, dtype=float)
    if len(rets) >= window:
        ma = np.convolve(rets, np.ones(window)/window, mode="valid")
    else:
        ma = None

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, rets, label="Return per episode")
    if ma is not None:
        plt.plot(np.arange(window, len(rets)+1), ma, label=f"{window}-episode average")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("On-Policy Actor–Critic with GAE(λ) on Pendulum-v1 — Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[Plot] Saved training curve to {out_path}")

# ---- Video recording helper ----
def rollout_and_record_video(policy, video_dir="videos", episodes=1, max_steps=1000, deterministic=True):
    """
    Roll out the (trained) policy and record video(s).
    - Gymnasium: uses RecordVideo and requires env with render_mode='rgb_array'
    - Classic Gym: uses Monitor wrapper
    """
    os.makedirs(video_dir, exist_ok=True)

    if GYMN:
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda e: True
        )
    else:
        env = gym.make("Pendulum-v1")
        env = gym.wrappers.Monitor(
            env,
            video_dir,
            force=True,
            video_callable=lambda e: True
        )

    total_returns = []
    for ep in range(episodes):
        if GYMN:
            obs, info = env.reset()
        else:
            obs = env.reset()
        done, steps, ep_ret = False, 0, 0.0

        while not done and steps < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            if deterministic:
                a = policy.act_deterministic(obs_t)
            else:
                a, _ = policy.act(obs_t)
            if GYMN:
                obs, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated
            else:
                obs, r, done, info = env.step(a)
            ep_ret += float(r)
            steps += 1

        total_returns.append(ep_ret)
        print(f"[Video] Episode {ep+1}/{episodes} return: {ep_ret:.1f}")

    env.close()
    print(f"[Video] Saved to: {os.path.abspath(video_dir)}")
    return total_returns

# ---- Training: On-Policy Actor–Critic with GAE(λ) and multi-critic updates ----
def train_actor_critic_gae_minibatch(
    episodes=1500,
    gamma=0.99,
    lam=0.95,
    lr_actor=3e-4,
    lr_critic=1e-3,
    batch_size=20,                # number of episodes per policy update
    critic_steps_per_update=20,   # multiple critic steps before actor step
    normalize_adv=True,           # normalize \hat{A} within each update batch
    render_every=None,
    target_avg_last100=-200.0,    # Pendulum: less negative is better
    device=None
):
    """
    Minibatch on-policy Actor–Critic with GAE(λ):
      - Collect 'batch_size' on-policy trajectories.
      - Build λ-return targets with current critic.
      - Take multiple critic steps toward a **fixed** v_targ (stabilizes).
      - Recompute GAE with the UPDATED critic for the actor step.
    """
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low, act_high = env.action_space.low, env.action_space.high
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNet(obs_dim, act_dim, act_low, act_high, hidden=128).to(device)
    valuef = ValueNet(obs_dim, hidden=128).to(device)
    opt_actor  = optim.Adam(policy.parameters(), lr=lr_actor)
    opt_critic = optim.Adam(valuef.parameters(), lr=lr_critic)

    returns_hist = []
    best_avg = -1e9
    start = time.time()
    episodes_run = 0

    while episodes_run < episodes:
        # ----- Collect a batch of trajectories -----
        S_list, S2_list, R_list, D_list, LP_list, A_list = [], [], [], [], [], []
        total_steps = 0
        batch_eps = min(batch_size, episodes - episodes_run)

        for _ in range(batch_eps):
            render = (render_every is not None and (episodes_run + 1) % render_every == 0)
            s, r, s2, d, lp, a_env, ep_ret, steps = rollout_episode(env, policy, render, device)
            returns_hist.append(ep_ret)
            episodes_run += 1
            total_steps += int(steps)

            S_list.extend(s)
            S2_list.extend(s2)
            R_list.extend(r)
            D_list.extend(d)
            LP_list.extend(lp)
            A_list.extend(a_env)

        # ----- Convert to tensors -----
        S  = torch.tensor(np.array(S_list),  dtype=torch.float32, device=device)     # (T, obs)
        S2 = torch.tensor(np.array(S2_list), dtype=torch.float32, device=device)     # (T, obs)
        R  = torch.tensor(np.array(R_list),  dtype=torch.float32, device=device)     # (T,)
        D  = torch.tensor(np.array(D_list),  dtype=torch.float32, device=device)     # (T,)
        A  = torch.tensor(np.array(A_list),  dtype=torch.float32, device=device)     # (T, act_dim)
        LP = torch.stack([lp if lp.dim()==0 else lp.squeeze(-1) for lp in LP_list])  # (T,)

        # ===== Build GAE advantages and λ-return targets with CURRENT critic =====
        with torch.no_grad():
            V_S  = valuef(S)
            V_S2 = valuef(S2)
            adv_init, v_targ_static, delta_init = compute_gae(R, V_S, V_S2, D, gamma=gamma, lam=lam)

        # ----- Critic phase: multiple steps toward FIXED λ-return targets -----
        for _ in range(int(critic_steps_per_update)):
            v_pred = valuef(S)
            critic_loss = (v_pred - v_targ_static).pow(2).mean()

            opt_critic.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(valuef.parameters(), max_norm=1.0)
            opt_critic.step()

        # ===== Recompute advantages with UPDATED critic for actor step =====
        with torch.no_grad():
            V_S  = valuef(S)
            V_S2 = valuef(S2)
            adv, _, _ = compute_gae(R, V_S, V_S2, D, gamma=gamma, lam=lam)

        if normalize_adv:
            std = adv.std().clamp_min(1e-6)
            adv = (adv - adv.mean()) / std

        # ----- Safety: recompute logp from current policy (MUST require grad) -----
        LP_cur = policy.log_prob(S, A)              # <-- no torch.no_grad() here!

        # ----- Actor step: policy gradient with advantage = \hat{A}^{(λ)} -----
        # adv is a constant target (already from no_grad); detach to be explicit
        actor_loss = -(LP_cur * adv.detach()).mean()

        opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt_actor.step()


        # ----- Logging -----
        avg100 = np.mean(returns_hist[-100:]) if len(returns_hist) >= 1 else 0.0
        best_avg = max(best_avg, avg100)
        if episodes_run % 10 == 0 or episodes_run == 1:
            print(
                f"Eps {episodes_run:4d} | LastRet {returns_hist[-1]:7.1f} | "
                f"Avg100 {avg100:7.1f} | Steps(batch) {total_steps:4d} | "
                f"ActorL {actor_loss.item():+.4f} | CriticSteps {critic_steps_per_update} | λ={lam:.2f}"
            )

        # Optional early stop
        if len(returns_hist) >= 100 and avg100 >= target_avg_last100:
            print(f"SOLVED in {episodes_run} episodes! Avg100={avg100:.1f}")
            break

    dur = time.time() - start
    print(f"Training done in {dur:.1f}s. Best Avg100={best_avg:.1f}")
    env.close()
    return policy, valuef, returns_hist

if __name__ == "__main__":
    policy, valuef, returns = train_actor_critic_gae_minibatch(
        episodes=5000,
        gamma=0.99,
        lam=0.95,               # GAE parameter (bias–variance trade-off)
        lr_actor=1e-4,          # smaller actor lr often helps stability on Pendulum
        lr_critic=1e-3,
        batch_size=128,          # multiple trajectories per update
        critic_steps_per_update=20,
        normalize_adv=True,
        render_every=None,
    )

    # ---- Plot and save training curve ----
    os.makedirs("videos", exist_ok=True)
    plot_training_curve(returns, window=50, out_path="videos/pendulum_returns.png")

    # ---- Roll out the final policy and record a video ----
    # Set deterministic=True for a stable video (greedy w.r.t. mean)
    rollout_and_record_video(policy, video_dir="videos", episodes=1, max_steps=1000, deterministic=True)
