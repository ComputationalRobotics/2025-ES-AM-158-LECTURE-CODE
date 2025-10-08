# REINFORCE on Pendulum-v1 (PyTorch) — value-converged critic per batch, single actor step
# ----------------------------------------------------------------------------------------
# pip install gymnasium torch matplotlib  (or: pip install gym torch matplotlib)

import os, random, time
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
from torch.nn import functional as F

# ---- Reproducibility ----
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def make_env(seed=SEED):
    # Continuous inverted pendulum (classic control)
    env = gym.make("Pendulum-v1")
    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    return env

# ---- Policy Network (Tanh-squashed Gaussian over continuous actions) ----
class PolicyNet(nn.Module):
    """
    Outputs a tanh-squashed Gaussian policy that is properly scaled to the env's
    action bounds. Provides:
      - act(): sample action in env space
      - act_deterministic(): mean action (tanh(mu)) in env space
      - log_prob(obs, act_env): exact log-prob of an env-space action given obs
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

        # Save action scaling for mapping (-1,1) -> [low, high]
        self.register_buffer("act_low", torch.tensor(act_low, dtype=torch.float32))
        self.register_buffer("act_high", torch.tensor(act_high, dtype=torch.float32))
        self.register_buffer("act_scale", (self.act_high - self.act_low) / 2.0)
        self.register_buffer("act_bias", (self.act_high + self.act_low) / 2.0)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    @torch.no_grad()
    def act(self, obs):
        """Sample a single env-space action (numpy array)."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        u = Normal(mu, std).sample()            # pre-tanh
        a = torch.tanh(u)                       # (-1,1)
        a_env = a * self.act_scale + self.act_bias
        return a_env.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def act_deterministic(self, obs):
        """Greedy action: tanh(mu) scaled to env bounds."""
        mu, _ = self.forward(obs)
        a = torch.tanh(mu)
        a_env = a * self.act_scale + self.act_bias
        return a_env.squeeze(0).cpu().numpy()

    def _atanh(self, x, eps=1e-6):
        x = x.clamp(min=-1+eps, max=1-eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))  # 0.5*ln((1+x)/(1-x))

    def log_prob(self, obs, act_env):
        """
        Exact log-prob for tanh-squashed, scaled Gaussian.
        obs: (B, obs_dim)
        act_env: (B, act_dim) in env bounds [low, high]
        returns: (B,)
        """
        mu, log_std = self.forward(obs)
        std = log_std.exp()

        # Map env action back to (-1,1), then to pre-tanh space via atanh
        a = (act_env - self.act_bias) / (self.act_scale + 1e-8)  # (-1,1)
        u = self._atanh(a)  # pre-tanh action

        base = Normal(mu, std)
        # log prob under Gaussian in u-space
        logp_u = base.log_prob(u).sum(dim=-1)

        # Change-of-variables for tanh: log|det(d tanh(u)/du)| = log(1 - tanh(u)^2)
        # Use the stable SAC-style expression:
        # log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2u))
        log_det_tanh = (2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u))).sum(dim=-1)

        # Scaling from (-1,1) -> [low, high]: y = scale * a + bias, so subtract log|scale|
        log_det_scale = torch.log(self.act_scale.abs() + 1e-8).sum(dim=-1)

        # Final log prob in env space
        return logp_u - log_det_tanh - log_det_scale

    def entropy_approx(self, obs):
        """Entropy of base Gaussian (ignores tanh+scaling) — ok if ent_coef small/zero."""
        mu, log_std = self.forward(obs)
        # Entropy of Normal = 0.5 * log(2πe σ^2)
        ent = 0.5 * (1.0 + np.log(2*np.pi)) + log_std
        return ent.sum(dim=-1).mean()

# ---- Value Network (state-value baseline V(s)) ----
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

# ---- Utilities ----
def discount_cumsum(rewards, gamma):
    """Return-to-go G_t = sum_{k=0}^{T-1-t} gamma^k r_{t+k}."""
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return list(reversed(out))

def rollout_episode(env, policy, gamma=0.99, render=False, device="cpu"):
    """
    Collect one episode and return:
      - states per step (Tensor [T, obs_dim])
      - actions per step (Tensor [T, act_dim], float)
      - returns-to-go per step (Tensor [T])
      - episode return (float)
      - episode length (int)
    """
    if GYMN:
        obs, info = env.reset()
    else:
        obs = env.reset()
    done = False
    rewards, states, acts = [], [], []
    ep_ret = 0.0
    steps = 0

    while not done:
        if render: env.render()
        states.append(np.asarray(obs, dtype=np.float32))
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a_env = policy.act(obs_t)
        if GYMN:
            next_obs, r, terminated, truncated, info = env.step(a_env)
            done = terminated or truncated
        else:
            next_obs, r, done, info = env.step(a_env)
        rewards.append(float(r))
        acts.append(np.asarray(a_env, dtype=np.float32))
        ep_ret += float(r)
        obs = next_obs
        steps += 1

    G = discount_cumsum(rewards, gamma)
    states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    acts_t = torch.tensor(np.array(acts), dtype=torch.float32, device=device)  # [T, act_dim]
    G_t = torch.tensor(G, dtype=torch.float32, device=device)
    return states_t, acts_t, G_t, ep_ret, steps

# ---- Plotting helper ----
def plot_training_curve(returns_hist, window=50, out_path="videos/pendulum_returns_value_converged.png"):
    """Plot per-episode returns and a moving average."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
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
    plt.title("REINFORCE on Pendulum-v1 — Critic Converged per Batch")
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
            episode_trigger=lambda e: True  # record all episodes we run here
        )
    else:
        env = gym.make("Pendulum-v1")
        env = gym.wrappers.Monitor(
            env,
            video_dir,
            force=True,
            video_callable=lambda e: True  # record all
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
                a = policy.act(obs_t)
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

# ---- Training: per-batch converged critic, then one actor step ----
def train_reinforce_value_converged(
    episodes=2000,
    gamma=0.99,
    lr_policy=1e-3,
    lr_value=1e-3,
    batch_size=40,                # number of episodes per update
    standardize_adv=True,         # normalize advantages across the WHOLE batch
    ent_coef=0.0,                 # entropy bonus (set >0 if exploration needs help)
    value_grad_clip=1.0,
    policy_grad_clip=1.0,
    # --- critic inner-loop convergence controls ---
    value_inner_max_steps=2000,   # hard cap on GD steps for critic
    value_check_every=10,         # check convergence every N steps
    value_convergence_tol=1e-6,   # relative improvement tolerance
    value_patience=10,            # consecutive checks without improvement
    target_avg_last100=-200.0,    # Pendulum: higher (less negative) is better
    render_every=None,
    device=None
):
    """
    Training scheme per minibatch:
      1) Collect a batch of on-policy trajectories with the *current* actor.
      2) Fit V(s) to Monte Carlo returns by running many GD steps on the batch
         until convergence (early-stopping with tolerance/patience).
      3) Compute advantages A = G - V(s) (critic is now converged on this batch).
      4) Do *one* actor (policy) update step using those advantages.
    """
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNet(obs_dim, act_dim, act_low, act_high, hidden=128).to(device)
    valuef = ValueNet(obs_dim, hidden=128).to(device)

    opt_pi = optim.Adam(policy.parameters(), lr=lr_policy)
    opt_v  = optim.Adam(valuef.parameters(), lr=lr_value)

    mse = nn.MSELoss(reduction="mean")

    returns_hist = []
    best_avg = -1e9
    start = time.time()
    episodes_run = 0

    while episodes_run < episodes:
        # ------- (1) Collect one batch of trajectories -------
        all_states, all_acts, all_G = [], [], []
        total_steps = 0
        batch_eps = min(batch_size, episodes - episodes_run)

        for _ in range(batch_eps):
            render = (render_every is not None and (episodes_run + 1) % render_every == 0)
            states_t, acts_t, G_t, ep_ret, steps = rollout_episode(env, policy, gamma, render, device)
            returns_hist.append(ep_ret)
            episodes_run += 1
            total_steps += int(steps)
            all_states.append(states_t)
            all_acts.append(acts_t)
            all_G.append(G_t)

        # Concatenate across all episodes in the batch
        S = torch.cat(all_states, dim=0)        # [N_steps, obs_dim]
        A = torch.cat(all_acts, dim=0)          # [N_steps, act_dim] (float)
        G = torch.cat(all_G, dim=0)             # [N_steps]

        # ------- (2) Critic inner loop: fit V(s) until convergence -------
        best_loss = float("inf")
        no_improve = 0
        steps_used = 0

        for k in range(1, value_inner_max_steps + 1):
            V = valuef(S)
            v_loss = mse(V, G)

            opt_v.zero_grad(set_to_none=True)
            v_loss.backward()
            if value_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(valuef.parameters(), max_norm=value_grad_clip)
            opt_v.step()

            steps_used = k
            if k % value_check_every == 0:
                cur = v_loss.item()
                rel_impr = (best_loss - cur) / max(1.0, abs(best_loss))
                if rel_impr <= value_convergence_tol:
                    no_improve += 1
                else:
                    no_improve = 0
                    best_loss = cur
                if no_improve >= value_patience:
                    break

        # ------- (3) Compute advantages with converged critic -------
        with torch.no_grad():
            V_final = valuef(S)
        adv = (G - V_final).detach()

        if standardize_adv:
            std = adv.std(unbiased=False)
            if std > 1e-8:
                adv = (adv - adv.mean()) / (std + 1e-8)
            else:
                adv = adv - adv.mean()

        # ------- (4) Single actor step (policy gradient) -------
        logp = policy.log_prob(S, A)            # exact log-prob of env actions
        entropy = policy.entropy_approx(S)      # approx; set ent_coef>0 if desired

        policy_loss = -(logp * adv).mean() - ent_coef * entropy

        opt_pi.zero_grad(set_to_none=True)
        policy_loss.backward()
        if policy_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=policy_grad_clip)
        opt_pi.step()

        # ------- Logging -------
        avg100 = np.mean(returns_hist[-100:]) if len(returns_hist) >= 1 else 0.0
        best_avg = max(best_avg, avg100)
        print(f"Eps {episodes_run:4d} | LastRet {returns_hist[-1]:7.1f} | "
              f"Avg100 {avg100:7.1f} | Steps(batch) {total_steps:4d} | "
              f"V-steps {steps_used:4d} | BestVLoss {best_loss:.4f} | "
              f"L_pi {policy_loss.item():.3f} | Ent {entropy.item():.3f}")

        if len(returns_hist) >= 100 and avg100 >= target_avg_last100:
            print(f"SOLVED in {episodes_run} episodes! Avg100={avg100:.1f}")
            break

    dur = time.time() - start
    print(f"Training done in {dur:.1f}s. Best Avg100={best_avg:.1f}")
    env.close()
    return policy, valuef, returns_hist

if __name__ == "__main__":
    policy, valuef, returns = train_reinforce_value_converged(
        episodes=3000,
        gamma=0.99,
        lr_policy=1e-3,
        lr_value=1e-3,
        batch_size=128,                # multiple trajectories per update
        standardize_adv=True,
        ent_coef=0.0,                 # try 1e-3 ~ 2e-2 if exploration needs help
        value_inner_max_steps=2000,
        value_check_every=10,
        value_convergence_tol=1e-6,
        value_patience=10,
        target_avg_last100=-200.0,
        render_every=None,
    )

    # ---- Plot and save training curve ----
    plot_training_curve(returns, window=50, out_path="videos/pendulum_returns_value_baseline.png")

    # ---- Roll out the final policy and record a video ----
    # Set deterministic=True for a stable video (greedy w.r.t. mean action)
    rollout_and_record_video(policy, video_dir="videos", episodes=1, max_steps=1000, deterministic=True)
