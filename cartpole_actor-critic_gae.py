# On-Policy Actor–Critic with GAE(λ) on CartPole-v1 — minibatch + multi-critic steps
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

# ---- Reproducibility ----
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def make_env(seed=SEED):
    env = gym.make("CartPole-v1")
    try:
        env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return env

# ---- Policy Network (Categorical over 2 actions) ----
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden=128, act_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim)  # logits
        )
    def forward(self, x):
        return self.net(x)  # logits

    def act(self, obs):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a.item(), logp

    def act_deterministic(self, obs):
        """Greedy action (argmax over logits) — handy for evaluation/video."""
        logits = self.forward(obs)
        a = torch.argmax(logits, dim=-1)
        return int(a.item())

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
    """
    if GYMN:
        obs, info = env.reset()
    else:
        obs = env.reset()
    done = False

    states, rewards, next_states, dones, logps = [], [], [], [], []
    ep_ret, steps = 0.0, 0

    while not done:
        if render: env.render()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a, logp = policy.act(obs_t)

        if GYMN:
            nobs, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
        else:
            nobs, r, done, info = env.step(a)

        states.append(np.array(obs, dtype=np.float32))
        rewards.append(float(r))
        next_states.append(np.array(nobs, dtype=np.float32))
        dones.append(float(done))
        logps.append(logp)

        ep_ret += float(r)
        obs = nobs
        steps += 1

    return states, rewards, next_states, dones, logps, ep_ret, steps

# ---- GAE(λ) helper ----
@torch.no_grad()
def compute_gae(R, V, V_next, D, gamma=0.99, lam=0.95):
    """
    Inputs are 1D tensors of the same length T (flattened over the batch):
      R: rewards_t
      V: V(s_t)
      V_next: V(s_{t+1})
      D: done flags in {0.,1.} (1. at terminal steps)
    Returns:
      adv: \hat{A}^{(\lambda)}_t  (T,)
      v_targ: \hat{V}^{targ}_t = adv + V(s_t)  (T,)
      delta: TD residuals (T,)
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
def plot_training_curve(returns_hist, window=50, out_path="cartpole_returns.png"):
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
    plt.title("On-Policy Actor–Critic with GAE(λ) on CartPole-v1 — Training Curve")
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
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda e: True  # record all episodes we run here
        )
    else:
        env = gym.make("CartPole-v1")
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
    episodes=800,
    gamma=0.99,
    lam=0.95,
    lr_actor=2.5e-3,
    lr_critic=2.5e-3,
    batch_size=10,                # number of episodes per policy update
    critic_steps_per_update=10,   # multiple critic steps before actor step
    normalize_adv=True,           # normalize \hat{A} within each update batch
    render_every=None,
    target_avg_last100=475.0,     # CartPole-v1 "solved" threshold
    device=None
):
    """
    Minibatch on-policy Actor–Critic with GAE(λ):
      - Collect 'batch_size' on-policy trajectories.
      - Build λ-return targets: v_targ = \hat{A}^{(λ)} + V(s_t) using current critic.
      - Take multiple critic steps toward a **fixed** v_targ (stabilizes optimization).
      - Recompute GAE with the UPDATED critic for the actor step (keeps actor in sync).
    """
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNet(obs_dim, hidden=128, act_dim=act_dim).to(device)
    valuef = ValueNet(obs_dim, hidden=128).to(device)
    opt_actor  = optim.Adam(policy.parameters(), lr=lr_actor)
    opt_critic = optim.Adam(valuef.parameters(), lr=lr_critic)

    returns_hist = []
    best_avg = -1e9
    start = time.time()
    episodes_run = 0

    while episodes_run < episodes:
        # ----- Collect a batch of trajectories -----
        S_list, S2_list, R_list, D_list, LP_list = [], [], [], [], []
        total_steps = 0
        batch_eps = min(batch_size, episodes - episodes_run)

        for _ in range(batch_eps):
            render = (render_every is not None and (episodes_run + 1) % render_every == 0)
            s, r, s2, d, lp, ep_ret, steps = rollout_episode(env, policy, render, device)
            returns_hist.append(ep_ret)
            episodes_run += 1
            total_steps += int(steps)

            S_list.extend(s)
            S2_list.extend(s2)
            R_list.extend(r)
            D_list.extend(d)
            LP_list.extend(lp)

        # ----- Convert to tensors -----
        S  = torch.tensor(np.array(S_list),  dtype=torch.float32, device=device)     # (T, obs)
        S2 = torch.tensor(np.array(S2_list), dtype=torch.float32, device=device)     # (T, obs)
        R  = torch.tensor(np.array(R_list),  dtype=torch.float32, device=device)     # (T,)
        D  = torch.tensor(np.array(D_list),  dtype=torch.float32, device=device)     # (T,)
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

        # ----- Actor step: policy gradient with advantage = \hat{A}^{(λ)} -----
        actor_loss = -(LP * adv).mean()

        opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt_actor.step()

        # ----- Logging -----
        avg100 = np.mean(returns_hist[-100:]) if len(returns_hist) >= 1 else 0.0
        best_avg = max(best_avg, avg100)
        if episodes_run % 10 == 0 or episodes_run == 1:
            print(
                f"Eps {episodes_run:4d} | LastRet {returns_hist[-1]:6.1f} | "
                f"Avg100 {avg100:6.1f} | Steps(batch) {total_steps:4d} | "
                f"ActorL {actor_loss.item():+.4f} | CriticSteps {critic_steps_per_update} | λ={lam:.2f}"
            )

        # Optional early stop when solved
        if len(returns_hist) >= 100 and avg100 >= target_avg_last100:
            print(f"SOLVED in {episodes_run} episodes! Avg100={avg100:.1f}")
            break

    dur = time.time() - start
    print(f"Training done in {dur:.1f}s. Best Avg100={best_avg:.1f}")
    env.close()
    return policy, valuef, returns_hist

if __name__ == "__main__":
    policy, valuef, returns = train_actor_critic_gae_minibatch(
        episodes=3000,
        gamma=0.99,
        lam=0.95,                 # GAE parameter (bias–variance trade-off)
        lr_actor=1e-3,
        lr_critic=1e-3,
        batch_size=20,            # multiple trajectories per update
        critic_steps_per_update=20,
        normalize_adv=True,
        render_every=None,
    )

    # ---- Plot and save training curve ----
    os.makedirs("videos", exist_ok=True)
    plot_training_curve(returns, window=50, out_path="videos/cartpole_returns.png")

    # ---- Roll out the final policy and record a video ----
    # Set deterministic=True for a stable video (greedy w.r.t. logits)
    rollout_and_record_video(policy, video_dir="videos", episodes=1, max_steps=1000, deterministic=True)
