# REINFORCE on CartPole-v1 (PyTorch) — minibatch, no baseline/advantage
# ----------------------------------------------------------------------
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
        # old gym API
        env.seed(seed)
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
    """Collect one episode and return (log-probs per step, returns-to-go per step, episode return, steps)."""
    if GYMN:
        obs, info = env.reset()
    else:
        obs = env.reset()
    done = False
    rewards, logps = [], []
    ep_ret = 0.0
    steps = 0

    while not done:
        if render: env.render()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a, logp = policy.act(obs_t)
        if GYMN:
            next_obs, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
        else:
            next_obs, r, done, info = env.step(a)
        rewards.append(float(r))
        logps.append(logp)
        ep_ret += float(r)
        obs = next_obs
        steps += 1

    G = discount_cumsum(rewards, gamma)
    G_t = torch.tensor(G, dtype=torch.float32, device=device)
    logps_t = torch.stack(logps).to(device)
    return logps_t, G_t, ep_ret, steps

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
    plt.title("REINFORCE on CartPole-v1 — Training Curve")
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

# ---- Training: minibatch REINFORCE (no baseline/advantage) ----
def train_reinforce_minibatch(
    episodes=800,
    gamma=0.99,
    lr=2.5e-3,
    batch_size=10,              # number of episodes per policy update
    normalize_by_steps=True,    # average loss over total timesteps in batch
    render_every=None,
    target_avg_last100=475.0,   # CartPole-v1 "solved" threshold
    device=None
):
    """
    Minibatch REINFORCE:
      - Collect 'batch_size' on-policy trajectories.
      - Compute loss L = - E_{t,episodes}[ log pi(a_t|s_t) * G_t ] (no baseline).
      - Single optimizer step per batch.
    """
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNet(obs_dim, hidden=128, act_dim=act_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    returns_hist = []
    best_avg = -1e9
    start = time.time()
    episodes_run = 0

    while episodes_run < episodes:
        # ----- Collect a batch of trajectories -----
        batch_loss = 0.0
        total_steps = 0
        batch_eps = min(batch_size, episodes - episodes_run)

        for b in range(batch_eps):
            render = (render_every is not None and (episodes_run + 1) % render_every == 0)
            logps, G_t, ep_ret, steps = rollout_episode(env, policy, gamma, render, device)
            returns_hist.append(ep_ret)
            episodes_run += 1
            total_steps += int(steps)

            # --- REINFORCE loss contribution (no baseline/advantage) ---
            # Sum over time of (- log pi * G_t)
            batch_loss = batch_loss + (-(logps * G_t).sum())

        # Normalize loss to keep LR scale stable
        if normalize_by_steps and total_steps > 0:
            batch_loss = batch_loss / float(total_steps)
        else:
            batch_loss = batch_loss / float(batch_eps)

        # ----- Single update -----
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging
        avg100 = np.mean(returns_hist[-100:]) if len(returns_hist) >= 1 else 0.0
        best_avg = max(best_avg, avg100)
        if episodes_run % 10 == 0 or episodes_run == 1:
            print(f"Eps {episodes_run:4d} | LastRet {returns_hist[-1]:6.1f} | "
                  f"Avg100 {avg100:6.1f} | Steps(batch) {total_steps:4d} | "
                  f"Loss {batch_loss.item():.3f}")

        # Optional early stop when solved
        if len(returns_hist) >= 100 and avg100 >= target_avg_last100:
            print(f"SOLVED in {episodes_run} episodes! Avg100={avg100:.1f}")
            break

    dur = time.time() - start
    print(f"Training done in {dur:.1f}s. Best Avg100={best_avg:.1f}")
    env.close()
    return policy, returns_hist

if __name__ == "__main__":
    policy, returns = train_reinforce_minibatch(
        episodes=2000,
        gamma=0.99,
        lr=2.5e-3,
        batch_size=20,            # multiple trajectories per update
        normalize_by_steps=True,  # average over all time steps collected in the batch
        render_every=None,
    )

    # ---- Plot and save training curve ----
    plot_training_curve(returns, window=50, out_path="videos/cartpole_returns.png")

    # ---- Roll out the final policy and record a video ----
    # Set deterministic=True for a stable video (greedy w.r.t. logits)
    rollout_and_record_video(policy, video_dir="videos", episodes=1, max_steps=1000, deterministic=True)
