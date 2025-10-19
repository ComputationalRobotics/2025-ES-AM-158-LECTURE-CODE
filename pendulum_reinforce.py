# REINFORCE on Pendulum-v1 (PyTorch) — minibatch, no baseline/advantage
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
    env = gym.make("Pendulum-v1")
    try:
        env.reset(seed=seed)
    except TypeError:
        # old gym API
        env.seed(seed)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    return env

# ---- Policy Network (Gaussian over continuous action, tanh-squashed) ----
class PolicyNet(nn.Module):
    """
    Outputs mean and log_std for a Gaussian over pre-tanh action u.
    The environment action is a = tanh(u) * act_scale.
    We apply the standard tanh-log-det-Jacobian correction to log_prob.
    """
    def __init__(self, obs_dim, hidden=128, act_dim=1, init_log_std=-0.5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * init_log_std)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std, -5.0, 2.0)  # keep std reasonable
        std = torch.exp(log_std)
        return mu, std, log_std

    @staticmethod
    def _squash(u, act_scale):
        a = torch.tanh(u) * act_scale
        return a

    @staticmethod
    def _log_prob_tanh_normal(u, mu, std):
        """
        Log prob of tanh(u) under a squashed Gaussian.
        log pi(a) = log N(u; mu, std) - sum log(1 - tanh(u)^2)
        (Constant log|scale| from a = scale*tanh(u) is omitted; it's grad-constant.)
        """
        base = torch.distributions.Normal(mu, std)
        log_prob_u = base.log_prob(u).sum(dim=-1)
        # tanh correction (per-dim), +eps for numerical stability
        correction = torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6).sum(dim=-1)
        return log_prob_u - correction

    def act(self, obs, act_scale):
        mu, std, _ = self.forward(obs)
        base = torch.distributions.Normal(mu, std)
        u = base.rsample()  # reparameterized sample
        a = self._squash(u, act_scale)
        logp = self._log_prob_tanh_normal(u, mu, std)
        return a, logp

    def act_deterministic(self, obs, act_scale):
        mu, std, _ = self.forward(obs)
        a = self._squash(mu, act_scale)
        return a

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
    Collect one episode and return (log-probs per step, returns-to-go per step, episode return, steps).
    Works for continuous action (Pendulum).
    """
    if GYMN:
        obs, info = env.reset()
    else:
        obs = env.reset()

    done = False
    rewards, logps = [], []
    ep_ret = 0.0
    steps = 0

    act_high = float(env.action_space.high[0])
    while not done:
        if render: env.render()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a_t, logp_t = policy.act(obs_t, act_scale=act_high)
        a_env = a_t.detach().cpu().numpy().astype(np.float32)[0]  # shape (), env expects array-like (1,)
        if a_env.ndim == 0:
            a_env = np.array([a_env], dtype=np.float32)

        if GYMN:
            next_obs, r, terminated, truncated, info = env.step(a_env)
            done = terminated or truncated
        else:
            next_obs, r, done, info = env.step(a_env)

        rewards.append(float(r))
        logps.append(logp_t.squeeze(0))  # scalar
        ep_ret += float(r)
        obs = next_obs
        steps += 1

    G = discount_cumsum(rewards, gamma)
    G_t = torch.tensor(G, dtype=torch.float32, device=device)
    logps_t = torch.stack(logps).to(device)
    return logps_t, G_t, ep_ret, steps

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
    plt.title("REINFORCE on Pendulum-v1 — Training Curve (no baseline)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[Plot] Saved training curve to {out_path}")

# ---- Video recording helper ----
def rollout_and_record_video(policy, video_dir="videos", episodes=1, max_steps=200, deterministic=True):
    """
    Roll out the (trained) policy and record video(s) on Pendulum.
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

    act_high = float(env.action_space.high[0])
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
                a_t = policy.act_deterministic(obs_t, act_scale=act_high)
                logp_dummy = None
            else:
                a_t, _ = policy.act(obs_t, act_scale=act_high)
            a_env = a_t.detach().cpu().numpy().astype(np.float32)[0]
            if a_env.ndim == 0:
                a_env = np.array([a_env], dtype=np.float32)

            if GYMN:
                obs, r, terminated, truncated, info = env.step(a_env)
                done = terminated or truncated
            else:
                obs, r, done, info = env.step(a_env)
            ep_ret += float(r)
            steps += 1

        total_returns.append(ep_ret)
        print(f"[Video] Episode {ep+1}/{episodes} return: {ep_ret:.1f}")

    env.close()
    print(f"[Video] Saved to: {os.path.abspath(video_dir)}")
    return total_returns

# ---- Training: minibatch REINFORCE (no baseline/advantage) ----
def train_reinforce_minibatch(
    episodes=1500,
    gamma=0.99,
    lr=3e-3,
    batch_size=10,              # number of episodes per policy update
    normalize_by_steps=True,    # average loss over total timesteps in batch
    render_every=None,
    device=None
):
    """
    Minibatch REINFORCE for continuous actions (Pendulum):
      - Collect 'batch_size' on-policy trajectories.
      - Compute loss L = - E_{t,episodes}[ log pi(a_t|s_t) * G_t ] (no baseline).
      - Single optimizer step per batch.
    """
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
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
            print(f"Eps {episodes_run:4d} | LastRet {returns_hist[-1]:7.1f} | "
                  f"Avg100 {avg100:7.1f} | Steps(batch) {total_steps:4d} | "
                  f"Loss {batch_loss.item():.3f}")

    dur = time.time() - start
    print(f"Training done in {dur:.1f}s. Best Avg100={best_avg:.1f}")
    env.close()
    return policy, returns_hist

if __name__ == "__main__":
    policy, returns = train_reinforce_minibatch(
        episodes=2000,
        gamma=0.99,
        lr=3e-3,
        batch_size=1024,             # multiple trajectories per update
        normalize_by_steps=True,   # average over all time steps collected in the batch
        render_every=None,
    )

    # ---- Plot and save training curve ----
    os.makedirs("videos", exist_ok=True)
    plot_training_curve(returns, window=50, out_path="videos/pendulum_returns.png")

    # ---- Roll out the final policy and record a video ----
    # Set deterministic=True for a stable video (greedy mean through tanh)
    rollout_and_record_video(policy, video_dir="videos", episodes=1, max_steps=200, deterministic=True)
