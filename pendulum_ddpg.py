# DDPG on Pendulum-v1 (PyTorch)
# ------------------------------
# Features:
#   • Deterministic actor with tanh squashing to env bounds
#   • Single critic Q(s, a)
#   • Target actor & target critic with Polyak averaging
#   • OU exploration noise (optionally Gaussian)
#   • Plots per-episode returns (learning curve)
#   • Saves evaluation rollouts as videos

import os, random, math, time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---- Gym import with fallback (Gymnasium preferred) ----
try:
    import gymnasium as gym
    GYMN = True
except Exception:
    import gym
    GYMN = False

# ----------------- Reproducibility ----------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def make_env(seed=SEED, render_for_video=False):
    """Create Pendulum env. For Gymnasium video, we need render_mode='rgb_array'."""
    if GYMN and render_for_video:
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
    else:
        env = gym.make("Pendulum-v1")
    try:
        env.reset(seed=seed)
    except TypeError:
        try: env.seed(seed)
        except Exception: pass
    return env

def make_video_env(video_dir, seed=SEED):
    """Wrap an env to record videos to `video_dir`."""
    os.makedirs(video_dir, exist_ok=True)
    env = make_env(seed=seed, render_for_video=True)
    try:
        from gymnasium.wrappers import RecordVideo as GymnRecordVideo
        env = GymnRecordVideo(env, video_folder=video_dir,
                              episode_trigger=lambda ep: True,
                              name_prefix="ddpg_pendulum")
    except Exception:
        if hasattr(gym.wrappers, "RecordVideo"):
            env = gym.wrappers.RecordVideo(env, video_folder=video_dir,
                                           episode_trigger=lambda ep: True)
        else:
            env = gym.wrappers.Monitor(env, video_dir, force=True)
    return env

# ----------------- Replay Buffer ------------------------
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.nxt = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.ptr = 0; self.size = 0; self.max_size = size

    def push(self, s, a, r, s2, d):
        self.obs[self.ptr] = s
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.nxt[self.ptr] = s2
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idxs = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x, dtype=torch.float32: torch.as_tensor(x[idxs], dtype=dtype, device=device)
        s   = to_t(self.obs)
        a   = to_t(self.act)
        r   = to_t(self.rew)
        s2  = to_t(self.nxt)
        d   = to_t(self.done)
        return s, a, r, s2, d

# ----------------- OU Noise -----------------------------
class OUNoise:
    """Ornstein–Uhlenbeck process for temporally correlated exploration noise."""
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size, self.mu, self.theta, self.sigma = size, mu, theta, sigma
        self.state = np.ones(size, dtype=np.float32) * mu
    def reset(self): self.state[:] = self.mu
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(*self.state.shape)
        self.state += dx
        return self.state.astype(np.float32)

# ----------------- Networks -----------------------------
def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for j in range(len(sizes)-1):
        layers += [nn.Linear(sizes[j], sizes[j+1])]
        if j < len(sizes)-2:
            layers += [act()]
        elif out_act is not None:
            layers += [out_act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    """Deterministic actor with tanh squashing to env bounds."""
    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden=(256,256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim])
        self.register_buffer("act_low",  torch.as_tensor(act_low,  dtype=torch.float32))
        self.register_buffer("act_high", torch.as_tensor(act_high, dtype=torch.float32))
        self._update_scale_bias()
    def _update_scale_bias(self):
        self.act_scale = (self.act_high - self.act_low) / 2.0
        self.act_bias  = (self.act_high + self.act_low) / 2.0
    def forward(self, s):
        u = self.net(s)
        a = torch.tanh(u) * self.act_scale + self.act_bias
        return a
    @torch.no_grad()
    def act(self, s_np):
        s = torch.as_tensor(s_np, dtype=torch.float32).unsqueeze(0)
        a = self.forward(s)
        return a.squeeze(0).cpu().numpy()

class Critic(nn.Module):
    """Q(s,a) critic; concatenates state and action."""
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.q = mlp([obs_dim+act_dim, *hidden, 1])
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q(x)

def soft_update(src, tgt, tau):
    with torch.no_grad():
        for p, p_t in zip(src.parameters(), tgt.parameters()):
            p_t.data.mul_(1 - tau)
            p_t.data.add_(tau * p.data)

# ----------------- Config -------------------------------
@dataclass
class Config:
    total_env_steps: int = 80_000
    start_random_steps: int = 1_000
    update_after: int = 1_000
    update_every: int = 50
    updates_per_step: int = 1

    buffer_size: int = 200_000
    batch_size: int = 256

    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 1e-3
    critic_lr: float = 1e-3

    hidden_actor: tuple = (256,256)
    hidden_critic: tuple = (256,256)

    device: str = "cpu"
    log_every: int = 2_000
    eval_episodes: int = 5

    # Exploration
    ou_theta: float = 0.15
    ou_sigma: float = 0.2
    noise_clip: float = 0.5         # clip exploration noise
    noise_decay: float = 1.0        # keep 1.0 or <1.0 to anneal per 10k steps

    # Video & plot
    video_episodes: int = 3
    video_dir: str = "videos_ddpg_pendulum"
    plot_path: str = "ddpg_pendulum_learning_curve.png"

# ----------------- Eval / Plot / Video ------------------
@torch.no_grad()
def evaluate_avg_return(env, actor, episodes=5):
    scores = []
    for _ in range(episodes):
        s, _ = env.reset() if GYMN else env.reset()
        done = False; ep_ret = 0.0
        while not done:
            a = actor.act(s)
            step_out = env.step(a)
            if GYMN:
                s, r, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                s, r, done, _ = step_out
            ep_ret += r
        scores.append(ep_ret)
    return float(np.mean(scores))

def record_videos(actor, cfg):
    env_v = make_video_env(cfg.video_dir, seed=SEED+123)
    try:
        for _ in range(cfg.video_episodes):
            s, _ = env_v.reset() if GYMN else env_v.reset()
            done = False
            while not done:
                a = actor.act(s)
                step_out = env_v.step(a)
                if GYMN:
                    s, r, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    s, r, done, _ = step_out
        print(f"[Video] Saved evaluation rollouts to: {os.path.abspath(cfg.video_dir)}")
    finally:
        env_v.close()

def plot_learning_curve(returns, cfg):
    if not returns:
        print("[Plot] No returns to plot.")
        return
    plt.figure(figsize=(7,4.5))
    xs = np.arange(1, len(returns)+1)
    plt.plot(xs, returns, linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DDPG on Pendulum-v1 — Learning Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(cfg.plot_path, dpi=160)
    try:
        plt.show()
    except Exception:
        pass
    print(f"[Plot] Saved learning curve to: {os.path.abspath(cfg.plot_path)}")

# ----------------- Training Loop ------------------------
def train_ddpg_pendulum(cfg=Config()):
    device = torch.device(cfg.device)
    env = make_env(SEED)
    obs_dim = env.observation_space.shape[0]     # 3
    act_dim = env.action_space.shape[0]          # 1
    act_low, act_high = env.action_space.low, env.action_space.high

    # Modules
    actor = Actor(obs_dim, act_dim, act_low, act_high, cfg.hidden_actor).to(device)
    actor_targ = Actor(obs_dim, act_dim, act_low, act_high, cfg.hidden_actor).to(device)
    actor_targ.load_state_dict(actor.state_dict())

    critic = Critic(obs_dim, act_dim, cfg.hidden_critic).to(device)
    critic_targ = Critic(obs_dim, act_dim, cfg.hidden_critic).to(device)
    critic_targ.load_state_dict(critic.state_dict())

    pi_optim = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    q_optim  = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    # Buffer & Noise
    rb = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size)
    ou = OUNoise(size=act_dim, theta=cfg.ou_theta, sigma=cfg.ou_sigma)

    # Run
    s, _ = env.reset() if GYMN else env.reset()
    ep_ret, ep_len, episodes = 0.0, 0, 0
    returns = []
    noise_scale = 1.0

    for t in range(1, cfg.total_env_steps + 1):
        # ----- Action with exploration noise -----
        if t < cfg.start_random_steps:
            a = env.action_space.sample()
        else:
            a = actor.act(s)
            noise = ou.sample() * noise_scale
            noise = np.clip(noise, -cfg.noise_clip, cfg.noise_clip)
            a = np.clip(a + noise, act_low, act_high)

        # ----- Step env -----
        step_out = env.step(a)
        if GYMN:
            s2, r, terminated, truncated, _ = step_out
            done = terminated or truncated
            d_mask = float(done)
        else:
            s2, r, done, _ = step_out
            d_mask = float(done)

        rb.push(s, a, r, s2, d_mask)
        ep_ret += r; ep_len += 1
        s = s2

        if done:
            episodes += 1
            returns.append(ep_ret)
            s, _ = env.reset() if GYMN else env.reset()
            ep_ret, ep_len = 0.0, 0
            ou.reset()
            # optional anneal of exploration noise per 10k steps
            if t % 10_000 == 0:
                noise_scale *= cfg.noise_decay

        # ----- Learn -----
        if t >= cfg.update_after and t % cfg.update_every == 0:
            for _ in range(cfg.updates_per_step * cfg.update_every):
                if rb.size < cfg.batch_size: break
                bs, ba, br, bs2, bd = rb.sample(cfg.batch_size, device)

                # --- Critic target ---
                with torch.no_grad():
                    a2 = actor_targ(bs2)
                    q_targ = critic_targ(bs2, a2)
                    y = br + cfg.gamma * (1.0 - bd) * q_targ

                # --- Critic update ---
                q = critic(bs, ba)
                q_loss = F.mse_loss(q, y)
                q_optim.zero_grad()
                q_loss.backward()
                q_optim.step()

                # --- Actor update (DPG ascent) ---
                # maximize Q(s, μ(s))  ⇔ minimize -Q
                a_pi = actor(bs)
                pi_loss = -critic(bs, a_pi).mean()
                pi_optim.zero_grad()
                pi_loss.backward()
                pi_optim.step()

                # --- Target networks (Polyak) ---
                soft_update(actor, actor_targ, cfg.tau)
                soft_update(critic, critic_targ, cfg.tau)

        # ----- Logging -----
        if t % cfg.log_every == 0:
            avg_recent = np.mean(returns[-10:]) if len(returns) >= 10 else (np.mean(returns) if returns else 0.0)
            eval_env = make_env(SEED+7)
            eval_score = evaluate_avg_return(eval_env, actor, cfg.eval_episodes) if len(returns) > 5 else float('nan')
            eval_env.close()
            print(f"[t={t}] episodes={episodes} recent10={avg_recent:.1f} eval_avg={eval_score:.1f} "
                  f"rb={rb.size} noise_scale={noise_scale:.2f}")

    env.close()
    print(f"Training done. Episodes: {episodes}. "
          f"Final 10-avg return: {(np.mean(returns[-10:]) if len(returns)>=10 else np.mean(returns)):.1f}")

    # --------- (1) Plot learning curve ----------
    plot_learning_curve(returns, cfg)

    # --------- (2) Record evaluation videos ----------
    record_videos(actor, cfg)

    return returns, actor, critic

if __name__ == "__main__":
    _ = train_ddpg_pendulum(Config(
        total_env_steps=50_000,
        start_random_steps=1_000,
        update_after=1_000,
        update_every=50,
        updates_per_step=1,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-3,
        critic_lr=1e-3,
        device="cpu",
        log_every=2_000,
        eval_episodes=5,
        # OU noise params
        ou_theta=0.15,
        ou_sigma=0.2,
        noise_clip=0.5,
        noise_decay=1.0,  # keep at 1.0; set <1.0 to anneal
        # outputs
        video_episodes=3,
        video_dir="videos_ddpg_pendulum",
        plot_path="ddpg_pendulum_learning_curve.png"
    ))
