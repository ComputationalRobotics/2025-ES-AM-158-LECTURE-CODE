# SAC (Continuous) on Pendulum-v1 with Twin Critics & Auto-Entropy
# ---------------------------------------------------------------
# Features:
#   • Reparameterized tanh-Gaussian policy (with correct tanh log-Jacobian)
#   • Twin critics + twin target critics (min backup)
#   • Optional automatic temperature tuning (default on)
#   • Plots per-episode return (learning curve)
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
                              name_prefix="sac_continuous")
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
    """Tanh-squashed Gaussian with reparameterization; returns action and log_prob (with tanh correction)."""
    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden=(256,256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, 2*act_dim])
        self.register_buffer("act_low",  torch.as_tensor(act_low,  dtype=torch.float32))
        self.register_buffer("act_high", torch.as_tensor(act_high, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(1e-6, dtype=torch.float32))
        self._update_scale_bias()

    def _update_scale_bias(self):
        self.act_scale = (self.act_high - self.act_low) / 2.0
        self.act_bias  = (self.act_high + self.act_low) / 2.0

    def _dist_params(self, obs):
        h = self.net(obs)
        act_dim = h.shape[-1] // 2
        mu, log_std = h[..., :act_dim], h[..., act_dim:]
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mu, std

    def rsample(self, obs):
        mu, std = self._dist_params(obs)
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        a_tanh = torch.tanh(u)
        a = self.act_scale * a_tanh + self.act_bias
        # log-prob with tanh correction (drop affine constant)
        logp_u = dist.log_prob(u).sum(dim=-1, keepdim=True)
        log_det = torch.log(1 - a_tanh.pow(2) + self.eps).sum(dim=-1, keepdim=True)
        logp = logp_u - log_det
        return a, logp, a_tanh, u

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mu, std = self._dist_params(obs_t)
        if deterministic:
            u = mu
        else:
            u = torch.distributions.Normal(mu, std).sample()
        a_tanh = torch.tanh(u)
        a = self.act_scale * a_tanh + self.act_bias
        return a.squeeze(0).cpu().numpy()

class CriticTwin(nn.Module):
    """Twin Q networks; input is state-action concatenation."""
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, *hidden, 1])
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)

def soft_update(net_src, net_tgt, tau):
    with torch.no_grad():
        for p, p_t in zip(net_src.parameters(), net_tgt.parameters()):
            p_t.data.mul_(1 - tau)
            p_t.data.add_(tau * p.data)

# ----------------- Config -------------------------------
@dataclass
class Config:
    total_env_steps: int = 50_000
    start_random_steps: int = 1_000
    update_after: int = 1_000
    update_every: int = 50
    updates_per_step: int = 1

    buffer_size: int = 200_000
    batch_size: int = 256

    gamma: float = 0.99
    tau: float = 0.005

    auto_alpha: bool = True
    alpha: float = 0.2           # used if auto_alpha=False
    target_entropy_scale: float = 1.0  # multiply by (-act_dim)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    hidden_actor: tuple = (256,256)
    hidden_critic: tuple = (256,256)

    device: str = "cpu"
    log_every: int = 2_000
    eval_episodes: int = 5
    video_episodes: int = 3
    video_dir: str = "videos_sac_pendulum"
    plot_path: str = "sac_continuous_pendulum_learning_curve.png"

# ----------------- Eval / Plot / Video ------------------
@torch.no_grad()
def evaluate_avg_return(env, actor, episodes=5, deterministic=True):
    scores = []
    for _ in range(episodes):
        s, _ = env.reset() if GYMN else env.reset()
        done = False; ep_ret = 0.0
        while not done:
            a = actor.act(s, deterministic=deterministic)
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
                a = actor.act(s, deterministic=True)
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
    plt.title("SAC (Continuous, Twin Critics) on Pendulum-v1 — Learning Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(cfg.plot_path, dpi=160)
    try:
        plt.show()
    except Exception:
        pass
    print(f"[Plot] Saved learning curve to: {os.path.abspath(cfg.plot_path)}")

# ----------------- Training Loop ------------------------
def train_sac_pendulum(cfg=Config()):
    device = torch.device(cfg.device)
    env = make_env(SEED)
    obs_dim = env.observation_space.shape[0]     # Pendulum: 3
    act_dim = env.action_space.shape[0]          # Pendulum: 1
    act_low, act_high = env.action_space.low, env.action_space.high

    # Modules
    actor = Actor(obs_dim, act_dim, act_low, act_high, cfg.hidden_actor).to(device)
    critic = CriticTwin(obs_dim, act_dim, cfg.hidden_critic).to(device)
    critic_targ = CriticTwin(obs_dim, act_dim, cfg.hidden_critic).to(device)
    critic_targ.load_state_dict(critic.state_dict())

    pi_optim = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    q_optim  = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    # Temperature
    if cfg.auto_alpha:
        target_entropy = - cfg.target_entropy_scale * float(act_dim)  # ≈ −dim(A)
        log_alpha = torch.tensor(math.log(cfg.alpha), requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)
        alpha = lambda: log_alpha.exp()
    else:
        target_entropy = None
        alpha = lambda: torch.tensor(cfg.alpha, device=device)

    # Buffer
    rb = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size)

    # Run
    s, _ = env.reset() if GYMN else env.reset()
    ep_ret, ep_len = 0.0, 0
    returns = []; episodes = 0

    for t in range(1, cfg.total_env_steps + 1):
        # ----- Action -----
        if t < cfg.start_random_steps:
            a = env.action_space.sample()
        else:
            a = actor.act(s, deterministic=False)

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

        # ----- Learn -----
        if t >= cfg.update_after and t % cfg.update_every == 0:
            for _ in range(cfg.updates_per_step * cfg.update_every):
                if rb.size < cfg.batch_size:
                    break
                bs, ba, br, bs2, bd = rb.sample(cfg.batch_size, device)

                # --- Target computation ---
                with torch.no_grad():
                    a2, logp2, _, _ = actor.rsample(bs2)
                    q1_t, q2_t = critic_targ(bs2, a2)
                    q_min_t = torch.min(q1_t, q2_t)
                    y = br + cfg.gamma * (1.0 - bd) * (q_min_t - alpha() * logp2)

                # --- Critic update ---
                q1, q2 = critic(bs, ba)
                q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                q_optim.zero_grad()
                q_loss.backward()
                q_optim.step()

                # --- Actor update (reparameterized) ---
                a_new, logp, _, _ = actor.rsample(bs)
                q1_pi, q2_pi = critic(bs, a_new)
                q_pi = torch.min(q1_pi, q2_pi)
                pi_loss = (alpha() * logp - q_pi).mean()
                pi_optim.zero_grad()
                pi_loss.backward()
                pi_optim.step()

                # --- Temperature (optional) ---
                if cfg.auto_alpha:
                    alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
                    alpha_optim.zero_grad()
                    alpha_loss.backward()
                    alpha_optim.step()

                # --- Target networks ---
                soft_update(critic, critic_targ, cfg.tau)

        # ----- Logging -----
        if t % cfg.log_every == 0:
            avg_recent = np.mean(returns[-10:]) if len(returns) >= 10 else (np.mean(returns) if returns else 0.0)
            eval_env = make_env(SEED+7)
            eval_score = evaluate_avg_return(eval_env, actor, cfg.eval_episodes, deterministic=True) \
                         if len(returns) > 5 else float('nan')
            eval_env.close()
            print(f"[t={t}] episodes={episodes} recent10={avg_recent:.1f} eval_avg={eval_score:.1f} "
                  f"rb={rb.size} alpha={alpha().item():.4f}")

    env.close()
    print(f"Training done. Episodes: {episodes}. "
          f"Final 10-avg return: {(np.mean(returns[-10:]) if len(returns)>=10 else np.mean(returns)):.1f}")

    # --------- (1) Plot learning curve ----------
    plot_learning_curve(returns, cfg)

    # --------- (2) Record evaluation videos ----------
    record_videos(actor, cfg)

    return returns, actor, critic

if __name__ == "__main__":
    _ = train_sac_pendulum(Config(
        total_env_steps=50_000,
        start_random_steps=1_000,
        update_after=1_000,
        update_every=50,
        updates_per_step=1,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        auto_alpha=True,     # set False to use fixed alpha below
        alpha=0.2,
        target_entropy_scale=1.0,  # target_entropy = -scale * act_dim
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        device="cpu",
        log_every=2_000,
        eval_episodes=5,
        video_episodes=3,
        video_dir="videos_sac_pendulum",
        plot_path="sac_continuous_pendulum_learning_curve.png"
    ))
