# MBPO (Model-Based Policy Optimization) with SAC on Pendulum-v1
# --------------------------------------------------------------
# Features:
#   • Dynamics ENSEMBLE learning Δs = s' - s
#   • Short model rollouts seeded from REAL states
#   • Off-policy learner = SAC (twin critics + twin targets, auto-α)
#   • Plots per-episode return (learning curve)
#   • Saves evaluation rollouts as videos

import os, random, math
from dataclasses import dataclass
from collections import deque
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
                              name_prefix="mbpo_sac_pendulum")
    except Exception:
        if hasattr(gym.wrappers, "RecordVideo"):
            env = gym.wrappers.RecordVideo(env, video_folder=video_dir,
                                           episode_trigger=lambda ep: True)
        else:
            env = gym.wrappers.Monitor(env, video_dir, force=True)
    return env

# ----------------- Replay Buffer ------------------------
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.nxt = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.ptr = 0; self.size = 0; self.max_size = size
        self.device = device

    def push(self, s, a, r, s2, d):
        self.obs[self.ptr] = s
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.nxt[self.ptr] = s2
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x, dtype=torch.float32: torch.as_tensor(x[idxs], dtype=dtype, device=self.device)
        return to_t(self.obs), to_t(self.act), to_t(self.rew), to_t(self.nxt), to_t(self.done)

# ----------------- Nets: MLP helper ---------------------
def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for j in range(len(sizes)-1):
        layers += [nn.Linear(sizes[j], sizes[j+1])]
        if j < len(sizes)-2:
            layers += [act()]
        elif out_act is not None:
            layers += [out_act()]
    return nn.Sequential(*layers)

# ----------------- SAC Components -----------------------
class Actor(nn.Module):
    """Tanh-squashed Gaussian; returns rsample action + log_prob with tanh correction."""
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
        d = h.shape[-1] // 2
        mu, log_std = h[..., :d], h[..., d:]
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mu, std
    def rsample(self, obs):
        mu, std = self._dist_params(obs)
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        a_tanh = torch.tanh(u)
        a = self.act_scale * a_tanh + self.act_bias
        logp_u = dist.log_prob(u).sum(dim=-1, keepdim=True)
        log_det = torch.log(1 - a_tanh.pow(2) + self.eps).sum(dim=-1, keepdim=True)
        logp = logp_u - log_det
        return a, logp
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mu, std = self._dist_params(obs_t)
        if deterministic:
            u = mu
        else:
            u = torch.distributions.Normal(mu, std).sample()
        a = torch.tanh(u) * self.act_scale + self.act_bias
        return a.squeeze(0).cpu().numpy()

class CriticTwin(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, *hidden, 1])
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)

def soft_update(src, tgt, tau):
    with torch.no_grad():
        for p, p_t in zip(src.parameters(), tgt.parameters()):
            p_t.data.mul_(1 - tau)
            p_t.data.add_(tau * p.data)

# ----------------- Dynamics Ensemble --------------------
class DynModel(nn.Module):
    """Predict Δs = s' - s."""
    def __init__(self, obs_dim, act_dim, hidden=(200,200)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, obs_dim])
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))

class Ensemble:
    def __init__(self, n_models, obs_dim, act_dim, device):
        self.members = [DynModel(obs_dim, act_dim).to(device) for _ in range(n_models)]
        self.optims  = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in self.members]
        self.device = device
        self.n = n_models
    def train_step(self, s, a, s2):
        target = s2 - s
        losses = []
        for m, opt in zip(self.members, self.optims):
            opt.zero_grad()
            pred = m(s, a)
            loss = F.mse_loss(pred, target)
            loss.backward(); opt.step()
            losses.append(loss.item())
        return float(np.mean(losses))
    @torch.no_grad()
    def step(self, s, a, sigma=0.0):
        m = random.choice(self.members)
        delta = m(s, a)
        if sigma > 0:
            delta = delta + sigma * torch.randn_like(delta)
        return s + delta

# ----------------- Pendulum Reward ----------------------
def pendulum_reward(state, action):
    # state = [cos(theta), sin(theta), theta_dot]; action shape (...,1)
    cos_th, sin_th, thdot = float(state[0]), float(state[1]), float(state[2])
    theta = math.atan2(sin_th, cos_th)
    u = float(np.clip(action, -2.0, 2.0))
    cost = theta**2 + 0.1 * thdot**2 + 0.001 * (u**2)
    return -cost

# ----------------- Config -------------------------------
@dataclass
class Config:
    # Environment & buffers
    total_env_steps: int = 50_000
    start_random_steps: int = 1_000
    update_after: int = 1_000
    update_every: int = 50
    real_buffer_size: int = 200_000
    model_buffer_size: int = 500_000
    batch_size_modelmix: int = 256
    model_ratio: float = 0.9  # fraction of model samples in mixed minibatch

    # SAC
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    auto_alpha: bool = True
    init_alpha: float = 0.2
    hidden_actor: tuple = (256,256)
    hidden_critic: tuple = (256,256)

    # Dynamics ensemble & rollouts
    ensemble_size: int = 5
    model_train_iters: int = 200
    rollout_horizon: int = 1       # keep short (1–5)
    rollout_seeds: int = 1000      # how many seed states per generation burst
    rollout_noise_sigma: float = 0.0

    # Training schedule
    sac_updates_per_step: float = 1.0

    # Device & logging
    device: str = "cpu"
    log_every: int = 2_000
    eval_episodes: int = 5

    # Outputs
    video_episodes: int = 3
    video_dir: str = "videos_mbpo_sac_pendulum"
    plot_path: str = "mbpo_sac_pendulum_learning_curve.png"

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
    # Gymnasium returns float reward; sum is float
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
    plt.xlabel("Episode"); plt.ylabel("Return")
    plt.title("MBPO (SAC) on Pendulum-v1 — Learning Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(cfg.plot_path, dpi=160)
    try:
        plt.show()
    except Exception:
        pass
    print(f"[Plot] Saved learning curve to: {os.path.abspath(cfg.plot_path)}")

# ----------------- MBPO + SAC Training ------------------
def train_mbpo_sac_pendulum(cfg=Config()):
    device = torch.device(cfg.device)
    env = make_env(SEED)
    obs_dim = env.observation_space.shape[0]     # 3
    act_dim = env.action_space.shape[0]          # 1
    act_low, act_high = env.action_space.low, env.action_space.high

    # Buffers
    rb_real  = ReplayBuffer(obs_dim, act_dim, cfg.real_buffer_size, device)
    rb_model = ReplayBuffer(obs_dim, act_dim, cfg.model_buffer_size, device)

    # SAC modules
    actor = Actor(obs_dim, act_dim, act_low, act_high, cfg.hidden_actor).to(device)
    critic = CriticTwin(obs_dim, act_dim, cfg.hidden_critic).to(device)
    critic_targ = CriticTwin(obs_dim, act_dim, cfg.hidden_critic).to(device)
    critic_targ.load_state_dict(critic.state_dict())

    pi_optim = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    q_optim  = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    # Temperature (auto)
    if cfg.auto_alpha:
        target_entropy = -float(act_dim)  # ≈ −dim(A)
        log_alpha = torch.tensor(math.log(cfg.init_alpha), requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)
        alpha_fn = lambda: log_alpha.exp()
    else:
        log_alpha = None
        alpha_fn = lambda: torch.tensor(cfg.init_alpha, device=device)

    # Dynamics ensemble
    ensemble = Ensemble(cfg.ensemble_size, obs_dim, act_dim, device)

    # Training vars
    s, _ = env.reset() if GYMN else env.reset()
    ep_ret, ep_len, episodes = 0.0, 0, 0
    returns = []

    for t in range(1, cfg.total_env_steps + 1):
        # ---- Action from current policy (stochastic to explore) ----
        if t < cfg.start_random_steps:
            a = env.action_space.sample()
        else:
            a = actor.act(s, deterministic=False)

        # ---- Step environment ----
        step_out = env.step(a)
        if GYMN:
            s2, r, terminated, truncated, _ = step_out
            done = terminated or truncated
            d_mask = float(done)
        else:
            s2, r, done, _ = step_out
            d_mask = float(done)

        rb_real.push(s, a, r, s2, d_mask)
        ep_ret += r; ep_len += 1
        s = s2

        if done:
            episodes += 1
            returns.append(ep_ret)
            s, _ = env.reset() if GYMN else env.reset()
            ep_ret, ep_len = 0.0, 0

        # ---- Training phases ----
        if t >= cfg.update_after and t % cfg.update_every == 0:
            # 1) Train dynamics on real buffer
            for _ in range(cfg.model_train_iters):
                if rb_real.size < cfg.batch_size_modelmix: break
                s_b, a_b, _r_b, s2_b, _d_b = rb_real.sample(cfg.batch_size_modelmix)
                ensemble.train_step(s_b, a_b, s2_b)

            # 2) Generate short model rollouts seeded from recent real states
            if rb_real.size > cfg.rollout_seeds:
                idxs = np.random.randint(max(0, rb_real.size - 5000), rb_real.size, size=cfg.rollout_seeds)
                seed_states = torch.as_tensor(rb_real.obs[idxs], dtype=torch.float32, device=device)
                s_m = seed_states.clone()
                for _h in range(cfg.rollout_horizon):
                    with torch.no_grad():
                        a_m, _logp_m = actor.rsample(s_m)
                        s2_m = ensemble.step(s_m, a_m, sigma=cfg.rollout_noise_sigma)
                        # Analytic reward for Pendulum
                        r_list = []
                        for i in range(s_m.shape[0]):
                            r_i = pendulum_reward(s_m[i].cpu().numpy(), a_m[i].cpu().numpy())
                            r_list.append([r_i])
                        r_m = torch.tensor(r_list, dtype=torch.float32, device=device)
                        d_m = torch.zeros_like(r_m)  # short model rollouts: no terminal flags
                    # push batch into model buffer
                    for i in range(s_m.shape[0]):
                        rb_model.push(s_m[i].cpu().numpy(),
                                      a_m[i].cpu().numpy(),
                                      r_m[i].cpu().numpy(),
                                      s2_m[i].cpu().numpy(),
                                      d_m[i].cpu().numpy())
                    s_m = s2_m

            # 3) Off-policy SAC updates with mixed batches
            total_updates = int(cfg.sac_updates_per_step * cfg.update_every)
            for _ in range(total_updates):
                if rb_real.size < cfg.batch_size_modelmix: continue

                bm = int(cfg.model_ratio * cfg.batch_size_modelmix)
                br = cfg.batch_size_modelmix - bm
                # Fall back to real if model buffer is small
                if rb_model.size < bm:
                    bm = 0
                    br = cfg.batch_size_modelmix

                if br > 0:
                    s_r, a_r, r_r, s2_r, d_r = rb_real.sample(br)
                if bm > 0:
                    s_mi, a_mi, r_mi, s2_mi, d_mi = rb_model.sample(bm)

                if bm > 0 and br > 0:
                    s_b  = torch.cat([s_r,  s_mi],  dim=0)
                    a_b  = torch.cat([a_r,  a_mi],  dim=0)
                    r_b  = torch.cat([r_r,  r_mi],  dim=0)
                    s2_b = torch.cat([s2_r, s2_mi], dim=0)
                    d_b  = torch.cat([d_r,  d_mi],  dim=0)
                elif bm > 0:
                    s_b, a_b, r_b, s2_b, d_b = s_mi, a_mi, r_mi, s2_mi, d_mi
                else:
                    s_b, a_b, r_b, s2_b, d_b = s_r, a_r, r_r, s2_r, d_r

                # Critic target
                with torch.no_grad():
                    a2, logp2 = actor.rsample(s2_b)
                    q1_t, q2_t = critic_targ(s2_b, a2)
                    q_targ = torch.min(q1_t, q2_t) - alpha_fn() * logp2
                    y = r_b + (1.0 - d_b) * cfg.gamma * q_targ

                # Critic update
                q1, q2 = critic(s_b, a_b)
                q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                q_optim.zero_grad()
                q_loss.backward()
                q_optim.step()

                # Actor update
                a_new, logp = actor.rsample(s_b)
                q1_pi, q2_pi = critic(s_b, a_new)
                q_pi = torch.min(q1_pi, q2_pi)
                pi_loss = (alpha_fn() * logp - q_pi).mean()
                pi_optim.zero_grad()
                pi_loss.backward()
                pi_optim.step()

                # Temperature
                if cfg.auto_alpha:
                    alpha_loss = -(log_alpha * (logp.detach() + (-float(act_dim)))).mean()
                    alpha_optim.zero_grad()
                    alpha_loss.backward()
                    alpha_optim.step()

                # Target nets
                soft_update(critic, critic_targ, cfg.tau)

        # ---- Logging ----
        if t % cfg.log_every == 0:
            avg_recent = np.mean(returns[-10:]) if len(returns) >= 10 else (np.mean(returns) if returns else 0.0)
            eval_env = make_env(SEED+7)
            eval_score = evaluate_avg_return(eval_env, actor, cfg.eval_episodes, deterministic=True) \
                         if len(returns) > 5 else float('nan')
            eval_env.close()
            alpha_val = float(alpha_fn().item())
            print(f"[t={t}] episodes={len(returns)} recent10={avg_recent:.1f} eval_avg={eval_score:.1f} "
                  f"real_buf={rb_real.size} model_buf={rb_model.size} alpha={alpha_val:.3f}")

    env.close()
    print(f"Training done. Episodes: {len(returns)}. "
          f"Final 10-avg return: {(np.mean(returns[-10:]) if len(returns)>=10 else np.mean(returns)):.1f}")

    # ---- Plot learning curve ----
    plot_learning_curve(returns, cfg)

    # ---- Record videos ----
    record_videos(actor, cfg)

    return returns, actor, critic

if __name__ == "__main__":
    _ = train_mbpo_sac_pendulum(Config(
        total_env_steps=50_000,
        start_random_steps=1_000,
        update_after=1_000,
        update_every=50,
        real_buffer_size=200_000,
        model_buffer_size=500_000,
        batch_size_modelmix=256,
        model_ratio=0.9,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        auto_alpha=True,
        init_alpha=0.2,
        hidden_actor=(256,256),
        hidden_critic=(256,256),
        ensemble_size=5,
        model_train_iters=200,
        rollout_horizon=2,      # try 1–3; increase when model is accurate
        rollout_seeds=1000,
        rollout_noise_sigma=0.0,
        sac_updates_per_step=1.0,
        device="cpu",
        log_every=2_000,
        eval_episodes=5,
        video_episodes=3,
        video_dir="videos_mbpo_sac_pendulum",
        plot_path="mbpo_sac_pendulum_learning_curve.png"
    ))
