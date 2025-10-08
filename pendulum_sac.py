# Soft Actor–Critic (SAC) on Pendulum-v1 — twin critics, target nets, auto-α, replay, video
# ------------------------------------------------------------------------------------------
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
        if hasattr(env, "seed"):
            env.seed(seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return env

# =========================
#    Tanh-Gaussian utils
# =========================
LOG_2PI = math.log(2.0 * math.pi)

def atanh(x):
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def gaussian_log_prob(u, mean, log_std):
    var = torch.exp(2.0 * log_std)
    return -0.5 * (((u - mean) ** 2) / var + 2.0 * log_std + LOG_2PI).sum(dim=-1)  # (B,)

def sample_squashed_gaussian(mean, log_std, act_scale):
    """
    Reparameterized sample a = scale * tanh(u), u ~ N(mean, std)
    Returns: a (B,A), logp(a|s) (B,)
    """
    std = torch.exp(log_std)
    eps = torch.randn_like(mean)
    u = mean + std * eps
    a = act_scale * torch.tanh(u)
    # log prob of squashed sample
    base_logp = gaussian_log_prob(u, mean, log_std)  # (B,)
    log_det = (torch.log(act_scale) + torch.log(1.0 - torch.tanh(u) ** 2 + 1e-6)).sum(dim=-1)
    logp = base_logp - log_det
    return a, logp

# =========================
#         Networks
# =========================
class Policy(nn.Module):
    """Tanh-squashed Gaussian policy."""
    def __init__(self, obs_dim, act_dim, hidden=256, log_std_min=-5.0, log_std_max=2.0, act_scale=1.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.register_buffer('act_scale', torch.as_tensor(act_scale, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, s):
        h = self.net(s)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    @torch.no_grad()
    def act(self, s, deterministic=False):
        mu, log_std = self.forward(s)
        if deterministic:
            a = self.act_scale * torch.tanh(mu)
            return a
        a, _ = sample_squashed_gaussian(mu, log_std, self.act_scale)
        return a

class QNet(nn.Module):
    """Q(s,a) network."""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)  # (B,)

# =========================
#      Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = int(capacity)
        self.idx = 0
        self.full = False
        self.s  = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.a  = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.r  = np.zeros((self.capacity,), dtype=np.float32)
        self.s2 = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.d  = np.zeros((self.capacity,), dtype=np.float32)
    def push(self, s, a, r, s2, d):
        self.s[self.idx]  = s
        self.a[self.idx]  = a
        self.r[self.idx]  = r
        self.s2[self.idx] = s2
        self.d[self.idx]  = d
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or (self.idx == 0)
    def __len__(self): return self.capacity if self.full else self.idx
    def sample(self, batch_size):
        n = len(self)
        idxs = np.random.randint(0, n, size=batch_size)
        return self.s[idxs], self.a[idxs], self.r[idxs], self.s2[idxs], self.d[idxs]

# =========================
#      Plotting / Video
# =========================
def plot_training_curve(returns_hist, window=50, out_path="videos/pendulum_sac_returns.png"):
    x = np.arange(1, len(returns_hist) + 1)
    rets = np.array(returns_hist, dtype=float)
    ma = np.convolve(rets, np.ones(window)/window, mode="valid") if len(rets) >= window else None
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, rets, label="Return per episode")
    if ma is not None:
        plt.plot(np.arange(window, len(rets)+1), ma, label=f"{window}-episode average")
    plt.xlabel("Episode"); plt.ylabel("Return")
    plt.title("SAC on Pendulum-v1 — Training Curve")
    plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[Plot] Saved training curve to {out_path}")

def rollout_and_record_video(policy, video_dir="videos", episodes=1, max_steps=200, deterministic=True):
    os.makedirs(video_dir, exist_ok=True)
    if GYMN:
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda e: True)
    else:
        env = gym.make("Pendulum-v1")
        env = gym.wrappers.Monitor(env, video_dir, force=True, video_callable=lambda e: True)
    dev = next(policy.parameters()).device
    total_returns = []
    for ep in range(episodes):
        if GYMN:
            obs, info = env.reset()
        else:
            obs = env.reset()
        done, steps, ep_ret = False, 0, 0.0
        while not done and steps < max_steps:
            s_t = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            with torch.no_grad():
                a = policy.act(s_t, deterministic=deterministic).squeeze(0).cpu().numpy()
            if GYMN:
                obs, r, terminated, truncated, _ = env.step(a.astype(np.float32))
                done = terminated or truncated
            else:
                obs, r, done, _ = env.step(a.astype(np.float32))
            ep_ret += float(r); steps += 1
        total_returns.append(ep_ret)
        print(f"[Video] Episode {ep+1}/{episodes} return: {ep_ret:.1f}")
    env.close()
    print(f"[Video] Saved to: {os.path.abspath(video_dir)}")
    return total_returns

# =========================
#          SAC
# =========================
def soft_update(target, source, tau=0.005):
    with torch.no_grad():
        for p_t, p in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1 - tau).add_(tau * p.data)

def train_sac_pendulum(
    episodes=800,
    gamma=0.99,
    tau=0.005,
    lr=3e-4,
    buffer_capacity=300_000,
    batch_size=256,
    start_steps=2_000,          # initial random exploration steps
    updates_per_step=1,         # gradient steps per env step after warmup
    critic_steps_per_update=1,  # extra critic steps (optional)
    target_entropy_scale=1.0,   # target entropy = -scale * act_dim
    device=None
):
    """
    SAC with:
      - Tanh-Gaussian actor (reparameterized)
      - Twin critics + target critics
      - Automatic temperature α tuning toward target entropy = -scale * act_dim
    """
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])  # Pendulum: 2.0
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Networks
    actor = Policy(obs_dim, act_dim, hidden=256, act_scale=act_scale).to(device)
    q1 = QNet(obs_dim, act_dim, hidden=256).to(device)
    q2 = QNet(obs_dim, act_dim, hidden=256).to(device)
    q1_targ = QNet(obs_dim, act_dim, hidden=256).to(device)
    q2_targ = QNet(obs_dim, act_dim, hidden=256).to(device)
    q1_targ.load_state_dict(q1.state_dict())
    q2_targ.load_state_dict(q2.state_dict())

    # Optimizers
    opt_actor = optim.Adam(actor.parameters(), lr=lr)
    opt_q1 = optim.Adam(q1.parameters(), lr=lr)
    opt_q2 = optim.Adam(q2.parameters(), lr=lr)

    # Temperature (auto-tuning)
    target_entropy = - target_entropy_scale * act_dim
    log_alpha = torch.tensor(0.0, requires_grad=True, device=device)  # α=1 init
    opt_alpha = optim.Adam([log_alpha], lr=lr)

    # Replay
    buf = ReplayBuffer(buffer_capacity, obs_dim, act_dim)
    to_t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)

    returns_hist = []
    best_avg = -1e9
    total_steps = 0
    last_actor_loss = None
    last_critic_loss = None
    last_alpha = float(torch.exp(log_alpha).item())

    for ep in range(1, episodes + 1):
        if GYMN:
            obs, info = env.reset(seed=SEED + ep)
        else:
            if hasattr(env, "seed"): env.seed(SEED + ep)
            obs = env.reset()
        done, ep_ret = False, 0.0

        while not done:
            # ----- Action selection -----
            if total_steps < start_steps:
                a = env.action_space.sample().astype(np.float32)  # pure random
            else:
                s_t = to_t(obs).unsqueeze(0)
                with torch.no_grad():
                    a = actor.act(s_t, deterministic=False).squeeze(0).cpu().numpy().astype(np.float32)

            # ----- Step env -----
            if GYMN:
                nobs, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
            else:
                nobs, r, done, _ = env.step(a)
            buf.push(obs, a, float(r), nobs, float(done))
            obs = nobs
            ep_ret += float(r)
            total_steps += 1

            # ----- Updates -----
            if len(buf) >= max(batch_size, start_steps):
                for _ in range(updates_per_step):
                    # multiple critic-only steps if requested
                    for _c in range(int(critic_steps_per_update)):
                        s, a_b, r_b, s2, d_b = buf.sample(batch_size)
                        S  = to_t(s)       # (B,obs)
                        A  = to_t(a_b)     # (B,act)
                        R  = to_t(r_b)     # (B,)
                        S2 = to_t(s2)      # (B,obs)
                        D  = to_t(d_b)     # (B,)

                        with torch.no_grad():
                            mu2, log_std2 = actor(S2)
                            a2, logp2 = sample_squashed_gaussian(mu2, log_std2, actor.act_scale)  # (B,A),(B,)
                            q1_t = q1_targ(S2, a2)
                            q2_t = q2_targ(S2, a2)
                            q_t_min = torch.min(q1_t, q2_t)
                            alpha = torch.exp(log_alpha)
                            y = R + gamma * (1.0 - D) * (q_t_min - alpha * logp2)  # (B,)

                        q1_pred = q1(S, A)
                        q2_pred = q2(S, A)
                        loss_q1 = (q1_pred - y).pow(2).mean()
                        loss_q2 = (q2_pred - y).pow(2).mean()

                        opt_q1.zero_grad(set_to_none=True)
                        loss_q1.backward()
                        torch.nn.utils.clip_grad_norm_(q1.parameters(), 1.0)
                        opt_q1.step()

                        opt_q2.zero_grad(set_to_none=True)
                        loss_q2.backward()
                        torch.nn.utils.clip_grad_norm_(q2.parameters(), 1.0)
                        opt_q2.step()

                        last_critic_loss = float(0.5 * (loss_q1.item() + loss_q2.item()))

                    # ---- Actor & α updates (one step) ----
                    s, a_b, r_b, s2, d_b = buf.sample(batch_size)
                    S = to_t(s)

                    mu, log_std = actor(S)
                    a_new, logp = sample_squashed_gaussian(mu, log_std, actor.act_scale)  # (B,A),(B,)
                    q1_pi = q1(S, a_new)
                    q2_pi = q2(S, a_new)
                    q_pi_min = torch.min(q1_pi, q2_pi)
                    alpha = torch.exp(log_alpha)

                    # Actor loss: minimize E[ α logπ - Q_min ]
                    actor_loss = (alpha.detach() * logp - q_pi_min).mean()
                    opt_actor.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    opt_actor.step()
                    last_actor_loss = float(actor_loss.item())

                    # Temperature update
                    alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
                    opt_alpha.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    opt_alpha.step()
                    last_alpha = float(torch.exp(log_alpha).item())

                    # Target networks
                    soft_update(q1_targ, q1, tau)
                    soft_update(q2_targ, q2, tau)

        returns_hist.append(ep_ret)
        avg100 = np.mean(returns_hist[-100:])
        best_avg = max(best_avg, avg100)

        if ep % 10 == 0 or ep == 1:
            aL = "n/a" if last_actor_loss is None else f"{last_actor_loss:+.4f}"
            cL = "n/a" if last_critic_loss is None else f"{last_critic_loss:.4f}"
            print(
                f"Ep {ep:4d} | Ret {ep_ret:7.1f} | Avg100 {avg100:7.1f} | "
                f"Steps {total_steps:6d} | ActorL {aL} | CriticL {cL} | α {last_alpha:.3f}"
            )

        # Early stop (illustrative threshold; Pendulum returns are negative)
        if len(returns_hist) >= 100 and avg100 >= -250.0:
            print(f"Reached target in {ep} episodes! Avg100={avg100:.1f}")
            break

    env.close()
    print("Training finished.")
    return actor, (q1, q2), returns_hist

# =========================
#           Main
# =========================
if __name__ == "__main__":
    actor, critics, returns = train_sac_pendulum(
        episodes=1000,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        buffer_capacity=300_000,
        batch_size=256,
        start_steps=2_000,
        updates_per_step=1,
        critic_steps_per_update=1,
        target_entropy_scale=1.0,   # target entropy = -1.0 * act_dim
    )

    # ---- Plot training curve ----
    os.makedirs("videos", exist_ok=True)
    plot_training_curve(returns, window=50, out_path="videos/pendulum_sac_returns.png")

    # ---- Save a rollout video ----
    rollout_and_record_video(actor, video_dir="videos", episodes=1, max_steps=200, deterministic=True)
