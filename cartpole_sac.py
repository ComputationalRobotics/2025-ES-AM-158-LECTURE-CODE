# SAC (Discrete) on CartPole-v1 with a Single Q, Single Target
# ------------------------------------------------------------
#
# - Fixed temperature alpha (no auto-tuning)
# - Single critic Q(s, ·) and a single Polyak-averaged target critic
# - Exact discrete expectations over actions (no action reparameterization)

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
    """Create CartPole env. For Gymnasium video, we need render_mode='rgb_array'."""
    if GYMN and render_for_video:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
    else:
        env = gym.make("CartPole-v1")
    try:
        env.reset(seed=seed)
    except TypeError:
        try: env.seed(seed)
        except Exception: pass
    return env

def make_video_env(video_dir, seed=SEED):
    """Wrap an env to record videos to `video_dir` (works for Gymnasium and older Gym)."""
    os.makedirs(video_dir, exist_ok=True)
    env = make_env(seed=seed, render_for_video=True)
    try:
        # Gymnasium / modern Gym
        from gymnasium.wrappers import RecordVideo as GymnRecordVideo
        env = GymnRecordVideo(env, video_folder=video_dir,
                              episode_trigger=lambda ep: True,
                              name_prefix="sac_discrete")
    except Exception:
        # Fallback to gym RecordVideo / Monitor
        if hasattr(gym.wrappers, "RecordVideo"):
            env = gym.wrappers.RecordVideo(env, video_folder=video_dir,
                                           episode_trigger=lambda ep: True)
        else:
            env = gym.wrappers.Monitor(env, video_dir, force=True)
    return env

# ----------------- Replay Buffer ------------------------
class ReplayBuffer:
    def __init__(self, obs_dim, size):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, 1), dtype=np.int64)
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
        a   = to_t(self.act, dtype=torch.int64)
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

class DiscretePolicy(nn.Module):
    """Outputs action probabilities π(a|s) for K discrete actions."""
    def __init__(self, obs_dim, n_actions, hidden=(128,128)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_actions])
    def forward(self, s):
        # returns both log π and π for numerical stability
        logits = self.net(s)                             # [B, K]
        logp   = torch.log_softmax(logits, dim=-1)       # [B, K]
        pi     = torch.exp(logp)                         # [B, K]
        return logp, pi
    @torch.no_grad()
    def sample(self, s_np):
        s = torch.as_tensor(s_np, dtype=torch.float32).unsqueeze(0)
        logp, pi = self.forward(s)
        dist = torch.distributions.Categorical(probs=pi.squeeze(0))
        a = dist.sample().item()
        return a

class QCritic(nn.Module):
    """Single Q-network mapping s → Q(s, :) (vector over actions)."""
    def __init__(self, obs_dim, n_actions, hidden=(128,128)):
        super().__init__()
        self.q = mlp([obs_dim, *hidden, n_actions])
    def forward(self, s):
        return self.q(s)  # [B, K]

def soft_update(net, target, tau):
    with torch.no_grad():
        for p, p_t in zip(net.parameters(), target.parameters()):
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

    # Fixed temperature (discrete entropy bonus)
    alpha: float = 0.2

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    hidden_actor: tuple = (128,128)
    hidden_critic: tuple = (128,128)

    device: str = "cpu"
    log_every: int = 2_000  # print every N env steps
    eval_episodes: int = 5  # quick evaluation
    video_episodes: int = 3
    video_dir: str = "videos_sac_cartpole"
    plot_path: str = "sac_discrete_cartpole_learning_curve.png"

# ----------------- Evaluation helpers -------------------
@torch.no_grad()
def act_greedy(policy, qnet, s_np, alpha=None, mode="q"):
    """
    Deterministic action for evaluation/videos.
    mode='q'  : argmax_a Q(s,a)
    mode='soft': argmax_a [Q(s,a) - alpha * log π(a|s)]
    """
    s = torch.as_tensor(s_np, dtype=torch.float32).unsqueeze(0)
    q = qnet(s)  # [1, K]
    if mode == "soft" and alpha is not None:
        logp, _ = policy(s)  # [1, K]
        score = q - alpha * logp
    else:
        score = q
    return int(torch.argmax(score, dim=-1).item())

def evaluate_avg_return(env, policy, qnet, episodes=5, greedy_mode="q", alpha=0.2):
    scores = []
    for _ in range(episodes):
        s, _ = env.reset() if GYMN else env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            a = act_greedy(policy, qnet, s, alpha=alpha, mode=greedy_mode)
            step_out = env.step(a)
            if GYMN:
                s, r, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                s, r, done, _ = step_out
            ep_ret += r
        scores.append(ep_ret)
    return float(np.mean(scores))

def record_videos(policy, qnet, cfg):
    """Roll out a few evaluation episodes and save videos to cfg.video_dir."""
    env_v = make_video_env(cfg.video_dir, seed=SEED+123)
    try:
        for ep in range(cfg.video_episodes):
            s, _ = env_v.reset() if GYMN else env_v.reset()
            done = False
            while not done:
                a = act_greedy(policy, qnet, s, alpha=cfg.alpha, mode="q")  # change to "soft" if desired
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
    """Plot per-episode return vs. episode index and save to disk."""
    if len(returns) == 0:
        print("[Plot] No returns to plot yet.")
        return
    plt.figure(figsize=(7,4.5))
    plt.plot(np.arange(1, len(returns)+1), returns, linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("SAC (Discrete, Single Q) on CartPole-v1 — Learning Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(cfg.plot_path, dpi=160)
    try:
        plt.show()
    except Exception:
        pass
    print(f"[Plot] Saved learning curve to: {os.path.abspath(cfg.plot_path)}")

# ----------------- Training Loop ------------------------
def train_discrete_sac_cartpole(cfg=Config()):
    device = torch.device(cfg.device)
    env = make_env(SEED)
    obs_dim = env.observation_space.shape[0]  # CartPole: 4
    n_actions = env.action_space.n           # CartPole: 2

    # Modules
    policy = DiscretePolicy(obs_dim, n_actions, cfg.hidden_actor).to(device)
    qnet   = QCritic(obs_dim, n_actions, cfg.hidden_critic).to(device)
    qtarget= QCritic(obs_dim, n_actions, cfg.hidden_critic).to(device)
    qtarget.load_state_dict(qnet.state_dict())

    pi_optim = torch.optim.Adam(policy.parameters(), lr=cfg.actor_lr)
    q_optim  = torch.optim.Adam(qnet.parameters(),   lr=cfg.critic_lr)

    # Buffer
    rb = ReplayBuffer(obs_dim, cfg.buffer_size)

    # Run
    s, _ = env.reset() if GYMN else env.reset()
    ep_ret, ep_len = 0.0, 0
    returns = []
    episodes = 0

    for t in range(1, cfg.total_env_steps + 1):
        # ----- Action -----
        if t < cfg.start_random_steps:
            a = env.action_space.sample()
        else:
            a = policy.sample(s)

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

                # Sample batch
                bs, ba, br, bs2, bd = rb.sample(cfg.batch_size, device)

                # --- Target computation ---
                with torch.no_grad():
                    # π(s') and log π(s')
                    logpi2, pi2 = policy(bs2)                   # [B, K] each
                    # Q_target(s', ·)
                    q2_targ_all = qtarget(bs2)                  # [B, K]
                    # V_tgt(s') = Σ_a' π(a'|s') [ Q̄(s',a') - α log π(a'|s') ]
                    v_tgt = (pi2 * (q2_targ_all - cfg.alpha * logpi2)).sum(dim=1, keepdim=True)  # [B,1]
                    y = br + cfg.gamma * (1.0 - bd) * v_tgt     # [B,1]

                # --- Critic update ---
                q_all = qnet(bs)                                # [B, K]
                q_sa = q_all.gather(1, ba)                      # [B, 1]
                q_loss = F.mse_loss(q_sa, y)

                q_optim.zero_grad()
                q_loss.backward()
                q_optim.step()

                # --- Actor update (exact discrete expectation) ---
                logpi, pi = policy(bs)                          # [B, K] each
                with torch.no_grad():
                    q_all_det = qnet(bs)                        # [B, K]
                # J_π = E_s Σ_a π(a|s)[ α log π(a|s) - Q(s,a) ]
                actor_obj = (pi * (cfg.alpha * logpi - q_all_det)).sum(dim=1).mean()
                pi_loss = actor_obj

                pi_optim.zero_grad()
                pi_loss.backward()
                pi_optim.step()

                # --- Target network (Polyak) ---
                soft_update(qnet, qtarget, cfg.tau)

        # ----- Logging -----
        if t % cfg.log_every == 0:
            avg_recent = np.mean(returns[-10:]) if len(returns) >= 10 else (np.mean(returns) if returns else 0.0)
            # quick deterministic eval with greedy Q
            eval_env = make_env(SEED+7)
            eval_score = evaluate_avg_return(eval_env, policy, qnet, cfg.eval_episodes, greedy_mode="q", alpha=cfg.alpha) \
                         if len(returns) > 5 else float('nan')
            eval_env.close()
            print(f"[t={t}] episodes={episodes}  recent10={avg_recent:.1f}  eval_avg={eval_score:.1f}  "
                  f"rb={rb.size}  alpha={cfg.alpha}")

    env.close()
    print(f"Training done. Episodes: {episodes}. Final 10-avg return: "
          f"{(np.mean(returns[-10:]) if len(returns)>=10 else np.mean(returns)):.1f}")

    # --------- (1) Plot learning curve ----------
    plot_learning_curve(returns, cfg)

    # --------- (2) Record evaluation videos ----------
    record_videos(policy, qnet, cfg)

    return returns, policy, qnet

if __name__ == "__main__":
    _ = train_discrete_sac_cartpole(Config(
        total_env_steps=80_000,
        start_random_steps=1_000,
        update_after=1_000,
        update_every=50,
        updates_per_step=1,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,          # Fixed temperature (tune e.g. 0.05–0.5)
        actor_lr=3e-4,
        critic_lr=3e-4,
        device="cpu",
        log_every=2_000,
        eval_episodes=5,
        video_episodes=3,
        video_dir="videos_sac_cartpole",
        plot_path="sac_discrete_cartpole_learning_curve.png"
    ))
