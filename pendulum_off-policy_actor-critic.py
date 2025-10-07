# Off-Policy Actor–Critic (continuous) on Pendulum-v1
# Replay + expected SARSA critic (sampled) + clipped IS for actor
# Tanh-Gaussian policy with proper log-prob correction
# Multiple critic steps per update iteration + VIDEO RECORDING
# ---------------------------------------------------------------
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

# ------------ Math helpers (tanh-squashed Gaussian) ------------
LOG_2PI = math.log(2.0 * math.pi)

def atanh(x):
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))  # numerically stable

def gaussian_log_prob(u, mean, log_std):
    """Elementwise log N(u; mean, std). Returns (B,)."""
    var = torch.exp(2.0 * log_std)
    return -0.5 * ((u - mean) ** 2 / var + 2.0 * log_std + LOG_2PI).sum(dim=-1)

def squashed_log_prob_from_action(a, mean, log_std, act_scale, eps=1e-6):
    """
    Given *squashed* action a in [-act_scale, act_scale], compute log prob under
    tanh-Gaussian with parameters (mean, log_std) in pre-squash space.
    """
    y = (a / act_scale).clamp(-1.0 + eps, 1.0 - eps)          # in (-1,1)
    u = atanh(y)                                              # pre-squash
    base_logp = gaussian_log_prob(u, mean, log_std)           # (B,)
    # log|det d(a)/d(u)| = sum_i [ log(act_scale) + log(1 - tanh(u_i)^2) ]
    log_det = (torch.log(act_scale) + torch.log(1.0 - torch.tanh(u) ** 2 + eps)).sum(dim=-1)
    return base_logp - log_det

def sample_squashed_gaussian(mean, log_std, act_scale):
    """
    Reparameterized sample from tanh-Gaussian and its log-prob.
    Returns: a (B,A), logp(a|s) (B,)
    """
    std = torch.exp(log_std)
    eps = torch.randn_like(mean)
    u = mean + std * eps
    a = act_scale * torch.tanh(u)
    # log prob of the *squashed* sample
    base_logp = gaussian_log_prob(u, mean, log_std)
    log_det = (torch.log(act_scale) + torch.log(1.0 - torch.tanh(u) ** 2 + 1e-6)).sum(dim=-1)
    logp = base_logp - log_det
    return a, logp

# -------------------- Networks --------------------
class PolicyGaussian(nn.Module):
    """
    Tanh-squashed Gaussian policy: a = act_scale * tanh(μ(s) + σ(s) * ε)
    """
    def __init__(self, obs_dim, act_dim, hidden=256, log_std_min=-5.0, log_std_max=2.0, act_scale=1.0):
        super().__init__()
        self.act_dim = act_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # Register act_scale as a buffer so it moves with .to(device)
        self.register_buffer('act_scale', torch.as_tensor(act_scale, dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    @torch.no_grad()
    def act_behavior(self, obs, explore_std=0.3):
        """
        Behavior μ: same mean as π but **inflated std** in pre-squash space.
        - obs: (1, obs_dim) tensor
        Returns: a_np (act_dim,), log_mu_prob (float)
        """
        mu_pi, log_std_pi = self.forward(obs)                  # (1,A)
        std_pi = torch.exp(log_std_pi)
        std_mu = torch.sqrt(std_pi ** 2 + explore_std ** 2)    # inflate variance
        log_std_mu = torch.log(std_mu + 1e-8)

        # sample under μ
        u = mu_pi + std_mu * torch.randn_like(std_mu)
        a = self.act_scale * torch.tanh(u)

        # log μ(a|s) with squashed correction
        log_mu = squashed_log_prob_from_action(a, mu_pi, log_std_mu, self.act_scale)
        return a.squeeze(0).cpu().numpy(), float(log_mu.item())

    @torch.no_grad()
    def act_deterministic(self, obs):
        mu, _ = self.forward(obs)
        a = self.act_scale * torch.tanh(mu)
        return a.squeeze(0).cpu().numpy()

class QNet(nn.Module):
    """State-action value network Q(s,a)."""
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

# -------------------- Replay Buffer --------------------
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
        self.logmu = np.zeros((self.capacity,), dtype=np.float32)  # log μ(a|s)
    def push(self, s, a, r, s2, d, logmu):
        self.s[self.idx] = s
        self.a[self.idx] = a
        self.r[self.idx] = r
        self.s2[self.idx] = s2
        self.d[self.idx] = d
        self.logmu[self.idx] = logmu
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or (self.idx == 0)
    def __len__(self):
        return self.capacity if self.full else self.idx
    def sample(self, batch_size):
        n = len(self)
        idxs = np.random.randint(0, n, size=batch_size)
        return (
            self.s[idxs], self.a[idxs], self.r[idxs],
            self.s2[idxs], self.d[idxs], self.logmu[idxs]
        )

# -------------------- Plotting --------------------
def plot_training_curve(returns_hist, window=50, out_path="videos/pendulum_offpolicy_ac_returns.png"):
    x = np.arange(1, len(returns_hist) + 1)
    rets = np.array(returns_hist, dtype=float)
    ma = np.convolve(rets, np.ones(window)/window, mode="valid") if len(rets) >= window else None

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, rets, label="Return per episode")
    if ma is not None:
        plt.plot(np.arange(window, len(rets)+1), ma, label=f"{window}-episode average")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Off-Policy Actor–Critic (continuous) on Pendulum-v1 — Training Curve")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[Plot] Saved training curve to {out_path}")

# -------------------- Video recording helper --------------------
def rollout_and_record_video(policy, video_dir="videos", episodes=1, max_steps=200, deterministic=True):
    """
    Roll out the (trained) policy on Pendulum-v1 and save video(s).
    - Gymnasium: uses RecordVideo and env with render_mode='rgb_array'
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

    dev = next(policy.parameters()).device

    total_returns = []
    for ep in range(episodes):
        if GYMN:
            obs, info = env.reset()
        else:
            obs = env.reset()
        done, steps, ep_ret = False, 0, 0.0

        while not done and steps < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            if deterministic:
                a = policy.act_deterministic(obs_t)
            else:
                with torch.no_grad():
                    mu, log_std = policy(obs_t)
                    a, _ = sample_squashed_gaussian(mu, log_std, policy.act_scale)
                    a = a.squeeze(0).cpu().numpy()

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

# -------------------- Training --------------------
def train_offpolicy_actor_critic_pendulum(
    episodes=600,
    gamma=0.99,
    lr_actor=3e-4,
    lr_critic=3e-4,
    buffer_capacity=200_000,
    batch_size=256,
    start_steps=1_000,            # warmup before learning
    updates_per_step=1,           # update iterations per env step
    critic_steps_per_update=4,    # critic-only steps before one actor step
    is_clip_c=10.0,               # clip on importance ratio
    explore_std=0.4,              # behavior std inflation in pre-squash space
    tau=0.005,                    # Polyak for target Q
    normalize_adv=True,           # normalize actor advantages
    v_samples=1,                  # samples for V_pi(s) estimate
    target_avg_last100=-250.0,    # "good" score threshold (Pendulum returns are negative)
    device=None
):
    """
    Off-policy AC for continuous control:
      - Behavior μ: squashed Gaussian with same mean as π but larger std (explore_std).
      - Critic: y = r + γ(1-d) * E_{a'~π}[ Q_target(s', a') ]  (one-sample MC or small average)
      - Actor: clipped-IS gradient E[ min(c, ρ) ∇ log π(a|s) * (Q - V_π) ].
    """
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])  # Pendulum: 2.0
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyGaussian(obs_dim, act_dim, hidden=256, act_scale=act_scale).to(device)
    qnet   = QNet(obs_dim, act_dim, hidden=256).to(device)
    q_targ = QNet(obs_dim, act_dim, hidden=256).to(device)
    q_targ.load_state_dict(qnet.state_dict())

    opt_actor  = optim.Adam(policy.parameters(), lr=lr_actor)
    opt_critic = optim.Adam(qnet.parameters(),   lr=lr_critic)

    buffer = ReplayBuffer(buffer_capacity, obs_dim, act_dim)

    def to_t(x): return torch.as_tensor(x, dtype=torch.float32, device=device)

    returns_hist = []
    best_avg = -1e9
    total_steps = 0
    start_time = time.time()

    # Safe logging even before updates begin
    last_actor_loss = None
    last_critic_loss = None

    for ep in range(1, episodes + 1):
        if GYMN:
            obs, info = env.reset(seed=SEED + ep)
        else:
            if hasattr(env, "seed"):
                env.seed(SEED + ep)
            obs = env.reset()
        done, ep_ret = False, 0.0

        while not done:
            # ---- Behavior action μ (inflated variance) ----
            obs_t = to_t(obs).unsqueeze(0)
            a_np, log_mu = policy.act_behavior(obs_t, explore_std=explore_std)
            a_clipped = np.clip(a_np, -act_scale, act_scale).astype(np.float32)

            # ---- Step env ----
            if GYMN:
                nobs, r, terminated, truncated, info = env.step(a_clipped)
                done = terminated or truncated
            else:
                nobs, r, done, info = env.step(a_clipped)

            buffer.push(obs, a_clipped, float(r), nobs, float(done), float(log_mu))
            ep_ret += float(r)
            obs = nobs
            total_steps += 1

            # ---- Learn after warmup ----
            if len(buffer) >= max(batch_size, start_steps):
                for _ in range(updates_per_step):

                    # ===== Critic phase: multiple steps =====
                    for _ in range(int(critic_steps_per_update)):
                        s, a_b, r_b, s2, d_b, logmu_b = buffer.sample(batch_size)
                        S   = to_t(s)                                # (B, obs)
                        A   = to_t(a_b)                               # (B, A)
                        R   = to_t(r_b)                               # (B,)
                        S2  = to_t(s2)                                # (B, obs)
                        D   = to_t(d_b)                               # (B,)

                        with torch.no_grad():
                            # sample a' ~ π(s') and compute Q_target(s',a')
                            mu2, log_std2 = policy(S2)                # (B,A)
                            q_next_accum = 0.0
                            for _vs in range(max(1, v_samples)):
                                a2, _ = sample_squashed_gaussian(mu2, log_std2, policy.act_scale)
                                q_next = q_targ(S2, a2)              # (B,)
                                q_next_accum = q_next if _vs == 0 else (q_next_accum + q_next)
                            v_next = q_next_accum / float(max(1, v_samples))  # (B,)

                            y = R + gamma * (1.0 - D) * v_next       # (B,)

                        q_pred = qnet(S, A)                           # (B,)
                        critic_loss = (q_pred - y).pow(2).mean()

                        opt_critic.zero_grad(set_to_none=True)
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=1.0)
                        opt_critic.step()
                        last_critic_loss = float(critic_loss.item())

                    # ===== Actor phase: one step using updated critic =====
                    s, a_b, r_b, s2, d_b, logmu_b = buffer.sample(batch_size)
                    S   = to_t(s)                                    # (B, obs)
                    A   = to_t(a_b)                                  # (B, A)
                    logMU = to_t(logmu_b)                            # (B,)

                    # log π(a|s) for buffer actions (with gradient)
                    mu, log_std = policy(S)                          # (B,A)
                    logp_pi_on_A = squashed_log_prob_from_action(A, mu, log_std, policy.act_scale)  # (B,)

                    # Importance ratios and advantage (no grad through these)
                    with torch.no_grad():
                        rho  = torch.exp(logp_pi_on_A - logMU).clamp(max=is_clip_c)  # (B,)
                        # Advantage: A = Q(S,A) - V_pi(S), with V_pi ≈ E_{a~π} Q(S,a)
                        q_sa = qnet(S, A).detach()                  # (B,)
                        v_accum = 0.0
                        for _vs in range(max(1, v_samples)):
                            a_pi, _ = sample_squashed_gaussian(mu, log_std, policy.act_scale)
                            v_accum = qnet(S, a_pi).detach() if _vs == 0 else (v_accum + qnet(S, a_pi).detach())
                        V_pi = v_accum / float(max(1, v_samples))
                        adv = q_sa - V_pi
                        if normalize_adv:
                            std = adv.std().clamp_min(1e-6)
                            adv = (adv - adv.mean()) / std

                    actor_loss = -(rho * logp_pi_on_A * adv).mean()

                    opt_actor.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    opt_actor.step()
                    last_actor_loss = float(actor_loss.item())

                    # ----- Target Q Polyak update -----
                    with torch.no_grad():
                        for p, tp in zip(qnet.parameters(), q_targ.parameters()):
                            tp.data.mul_(1 - tau).add_(tau * p.data)

        returns_hist.append(ep_ret)
        avg100 = np.mean(returns_hist[-100:])
        best_avg = max(best_avg, avg100)

        # Safe logging (handles warmup episodes with no updates yet)
        aL = "n/a" if last_actor_loss is None else f"{last_actor_loss:+.4f}"
        cL = "n/a" if last_critic_loss is None else f"{last_critic_loss:.4f}"
        if ep % 10 == 0 or ep == 1:
            print(
                f"Ep {ep:4d} | Ret {ep_ret:7.1f} | Avg100 {avg100:7.1f} | "
                f"Steps {total_steps:6d} | ActorL {aL} | CriticL {cL} | CriticSteps/Upd {critic_steps_per_update}"
            )

        # Early stop (Pendulum is "better" when closer to 0; threshold is illustrative)
        if len(returns_hist) >= 100 and avg100 >= target_avg_last100:
            print(f"Reached target in {ep} episodes! Avg100={avg100:.1f}")
            break

    dur = time.time() - start_time
    print(f"Training done in {dur:.1f}s. Best Avg100={best_avg:.1f}")
    env.close()
    return policy, qnet, returns_hist

# -------------------- Main --------------------
if __name__ == "__main__":
    policy, qnet, returns = train_offpolicy_actor_critic_pendulum(
        episodes=1000,
        gamma=0.99,
        lr_actor=3e-4,
        lr_critic=3e-4,
        buffer_capacity=300_000,
        batch_size=256,
        start_steps=2_000,
        updates_per_step=1,
        critic_steps_per_update=8,   # try 4–16
        is_clip_c=10.0,
        explore_std=0.5,             # behavior variance (pre-squash); try 0.3–0.8
        tau=0.005,
        normalize_adv=True,
        v_samples=1,                 # try >1 to reduce variance (slower)
        target_avg_last100=-250.0,
    )

    # ---- Plot and save training curve ----
    os.makedirs("videos", exist_ok=True)
    plot_training_curve(returns, window=50, out_path="videos/pendulum_offpolicy_ac_returns.png")

    # ---- Roll out the final policy and RECORD a video ----
    # Set deterministic=True for a stable video (greedy w.r.t. mean action)
    rollout_and_record_video(policy, video_dir="videos", episodes=5, max_steps=200, deterministic=True)
