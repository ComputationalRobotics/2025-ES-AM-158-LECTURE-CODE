# Off-Policy Actor–Critic (discrete) on CartPole-v1
# Replay + expected SARSA critic + clipped IS for actor
# Now with MULTIPLE CRITIC STEPS per update iteration
# ----------------------------------------------------
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
        self.act_dim = act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim)  # logits
        )
    def forward(self, x):
        return self.net(x)  # logits (B, A)

    @torch.no_grad()
    def act_behavior(self, obs, epsilon=0.1):
        """
        Epsilon-soft behavior policy μ:
          with prob ε: uniform random action;
          with prob 1-ε: sample from π (Categorical of logits).
        Returns: a (int), mu_prob (float), pi_probs (1D tensor length A)
        """
        logits = self.forward(obs)  # (1, A)
        pi_probs = torch.softmax(logits, dim=-1).squeeze(0)  # (A,)
        A = pi_probs.numel()
        if torch.rand((), device=obs.device) < epsilon:
            a = torch.randint(high=A, size=(1,), device=obs.device).item()
        else:
            a = torch.distributions.Categorical(probs=pi_probs).sample().item()
        mu_prob = (1.0 - epsilon) * pi_probs[a].item() + epsilon * (1.0 / A)
        return a, mu_prob, pi_probs.cpu()

    @torch.no_grad()
    def act_deterministic(self, obs):
        """Greedy action (argmax over logits) — handy for evaluation/video."""
        logits = self.forward(obs)
        a = torch.argmax(logits, dim=-1)
        return int(a.item())

# ---- Q-Critic (discrete actions) ----
class QNet(nn.Module):
    def __init__(self, obs_dim, hidden=128, act_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)  # Q-values for each action
        )
    def forward(self, x):
        return self.net(x)  # (B, A)

# ---- Simple Replay Buffer ----
class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity = int(capacity)
        self.idx = 0
        self.full = False
        self.s = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity,), dtype=np.int64)
        self.r = np.zeros((self.capacity,), dtype=np.float32)
        self.s2 = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.d = np.zeros((self.capacity,), dtype=np.float32)
        self.mu = np.zeros((self.capacity,), dtype=np.float32)  # μ(a|s) at collection time
    def push(self, s, a, r, s2, d, mu_prob):
        self.s[self.idx] = s
        self.a[self.idx] = a
        self.r[self.idx] = r
        self.s2[self.idx] = s2
        self.d[self.idx] = d
        self.mu[self.idx] = mu_prob
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or (self.idx == 0)
    def __len__(self):
        return self.capacity if self.full else self.idx
    def sample(self, batch_size):
        n = len(self)
        idxs = np.random.randint(0, n, size=batch_size)
        return (
            self.s[idxs], self.a[idxs], self.r[idxs],
            self.s2[idxs], self.d[idxs], self.mu[idxs]
        )

# ---- Plotting helper ----
def plot_training_curve(returns_hist, window=50, out_path="videos/cartpole_offpolicy_ac_returns.png"):
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
    plt.title("Off-Policy Actor–Critic (discrete) on CartPole-v1 — Training Curve")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
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
                # sample from π (not μ) during evaluation
                logits = policy(obs_t)
                a = torch.distributions.Categorical(logits=logits).sample().item()
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

# ---- Training: Off-Policy Actor–Critic (replay + clipped IS, expected SARSA critic) ----
def train_offpolicy_actor_critic(
    episodes=800,
    gamma=0.99,
    lr_actor=1e-3,
    lr_critic=1e-3,
    buffer_capacity=100_000,
    batch_size=256,
    start_steps=1_000,            # warmup before learning
    updates_per_step=1,           # how many update-iterations per env step
    critic_steps_per_update=4,    # <<< multiple critic-only steps before one actor step
    is_clip_c=10.0,               # importance-ratio clip for actor
    epsilon_start=0.2,            # μ = ε-soft policy (exploration at start)
    epsilon_end=0.01,             # exploration at end
    epsilon_decay_steps=50_000,   # linear decay schedule
    tau=0.005,                    # Polyak for target Q
    normalize_adv=True,           # normalize actor advantages in the batch
    target_avg_last100=475.0,     # CartPole-v1 "solved" threshold
    device=None
):
    """
    Experience-replay off-policy Actor–Critic for discrete actions:
      - Behavior policy μ: ε-soft around π (stores μ(a|s)).
      - Critic: Q(s,a) with expected-SARSA target under π and target Q (Polyak).
      - Actor: clipped-IS gradient  E[ min(c, ρ_t) ∇ log π(a|s) * (Q - V_π) ].
      - NEW: Do `critic_steps_per_update` critic steps before ONE actor step each update iteration.
    """
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNet(obs_dim, hidden=128, act_dim=act_dim).to(device)
    qnet   = QNet(obs_dim, hidden=128, act_dim=act_dim).to(device)
    q_targ = QNet(obs_dim, hidden=128, act_dim=act_dim).to(device)
    q_targ.load_state_dict(qnet.state_dict())

    opt_actor  = optim.Adam(policy.parameters(), lr=lr_actor)
    opt_critic = optim.Adam(qnet.parameters(),   lr=lr_critic)

    buffer = ReplayBuffer(buffer_capacity, obs_dim)

    def to_t(x): return torch.as_tensor(x, dtype=torch.float32, device=device)

    returns_hist = []
    best_avg = -1e9
    total_steps = 0
    start_time = time.time()

    eps = epsilon_start
    eps_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)

    # Track last losses for safe logging even before learning starts
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
            # ---- Behavior action (μ) ----
            obs_t = to_t(obs).unsqueeze(0)
            a, mu_prob, _ = policy.act_behavior(obs_t, epsilon=eps)
            # ---- Step env ----
            if GYMN:
                nobs, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated
            else:
                nobs, r, done, info = env.step(a)

            buffer.push(obs, a, float(r), nobs, float(done), float(mu_prob))
            ep_ret += float(r)
            obs = nobs
            total_steps += 1

            # Decay ε linearly
            if eps > epsilon_end:
                eps = max(epsilon_end, eps - eps_decay)

            # ---- Learn after warmup ----
            if len(buffer) >= max(batch_size, start_steps):
                for _ in range(updates_per_step):

                    # ====== Critic phase: multiple steps before actor ======
                    for _cs in range(int(critic_steps_per_update)):
                        # Sample batch
                        s, a_b, r_b, s2, d_b, mu = buffer.sample(batch_size)
                        S   = to_t(s)                               # (B, obs)
                        A   = torch.as_tensor(a_b, device=device)   # (B,)
                        R   = to_t(r_b)                              # (B,)
                        S2  = to_t(s2)                               # (B, obs)
                        D   = to_t(d_b)                              # (B,)

                        # Critic target under π with target Q (expected SARSA)
                        with torch.no_grad():
                            logits_next = policy(S2)                         # (B, A)
                            pi_next     = torch.softmax(logits_next, dim=-1)# (B, A)
                            q_next_targ = q_targ(S2)                         # (B, A)
                            v_next      = (pi_next * q_next_targ).sum(dim=1) # (B,)
                            y = R + gamma * (1.0 - D) * v_next               # (B,)

                        q_all = qnet(S)                                      # (B, A)
                        q_sa  = q_all.gather(1, A.view(-1, 1)).squeeze(1)    # (B,)
                        critic_loss = (q_sa - y).pow(2).mean()

                        opt_critic.zero_grad(set_to_none=True)
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=1.0)
                        opt_critic.step()
                        last_critic_loss = float(critic_loss.item())

                    # ====== Actor phase: one step using updated critic ======
                    s, a_b, r_b, s2, d_b, mu = buffer.sample(batch_size)
                    S   = to_t(s)
                    A   = torch.as_tensor(a_b, device=device)
                    MU  = to_t(mu)

                    # Policy with gradient
                    logits_actor = policy(S)                                 # (B, A)
                    log_probs     = torch.log_softmax(logits_actor, dim=-1)  # (B, A)
                    pi_probs      = log_probs.exp()                          # (B, A)
                    logp_a        = log_probs.gather(1, A.view(-1, 1)).squeeze(1)  # (B,)

                    # ρ = π(a|s)/μ(a|s) and advantage = Q - V_π (no grad through these)
                    with torch.no_grad():
                        pi_a = pi_probs.gather(1, A.view(-1, 1)).squeeze(1).clamp_min(1e-8)
                        rho  = (pi_a / MU.clamp_min(1e-8))
                        rho_c = torch.clamp(rho, max=is_clip_c)

                        q_all_det = qnet(S).detach()                          # (B, A)
                        q_sa_det  = q_all_det.gather(1, A.view(-1, 1)).squeeze(1)
                        v_pi      = (pi_probs.detach() * q_all_det).sum(dim=1)
                        adv       = q_sa_det - v_pi
                        if normalize_adv:
                            std = adv.std().clamp_min(1e-6)
                            adv = (adv - adv.mean()) / std

                    actor_loss = -(rho_c * logp_a * adv).mean()

                    opt_actor.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    opt_actor.step()
                    last_actor_loss = float(actor_loss.item())

                    # ----- Target Q Polyak update (once per update iteration) -----
                    with torch.no_grad():
                        for p, tp in zip(qnet.parameters(), q_targ.parameters()):
                            tp.data.mul_(1 - tau).add_(tau * p.data)

        returns_hist.append(ep_ret)
        avg100 = np.mean(returns_hist[-100:])
        best_avg = max(best_avg, avg100)

        # ---- Safe logging (handles warmup episodes without losses) ----
        actorL_str  = f"{last_actor_loss:+.4f}" if last_actor_loss is not None else "   n/a"
        criticL_str = f"{last_critic_loss:.4f}" if last_critic_loss is not None else "  n/a"

        if ep % 10 == 0 or ep == 1:
            print(
                f"Ep {ep:4d} | Ret {ep_ret:6.1f} | Avg100 {avg100:6.1f} | "
                f"Steps {total_steps:6d} | ε {eps:5.3f} | "
                f"ActorL {actorL_str} | CriticL {criticL_str} | "
                f"CriticSteps/Update {critic_steps_per_update}"
            )

        if len(returns_hist) >= 100 and avg100 >= target_avg_last100:
            print(f"SOLVED in {ep} episodes! Avg100={avg100:.1f}")
            break

    dur = time.time() - start_time
    print(f"Training done in {dur:.1f}s. Best Avg100={best_avg:.1f}")
    env.close()
    return policy, qnet, returns_hist

if __name__ == "__main__":
    policy, qnet, returns = train_offpolicy_actor_critic(
        episodes=2000,
        gamma=0.99,
        lr_actor=1e-3,
        lr_critic=1e-3,
        buffer_capacity=200_000,
        batch_size=256,
        start_steps=2_000,
        updates_per_step=1,          # number of update iterations per env step
        critic_steps_per_update=4,   # <<< tune this for a stronger critic
        is_clip_c=10.0,
        epsilon_start=0.20,
        epsilon_end=0.02,
        epsilon_decay_steps=80_000,
        tau=0.005,
        normalize_adv=True,
        target_avg_last100=475.0,
    )

    # ---- Plot and save training curve ----
    os.makedirs("videos", exist_ok=True)
    plot_training_curve(returns, window=50, out_path="videos/cartpole_offpolicy_ac_returns.png")

    # ---- Roll out the final policy and record a video ----
    # Set deterministic=True for a stable video (greedy w.r.t. logits)
    rollout_and_record_video(policy, video_dir="videos", episodes=1, max_steps=1000, deterministic=True)
