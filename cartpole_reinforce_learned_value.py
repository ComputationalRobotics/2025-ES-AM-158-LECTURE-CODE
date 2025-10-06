# REINFORCE on CartPole-v1 (PyTorch) — value-converged critic per batch, single actor step
# ----------------------------------------------------------------------------------------
# pip install gymnasium torch matplotlib  (or: pip install gym torch matplotlib)

import os, random, time
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
        return int(a.item())

    def act_deterministic(self, obs):
        """Greedy action (argmax over logits) — handy for evaluation/video."""
        logits = self.forward(obs)
        a = torch.argmax(logits, dim=-1)
        return int(a.item())

# ---- Value Network (state-value baseline V(s)) ----
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
    Collect one episode and return:
      - states per step (Tensor [T, obs_dim])
      - actions per step (Tensor [T], long)
      - returns-to-go per step (Tensor [T])
      - episode return (float)
      - episode length (int)
    """
    if GYMN:
        obs, info = env.reset()
    else:
        obs = env.reset()
    done = False
    rewards, states, acts = [], [], []
    ep_ret = 0.0
    steps = 0

    while not done:
        if render: env.render()
        states.append(np.asarray(obs, dtype=np.float32))
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a = policy.act(obs_t)
        if GYMN:
            next_obs, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
        else:
            next_obs, r, done, info = env.step(a)
        rewards.append(float(r))
        acts.append(int(a))
        ep_ret += float(r)
        obs = next_obs
        steps += 1

    G = discount_cumsum(rewards, gamma)
    states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    acts_t = torch.tensor(acts, dtype=torch.long, device=device)
    G_t = torch.tensor(G, dtype=torch.float32, device=device)
    return states_t, acts_t, G_t, ep_ret, steps

# ---- Plotting helper ----
def plot_training_curve(returns_hist, window=50, out_path="videos/cartpole_returns_value_converged.png"):
    """Plot per-episode returns and a moving average."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
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
    plt.title("REINFORCE on CartPole-v1 — Critic Converged per Batch")
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
                a = policy.act(obs_t)
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

# ---- Training: per-batch converged critic, then one actor step ----
def train_reinforce_value_converged(
    episodes=2000,
    gamma=0.99,
    lr_policy=1e-3,
    lr_value=1e-3,
    batch_size=40,                # number of episodes per update
    standardize_adv=True,         # normalize advantages across the WHOLE batch
    ent_coef=1e-2,                # entropy bonus (try 5e-3 ~ 2e-2)
    value_grad_clip=1.0,
    policy_grad_clip=1.0,
    # --- critic inner-loop convergence controls ---
    value_inner_max_steps=2000,   # hard cap on GD steps for critic
    value_check_every=10,         # check convergence every N steps
    value_convergence_tol=1e-6,   # relative improvement tolerance
    value_patience=10,            # consecutive checks without improvement
    target_avg_last100=475.0,
    render_every=None,
    device=None
):
    """
    Training scheme per minibatch:
      1) Collect a batch of on-policy trajectories with the *current* actor.
      2) Fit V(s) to Monte Carlo returns by running many GD steps on the batch
         until convergence (early-stopping with tolerance/patience).
      3) Compute advantages A = G - V(s) (critic is now converged on this batch).
      4) Do *one* actor (policy) update step using those advantages.
    """
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNet(obs_dim, hidden=128, act_dim=act_dim).to(device)
    valuef = ValueNet(obs_dim, hidden=128).to(device)

    opt_pi = optim.Adam(policy.parameters(), lr=lr_policy)
    opt_v  = optim.Adam(valuef.parameters(), lr=lr_value)

    mse = nn.MSELoss(reduction="mean")

    returns_hist = []
    best_avg = -1e9
    start = time.time()
    episodes_run = 0

    while episodes_run < episodes:
        # ------- (1) Collect one batch of trajectories -------
        all_states, all_acts, all_G = [], [], []
        total_steps = 0
        batch_eps = min(batch_size, episodes - episodes_run)

        for _ in range(batch_eps):
            render = (render_every is not None and (episodes_run + 1) % render_every == 0)
            states_t, acts_t, G_t, ep_ret, steps = rollout_episode(env, policy, gamma, render, device)
            returns_hist.append(ep_ret)
            episodes_run += 1
            total_steps += int(steps)
            all_states.append(states_t)
            all_acts.append(acts_t)
            all_G.append(G_t)

        # Concatenate across all episodes in the batch
        S = torch.cat(all_states, dim=0)        # [N_steps, obs_dim]
        A = torch.cat(all_acts, dim=0)          # [N_steps]
        G = torch.cat(all_G, dim=0)             # [N_steps]

        # ------- (2) Critic inner loop: fit V(s) until convergence -------
        best_loss = float("inf")
        no_improve = 0
        steps_used = 0

        for k in range(1, value_inner_max_steps + 1):
            V = valuef(S)
            v_loss = mse(V, G)

            opt_v.zero_grad(set_to_none=True)
            v_loss.backward()
            if value_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(valuef.parameters(), max_norm=value_grad_clip)
            opt_v.step()

            steps_used = k
            # check convergence every 'value_check_every' steps
            if k % value_check_every == 0:
                cur = v_loss.item()
                # relative improvement w.r.t. best so far
                rel_impr = (best_loss - cur) / max(1.0, abs(best_loss))
                if rel_impr <= value_convergence_tol:
                    no_improve += 1
                else:
                    no_improve = 0
                    best_loss = cur
                if no_improve >= value_patience:
                    break

        # ------- (3) Compute advantages with converged critic -------
        with torch.no_grad():
            V_final = valuef(S)
        adv = (G - V_final).detach()

        if standardize_adv:
            std = adv.std(unbiased=False)
            if std > 1e-8:
                adv = (adv - adv.mean()) / (std + 1e-8)
            else:
                adv = adv - adv.mean()

        # ------- (4) Single actor step (policy gradient) -------
        logits_now = policy(S)
        dist_now = torch.distributions.Categorical(logits=logits_now)
        logp = dist_now.log_prob(A)             # gradient flows through policy
        entropy = dist_now.entropy().mean()     # exploration bonus

        policy_loss = -(logp * adv).mean() - ent_coef * entropy

        opt_pi.zero_grad(set_to_none=True)
        policy_loss.backward()
        if policy_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=policy_grad_clip)
        opt_pi.step()

        # ------- Logging -------
        avg100 = np.mean(returns_hist[-100:]) if len(returns_hist) >= 1 else 0.0
        best_avg = max(best_avg, avg100)
        print(f"Eps {episodes_run:4d} | LastRet {returns_hist[-1]:6.1f} | "
              f"Avg100 {avg100:6.1f} | Steps(batch) {total_steps:4d} | "
              f"V-steps {steps_used:4d} | BestVLoss {best_loss:.4f} | "
              f"L_pi {policy_loss.item():.3f} | Ent {entropy.item():.3f}")

        if len(returns_hist) >= 100 and avg100 >= target_avg_last100:
            print(f"SOLVED in {episodes_run} episodes! Avg100={avg100:.1f}")
            break

    dur = time.time() - start
    print(f"Training done in {dur:.1f}s. Best Avg100={best_avg:.1f}")
    env.close()
    return policy, valuef, returns_hist

if __name__ == "__main__":
    policy, valuef, returns = train_reinforce_value_converged(
        episodes=5000,
        gamma=0.99,
        lr_policy=1e-2,
        lr_value=1e-3,
        batch_size=50,                # multiple trajectories per update
        standardize_adv=True,
        ent_coef=0,
        value_inner_max_steps=2000,
        value_check_every=10,
        value_convergence_tol=1e-6,
        value_patience=10,
        render_every=None,
    )

    # ---- Plot and save training curve ----
    plot_training_curve(returns, window=50, out_path="videos/cartpole_returns_value_baseline.png")

    # ---- Roll out the final policy and record a video ----
    # Set deterministic=True for a stable video (greedy w.r.t. logits)
    rollout_and_record_video(policy, video_dir="videos", episodes=1, max_steps=1000, deterministic=True)
