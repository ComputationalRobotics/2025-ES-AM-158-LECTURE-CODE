# REINFORCE on CartPole-v1 (PyTorch) — minibatch with learned value baseline (MSE critic)
# ---------------------------------------------------------------------------------------
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
        logp = dist.log_prob(a)
        return a.item(), logp

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
      - log-probs per step (Tensor [T])
      - returns-to-go per step (Tensor [T])
      - episode return (float)
      - episode length (int)
    """
    if GYMN:
        obs, info = env.reset()
    else:
        obs = env.reset()
    done = False
    rewards, logps, states = [], [], []
    ep_ret = 0.0
    steps = 0

    while not done:
        if render: env.render()
        states.append(np.asarray(obs, dtype=np.float32))
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
    states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    G_t = torch.tensor(G, dtype=torch.float32, device=device)
    logps_t = torch.stack(logps).to(device)
    return states_t, logps_t, G_t, ep_ret, steps

# ---- Plotting helper ----
def plot_training_curve(returns_hist, window=50, out_path="videos/cartpole_returns_value_baseline.png"):
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
    plt.title("REINFORCE on CartPole-v1 — Value Baseline")
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

# ---- Training: minibatch REINFORCE with learned value baseline (MSE critic) ----
def train_reinforce_minibatch_value_baseline(
    episodes=2000,
    gamma=0.99,
    lr_policy=1e-3,
    lr_value=1e-3,
    batch_size=40,               # number of episodes per update (larger => lower variance)
    standardize_adv=True,        # normalize advantages across the WHOLE batch
    ent_coef=1e-2,               # entropy bonus (try 5e-3 ~ 2e-2)
    value_grad_clip=1.0,
    policy_grad_clip=1.0,
    target_avg_last100=475.0,    # CartPole-v1 "solved" threshold
    render_every=None,
    device=None
):
    """
    Minibatch REINFORCE with a learned value baseline:
      - Policy loss: L_pi = - E[ log pi(a_t|s_t) * (G_t - V(s_t)) ] - ent_coef * E[H(pi(.|s_t))]
      - Value loss:  L_V  =   E[ (G_t - V(s_t))^2 ]
    """
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNet(obs_dim, hidden=128, act_dim=act_dim).to(device)
    valuef = ValueNet(obs_dim, hidden=128).to(device)

    opt_pi = optim.Adam(policy.parameters(), lr=lr_policy)
    opt_v  = optim.Adam(valuef.parameters(), lr=lr_value)

    returns_hist = []
    best_avg = -1e9
    start = time.time()
    episodes_run = 0

    while episodes_run < episodes:
        # ------- Collect one batch of trajectories -------
        all_states, all_logps, all_G = [], [], []
        total_steps = 0
        batch_eps = min(batch_size, episodes - episodes_run)

        for _ in range(batch_eps):
            render = (render_every is not None and (episodes_run + 1) % render_every == 0)
            states_t, logps_t, G_t, ep_ret, steps = rollout_episode(env, policy, gamma, render, device)
            returns_hist.append(ep_ret)
            episodes_run += 1
            total_steps += int(steps)

            all_states.append(states_t)   # [T_i, obs_dim]
            all_logps.append(logps_t)     # [T_i]
            all_G.append(G_t)             # [T_i]

        # Concatenate across all episodes in the batch
        S = torch.cat(all_states, dim=0)        # [N_steps, obs_dim]
        LOGP = torch.cat(all_logps, dim=0)      # [N_steps]
        G = torch.cat(all_G, dim=0)             # [N_steps]

        # ------- Critic: V(s) and advantages (detach for actor) -------
        V = valuef(S)                            # [N_steps]
        adv = (G - V.detach())                   # baseline is action-independent

        if standardize_adv:
            std = adv.std(unbiased=False)
            if std > 1e-8:
                adv = (adv - adv.mean()) / (std + 1e-8)
            else:
                adv = adv - adv.mean()

        # ------- Policy loss (with entropy bonus; entropy needs gradient!) -------
        policy_loss = -(LOGP * adv).mean()
        logits_now = policy(S)
        dist_now = torch.distributions.Categorical(logits=logits_now)
        entropy = dist_now.entropy().mean()
        policy_loss = policy_loss - ent_coef * entropy

        # ------- Value loss (MSE to MC returns) -------
        value_loss = (G - V).pow(2).mean()

        # ------- Update actor -------
        opt_pi.zero_grad(set_to_none=True)
        policy_loss.backward()
        if policy_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=policy_grad_clip)
        opt_pi.step()

        # ------- Update critic -------
        opt_v.zero_grad(set_to_none=True)
        value_loss.backward()
        if value_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(valuef.parameters(), max_norm=value_grad_clip)
        opt_v.step()

        # ------- Logging -------
        avg100 = np.mean(returns_hist[-100:]) if len(returns_hist) >= 1 else 0.0
        best_avg = max(best_avg, avg100)
        if episodes_run % 10 == 0 or episodes_run == 1:
            print(f"Eps {episodes_run:4d} | LastRet {returns_hist[-1]:6.1f} | "
                  f"Avg100 {avg100:6.1f} | Steps(batch) {total_steps:4d} | "
                  f"L_pi {policy_loss.item():.3f} | Ent {entropy.item():.3f} | L_V {value_loss.item():.3f}")

        if len(returns_hist) >= 100 and avg100 >= target_avg_last100:
            print(f"SOLVED in {episodes_run} episodes! Avg100={avg100:.1f}")
            break

    dur = time.time() - start
    print(f"Training done in {dur:.1f}s. Best Avg100={best_avg:.1f}")
    env.close()
    return policy, valuef, returns_hist

if __name__ == "__main__":
    policy, valuef, returns = train_reinforce_minibatch_value_baseline(
        episodes=2000,
        gamma=0.99,
        lr_policy=1e-3,
        lr_value=1e-3,
        batch_size=40,             # multiple trajectories per update
        standardize_adv=True,
        ent_coef=1e-2,
        render_every=None,
    )

    # ---- Plot and save training curve ----
    plot_training_curve(returns, window=50, out_path="videos/cartpole_returns_value_baseline.png")

    # ---- Roll out the final policy and record a video ----
    # Set deterministic=True for a stable video (greedy w.r.t. logits)
    rollout_and_record_video(policy, video_dir="videos", episodes=1, max_steps=1000, deterministic=True)
