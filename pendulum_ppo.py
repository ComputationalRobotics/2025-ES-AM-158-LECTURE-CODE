# PPO on Pendulum-v1 (PyTorch)
# ----------------------------
# pip install gymnasium torch matplotlib  (or: pip install gym torch matplotlib)

import os, time, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- Gym import with fallback (Gymnasium preferred) ---
try:
    import gymnasium as gym
    GYMN = True
except Exception:
    import gym
    GYMN = False

# ---------- Reproducibility ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def make_env(seed=SEED, render_mode=None):
    env = gym.make("Pendulum-v1", render_mode=render_mode)
    try:
        env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"): env.seed(seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return env

# ---------- Policy (tanh-Gaussian) + Value ----------
class PolicyNet(nn.Module):
    """
    Tanh-squashed Gaussian policy scaled to env bounds.
    Provides: act(), log_prob(obs, act_env), entropy_approx(obs)
    """
    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden=64, log_std_bounds=(-5, 2)):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)
        self.log_std_min, self.log_std_max = log_std_bounds

        act_low = torch.as_tensor(act_low, dtype=torch.float32)
        act_high = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("act_low", act_low)
        self.register_buffer("act_high", act_high)
        self.register_buffer("act_scale", (act_high - act_low) / 2.0)
        self.register_buffer("act_bias", (act_high + act_low) / 2.0)

    def forward(self, x):
        h = self.body(x)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def _atanh(self, x, eps=1e-6):
        x = x.clamp(min=-1+eps, max=1-eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def log_prob(self, obs, act_env):
        """Exact env-space log-prob with tanh + scaling change-of-variables."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        # env -> (-1,1) -> pre-tanh
        a = (act_env - self.act_bias) / (self.act_scale + 1e-8)
        u = self._atanh(a)
        # base Gaussian
        var = std.pow(2)
        logp_u = -0.5 * (((u - mu) ** 2) / (var + 1e-8) + 2*log_std + math.log(2*math.pi))
        logp_u = logp_u.sum(dim=-1)
        # |det d(tanh)/du|; stable expression from SAC
        log_det_tanh = (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(dim=-1)
        # scaling (-1,1)->[low,high]
        log_det_scale = torch.log(self.act_scale.abs() + 1e-8).sum(dim=-1)
        return logp_u - log_det_tanh - log_det_scale

    def entropy_approx(self, obs):
        """Entropy of base Gaussian (ignores tanh+scale) — standard approximation."""
        _, log_std = self.forward(obs)
        # Entropy of Normal = 0.5 * log(2πe σ^2) per dim
        ent = 0.5 * (1.0 + math.log(2*math.pi)) + log_std
        return ent.sum(dim=-1)  # (B,)

    @torch.no_grad()
    def act(self, obs):
        """Sample env-space action and log-prob for rollout; also return pre-update value."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        u = torch.normal(mean=mu, std=std)
        a = torch.tanh(u)
        a_env = a * self.act_scale + self.act_bias
        logp = self.log_prob(obs, a_env)
        return a_env, logp

class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.v(x).squeeze(-1)

# ---------- GAE(λ) ----------
def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """
    Vectorized GAE over a rollout of length T (1D tensors).
    'dones' should be 1 for true terminal (not time-limit), else 0.
    """
    T = rewards.size(0)
    adv = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    last_adv = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        adv[t] = last_adv
    returns = adv + values
    return adv, returns

# ---------- Rollout Buffer ----------
class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.device = device
        self.size = size
        self.reset(obs_dim, act_dim)

    def reset(self, obs_dim, act_dim):
        self.obs = torch.zeros((self.size, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.size, act_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.size,), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.size,), dtype=torch.float32, device=self.device)
        self.logprobs = torch.zeros((self.size,), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((self.size,), dtype=torch.float32, device=self.device)
        self.ptr = 0

    def add(self, obs, action, reward, done, logprob, value):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.logprobs[self.ptr] = logprob
        self.values[self.ptr] = value
        self.ptr += 1

    def full(self):
        return self.ptr >= self.size

# ---------- PPO Training ----------
def ppo_train_pendulum(
    total_timesteps=200_000,
    steps_per_update=4096,
    minibatch_size=256,
    update_epochs=10,
    gamma=0.99,
    lam=0.95,
    clip_coef=0.2,
    ent_coef=0.0,       # Pendulum usually learns fine with little/no entropy bonus
    vf_coef=0.5,
    max_grad_norm=1.0,
    target_kl=0.015,
    lr=3e-4,
    seed=SEED,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low, act_high = env.action_space.low, env.action_space.high

    policy = PolicyNet(obs_dim, act_dim, act_low, act_high, hidden=128).to(device)
    valuef = ValueNet(obs_dim, hidden=128).to(device)
    optim_all = optim.Adam(list(policy.parameters()) + list(valuef.parameters()), lr=lr)

    # logging
    ep_returns = []
    returns_plot_path = "videos/ppo_pendulum_returns.png"
    os.makedirs("videos", exist_ok=True)

    buffer = RolloutBuffer(obs_dim, act_dim, steps_per_update, device)

    # reset env
    if GYMN:
        obs, info = env.reset()
        terminated, truncated = False, False
        done_flag_for_stats = False
    else:
        obs = env.reset()
        done_flag_for_stats = False

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    global_steps = 0
    start = time.time()

    while global_steps < total_timesteps:
        buffer.reset(obs_dim, act_dim)
        # --------- Rollout collection ---------
        while not buffer.full():
            with torch.no_grad():
                value = valuef(obs_t.unsqueeze(0)).squeeze(0)
                action_env, logp = policy.act(obs_t.unsqueeze(0))
            action_env_np = action_env.squeeze(0).cpu().numpy().astype(np.float32)

            if GYMN:
                next_obs, reward, terminated, truncated, info = env.step(action_env_np)
                done = bool(terminated)  # treat true terminal as done; bootstrap through truncation
                done_flag_for_stats = terminated or truncated
            else:
                next_obs, reward, done, info = env.step(action_env_np)
                done_flag_for_stats = done

            buffer.add(
                obs_t,
                torch.tensor(action_env_np, dtype=torch.float32, device=device),
                torch.tensor(reward, dtype=torch.float32, device=device),
                torch.tensor(float(done), dtype=torch.float32, device=device),
                logp.detach(),
                value.detach()
            )

            global_steps += 1
            # episode bookkeeping for returns
            if not hasattr(ppo_train_pendulum, "_cur_ep_ret"):
                ppo_train_pendulum._cur_ep_ret = 0.0
            ppo_train_pendulum._cur_ep_ret += float(reward)

            # step env
            if done_flag_for_stats:
                ep_returns.append(ppo_train_pendulum._cur_ep_ret)
                ppo_train_pendulum._cur_ep_ret = 0.0
                if GYMN:
                    obs, info = env.reset()
                else:
                    obs = env.reset()
            else:
                obs = next_obs

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

        # bootstrap value for last state
        with torch.no_grad():
            last_value = valuef(obs_t.unsqueeze(0)).squeeze(0)

        # --------- Compute advantages & returns ---------
        adv, v_targ = compute_gae(
            rewards=buffer.rewards,
            values=buffer.values,
            dones=buffer.dones,
            last_value=last_value,
            gamma=gamma,
            lam=lam
        )
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # --------- PPO update (multiple epochs over minibatches) ----------
        b_inds = np.arange(steps_per_update)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start_idx in range(0, steps_per_update, minibatch_size):
                mb_inds = b_inds[start_idx:start_idx + minibatch_size]
                mb_obs = buffer.obs[mb_inds]
                mb_actions = buffer.actions[mb_inds]
                mb_old_logp = buffer.logprobs[mb_inds]
                mb_adv = adv[mb_inds]
                mb_vtarg = v_targ[mb_inds]

                # new logprob, entropy, value
                new_logp = policy.log_prob(mb_obs, mb_actions)
                entropy = policy.entropy_approx(mb_obs).mean()
                v_pred = valuef(mb_obs)

                # policy loss (clipped)
                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                value_loss = 0.5 * (mb_vtarg - v_pred).pow(2).mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                optim_all.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(policy.parameters()) + list(valuef.parameters()), max_grad_norm)
                optim_all.step()

            # KL early stopping
            with torch.no_grad():
                # approximate KL (old - new); expectation over rollout
                new_logp_all = policy.log_prob(buffer.obs, buffer.actions)
                approx_kl = (buffer.logprobs - new_logp_all).mean().item()
            if approx_kl > target_kl:
                print(f"[Update] Early stop at epoch {epoch+1}/{update_epochs} (KL={approx_kl:.4f} > {target_kl})")
                break

        # --------- Logging ---------
        if len(ep_returns) > 0:
            avg100 = np.mean(ep_returns[-100:])
            print(f"Steps {global_steps:7d} | EpRet {ep_returns[-1]:7.1f} | Avg100 {avg100:7.1f} | KL {approx_kl:.4f}")

    env.close()
    dur = time.time() - start
    print(f"Training finished in {dur/60:.1f} min. Episodes: {len(ep_returns)}")
    return policy, valuef, ep_returns

# ---------- Plotting ----------
def plot_returns(returns, window=50, path="videos/ppo_pendulum_returns.png"):
    x = np.arange(1, len(returns) + 1)
    rets = np.array(returns, dtype=float)
    ma = None
    if len(rets) >= window:
        ma = np.convolve(rets, np.ones(window)/window, mode="valid")

    plt.figure(figsize=(8,4.5))
    plt.plot(x, rets, label="Return/episode")
    if ma is not None:
        plt.plot(np.arange(window, len(rets)+1), ma, label=f"{window}-episode avg")
    plt.xlabel("Episode"); plt.ylabel("Return")
    plt.title("PPO on Pendulum-v1")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"[Plot] saved to {path}")

# ---------- Video ----------
def record_video(policy, video_dir="videos", episodes=1, max_steps=1000, deterministic=True):
    os.makedirs(video_dir, exist_ok=True)
    if GYMN:
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda e: True)
        obs, info = env.reset()
    else:
        env = gym.make("Pendulum-v1")
        env = gym.wrappers.Monitor(env, video_dir, force=True, video_callable=lambda e: True)
        obs = env.reset()

    total_returns = []
    for ep in range(episodes):
        done, steps, ep_ret = False, 0, 0.0
        while not done and steps < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mu, _ = policy.forward(obs_t)
                a = torch.tanh(mu) * policy.act_scale + policy.act_bias if deterministic else policy.act(obs_t)[0]
                a = a.squeeze(0).cpu().numpy().astype(np.float32)
            if GYMN:
                obs, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
            else:
                obs, r, done, _ = env.step(a)
            ep_ret += float(r); steps += 1
            if done and ep < episodes-1:
                if GYMN: obs, _ = env.reset()
                else: obs = env.reset()
        total_returns.append(ep_ret)
        print(f"[Video] Episode {ep+1}/{episodes} return: {ep_ret:.1f}")
    env.close()
    print(f"[Video] saved to: {os.path.abspath(video_dir)}")
    return total_returns

# ---------- Main ----------
if __name__ == "__main__":
    policy, valuef, ep_returns = ppo_train_pendulum(
        total_timesteps=1500_000,   # adjust for your compute budget
        steps_per_update=4096,
        minibatch_size=256,
        update_epochs=10,
        gamma=0.99,
        lam=0.95,
        clip_coef=0.2,
        ent_coef=0.0,     # try 1e-3 ~ 2e-2 if exploration is sluggish
        vf_coef=0.5,
        max_grad_norm=1.0,
        target_kl=0.02,
        lr=3e-4,
    )

    plot_returns(ep_returns, window=50, path="videos/ppo_pendulum_returns.png")
    record_video(policy, video_dir="videos", episodes=1, max_steps=1000, deterministic=True)
