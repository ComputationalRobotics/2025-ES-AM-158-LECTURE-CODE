# deep_sarsa_acrobot_min.py
import os, glob
import gymnasium as gym
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, random
import matplotlib.pyplot as plt

import copy
from torch.optim import AdamW

# --- env & seed ---
env = gym.make("MountainCar-v0")
env_max_step = env.spec.max_episode_steps
# Acrobot-v1, CartPole-v1, MountainCar-v0
# acrobot: 6D state, 3 actions
# cartpole: 4D state, 2 actions
seed = 42
env.reset(seed=seed); env.action_space.seed(seed); env.observation_space.seed(seed)
np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)

state_dims = int(np.prod(env.observation_space.shape))  # 4/6 都兼容
num_actions = env.action_space.n
print(f"State dimensions: {state_dims}")
print(f" Number of actions: {num_actions}")

# --- wrapper: return torch tensors ---
class PreprocessEnv(gym.Wrapper):
    def __init__(self, env): super().__init__(env)
    def reset(self, seed=None, options=None):
        out = self.env.reset(seed=seed, options=options)
        state = out[0] if isinstance(out, tuple) else out
        return torch.from_numpy(np.asarray(state, np.float32)).unsqueeze(0)
    def step(self, action):
        a = int(action.item()) if isinstance(action, torch.Tensor) else int(action)
        ns, r, term, trunc, info = self.env.step(a)
        done = bool(term or trunc)
        ns_t = torch.from_numpy(np.asarray(ns, np.float32)).unsqueeze(0)
        r_t  = torch.tensor([[r]], dtype=torch.float32)
        d_t  = torch.tensor([[done]], dtype=torch.bool)
        return ns_t, r_t, d_t, info
env = PreprocessEnv(env)

# --- Q-net ---
class QNetwork(nn.Module):
    def __init__(self, s_dim, nA):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(s_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, nA)
        )
    def forward(self, s): return self.layers(s)   # s: [B, s_dim] -> [B, nA]

q_network = QNetwork(state_dims, num_actions)

# --- epsilon-greedy policy ---
def policy(state, epsilon=0.):
    if torch.rand(1) < epsilon:
        return torch.randint(num_actions, (1, 1))
    av = q_network(state).detach()
    return torch.argmax(av, dim=-1, keepdim=True)

# --- replay memory ---
class ReplayMemory:
    def __init__(self, capacity=1_000_000):
        self.capacity, self.memory, self.pos = capacity, [], 0
    def insert(self, transition):
        if len(self.memory) < self.capacity: self.memory.append(None)
        self.memory[self.pos] = transition; self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        batch = list(zip(*random.sample(self.memory, batch_size)))
        return [torch.cat(items) for items in batch]
    def can_sample(self, batch_size): return len(self.memory) >= batch_size * 10
    def __len__(self): return len(self.memory)
memory = ReplayMemory()

class DeepSARSA:
    def __init__(self, q_net, policy, env, memory, gamma=0.995, epsilon=0.05, alpha=1e-3, batch_size=64,
                 ):  # ← 选两维来可视化
        self.q_network = q_net
        self.target_q_network = copy.deepcopy(q_net).eval()
        self.policy, self.env, self.memory = policy, env, memory
        self.gamma, self.epsilon, self.batch_size = gamma, epsilon, batch_size
        self.optim = AdamW(q_net.parameters(), lr=alpha)

        # 观测空间边界（可能含 ±inf）
        obs_space = self.env.unwrapped.observation_space
        self.low  = np.asarray(getattr(obs_space, "low", np.full(state_dims, -1.0)),  dtype=np.float32)
        self.high = np.asarray(getattr(obs_space, "high", np.full(state_dims,  1.0)), dtype=np.float32)
        self.A    = int(self.env.unwrapped.action_space.n)

        self.state_traj = []     # 保存完整维度轨迹（4D/6D）
        self.action_traj = []
        self.anchor = np.zeros(state_dims, dtype=np.float32)  # 切片的锚点
        
    def evaluate_success_rate(self, episodes: int = 200, max_steps: int = 500, seed: int = 0):
        successes, rets, steps_list = 0, [], []
        for ep in range(episodes):
            obs = self.env.reset(seed=seed + ep)
            total, steps = 0.0, 0
            done = False
            while not done:
                a = self.policy(torch.from_numpy(np.asarray(obs, np.float32)).unsqueeze(0), epsilon=0.0)
                obs, r, done, _ = self.env.step(a)
                total += float(r); steps += 1
                if steps >= max_steps:
                    break
            if steps <= max_steps - 10:
                successes += 1
            rets.append(total); steps_list.append(steps)
        env.close()
        rate = successes / episodes
        print(f"[Eval] Success rate: {successes}/{episodes} = {rate:.2%} | "
            f"mean return {np.mean(rets):.2f} | mean steps {np.mean(steps_list):.1f}")
        return rate, rets, steps_list

    @torch.no_grad()
    def Q(self, s_np, a: int) -> float:
        s_t = torch.from_numpy(np.asarray(s_np, np.float32)).unsqueeze(0)
        q   = self.q_network(s_t)[0, a]
        return float(q.item())

    def train(self, episodes):
        stats = {'MSE Loss': [], 'Returns': []}
        for episode in range(1, episodes + 1):
            state, ep_return, done = self.env.reset(), 0.0, False
            self.state_traj, self.action_traj = [], []

            while not done:
                action = self.policy(state, self.epsilon)
                next_state, reward, done_t, _ = self.env.step(action)

                # 记录完整维度的 next_state 和所采取的 action
                self.state_traj.append(next_state.numpy().flatten())
                self.action_traj.append(int(action.item()))

                self.memory.insert([state, action, reward, done_t, next_state])

                if self.memory.can_sample(self.batch_size):
                    s_b, a_b, r_b, d_b, ns_b = self.memory.sample(self.batch_size)
                    qsa   = self.q_network(s_b).gather(1, a_b)
                    
                    # # SARSA, on-policy
                    # with torch.no_grad():
                    #     na_b  = self.policy(ns_b, self.epsilon)
                    #     nextq = self.target_q_network(ns_b).gather(1, na_b)
                    #     target = r_b + (~d_b) * self.gamma * nextq

                    # # DQN, off-policy
                    # with torch.no_grad():
                    #     nextq_max = self.target_q_network(ns_b).max(dim=1, keepdim=True).values
                    #     target = r_b + (~d_b) * self.gamma * nextq_max
                        
                    # Double DQN, off-policy
                    with torch.no_grad():
                        na_b      = self.q_network(ns_b).max(dim=1, keepdim=True).indices
                        # this is the same as SARSA without epsilon-greedy
                        nextq_dd  = self.target_q_network(ns_b).gather(1, na_b)
                        target    = r_b + (~d_b) * self.gamma * nextq_dd
                        
                    loss = F.mse_loss(qsa, target)
                    self.optim.zero_grad(set_to_none=True); loss.backward(); self.optim.step()
                    stats['MSE Loss'].append(loss.item())

                state, ep_return, done = next_state, ep_return + reward.item(), bool(done_t.item())

            stats['Returns'].append(ep_return)

            # 用本轮轨迹的均值当作切片 anchor（让切片更“靠近”真实访问区域）
            if len(self.state_traj) > 0:
                self.anchor = np.mean(np.asarray(self.state_traj, dtype=np.float32), axis=0)

            if episode % 50 == 0:
                w = min(50, len(stats['Returns']))
                avg_ret  = float(np.mean(stats['Returns'][-w:]))
                avg_loss = float(np.mean(stats['MSE Loss'][-w:])) if stats['MSE Loss'] else float('nan')
                print(f"Episode {episode}/{episodes} | Avg Return({w}): {avg_ret:.2f} | MSE Loss({w}): {avg_loss:.4f}")
                # self.evaluate_success_rate(episodes=50, max_steps=env_max_step, seed=seed)

            # 硬同步 target
            if episode % 10 == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())
        return stats

agent = DeepSARSA(q_network, policy, env, memory, epsilon=0.02) 

# --- train ---
stats = agent.train(episodes=1500)
# save model
os.makedirs("output/acrobot_deep_sarsa", exist_ok=True)
torch.save(q_network.state_dict(), "output/acrobot_deep_sarsa/q_network.pth")

# --- plot stats (轻量平滑) ---
def plot_stats(stats, outdir="output", win=21):
    os.makedirs(outdir, exist_ok=True)
    def smooth(x, w):
        if len(x) < w: return x
        k = w // 2
        return [np.mean(x[max(0,i-k):min(len(x), i+k+1)]) for i in range(len(x))]
    fig, ax = plt.subplots(len(stats), 1, figsize=(10, 6))
    if len(stats) == 1: ax = [ax]
    for i, k in enumerate(stats):
        ax[i].plot(smooth(stats[k], win)); ax[i].set_title(k)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "training_stats.png")); plt.close()
plot_stats(stats, "output/acrobot_deep_sarsa")


