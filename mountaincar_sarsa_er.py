import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# ======================== Utilities ========================
def normalize_obs(obs, low, high):
    center = 0.5 * (low + high)
    scale = 0.5 * (high - low)
    scale = np.where(scale == 0.0, 1.0, scale)
    return (obs - center) / scale

def epsilon_greedy(q, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, q.shape[0]))
    return int(np.argmax(q))

def expected_q(q, epsilon):
    A = q.shape[0]
    greedy = int(np.argmax(q))
    probs = np.full(A, epsilon / A)
    probs[greedy] += (1.0 - epsilon)
    return float(np.dot(probs, q))

def huber(delta, k=1.0):
    return np.clip(delta, -k, k)

# ======================== Optimizer (Adam) ========================
class Adam:
    def __init__(self, shapes, lr=3e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t = 0

    def step(self, params, grads, max_grad_norm=None):
        if max_grad_norm is not None:
            flat = np.concatenate([g.ravel() for g in grads])
            norm = np.linalg.norm(flat)
            if norm > max_grad_norm and norm > 0:
                scale = max_grad_norm / norm
                grads = [g * scale for g in grads]

        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t)
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*g
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*(g*g)
            p = p + lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)
            new_params.append(p)
        return new_params

# ======================== Q Network ========================
class QNetwork:
    """
    Tiny 2-layer MLP for Q(s, ·; θ). Manual backprop (NumPy only).
    """
    def __init__(self, state_dim, action_n, hidden=64, seed=0, lr=3e-3, max_grad_norm=5.0):
        self.state_dim = state_dim
        self.action_n = action_n
        self.hidden = hidden
        self.rng = np.random.default_rng(seed)
        # Xavier/Glorot init
        limit1 = np.sqrt(6.0 / (state_dim + hidden))
        limit2 = np.sqrt(6.0 / (hidden + action_n))
        self.W1 = self.rng.uniform(-limit1, limit1, size=(hidden, state_dim))
        self.b1 = np.zeros(hidden)
        self.W2 = self.rng.uniform(-limit2, limit2, size=(action_n, hidden))
        self.b2 = np.zeros(action_n)

        self.opt = Adam(
            shapes=[self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape],
            lr=lr
        )
        self.max_grad_norm = max_grad_norm

    def forward(self, s):
        a1 = self.W1 @ s + self.b1            # (H,)
        h1 = np.tanh(a1)                       # (H,)
        q = self.W2 @ h1 + self.b2             # (A,)
        cache = (s, a1, h1, q)
        return q, cache

    def grad_Q_params(self, cache, action_index):
        s, a1, h1, q = cache
        a = int(action_index)

        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)

        dW2[a, :] = h1
        db2[a] = 1.0

        dh1 = self.W2[a, :]
        da1 = (1.0 - np.tanh(a1)**2) * dh1
        dW1 += np.outer(da1, s)
        db1 += da1

        return dW1, db1, dW2, db2

    def apply_update(self, dW1, db1, dW2, db2):
        params = [self.W1, self.b1, self.W2, self.b2]
        grads  = [dW1,      db1,      dW2,      db2]
        new_params = self.opt.step(params, grads, max_grad_norm=self.max_grad_norm)
        self.W1, self.b1, self.W2, self.b2 = new_params

    # ------- target-net helpers -------
    def get_weights(self):
        return (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

    def set_weights(self, weights):
        self.W1, self.b1, self.W2, self.b2 = [w.copy() for w in weights]

    def hard_update_from(self, other):
        self.set_weights(other.get_weights())

# ======================== Replay Buffer ========================
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = int(capacity)
        self.ptr = 0
        self.full = False
        self.s   = np.zeros((capacity, state_dim), dtype=np.float64)
        self.a   = np.zeros(capacity, dtype=np.int32)
        self.r   = np.zeros(capacity, dtype=np.float64)
        self.s2  = np.zeros((capacity, state_dim), dtype=np.float64)
        self.a2  = np.zeros(capacity, dtype=np.int32)   # only used for SARSA
        self.d   = np.zeros(capacity, dtype=np.bool_)   # done
        self.eps = np.zeros(capacity, dtype=np.float64) # behavior epsilon at t

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def add(self, s, a, r, s2, a2, done, eps_t):
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.a2[i] = a2
        self.d[i] = done
        self.eps[i] = eps_t
        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or (self.ptr == 0)

    def sample(self, batch_size, rng):
        n = len(self)
        idxs = rng.integers(0, n, size=batch_size)
        return (self.s[idxs], self.a[idxs], self.r[idxs],
                self.s2[idxs], self.a2[idxs], self.d[idxs], self.eps[idxs])

# ======================== Training ========================
def train(env_id="MountainCar-v0", algo="sarsa", episodes=2500, gamma=0.999, alpha=3e-3,
          epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.997, hidden=64, seed=0,
          huber_k=1.0, max_grad_norm=5.0, print_every=100,
          # ---- replay options ----
          use_replay=True, replay_size=50000, batch_size=64, warmup=1000, updates_per_step=4,
          use_target=True, target_sync=500):
    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    obs, info = env.reset(seed=seed)
    low, high = env.observation_space.low, env.observation_space.high
    action_n = env.action_space.n
    net = QNetwork(state_dim=obs.shape[0], action_n=action_n, hidden=hidden,
                   seed=seed, lr=alpha, max_grad_norm=max_grad_norm)

    # Target network (optional)
    if use_target:
        target_net = QNetwork(state_dim=obs.shape[0], action_n=action_n, hidden=hidden,
                              seed=seed+1234, lr=alpha, max_grad_norm=max_grad_norm)
        target_net.hard_update_from(net)
    else:
        target_net = net

    # Replay buffer (optional)
    if use_replay:
        rb = ReplayBuffer(replay_size, state_dim=obs.shape[0])

    returns = []
    eps = float(epsilon_start)
    global_step = 0

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        s_n = normalize_obs(obs, low, high)
        q, cache = net.forward(s_n)
        a = epsilon_greedy(q, eps, rng)
        total = 0.0
        done = False

        while not done:
            next_obs, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            total += r

            s2_n = normalize_obs(next_obs, low, high)
            q2, cache2 = net.forward(s2_n)

            # Choose a2 for behavior (even if algo == expected_sarsa)
            a2 = epsilon_greedy(q2, eps, rng)

            if not use_replay:
                # ----- original on-policy (immediate) update -----
                if algo == "sarsa":
                    target = r + (0.0 if done else gamma * q2[a2])
                elif algo == "expected_sarsa":
                    target = r + (0.0 if done else gamma * expected_q(q2, eps))
                else:
                    raise ValueError("algo must be 'sarsa' or 'expected_sarsa'")

                delta = target - q[a]
                delta_eff = huber(delta, k=huber_k)
                dW1, db1, dW2, db2 = net.grad_Q_params(cache, a)
                dW1 *= delta_eff; db1 *= delta_eff; dW2 *= delta_eff; db2 *= delta_eff
                net.apply_update(dW1, db1, dW2, db2)
            else:
                # ----- store transition for replay -----
                rb.add(s_n, a, r, s2_n, a2, done, eps)

                # ----- sampled updates -----
                if len(rb) >= warmup:
                    for _ in range(updates_per_step):
                        batch = rb.sample(batch_size, rng)
                        s_b, a_b, r_b, s2_b, a2_b, d_b, eps_b = batch

                        # accumulate grads across batch
                        sum_dW1 = np.zeros_like(net.W1)
                        sum_db1 = np.zeros_like(net.b1)
                        sum_dW2 = np.zeros_like(net.W2)
                        sum_db2 = np.zeros_like(net.b2)

                        for i in range(s_b.shape[0]):
                            s_i = s_b[i]; a_i = int(a_b[i]); r_i = r_b[i]
                            s2_i = s2_b[i]; a2_i = int(a2_b[i]); d_i = bool(d_b[i]); eps_i = float(eps_b[i])

                            q_i, cache_i = net.forward(s_i)
                            # bootstrap from target network if enabled
                            q2_i, _ = target_net.forward(s2_i)

                            if algo == "sarsa":
                                tgt = r_i + (0.0 if d_i else gamma * q2_i[a2_i])
                            elif algo == "expected_sarsa":
                                tgt = r_i + (0.0 if d_i else gamma * expected_q(q2_i, eps_i))
                            else:
                                raise ValueError("algo must be 'sarsa' or 'expected_sarsa'")

                            delta_i = huber(tgt - q_i[a_i], k=huber_k)
                            dW1_i, db1_i, dW2_i, db2_i = net.grad_Q_params(cache_i, a_i)
                            sum_dW1 += dW1_i * delta_i
                            sum_db1 += db1_i * delta_i
                            sum_dW2 += dW2_i * delta_i
                            sum_db2 += db2_i * delta_i

                        # average gradients
                        bs = float(s_b.shape[0])
                        net.apply_update(sum_dW1/bs, sum_db1/bs, sum_dW2/bs, sum_db2/bs)

                        global_step += 1
                        if use_target and (global_step % int(target_sync) == 0):
                            target_net.hard_update_from(net)

            # advance rollout (action for next step already sampled)
            obs, s_n, q, cache = next_obs, s2_n, q2, cache2
            a = a2

        returns.append(total)
        eps = max(epsilon_end, eps * epsilon_decay)

        if (ep + 1) % print_every == 0:
            recent = returns[-print_every:]
            print(f"[{algo}{' + replay' if use_replay else ''}] ep {ep+1}/{episodes} | "
                  f"meanR={np.mean(recent):.1f} bestR={np.max(returns):.1f} eps={eps:.3f}")

    env.close()
    return net, np.array(returns), (low, high, action_n)

# ======================== Visualization ========================
def policy_map(net, low, high, grid=101):
    xs = np.linspace(low[0], high[0], grid)
    vs = np.linspace(low[1], high[1], grid)
    pm = np.zeros((grid, grid), dtype=np.int32)
    for i, v in enumerate(vs):
        for j, x in enumerate(xs):
            s = np.array([x, v], dtype=np.float64)
            s_n = normalize_obs(s, low, high)
            q, _ = net.forward(s_n)
            pm[i, j] = int(np.argmax(q))
    return xs, vs, pm

def plot_policy_map(net, low, high, title, grid=101, savepath=None):
    xs, vs, pm = policy_map(net, low, high, grid=grid)
    plt.figure(figsize=(7,5))
    extent = [xs.min(), xs.max(), vs.min(), vs.max()]
    plt.imshow(pm, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(label="Greedy action index (0=left,1=idle,2=right)")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title(title)
    if savepath:
        plt.savefig(savepath, bbox_inches="tight"); plt.close()
    else:
        plt.show()

def plot_returns(returns, title, window=50, savepath=None):
    plt.figure(figsize=(7,4))
    R = np.array(returns, dtype=float)
    if len(R) >= window:
        ma = np.convolve(R, np.ones(window)/float(window), mode="valid")
        plt.plot(np.arange(len(ma)), ma)
        plt.xlabel("Episode (smoothed)")
        plt.ylabel(f"Return ({window}-ep MA)")
    else:
        plt.plot(R)
        plt.xlabel("Episode")
        plt.ylabel("Return")
    plt.title(title)
    if savepath:
        plt.savefig(savepath, bbox_inches="tight"); plt.close()
    else:
        plt.show()

def greedy_rollout(env_id, net, low, high, max_steps=500, seed=0):
    env = gym.make(env_id, render_mode=None)
    obs, info = env.reset(seed=seed)
    traj = [obs.copy()]
    total = 0.0
    done = False
    steps = 0
    while not done and steps < max_steps:
        s_n = normalize_obs(obs, low, high)
        q, _ = net.forward(s_n)
        a = int(np.argmax(q))
        obs, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        total += r
        traj.append(obs.copy())
        steps += 1
    env.close()
    return np.array(traj), total

def plot_rollout_overlay(env_id, net, low, high, title, grid=101, seed=0, savepath=None):
    xs, vs, pm = policy_map(net, low, high, grid=grid)
    traj, ret = greedy_rollout(env_id, net, low, high, max_steps=500, seed=seed)
    plt.figure(figsize=(7,5))
    extent = [xs.min(), xs.max(), vs.min(), vs.max()]
    plt.imshow(pm, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(label="Greedy action index (0=left,1=idle,2=right)")
    plt.plot(traj[:,0], traj[:,1])
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title(title + f" — greedy rollout (return {ret:.0f})")
    if savepath:
        plt.savefig(savepath, bbox_inches="tight"); plt.close()
    else:
        plt.show()

# ======================== Video ========================
def record_greedy_rollout_video(env_id, net, low, high, path="mountaincar_greedy.mp4",
                                max_steps=500, seed=0):
    dirname = os.path.dirname(path) or "."
    basename = os.path.splitext(os.path.basename(path))[0]

    os.makedirs(dirname, exist_ok=True)
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=dirname,
        name_prefix=basename,
        episode_trigger=lambda e: True
    )
    obs, info = env.reset(seed=seed)
    total = 0.0
    done = False
    steps = 0
    while not done and steps < max_steps:
        s_n = normalize_obs(obs, low, high)
        q, _ = net.forward(s_n)
        a = int(np.argmax(q))
        obs, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        total += r
        steps += 1
    env.close()

    candidates = sorted(glob.glob(os.path.join(dirname, f"{basename}*.mp4")))
    out_path = candidates[-1] if candidates else path
    print(f"[Video] Greedy rollout saved to: {out_path} (return {total:.0f})")
    return out_path, total

# ======================== CLI ========================
def main():
    parser = argparse.ArgumentParser(description="Gym MountainCar-v0 with Semi-gradient SARSA(0)/Expected-SARSA + Experience Replay (NumPy NN)")
    parser.add_argument("--algo", type=str, default="both", choices=["sarsa", "expected_sarsa", "both"], help="Algorithm to run")
    parser.add_argument("--episodes", type=int, default=3000, help="Training episodes")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--alpha", type=float, default=1e-3, help="Learning rate (Adam)")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.997, help="Multiplicative epsilon decay per episode")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden units")
    parser.add_argument("--seed", type=int, default=50, help="Random seed")
    parser.add_argument("--save", action="store_true", help="Save plots instead of showing")
    parser.add_argument("--env-id", type=str, default="MountainCar-v0", help="Gymnasium env id")
    parser.add_argument("--max-grad-norm", type=float, default=10, help="Global grad-norm clip (None to disable)")
    parser.add_argument("--huber-k", type=float, default=5.0, help="Huber clipping for TD-error")
    parser.add_argument("--video", action="store_true", help="Record a greedy rollout video after training")
    parser.add_argument("--video-path", type=str, default=None, help="Output path for rollout video (.mp4)")
    # ---- replay args ----
    parser.add_argument("--no-replay", action="store_true", help="Disable experience replay (use pure on-policy updates)")
    parser.add_argument("--replay-size", type=int, default=50000, help="Replay buffer capacity")
    parser.add_argument("--batch-size", type=int, default=64, help="Replay mini-batch size")
    parser.add_argument("--warmup", type=int, default=1000, help="Number of transitions before starting replay updates")
    parser.add_argument("--updates-per-step", type=int, default=4, help="Gradient updates per environment step (when replay enabled)")
    parser.add_argument("--no-target", action="store_true", help="Disable target network (targets come from online net)")
    parser.add_argument("--target-sync", type=int, default=500, help="Hard target-net sync period (steps)")
    args = parser.parse_args()

    runs = ["sarsa", "expected_sarsa"] if args.algo == "both" else [args.algo]
    for i, algo in enumerate(runs):
        seed = args.seed + i
        net, returns, (low, high, action_n) = train(
            env_id=args.env_id, algo=algo, episodes=args.episodes, gamma=args.gamma, alpha=args.alpha,
            epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end, epsilon_decay=args.epsilon_decay,
            hidden=args.hidden, seed=seed, huber_k=args.huber_k, max_grad_norm=args.max_grad_norm,
            use_replay=not args.no_replay, replay_size=args.replay_size, batch_size=args.batch_size,
            warmup=args.warmup, updates_per_step=args.updates_per_step,
            use_target=not args.no_target, target_sync=args.target_sync
        )
        title = f"Semi-gradient {'Expected ' if algo=='expected_sarsa' else ''}SARSA(0){' + Replay' if not args.no_replay else ''}"
        base = f"gym_{algo}_eps{args.episodes}_seed{seed}"

        if args.save:
            plot_policy_map(net, low, high, title + " — Greedy Policy", grid=101, savepath=base + "_policy.png")
            plot_returns(returns, title + " — training returns (smoothed)", window=50, savepath=base + "_returns.png")
            plot_rollout_overlay(args.env_id, net, low, high, title, grid=101, seed=seed, savepath=base + "_rollout.png")
            print(f"[Saved] {base}_policy.png, {base}_returns.png, {base}_rollout.png")
        else:
            plot_policy_map(net, low, high, title + " — Greedy Policy", grid=101, savepath=None)
            plot_returns(returns, title + " — training returns (smoothed)", window=50, savepath=None)
            plot_rollout_overlay(args.env_id, net, low, high, title, grid=101, seed=seed, savepath=None)

        if args.video or args.save:
            vid_path = args.video_path or (base + "_greedy.mp4")
            out_path, ret = record_greedy_rollout_video(args.env_id, net, low, high, path=vid_path, seed=seed)
            print(f"[Video ready] {out_path} — return {ret:.0f}")

if __name__ == "__main__":
    main()
