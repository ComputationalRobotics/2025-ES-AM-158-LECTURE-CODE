import numpy as np
import matplotlib.pyplot as plt

# ------------------ Environment ------------------
class GridWorld:
    def __init__(self, n=5, start=(4,0), goal=(0,4), obstacles=None, max_steps=None, seed=0):
        self.n = n
        self.start = start
        self.goal = goal
        if obstacles is None:
            obstacles = {(1,1),(2,1),(3,1)}  # wall
        self.obstacles = set(obstacles)
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps or (n*n*4)
        self.reset()
        
    @property
    def n_states(self):
        return self.n * self.n
    
    @property
    def n_actions(self):
        return 4
    
    def to_state(self, pos):
        r, c = pos
        return r * self.n + c
    
    def from_state(self, s):
        return divmod(s, self.n)
    
    def reset(self, random_start=False):
        if random_start:
            free = [(r,c) for r in range(self.n) for c in range(self.n)
                    if (r,c) != self.goal and (r,c) not in self.obstacles]
            self.pos = free[self.rng.integers(len(free))]
        else:
            self.pos = self.start
        self.t = 0
        return self.to_state(self.pos)
    
    def step(self, a):
        self.t += 1
        r, c = self.pos
        if a == 0: nr, nc = r-1, c       # up
        elif a == 1: nr, nc = r, c+1     # right
        elif a == 2: nr, nc = r+1, c     # down
        elif a == 3: nr, nc = r, c-1     # left
        else: raise ValueError("Invalid action")
        
        if not (0 <= nr < self.n and 0 <= nc < self.n) or (nr,nc) in self.obstacles:
            nr, nc = r, c  # bump
        
        self.pos = (nr, nc)
        s_next = self.to_state(self.pos)
        done = (self.pos == self.goal) or (self.t >= self.max_steps)
        rwd = 0.0 if self.pos == self.goal else -1.0
        return s_next, rwd, done, {}
    
    def render_policy(self, policy):
        arrows = {0:'^', 1:'>', 2:'v', 3:'<'}
        lines = []
        for r in range(self.n):
            line = []
            for c in range(self.n):
                if (r,c) in self.obstacles:
                    line.append('#')
                elif (r,c) == self.goal:
                    line.append('G')
                else:
                    s = self.to_state((r,c))
                    line.append(arrows[int(policy[s])])
            lines.append(' '.join(line))
        return '\n'.join(lines)


# --------------- DP: compute Q* via value iteration ---------------
def deterministic_step(env, s, a):
    r, c = env.from_state(s)
    if (r,c) == env.goal:
        return s, 0.0, True
    if a == 0: nr, nc = r-1, c
    elif a == 1: nr, nc = r, c+1
    elif a == 2: nr, nc = r+1, c
    elif a == 3: nr, nc = r, c-1
    else: raise ValueError
    
    if not (0 <= nr < env.n and 0 <= nc < env.n) or (nr,nc) in env.obstacles:
        nr, nc = r, c
    s_next = env.to_state((nr,nc))
    done = (nr, nc) == env.goal
    rwd = 0.0 if done else -1.0
    return s_next, rwd, done

def value_iteration_Qstar(env, gamma=0.99, tol=1e-10, max_iter=10_000):
    nS, nA = env.n_states, env.n_actions
    V = np.zeros(nS, dtype=float)
    goal_s = env.to_state(env.goal)
    for it in range(max_iter):
        delta = 0.0
        V_new = V.copy()
        for s in range(nS):
            if s == goal_s or env.from_state(s) in env.obstacles:
                V_new[s] = 0.0
                continue
            q_vals = []
            for a in range(nA):
                s_next, rwd, done = deterministic_step(env, s, a)
                q_vals.append(rwd + gamma * V[s_next] * (0.0 if done else 1.0))
            best = max(q_vals)
            delta = max(delta, abs(best - V[s]))
            V_new[s] = best
        V = V_new
        if delta < tol:
            break
    Q_star = np.zeros((nS, nA), dtype=float)
    for s in range(nS):
        for a in range(nA):
            s_next, rwd, done = deterministic_step(env, s, a)
            Q_star[s, a] = rwd + gamma * V[s_next] * (0.0 if done else 1.0)
        if s == goal_s or env.from_state(s) in env.obstacles:
            Q_star[s,:] = 0.0
    return Q_star

# --------------- Helpers ---------------
def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return rng.integers(Q.shape[1])
    q = Q[s]
    max_q = np.max(q)
    max_acts = np.flatnonzero(q == max_q)
    return int(rng.choice(max_acts))

def greedy_policy_from_Q(Q):
    return np.argmax(Q, axis=1)

def rollout(env, policy, max_steps=None):
    s = env.reset(random_start=False)
    total_r = 0.0
    steps = 0
    max_steps = max_steps or env.max_steps
    done = False
    while not done and steps < max_steps:
        a = int(policy[s])
        s, r, done, _ = env.step(a)
        total_r += r
        steps += 1
    return total_r, steps, done

def mask_valid_states(env):
    valid = np.ones(env.n_states, dtype=bool)
    for (r,c) in env.obstacles:
        valid[env.to_state((r,c))] = False
    return valid

def rmse_Q(Q, Qstar, valid_mask):
    diff = (Q - Qstar)[valid_mask]
    return float(np.sqrt(np.mean(diff**2)))

def linf_Q(Q, Qstar, valid_mask):
    diff = (Q - Qstar)[valid_mask]
    return float(np.max(np.abs(diff)))

# --------------- Schedules ---------------
def harmonic_eps(ep, eps0=1.0, eps_min=0.01, scale=50.0):
    eps = eps0 / (1.0 + ep/scale)
    return eps if eps > eps_min else eps_min

def stepsize_from_count(count, alpha0=1.0, beta=1.0):
    return alpha0 / (count ** beta)

# --------------- Algorithms with decaying step sizes ---------------
def mc_control_with_error(env, episodes=5000, gamma=0.99, eps0=1.0, eps_min=0.01, Q_star=None, valid_mask=None, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    N = np.zeros_like(Q, dtype=np.int64)
    rmse_hist, linf_hist = [], []
    
    for ep in range(episodes):
        eps = harmonic_eps(ep, eps0=eps0, eps_min=eps_min, scale=50.0)
        s = env.reset(random_start=True)
        traj = []
        done = False
        while not done:
            a = epsilon_greedy(Q, s, eps, rng)
            s_next, rwd, done, _ = env.step(a)
            traj.append((s, a, rwd))
            s = s_next
            if len(traj) >= env.max_steps: break
      
        # reverse traversal to compute returns
        G = 0.0
        returns = [0.0] * len(traj)
        for t in reversed(range(len(traj))):
            _, _, r_t1 = traj[t]
            G = gamma * G + r_t1
            returns[t] = G

        # Every-visit MC update with sample-average stepsizes
        for t, (s_t, a_t, _) in enumerate(traj):
            N[s_t, a_t] += 1
            alpha = 1 / N[s_t, a_t] 
            Q[s_t, a_t] += alpha * (returns[t] - Q[s_t, a_t])
        
        if Q_star is not None:
            rmse_hist.append(rmse_Q(Q, Q_star, valid_mask))
            linf_hist.append(linf_Q(Q, Q_star, valid_mask))
    return Q, np.array(rmse_hist), np.array(linf_hist)

def sarsa_with_error(env, episodes=5000, gamma=0.99, alpha0=1.0, beta=1.0, eps0=1.0, eps_min=0.01, Q_star=None, valid_mask=None, seed=1):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    N = np.zeros_like(Q, dtype=np.int64)
    rmse_hist, linf_hist = [], []
    for ep in range(episodes):
        eps = harmonic_eps(ep, eps0=eps0, eps_min=eps_min, scale=50.0)
        s = env.reset(random_start=False)
        a = epsilon_greedy(Q, s, eps, rng)
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            s_next, rwd, done, _ = env.step(a)
            if not done:
                a_next = epsilon_greedy(Q, s_next, eps, rng)
                target = rwd + gamma * Q[s_next, a_next]
            else:
                target = rwd
            N[s, a] += 1
            alpha = stepsize_from_count(N[s, a], alpha0=alpha0, beta=beta)
            td = target - Q[s, a]
            Q[s, a] += alpha * td
            s, a = s_next, (a_next if not done else a)
            steps += 1
        if Q_star is not None:
            rmse_hist.append(rmse_Q(Q, Q_star, valid_mask))
            linf_hist.append(linf_Q(Q, Q_star, valid_mask))
    return Q, np.array(rmse_hist), np.array(linf_hist)

def expected_sarsa_with_error(env, episodes=5000, gamma=0.99, alpha0=1.0, beta=1.0, eps0=1.0, eps_min=0.01, Q_star=None, valid_mask=None, seed=2):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    N = np.zeros_like(Q, dtype=np.int64)
    rmse_hist, linf_hist = [], []
    for ep in range(episodes):
        eps = harmonic_eps(ep, eps0=eps0, eps_min=eps_min, scale=50.0)
        s = env.reset(random_start=False)
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            a = epsilon_greedy(Q, s, eps, rng)
            s_next, rwd, done, _ = env.step(a)
            if not done:
                q_next = Q[s_next]
                a_star_idx = np.flatnonzero(q_next == np.max(q_next))
                pi = np.full(env.n_actions, eps / env.n_actions)
                pi[a_star_idx] += (1 - eps) / len(a_star_idx)
                exp_q = float(np.dot(pi, q_next))
                target = rwd + gamma * exp_q
            else:
                target = rwd
            N[s, a] += 1
            alpha = stepsize_from_count(N[s, a], alpha0=alpha0, beta=beta)
            td = target - Q[s, a]
            Q[s, a] += alpha * td
            s = s_next
            steps += 1
        if Q_star is not None:
            rmse_hist.append(rmse_Q(Q, Q_star, valid_mask))
            linf_hist.append(linf_Q(Q, Q_star, valid_mask))
    return Q, np.array(rmse_hist), np.array(linf_hist)

def q_learning_with_error(env, episodes=5000, gamma=0.99, alpha0=1.0, beta=1.0,
                          eps0=1.0, eps_min=0.01, Q_star=None, valid_mask=None, seed=3):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    N = np.zeros_like(Q, dtype=np.int64)
    rmse_hist, linf_hist = [], []
    for ep in range(episodes):
        eps = harmonic_eps(ep, eps0=eps0, eps_min=eps_min, scale=50.0)
        s = env.reset(random_start=False)
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            a = epsilon_greedy(Q, s, eps, rng)  # behavior
            s_next, rwd, done, _ = env.step(a)
            if not done:
                target = rwd + gamma * np.max(Q[s_next])  # greedy bootstrap
            else:
                target = rwd
            N[s, a] += 1
            alpha = stepsize_from_count(N[s, a], alpha0=alpha0, beta=beta)
            td = target - Q[s, a]
            Q[s, a] += alpha * td
            s = s_next
            steps += 1
        if Q_star is not None:
            rmse_hist.append(rmse_Q(Q, Q_star, valid_mask))
            linf_hist.append(linf_Q(Q, Q_star, valid_mask))
    return Q, np.array(rmse_hist), np.array(linf_hist)


def double_q_learning_with_error(env, episodes=5000, gamma=0.99, alpha0=1.0, beta=1.0,
                                 eps0=1.0, eps_min=0.01, Q_star=None, valid_mask=None, seed=4):
    rng = np.random.default_rng(seed)
    QA = np.zeros((env.n_states, env.n_actions))
    QB = np.zeros((env.n_states, env.n_actions))
    NA = np.zeros_like(QA, dtype=np.int64)
    NB = np.zeros_like(QB, dtype=np.int64)
    rmse_hist, linf_hist = [], []
    for ep in range(episodes):
        eps = harmonic_eps(ep, eps0=eps0, eps_min=eps_min, scale=50.0)
        s = env.reset(random_start=False)
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            # Behavior uses ε-greedy wrt the combined estimate
            Qsum = QA + QB
            a = epsilon_greedy(Qsum, s, eps, rng)
            s_next, rwd, done, _ = env.step(a)

            # Randomly choose which table to update (A or B)
            if rng.random() < 0.5:
                NA[s, a] += 1
                alpha = stepsize_from_count(NA[s, a], alpha0=alpha0, beta=beta)
                if not done:
                    a_star = np.argmax(QA[s_next])          # argmax w.r.t. QA
                    target = rwd + gamma * QB[s_next, a_star]  # evaluate with QB
                else:
                    target = rwd
                td = target - QA[s, a]
                QA[s, a] += alpha * td
            else:
                NB[s, a] += 1
                alpha = stepsize_from_count(NB[s, a], alpha0=alpha0, beta=beta)
                if not done:
                    a_star = np.argmax(QB[s_next])          # argmax w.r.t. QB
                    target = rwd + gamma * QA[s_next, a_star]  # evaluate with QA
                else:
                    target = rwd
                td = target - QB[s, a]
                QB[s, a] += alpha * td

            s = s_next
            steps += 1

        if Q_star is not None:
            Qavg = 0.5 * (QA + QB)
            rmse_hist.append(rmse_Q(Qavg, Q_star, valid_mask))
            linf_hist.append(linf_Q(Qavg, Q_star, valid_mask))
    Qavg = 0.5 * (QA + QB)
    return Qavg, np.array(rmse_hist), np.array(linf_hist)


# ------------------- Run & evaluate -------------------
env = GridWorld(max_steps=500)
gamma = 0.99
Q_star = value_iteration_Qstar(env, gamma=gamma)
valid_mask = mask_valid_states(env)

episodes = 5000
Q_mc, mc_rmse, mc_linf = mc_control_with_error(env, episodes=episodes, gamma=gamma,
                                               eps0=1.0, eps_min=0.3,
                                               Q_star=Q_star, valid_mask=valid_mask, seed=0)
Q_sa, sa_rmse, sa_linf = sarsa_with_error(env, episodes=episodes, gamma=gamma,
                                          alpha0=1.0, beta=1.0,
                                          eps0=1.0, eps_min=0.01,
                                          Q_star=Q_star, valid_mask=valid_mask, seed=1)
Q_es, es_rmse, es_linf = expected_sarsa_with_error(env, episodes=episodes, gamma=gamma,
                                                   alpha0=1.0, beta=1.0,
                                                   eps0=1.0, eps_min=0.01,
                                                   Q_star=Q_star, valid_mask=valid_mask, seed=2)
Q_ql, ql_rmse, ql_linf = q_learning_with_error(env, episodes=episodes, gamma=gamma,
                                               alpha0=1.0, beta=1.0,
                                               eps0=1.0, eps_min=0.01,
                                               Q_star=Q_star, valid_mask=valid_mask, seed=3)

Q_dq, dq_rmse, dq_linf = double_q_learning_with_error(
    env, episodes=episodes, gamma=0.99, alpha0=1.0, beta=1.0,
    eps0=1.0, eps_min=0.01, Q_star=Q_star, valid_mask=valid_mask, seed=4
)

# Final errors
def sup_errors(Q, Q_star, mask):
    return float(np.max(np.abs((Q-Q_star)[mask])))
print("Final Linf errors:")
print("MC:", sup_errors(Q_mc, Q_star, valid_mask))
print("SARSA:", sup_errors(Q_sa, Q_star, valid_mask))
print("Expected SARSA:", sup_errors(Q_es, Q_star, valid_mask))
print("Q-Learning:", sup_errors(Q_ql, Q_star, valid_mask))
print("Double Q-Learning:", sup_errors(Q_dq, Q_star, valid_mask))

# Plot RMSE vs episodes
plt.figure(figsize=(8,4.2))
plt.plot(mc_rmse, label="MC Control")
plt.plot(sa_rmse, label="SARSA (decay)")
plt.plot(es_rmse, label="Expected SARSA (decay)")
plt.plot(ql_rmse, label="Q-Learning (decay)")
plt.plot(dq_rmse, label="Double Q-Learning (decay)")
plt.xlabel("Episodes")
plt.ylabel("RMSE(Q, Q*)")
plt.title("Convergence of Q-estimates to Q* on GridWorld")
plt.legend()
plt.tight_layout()
plt.show()

# Policies for inspection
pol_mc = greedy_policy_from_Q(Q_mc)
pol_sa = greedy_policy_from_Q(Q_sa)
pol_es = greedy_policy_from_Q(Q_es)
pol_ql = greedy_policy_from_Q(Q_ql)
pol_dq = greedy_policy_from_Q(Q_dq)
print("\nGreedy policies (arrows):")
print("\nMC Control:\n", env.render_policy(pol_mc))
print("\nSARSA:\n", env.render_policy(pol_sa))
print("\nExpected SARSA:\n", env.render_policy(pol_es))
print("\nQ-Learning:\n", env.render_policy(pol_ql))
print("\nDouble Q-Learning greedy policy:\n", env.render_policy(pol_dq))
