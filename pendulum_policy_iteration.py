import numpy as np
import matplotlib.pyplot as plt

# ----- Physical & MDP parameters -----
g, l, m, c = 9.81, 1.0, 1.0, 0.1
dt = 0.05
gamma = 0.97
eps = 1e-8

# Grids
N_theta = 51
N_thetadot = 51
N_u = 21

theta_grid = np.linspace(-1.5*np.pi, 1.5*np.pi, N_theta)
thetadot_grid = np.linspace(-1.5*np.pi, 1.5*np.pi, N_thetadot)
u_max = 0.5 * m * g * l
u_grid = np.linspace(-u_max, u_max, N_u)

# Helpers to index/unwrap
def wrap_angle(x):
    return np.arctan2(np.sin(x), np.cos(x))

def state_index(i, j):
    return i * N_thetadot + j

def index_to_state(idx):
    i = idx // N_thetadot
    j = idx % N_thetadot
    return theta_grid[i], thetadot_grid[j]

S = N_theta * N_thetadot
A = N_u

# ----- Dynamics step (continuous -> one Euler step) -----
def step_euler(theta, thetadot, u):
    theta_next = wrap_angle(theta + dt * thetadot)
    thetadot_next = thetadot + dt * ((g/l) * np.sin(theta) + (1/(m*l*l))*u - c*thetadot)
    # clip angular velocity to grid range (bounded MDP)
    thetadot_next = np.clip(thetadot_next, thetadot_grid[0], thetadot_grid[-1])
    return theta_next, thetadot_next

# ----- Find 3 nearest grid states and probability weights (inverse-distance) -----
grid_pts = np.stack(np.meshgrid(theta_grid, thetadot_grid, indexing='ij'), axis=-1).reshape(-1, 2)

def nearest3_probs(theta_next, thetadot_next):
    x = np.array([theta_next, thetadot_next])
    dists = np.linalg.norm(grid_pts - x[None, :], axis=1)
    nn_idx = np.argpartition(dists, 3)[:3]      # three smallest (unordered)
    nn_idx = nn_idx[np.argsort(dists[nn_idx])]  # sort those 3 by distance
    d = dists[nn_idx]
    w = 1.0 / (d + eps)
    p = w / w.sum()
    return nn_idx.astype(int), p

# ----- Reward -----
def reward(theta, thetadot, u):
    return -(theta**2 + 0.1*thetadot**2 + 0.01*u**2)

# ----- Build tabular MDP: R[s,a] and sparse P[s,a,3] -----
R = np.zeros((S, A))
NS_idx = np.zeros((S, A, 3), dtype=int)   # next-state indices (3 nearest)
NS_prob = np.zeros((S, A, 3))             # their probabilities

for i, th in enumerate(theta_grid):
    for j, thd in enumerate(thetadot_grid):
        s = state_index(i, j)
        for a, u in enumerate(u_grid):
            # reward at current (s,a)
            R[s, a] = reward(th, thd, u)
            # next continuous state
            th_n, thd_n = step_euler(th, thd, u)
            # map to 3 nearest grid states
            nn_idx, p = nearest3_probs(th_n, thd_n)
            NS_idx[s, a, :] = nn_idx
            NS_prob[s, a, :] = p

# =======================
#     POLICY ITERATION
# =======================

# Represent policy as a deterministic action index per state: pi[s] in {0..A-1}
# Start from uniform-random policy (deterministic tie-breaker: middle action)
pi = np.full(S, A // 2, dtype=int)

def policy_evaluation(pi, V_init=None, tol=1e-6, max_iters=10000):
    """Iterative policy evaluation for deterministic pi (action index per state)."""
    V = np.zeros(S) if V_init is None else V_init.copy()
    for k in range(max_iters):
        # For each state s, use chosen action a = pi[s]
        a = pi  # shape (S,)
        # Expected next value under chosen action
        EV_next = (NS_prob[np.arange(S), a] * V[NS_idx[np.arange(S), a]]).sum(axis=1)  # (S,)
        V_new = R[np.arange(S), a] + gamma * EV_next
        if np.max(np.abs(V_new - V)) < tol:
            # print(f"Policy evaluation converged in {k+1} iterations.")
            return V_new
        V = V_new
    # print("Policy evaluation reached max_iters without meeting tolerance.")
    return V

def policy_improvement(V, pi_old=None):
    """Greedy improvement: pi'(s) = argmax_a [ R(s,a) + gamma * E[V(s')] ]."""
    # Compute Q(s,a) = R + gamma * sum_j P(s,a,j) V(ns_j)
    EV_next = (NS_prob * V[NS_idx]).sum(axis=2)      # (S, A)
    Q = R + gamma * EV_next                           # (S, A)
    pi_new = np.argmax(Q, axis=1).astype(int)         # greedy deterministic policy
    stable = (pi_old is not None) and np.array_equal(pi_new, pi_old)
    return pi_new, stable

# Main PI loop
max_pi_iters = 100
V = np.zeros(S)
for it in range(max_pi_iters):
    # Policy evaluation
    V = policy_evaluation(pi, V_init=V, tol=1e-6, max_iters=10000)
    # Policy improvement
    pi_new, stable = policy_improvement(V, pi_old=pi)
    print(f"[PI] Iter {it+1}: policy changed = {not stable}")
    pi = pi_new
    if stable:
        print("Policy iteration converged: policy stable.")
        break
else:
    print("Reached max_pi_iters without policy stability (may still be near-optimal).")

# ----- Visualization -----
V_grid = V.reshape(N_theta, N_thetadot)

fig, ax = plt.subplots(figsize=(7,5), dpi=120)
im = ax.imshow(
    V_grid,
    origin="lower",
    extent=[thetadot_grid.min(), thetadot_grid.max(),
            theta_grid.min(), theta_grid.max()],
    aspect="auto",
    cmap="viridis"
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$V^{\pi}(\theta,\dot{\theta})$ (final PI)")

ax.set_xlabel(r"$\dot{\theta}$")
ax.set_ylabel(r"$\theta$")
ax.set_title(r"State-value $V$ after Policy Iteration")

plt.tight_layout()
plt.show()

# Visualize the greedy action *value* (torque)
pi_grid = pi.reshape(N_theta, N_thetadot)          # action indices
action_values = u_grid[pi_grid]                    # map indices -> torques

plt.figure(figsize=(7,5), dpi=120)
im = plt.imshow(action_values,
           origin="lower",
           extent=[thetadot_grid.min(), thetadot_grid.max(),
                   theta_grid.min(), theta_grid.max()],
           aspect="auto", cmap="coolwarm")         # diverging colormap good for ± torque
cbar = plt.colorbar(im)
cbar.set_label("Greedy action value (torque)")

plt.xlabel(r"$\dot{\theta}$")
plt.ylabel(r"$\theta$")
plt.title("Greedy policy (torque) after PI")
plt.tight_layout()
plt.show()

# =======================
#   ROLLOUT UNDER pi*
# =======================
def nearest_state_idx(theta, thetadot):
    """Index of the closest grid state to (theta, thetadot)."""
    x = np.array([theta, thetadot])
    dists = np.linalg.norm(grid_pts - x[None, :], axis=1)
    return int(np.argmin(dists))

# Initial state and rollout settings
theta0, thetadot0 = -np.pi, 0.0
H = 400  # horizon (steps); 400 * 0.05s = 20 seconds of simulated time

thetas = np.zeros(H+1)
thetadots = np.zeros(H+1)
us = np.zeros(H)
rewards = np.zeros(H)
discounts = np.power(gamma, np.arange(H))

thetas[0], thetadots[0] = wrap_angle(theta0), thetadot0

for t in range(H):
    # Choose greedy action from the nearest tabular state
    s_idx = nearest_state_idx(thetas[t], thetadots[t])
    a_idx = pi[s_idx]
    u = u_grid[a_idx]
    us[t] = u

    # Reward and next state
    rewards[t] = reward(thetas[t], thetadots[t], u)
    th_next, thd_next = step_euler(thetas[t], thetadots[t], u)

    thetas[t+1] = th_next
    thetadots[t+1] = thd_next

G = float((discounts * rewards).sum())
print(f"Discounted return from [-pi, 0] under pi*: {G:.3f}")

# ----- Plot time series -----
t_axis = np.arange(H+1) * dt

fig, axs = plt.subplots(3, 1, figsize=(9,8), sharex=True, dpi=120)
axs[0].plot(t_axis, thetas, lw=1.5)
axs[0].set_ylabel(r"$\theta$ (rad)")
axs[0].set_title(r"Rollout under $\pi^*$ from $[-\pi,0]$")

axs[1].plot(t_axis, thetadots, lw=1.5)
axs[1].set_ylabel(r"$\dot{\theta}$ (rad/s)")

axs[2].plot(t_axis[:-1], us, lw=1.5)
axs[2].set_xlabel("time (s)")
axs[2].set_ylabel("torque u")

plt.tight_layout()
plt.show()

# ----- Overlay trajectory on value heatmap -----
fig, ax = plt.subplots(figsize=(7,5), dpi=120)
im = ax.imshow(
    V_grid,
    origin="lower",
    extent=[thetadot_grid.min(), thetadot_grid.max(),
            theta_grid.min(), theta_grid.max()],
    aspect="auto",
    cmap="viridis"
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$V^{\pi^*}(\theta,\dot{\theta})$")

ax.plot(thetadots, thetas, color="white", lw=1.5, alpha=0.9, label="trajectory")
ax.scatter([thetadots[0]], [thetas[0]], c="red", s=30, label="start")
ax.set_xlabel(r"$\dot{\theta}$")
ax.set_ylabel(r"$\theta$")
ax.set_title(r"Trajectory under $\pi^*$ on value landscape")
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()


