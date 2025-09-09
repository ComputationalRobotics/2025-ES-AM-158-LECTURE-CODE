import numpy as np
import matplotlib.pyplot as plt

# ----- Physical & MDP parameters -----
g, l, m, c = 9.81, 1.0, 1.0, 0.1
dt = 0.05
gamma = 0.97
eps = 1e-8

# Grids
N_theta = 101
N_thetadot = 101
N_u = 51

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
            R[s, a] = reward(th, thd, u)
            th_n, thd_n = step_euler(th, thd, u)
            nn_idx, p = nearest3_probs(th_n, thd_n)
            NS_idx[s, a, :] = nn_idx
            NS_prob[s, a, :] = p

# =======================
#       VALUE ITERATION
# =======================

# Bellman optimality update:
# V_{k+1}(s) = max_a [ R(s,a) + gamma * sum_j P(s,a,j) * V_k(ns_j) ]
V = np.zeros(S)
tol = 1e-6
max_vi_iters = 1000

for k in range(max_vi_iters):
    # Expected next V for every (s,a), given current V_k
    EV_next = (NS_prob * V[NS_idx]).sum(axis=2)   # shape (S, A)
    Q = R + gamma * EV_next                       # shape (S, A)
    V_new = np.max(Q, axis=1)                     # greedy backup over actions

    delta = np.max(np.abs(V_new - V))
    # Optional: a stopping rule aligned with policy loss bound could scale tol
    # e.g., stop when delta <= tol * (1 - gamma) / (2 * gamma)
    if delta < tol:
        V = V_new
        print(f"Value Iteration converged in {k+1} iterations (sup-norm change {delta:.2e}).")
        break
    V = V_new
else:
    print(f"Reached max_vi_iters={max_vi_iters} (last sup-norm change {delta:.2e}).")

# Greedy policy extraction from the final V
EV_next = (NS_prob * V[NS_idx]).sum(axis=2)   # recompute with final V
Q = R + gamma * EV_next
pi = np.argmax(Q, axis=1)                     # deterministic greedy policy (indices)

# ----- Visualization: Value function -----
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
cbar.set_label(r"$V^*(\theta,\dot{\theta})$ (Value Iteration)")

ax.set_xlabel(r"$\dot{\theta}$")
ax.set_ylabel(r"$\theta$")
ax.set_title(r"State-value $V$ after Value Iteration")

plt.tight_layout()
plt.show()

# ----- Visualization: Greedy torque field -----
pi_grid = pi.reshape(N_theta, N_thetadot)   # action indices
action_values = u_grid[pi_grid]             # map indices -> torques

plt.figure(figsize=(7,5), dpi=120)
im = plt.imshow(
    action_values,
    origin="lower",
    extent=[thetadot_grid.min(), thetadot_grid.max(),
            theta_grid.min(), theta_grid.max()],
    aspect="auto",
    cmap="coolwarm"   # good for ± torque
)
cbar = plt.colorbar(im)
cbar.set_label("Greedy action value (torque)")

plt.xlabel(r"$\dot{\theta}$")
plt.ylabel(r"$\theta$")
plt.title("Greedy policy (torque) extracted from Value Iteration")
plt.tight_layout()
plt.show()
