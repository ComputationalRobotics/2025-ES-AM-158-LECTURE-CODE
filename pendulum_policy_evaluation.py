import numpy as np
import matplotlib.pyplot as plt

# ----- Physical & MDP parameters -----
g, l, m, c = 9.81, 1.0, 1.0, 0.1
dt = 0.05
gamma = 0.97
eps = 1e-8

# Grids
N_theta = 21
N_thetadot = 21
N_u = 1

theta_grid = np.linspace(-np.pi, np.pi, N_theta)
thetadot_grid = np.linspace(-np.pi, np.pi, N_thetadot)
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
# Pre-compute all grid points for fast nearest neighbor search
grid_pts = np.stack(np.meshgrid(theta_grid, thetadot_grid, indexing='ij'), axis=-1).reshape(-1, 2)

def nearest3_probs(theta_next, thetadot_next):
    x = np.array([theta_next, thetadot_next])
    dists = np.linalg.norm(grid_pts - x[None, :], axis=1)
    nn_idx = np.argpartition(dists, 3)[:3]  # three smallest (unordered)
    # sort those 3 by distance for stability
    nn_idx = nn_idx[np.argsort(dists[nn_idx])]
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
NS_prob = np.zeros((S, A, 3))            # their probabilities

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

# ----- Fixed policy: uniform over actions -----
Pi = np.full((S, A), 1.0 / A)

# ----- Iterative policy evaluation -----
V = np.zeros(S)  # initialization (any vector works)
tol = 1e-6
max_iters = 10000

for k in range(max_iters):
    V_new = np.zeros_like(V)
    # Compute Bellman update: V_{k+1}(s) = sum_a Pi(s,a)[ R(s,a) + gamma * sum_j P(s,a,j) V_k(ns_j) ]
    # First, expected next V for each (s,a)
    EV_next = (NS_prob * V[NS_idx]).sum(axis=2)  # shape: (S, A)
    # Then expectation over actions under Pi
    V_new = (Pi * (R + gamma * EV_next)).sum(axis=1)  # shape: (S,)
    # Check convergence
    if np.max(np.abs(V_new - V)) < tol:
        V = V_new
        print(f"Converged in {k+1} iterations (sup-norm change < {tol}).")
        break
    V = V_new
else:
    print(f"Reached max_iters={max_iters} without meeting tolerance {tol}.")

V_grid = V.reshape(N_theta, N_thetadot)

# V_grid: shape (N_theta, N_thetadot)
# theta_grid, thetadot_grid already defined
fig, ax = plt.subplots(figsize=(7,5), dpi=120)
im = ax.imshow(
    V_grid,
    origin="lower",
    extent=[thetadot_grid.min(), thetadot_grid.max(),
            theta_grid.min(), theta_grid.max()],
    aspect="auto",
    cmap="viridis"  # any matplotlib colormap, e.g., "plasma", "inferno"
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$V^\pi(\theta,\dot{\theta})$")

ax.set_xlabel(r"$\dot{\theta}$")
ax.set_ylabel(r"$\theta$")
ax.set_title(r"State-value $V^\pi$ (tabular policy evaluation)")

plt.tight_layout()
plt.show()

