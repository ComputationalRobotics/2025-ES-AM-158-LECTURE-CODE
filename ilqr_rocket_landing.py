"""
iLQR for a planar rocket (2D) soft landing.

State:   x = [px, py, vx, vy, theta, omega]
Control: u = [T, tau]
  - T    : thrust magnitude (acts along the body axis; positive "through the engine")
  - tau  : control torque (simplified as a direct input to angular acceleration)

Continuous-time dynamics:
    pxdot   = vx
    pydot   = vy
    vxdot   = (T/m) * sin(theta)
    vydot   = (T/m) * cos(theta) - g
    thetadot= omega
    omegadot= tau / I

We discretize with RK4 for the *forward rollout* (true dynamics).
For linearization, we use the continuous-time Jacobians (Ac, Bc) and map to discrete via
    A = I + dt * Ac,   B = dt * Bc
which is a standard first-order accurate approximation.

Cost:
  Running:    0.5 * (x - x_goal)^T Q (x - x_goal) + 0.5 * u^T R u
  Terminal:    0.5 * (x_N - x_goal)^T Qf (x_N - x_goal)

No control limits are imposed (as requested).
We perform iLQR with line search that scales only the feedforward term (alpha * k_t),
NOT the feedback (K_t).

This script plots:
  1) Initial nominal trajectory (dashed)
  2) Intermediate iLQR trajectories over iterations (light)
  3) Final optimized trajectory (bold)

Author notes:
- The model is intentionally simple (thrust along body axis, torque is direct).
- For more realism, add dry mass/prop mass, thrust vectoring limits, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

# -----------------------------
# Problem parameters
# -----------------------------
g   = 9.81         # gravity [m/s^2]
m   = 1.0          # mass [kg]
Izz = 0.2          # planar moment of inertia [kg·m^2]

dt       = 0.05     # discretization step [s]
T_final  = 6.0      # horizon [s]
N        = int(T_final / dt)

# Initial state: start high, a bit of horizontal drift, slight tilt
x0 = np.array([
    5.0,     # px [m] (horizontal offset)
    10.0,    # py [m] (altitude)
    -0.5,    # vx [m/s]
    -1.0,    # vy [m/s]
    np.deg2rad(10.0),  # theta [rad]
    0.0      # omega [rad/s]
])

# Goal (soft landing at origin, upright, zero velocities)
x_goal = np.zeros(6)  # [0,0,0,0,0,0]

# Cost weights (tune as desired)
Q  = np.diag([  1.0,   2.0,  0.5,  0.5,   2.0,  0.5])   # running state weights
R  = np.diag([1e-3, 1e-3])                              # running control weights
Qf = np.diag([200.0, 300.0, 50.0, 50.0, 300.0, 50.0])   # terminal weights

# Line search candidates (alpha scales feedforward only)
ALPHAS = [1.0, 0.5, 0.25, 0.125, 0.0625]

# iLQR parameters
MAX_ILQR_ITERS = 60
COST_TOL       = 1e-6
REG_INIT       = 1e-6       # Levenberg-Marquardt lambda
REG_UP         = 10.0
REG_DOWN       = 0.3
REG_MAX        = 1e8

# -----------------------------
# Rocket continuous dynamics + Jacobians
# -----------------------------
def f_continuous(x, u):
    px, py, vx, vy, th, om = x
    T, tau = u
    s = np.sin(th); c = np.cos(th)
    xdot = np.array([
        vx,
        vy,
        (T/m) * s,
        (T/m) * c - g,
        om,
        tau / Izz
    ])
    return xdot

def jacobians_continuous(x, u):
    """Return Ac = df/dx, Bc = df/d u (continuous time)."""
    px, py, vx, vy, th, om = x
    T, tau = u
    s = np.sin(th); c = np.cos(th)

    Ac = np.zeros((6,6))
    # d/dx
    # pxdot depends on vx
    Ac[0,2] = 1.0
    # pydot depends on vy
    Ac[1,3] = 1.0
    # vxdot depends on theta via sin(theta) * T/m
    Ac[2,4] = (T/m) * c
    # vydot depends on theta via cos(theta) * T/m -> derivative -sin * T/m
    Ac[3,4] = -(T/m) * s
    # thetadot depends on omega
    Ac[4,5] = 1.0
    # omegadot no state dependence (in this simple model)

    Bc = np.zeros((6,2))
    # d/dT
    Bc[2,0] = (1.0/m) * s
    Bc[3,0] = (1.0/m) * c
    # d/dtau
    Bc[5,1] = 1.0 / Izz

    return Ac, Bc

# -----------------------------
# Discretization helpers
# -----------------------------
def rk4_step(x, u, dt):
    """One RK4 step for the continuous dynamics."""
    k1 = f_continuous(x, u)
    k2 = f_continuous(x + 0.5*dt*k1, u)
    k3 = f_continuous(x + 0.5*dt*k2, u)
    k4 = f_continuous(x + dt*k3,     u)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def discretize_jacobians(x, u, dt):
    """First-order discrete-time Jacobians from continuous-time ones."""
    Ac, Bc = jacobians_continuous(x, u)
    n, m = Ac.shape[0], Bc.shape[1]
    A = np.eye(n) + dt * Ac
    B = dt * Bc
    return A, B

# -----------------------------
# Cost functions + derivatives
# -----------------------------
def stage_cost(x, u):
    dx = x - x_goal
    return 0.5 * dx @ (Q @ dx) + 0.5 * u @ (R @ u)

def terminal_cost(xN):
    dx = xN - x_goal
    return 0.5 * dx @ (Qf @ dx)

def stage_cost_derivatives(x, u):
    """Return lx, lu, lxx, luu, lux for quadratic cost."""
    dx = x - x_goal
    lx  = Q @ dx
    lu  = R @ u
    lxx = Q
    luu = R
    lux = np.zeros((u.shape[0], x.shape[0]))  # no cross terms in this quadratic
    return lx, lu, lxx, luu, lux

def terminal_cost_derivatives(xN):
    dx  = xN - x_goal
    Vx  = Qf @ dx
    Vxx = Qf
    return Vx, Vxx

def trajectory_cost(xs, us):
    J = np.sum([stage_cost(xs[t], us[t]) for t in range(us.shape[0])])
    J += terminal_cost(xs[-1])
    return J

# -----------------------------
# Rollouts
# -----------------------------
def rollout(x0, us):
    xs = np.zeros((us.shape[0] + 1, x0.shape[0]))
    xs[0] = x0
    for t in range(us.shape[0]):
        xs[t+1] = rk4_step(xs[t], us[t], dt)
    return xs

# -----------------------------
# iLQR
# -----------------------------
def ilqr(x0, us_init, max_iters=MAX_ILQR_ITERS, reg_init=REG_INIT):
    n = x0.shape[0]
    N = us_init.shape[0]
    m = us_init.shape[1]

    # Nominal rollout
    us = us_init.copy()
    xs = rollout(x0, us)
    J  = trajectory_cost(xs, us)

    mu = reg_init                       # LM regularization on Vxx (next)
    mu_min, mu_max = 1e-9, REG_MAX

    # For plotting intermediate rollouts
    intermediates = [xs.copy()]

    for it in range(max_iters):
        # --- Backward pass ---
        Ks = np.zeros((N, m, n))
        ks = np.zeros((N, m))

        Vx, Vxx = terminal_cost_derivatives(xs[-1])
        # Regularize Vxx_next in the backward recursion to improve PD-ness
        success = True

        for t in reversed(range(N)):
            x = xs[t]; u = us[t]
            # Linearization
            A, B = discretize_jacobians(x, u, dt)

            # Cost derivatives at stage
            lx, lu, lxx, luu, lux = stage_cost_derivatives(x, u)

            # Regularize Vxx (Levenberg–Marquardt style)
            Vxx_reg = Vxx + mu * np.eye(n)

            # Q-function blocks
            Qx  = lx + A.T @ Vx
            Qu  = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx_reg @ A
            Quu = luu + B.T @ Vxx_reg @ B
            Qux = lux + B.T @ Vxx_reg @ A  # (u,x)

            # Ensure symmetric (numerically)
            Quu = 0.5 * (Quu + Quu.T)
            Qxx = 0.5 * (Qxx + Qxx.T)

            # Check PD of Quu via Cholesky; if fails, increase mu and restart BP
            try:
                # Cholesky factor for stable solves
                L = np.linalg.cholesky(Quu)
                # Solve Quu * v = rhs using chol factors
                def solve_Quu(rhs):
                    y = np.linalg.solve(L, rhs)
                    return np.linalg.solve(L.T, y)

                K = - solve_Quu(Qux)
                k = - solve_Quu(Qu)

            except np.linalg.LinAlgError:
                # Not PD: increase regularization and redo full backward pass
                success = False
                break

            # Value recursion
            Ks[t] = K
            ks[t] = k
            Vx  = Qx + Qux.T @ k + K.T @ Qu + K.T @ Quu @ k
            Vxx = Qxx + Qux.T @ K + K.T @ Qux + K.T @ Quu @ K
            Vxx = 0.5 * (Vxx + Vxx.T)  # symmetrize

        if not success:
            mu = min(mu * REG_UP, mu_max)
            # If regularization exploded, abort to avoid infinite loop
            if mu >= mu_max:
                print("Regularization too large; stopping.")
                break
            continue

        # --- Forward line search on true dynamics ---
        improved = False
        best_xs, best_us, best_J, best_alpha = None, None, None, None
        for alpha in ALPHAS:
            xs_new = np.zeros_like(xs)
            us_new = np.zeros_like(us)
            xs_new[0] = x0

            for t in range(N):
                # Only scale feedforward term by alpha; DO NOT scale K
                du = alpha * ks[t] + Ks[t] @ (xs_new[t] - xs[t])
                us_new[t] = us[t] + du
                xs_new[t+1] = rk4_step(xs_new[t], us_new[t], dt)

            J_new = trajectory_cost(xs_new, us_new)
            if J_new < J:
                improved = True
                best_xs, best_us, best_J, best_alpha = xs_new, us_new, J_new, alpha
                break  # pick the first improving alpha

        if improved:
            xs, us, J = best_xs, best_us, best_J
            intermediates.append(xs.copy())
            # decrease regularization (we're in a good region)
            mu = max(mu * REG_DOWN, mu_min)

            # Convergence check: tiny relative improvement
            if len(intermediates) >= 2:
                J_prev = trajectory_cost(intermediates[-2], us)  # rough check
                rel_improve = (J_prev - J) / max(1.0, abs(J_prev))
                if rel_improve < COST_TOL:
                    print(f"Converged at iter {it+1}, alpha={best_alpha:.3f}, J={J:.6f}")
                    break
        else:
            # Increase regularization and try again
            mu = min(mu * REG_UP, mu_max)
            if mu >= mu_max:
                print("Regularization too large; stopping.")
                break

    return xs, us, intermediates

# -----------------------------
# Initial nominal controls and rollout
# -----------------------------
# Hover thrust as a reasonable starting point, zero torque
u_hover = np.array([m * g, 0.0])  # hover thrust, no torque
us_init = np.tile(u_hover, (N, 1))

# Initial nominal rollout
xs_init = rollout(x0, us_init)
J_init = trajectory_cost(xs_init, us_init)
print(f"Initial cost: {J_init:.3f}")

# -----------------------------
# Run iLQR
# -----------------------------
xs_opt, us_opt, intermediates = ilqr(x0, us_init, max_iters=MAX_ILQR_ITERS, reg_init=REG_INIT)
J_final = trajectory_cost(xs_opt, us_opt)
print(f"Final cost:   {J_final:.3f}")


def plot_rocket_boxes(ax, xs, step=6, w=0.5, h=1.6, color='C0', edgealpha=0.9, linealpha=0.6, label=None, zorder=3):
    """
    Draw the rocket as a sequence of oriented rectangles ("boxes") along a trajectory.
    """
    if label is not None:
        ax.plot([], [], color=color, label=label)

    for t in range(0, xs.shape[0], step):
        px, py, vx, vy, th, om = xs[t]

        rect = Rectangle((-w/2, -h/2), w, h, linewidth=0.9,
                         edgecolor=color, facecolor='none', alpha=edgealpha, zorder=zorder)
        trans = Affine2D().rotate_around(0.0, 0.0, th).translate(px, py) + ax.transData
        rect.set_transform(trans)
        ax.add_patch(rect)

        # small heading marker toward the body "top"
        tip_local = np.array([0.0, 0.55 * h])
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]])
        tip_world = np.array([px, py]) + R @ tip_local
        ax.plot([px, tip_world[0]], [py, tip_world[1]],
                color=color, alpha=linealpha, linewidth=1.0, zorder=zorder)

# --- New figure: oriented boxes for initial, intermediate, and final trajectories (no y inversion) ---
fig, ax = plt.subplots(figsize=(7.5, 6.5))

# Initial nominal (path + boxes)
ax.plot(xs_init[:,0], xs_init[:,1], 'k--', lw=1.2, label='Initial path')
plot_rocket_boxes(ax, xs_init, step=10, w=0.45, h=1.4, color='k', edgealpha=0.5, linealpha=0.4, label='Initial rocket', zorder=2)

# Intermediates (light)
for i, xs_mid in enumerate(intermediates[:-1]):  # exclude the final rollout (plotted below)
    c = 'C1'
    if i == 0:
        ax.plot(xs_mid[:,0], xs_mid[:,1], color=c, alpha=0.25, lw=1.0, label='Intermediate paths')
    else:
        ax.plot(xs_mid[:,0], xs_mid[:,1], color=c, alpha=0.2, lw=0.9)
    plot_rocket_boxes(ax, xs_mid, step=12, w=0.42, h=1.3, color=c, edgealpha=0.25, linealpha=0.2, zorder=1)

# Final optimized (path + boxes)
ax.plot(xs_opt[:,0], xs_opt[:,1], color='C0', lw=2.0, label='Final (iLQR) path')
plot_rocket_boxes(ax, xs_opt, step=6, w=0.5, h=1.6, color='C0', edgealpha=0.95, linealpha=0.8, label='Final rocket', zorder=4)

# Start/goal markers
ax.scatter([x0[0]], [x0[1]], c='tab:green', marker='o', s=50, label='Start', zorder=5)
ax.scatter([0.0], [0.0], c='tab:red', marker='*', s=100, label='Goal', zorder=5)

# === Key change: DO NOT invert the y-axis ===
# Compute sensible y-limits from all trajectories so downward motion appears downward.
ys_all = [xs_init[:,1], xs_opt[:,1]] + [xs_mid[:,1] for xs_mid in intermediates[:-1]]
y_min = min(map(np.min, ys_all))
y_max = max(map(np.max, ys_all))
pad = 0.05 * max(1.0, y_max - y_min)
ax.set_ylim(y_min - pad, y_max + pad)

ax.set_aspect('equal', 'box')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Planar Rocket Landing: Oriented-Box Visualization (iLQR)')
ax.grid(True, alpha=0.3)
ax.legend(loc='best')
plt.tight_layout()
plt.show()