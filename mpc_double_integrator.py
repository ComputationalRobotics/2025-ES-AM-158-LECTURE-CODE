# Minimal MPC example (double integrator with box constraints) -----------------
# - Discrete-time dynamics: s_{k+1} = A s_k + B u_k
# - Stage cost:     (x_k - x_goal)^T Q (x_k - x_goal) + u_k^T R u_k
# - Terminal cost:  (x_N - x_goal)^T Qf (x_N - x_goal)
# - Receding horizon: apply u_0*, shift/warm-start, repeat
# ------------------------------------------------------------------------------

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ----------------------------
# System model & MPC settings
# ----------------------------
dt = 0.1
A = np.array([[1.0, dt],
              [0.0, 1.0]])
B = np.array([[0.5*dt**2],
              [dt]])

n, m = A.shape[0], B.shape[1]         # n=2 states [position, velocity], m=1 input [acceleration]
N = 20                                 # Horizon length

# Cost weights (tune as needed)
Q  = np.diag([10.0, 1.0])              # Penalize position more than velocity
R  = np.diag([0.05])                   # Penalize control effort
Qf = np.diag([50.0, 5.0])              # Terminal state penalty

# State and input bounds (componentwise)
x_min = np.array([-5.0, -3.0])
x_max = np.array([ 5.0,  3.0])
u_min = np.array([-2.0])
u_max = np.array([ 2.0])

# Goal (can be updated online if desired)
x_goal_val = np.array([0.0, 0.0])      # regulate to origin

# Simulation parameters
T_steps = 120                          # total closed-loop steps
x0_true  = np.array([4.0, 0.0])        # initial true state
disturb_std = np.array([0.02, 0.05]) # (optional) process noise std dev

# ----------------------------------------------------
# Build one MPC problem (solve repeatedly with x0= s_t)
# ----------------------------------------------------
x = cp.Variable((n, N+1))              # decision: state trajectory
u = cp.Variable((m, N))                # decision: control trajectory

x0_param   = cp.Parameter(n)           # current measured s_t
x_goal_par = cp.Parameter(n)           # goal (constant or time-varying)

cost = 0
constr = [x[:, 0] == x0_param]         # Enforce x0 = s_t  (Eq. x0 = s_t)

# Dynamics, bounds, and stage costs
for k in range(N):
    cost += cp.quad_form(x[:, k] - x_goal_par, Q) + cp.quad_form(u[:, k], R)
    constr += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]           # Dynamics x_{k+1} = f(x_k,u_k)
    constr += [x_min <= x[:, k], x[:, k] <= x_max]               # State bounds
    constr += [u_min <= u[:, k], u[:, k] <= u_max]               # Input bounds

# Terminal cost + terminal bound(s)
cost += cp.quad_form(x[:, N] - x_goal_par, Qf)
constr += [x_min <= x[:, N], x[:, N] <= x_max]

prob = cp.Problem(cp.Minimize(cost), constr)

# ---------------------------------------------
# Closed-loop simulation (receding-horizon MPC)
# ---------------------------------------------
x_traj = [x0_true.copy()]
u_traj = []
mpc_obj_vals = []

# Warm-start buffers (filled after first solve)
u_warm = np.zeros((m, N))
x_warm = np.tile(x0_true.reshape(-1, 1), (1, N+1))

for t in range(T_steps):
    # Set the current parameter values
    x0_param.value   = x_traj[-1]
    x_goal_par.value = x_goal_val

    # Warm-start decision variables (shift previous solution)
    if t > 0:
        # Shift u: [u1*, u2*, ..., u_{N-1}*, u_{N-1}*] as an initial guess
        u_warm = np.hstack([u_last[:, 1:], u_last[:, [-1]]])
        # Roll out x_warm through dynamics from the current measured state
        x_warm[:, [0]] = x_traj[-1].reshape(-1, 1)
        for k in range(N):
            x_warm[:, [k+1]] = A @ x_warm[:, [k]] + B @ u_warm[:, [k]]
        u.value = u_warm
        x.value = x_warm

    # Solve the QP (OSQP is fast; SCS/ECOS also work if OSQP unavailable)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    except cp.SolverError:
        # Fallback to ECOS if OSQP isn't installed
        prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"MPC solve failed at step {t}: status = {prob.status}")

    u_star = u.value.copy()            # full open-loop sequence u_0..u_{N-1}
    x_star = x.value.copy()            # predicted states
    u_last = u_star                    # store for warm-start at next step

    # Apply only the first control (Eq. a_t = u_0*)
    u_apply = u_star[:, 0]
    u_traj.append(u_apply)

    # Evolve the true system with (optional) disturbance w_t
    w = np.random.randn(n) * disturb_std
    x_next = A @ x_traj[-1] + (B @ u_apply.reshape(-1, 1)).ravel() + w
    x_traj.append(x_next)

    # (Optional) log objective value
    mpc_obj_vals.append(prob.value)

# -----------------
# Visualization
# -----------------
x_traj = np.array(x_traj)                  # shape (T_steps+1, n)
u_traj = np.array(u_traj)                  # shape (T_steps, m)
tgrid_x = np.arange(T_steps+1) * dt
tgrid_u = np.arange(T_steps) * dt

fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
axs[0].plot(tgrid_x, x_traj[:, 0], label='position')
axs[0].axhline(x_goal_val[0], linestyle='--', label='p goal')
axs[0].set_ylabel('position')
axs[0].legend(loc='best')

axs[1].plot(tgrid_x, x_traj[:, 1], label='velocity')
axs[1].axhline(x_goal_val[1], linestyle='--', label='v goal')
axs[1].set_ylabel('velocity')
axs[1].legend(loc='best')

axs[2].step(tgrid_u, u_traj[:, 0], where='post', label='accel (control)')
axs[2].axhline(u_min[0], linestyle='--', alpha=0.6)
axs[2].axhline(u_max[0], linestyle='--', alpha=0.6)
axs[2].set_ylabel('u')
axs[2].set_xlabel('time [s]')
axs[2].legend(loc='best')

fig.suptitle('MPC (receding horizon) on a double integrator')
plt.tight_layout()
plt.show()
