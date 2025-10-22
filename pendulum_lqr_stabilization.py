# LQR stabilization of an inverted pendulum with RK4 simulation
# --------------------------------------------------------------
# pip install numpy scipy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

# ---------- Nonlinear dynamics in shifted coordinates ----------
def pendulum_f_z(z, u, m, g, l, b):
    """
    z = [theta - pi, theta_dot]
    zdot = [z2,
            (u - b*z2 + m*g*l*sin(z1)) / (m*l^2)]
    """
    z1, z2 = z
    return np.array([
        z2,
        (u - b * z2 + m * g * l * np.sin(z1)) / (m * l * l),
    ], dtype=float)

# ---------- Integrators ----------
def euler_step(f, z, u, dt, args):
    return z + dt * f(z, u, *args)

def rk4_step(f, z, u, dt, args):
    """
    Classic 4th-order Runge–Kutta with zero-order hold on the control u.
    """
    k1 = f(z, u, *args)
    k2 = f(z + 0.5 * dt * k1, u, *args)
    k3 = f(z + 0.5 * dt * k2, u, *args)
    k4 = f(z + dt * k3, u, *args)
    return z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ---------- Discrete-time LQR ----------
def dlqr(A, B, Q, R):
    S = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ S @ B, B.T @ S @ A)
    return K, S

# ---------- Problem setup ----------
m, g, l, b = 1.0, 9.8, 1.0, 0.1

# Linearize about upright (z=0, u=0) in continuous time
Ac = np.array([[0.0,      1.0],
               [g / l, -b / (m * l * l)]], dtype=float)
Bc = np.array([[0.0],
               [1.0 / (m * l * l)]], dtype=float)

# Forward-Euler discretization for LQR design (sample-and-hold controller)
dt = 0.01
A = np.eye(2) + dt * Ac
B = dt * Bc

# LQR weights
Q = np.eye(2)
R = np.array([[1.0]])

# Gain and closed-loop check
K, S = dlqr(A, B, Q, R)
lam_cl = np.linalg.eigvals(A - B @ K)
print("LQR gain K:\n", K)
print("Closed-loop eigenvalues (A - B K):", lam_cl)

# ---------- Simulation parameters ----------
z0 = np.array([0.1, 0.1])       # initial deviation [rad, rad/s]
T_final = 10.0
num_steps = int(T_final / dt)

# Choose integrator: "euler" or "rk4"
INTEGRATOR = "rk4"      # set to "euler" to compare
SUBSTEPS = 1            # you can increase (e.g., 5) for even higher accuracy

# ---------- Run simulation of the nonlinear system ----------
z_traj = np.zeros((2, num_steps + 1))
u_traj = np.zeros(num_steps)
z = z0.copy()
z_traj[:, 0] = z

args_dyn = (m, g, l, b)

for k in range(num_steps):
    # sample-and-hold control (computed once per step)
    u = float(-(K @ z))
    # u = float(np.clip(u, -5.0, 5.0))  # optional saturation

    # integrate dynamics over [t_k, t_{k+1}] with zero-order hold on u
    z_step = z.copy()
    h = dt / SUBSTEPS
    for _ in range(SUBSTEPS):
        if INTEGRATOR == "rk4":
            z_step = rk4_step(pendulum_f_z, z_step, u, h, args_dyn)
        elif INTEGRATOR == "euler":
            z_step = euler_step(pendulum_f_z, z_step, u, h, args_dyn)
        else:
            raise ValueError("INTEGRATOR must be 'rk4' or 'euler'.")

    z = z_step
    z_traj[:, k + 1] = z
    u_traj[k] = u

# ---------- Plot ----------
t = np.arange(num_steps) * dt
fs_label, fs_tick = 20, 14

fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
axs[0].plot(t, z_traj[0, :-1], linewidth=2)
axs[0].set_ylabel(r"$z_1$ (rad)", fontsize=fs_label)
axs[0].grid(True); axs[0].tick_params(labelsize=fs_tick)

axs[1].plot(t, z_traj[1, :-1], linewidth=2)
axs[1].set_ylabel(r"$z_2$ (rad/s)", fontsize=fs_label)
axs[1].grid(True); axs[1].tick_params(labelsize=fs_tick)

axs[2].plot(t, u_traj, linewidth=2)
axs[2].set_ylabel(r"$u$ (N·m)", fontsize=fs_label)
axs[2].set_xlabel("time (s)", fontsize=fs_label)
axs[2].grid(True); axs[2].tick_params(labelsize=fs_tick)

fig.suptitle(
    f"LQR stabilization (nonlinear sim, {INTEGRATOR.upper()} with SUBSTEPS={SUBSTEPS})",
    fontsize=18
)
fig.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.show()
