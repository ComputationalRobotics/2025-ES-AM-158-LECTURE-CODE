"""
TVLQR tracking on a *nonlinear* unicycle (planar car) model
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
def wrap_angle(a):
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi

def f_unicycle(x, u, dt):
    """One-step discrete dynamics (Euler) for the unicycle."""
    px, py, th = x
    v, om = u
    nx = np.empty_like(x)
    nx[0] = px + dt * v * np.cos(th)
    nx[1] = py + dt * v * np.sin(th)
    nx[2] = wrap_angle(th + dt * om)
    return nx

def linearize_unicycle(xbar, ubar, dt):
    """
    Discrete-time Jacobians A = ∂f/∂x, B = ∂f/∂u for the unicycle dynamics,
    evaluated at (x̄, ū). Exact for the Euler-discretized model.
    """
    px, py, th = xbar
    v, om = ubar
    c = np.cos(th); s = np.sin(th)

    # State Jacobian
    A = np.array([
        [1.0, 0.0, -dt * v * s],
        [0.0, 1.0,  dt * v * c],
        [0.0, 0.0,  1.0       ]
    ])

    # Control Jacobian
    B = np.array([
        [dt * c, 0.0],
        [dt * s, 0.0],
        [0.0,    dt ]
    ])
    return A, B

def tvlqr(A_seq, B_seq, Q_seq, R_seq, QN):
    """
    Finite-horizon TVLQR backward Riccati recursion.
    Returns K_t gains and S_t cost-to-go matrices.
    """
    N = len(A_seq)
    S_seq = [None] * (N + 1)
    K_seq = [None] * N
    S = QN.copy()
    S_seq[N] = S

    for t in reversed(range(N)):
        A = A_seq[t]; B = B_seq[t]
        Q = Q_seq[t]; R = R_seq[t]

        G = R + B.T @ S @ B                      # Q_uu
        K = np.linalg.solve(G, B.T @ S @ A)      # (R + B'SB)^{-1} B'SA
        K_seq[t] = K
        S = Q + A.T @ (S - S @ B @ K) @ A        # Riccati update
        S_seq[t] = S

    return K_seq, S_seq

# -----------------------------
# Nominal trajectory (nonlinear)
# -----------------------------
dt = 0.02
T_final = 12.0
N = int(T_final / dt)

# Choose a circle via constant (v̄, ω̄).
v_bar = 1.2                # m/s
omega_bar = 0.4            # rad/s
R = v_bar / omega_bar      # circle radius

# Nominal control sequence
ubar_seq = np.tile(np.array([v_bar, omega_bar]), (N,1))

# Nominal state (start at (R, 0) with heading pi/2 to be tangent)
xbar_seq = np.zeros((N+1, 3))
xbar_seq[0] = np.array([R, 0.0, np.pi/2])   # at (R, 0), heading upward

for t in range(N):
    xbar_seq[t+1] = f_unicycle(xbar_seq[t], ubar_seq[t], dt)

# -----------------------------
# Linearize along the nominal
# -----------------------------
A_seq = []
B_seq = []
for t in range(N):
    A_t, B_t = linearize_unicycle(xbar_seq[t], ubar_seq[t], dt)
    A_seq.append(A_t)
    B_seq.append(B_t)

# -----------------------------
# Deviation-cost design
# -----------------------------
Q = np.diag([30.0, 30.0, 5.0])
QN = np.diag([60.0, 60.0, 8.0])
R = np.diag([0.2, 0.2])

Q_seq = [Q for _ in range(N)]
R_seq = [R for _ in range(N)]

K_seq, S_seq = tvlqr(A_seq, B_seq, Q_seq, R_seq, QN)

# -----------------------------
# Disturbances and initial deviation
# -----------------------------
rng = np.random.default_rng(7)

# Process noise each step (position ~ centimeters, heading ~ small)
sigma = np.array([0.01, 0.01, np.deg2rad(0.2)])
W = np.diag(sigma**2)
w_seq = rng.multivariate_normal(mean=np.zeros(3), cov=W, size=N)

# Add a brief "gust" that knocks heading and position
gust_start = int(4.0 / dt)
gust_end   = gust_start + int(0.8 / dt)
w_seq[gust_start:gust_end, 2] += np.deg2rad(1.8)   # heading kick
w_seq[gust_start:gust_end, 0] += 0.01              # small x drift

# Start away from nominal
x0 = xbar_seq[0].copy()
x0[0] += -0.25      # -25 cm in x
x0[1] +=  0.15      # +15 cm in y
x0[2]  = wrap_angle(x0[2] + np.deg2rad(8.0))  # +8 deg heading error

# Optional control limits (for closed-loop; nominal values are within limits)
v_min, v_max = 0.2, 2.0
om_min, om_max = -1.0, 1.0

# -----------------------------
# Closed-loop simulation (TVLQR)
# -----------------------------
x_traj = np.zeros_like(xbar_seq)
u_traj = np.zeros_like(ubar_seq)
dx_traj = np.zeros_like(xbar_seq)

x = x0.copy()
x_traj[0] = x
dx = x - xbar_seq[0]
dx[2] = wrap_angle(dx[2])
dx_traj[0] = dx

for t in range(N):
    K = K_seq[t]
    xbar = xbar_seq[t]; ubar = ubar_seq[t]

    dx = x - xbar
    dx[2] = wrap_angle(dx[2])           # keep heading deviation small
    du = - K @ dx
    u = ubar + du

    # apply control saturation (common in vehicles)
    u[0] = np.clip(u[0], v_min, v_max)
    u[1] = np.clip(u[1], om_min, om_max)

    # propagate nonlinear dynamics with disturbance
    w = w_seq[t]
    x = f_unicycle(x, u, dt)
    x[0] += w[0]
    x[1] += w[1]
    x[2] = wrap_angle(x[2] + w[2])

    u_traj[t] = u
    x_traj[t+1] = x
    dx_traj[t+1] = np.array([x[0]-xbar_seq[t+1,0],
                              x[1]-xbar_seq[t+1,1],
                              wrap_angle(x[2]-xbar_seq[t+1,2])])

# -----------------------------
# NEW: Open-loop simulation (nominal control + noise, NO feedback)
# -----------------------------
x_traj_open = np.zeros_like(xbar_seq)
u_traj_open = np.copy(ubar_seq)

x_open = x0.copy()
x_traj_open[0] = x_open

for t in range(N):
    u = u_traj_open[t]              # exactly nominal control
    w = w_seq[t]                    # SAME disturbances for apples-to-apples
    x_open = f_unicycle(x_open, u, dt)
    x_open[0] += w[0]
    x_open[1] += w[1]
    x_open[2] = wrap_angle(x_open[2] + w[2])
    x_traj_open[t+1] = x_open

# -----------------------------
# Diagnostics
# -----------------------------
t_grid = dt * np.arange(N+1)
pos_err = np.linalg.norm(dx_traj[:, :2], axis=1)
ang_err = np.abs(dx_traj[:, 2])

rms_pos = np.sqrt(np.mean(pos_err**2))
rms_ang = np.rad2deg(np.sqrt(np.mean(ang_err**2)))

print(f"RMS position error (TVLQR): {rms_pos:.3f} m")
print(f"RMS heading error  (TVLQR): {rms_ang:.2f} deg")

# -----------------------------
# Figure 1: Trajectory in the plane (now with open-loop baseline)
# -----------------------------
plt.figure(figsize=(6.4,6.1))
plt.plot(xbar_seq[:,0], xbar_seq[:,1], 'k--', lw=1.2, label='Nominal path')
plt.plot(x_traj[:,0],   x_traj[:,1],   lw=1.8, label='Tracked (TVLQR)')
plt.plot(x_traj_open[:,0], x_traj_open[:,1], lw=1.6, alpha=0.9,
         label='Open-loop (ū + noise, no feedback)')
plt.scatter(x_traj[0,0], x_traj[0,1], marker='o', s=40, label='Start', zorder=3)
plt.axis('equal')
plt.title('Nonlinear Unicycle: TVLQR vs. Open-loop (same disturbances)')
plt.xlabel('x [m]'); plt.ylabel('y [m]')
plt.legend(loc='best'); plt.grid(True)

# Figure 2: Tracking errors (TVLQR)
plt.figure(figsize=(10,4.2))
plt.plot(t_grid, pos_err, label='‖δpos‖ [m]')
plt.plot(t_grid, np.rad2deg(ang_err), label='|δθ| [deg]')
plt.axvspan(gust_start*dt, gust_end*dt, alpha=0.15, label='gust')
plt.title('TVLQR Tracking Errors vs Time')
plt.xlabel('time [s]'); plt.ylabel('error')
plt.legend(); plt.grid(True)

# Figure 3: Controls (nominal vs closed-loop) with gust label
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

# Top subplot: speed v
ax1.plot(t_grid[:-1], ubar_seq[:,0], '--', label='v̄')
ax1.plot(t_grid[:-1], u_traj[:,0], label='v (TVLQR)')
# Shade the gust window and label it
ax1.axvspan(gust_start*dt, gust_end*dt, alpha=0.15, color='gray', label='gust window')
ax1.set_ylabel('speed v [m/s]')
ax1.legend()
ax1.grid(True)

# Bottom subplot: yaw rate ω
ax2.plot(t_grid[:-1], ubar_seq[:,1], '--', label='ω̄')
ax2.plot(t_grid[:-1], u_traj[:,1], label='ω (TVLQR)')
# Shade the gust window (no extra legend entry to avoid duplicates)
ax2.axvspan(gust_start*dt, gust_end*dt, alpha=0.15, color='gray')
ax2.set_xlabel('time [s]')
ax2.set_ylabel('yaw rate ω [rad/s]')
ax2.legend()
ax2.grid(True)

fig.suptitle('Control Inputs: nominal vs. closed-loop (gust window shaded)')
plt.tight_layout()
plt.show()
