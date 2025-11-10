# Planar Rocket Soft-Landing (cyipopt) — Euler discretization + analytic Jacobians
# --------------------------------------------------------------------------------
# State  z = [px, py, vx, vy, m],  Control u = [Tx, Ty]
# Continuous dynamics:
#   ẋ = vx
#   ẏ = vy
#   v̇x = Tx / m
#   v̇y = Ty / m - g
#   ṁ = -alpha * ||u||
#
# Discretization (forward Euler):
#   z_{k+1} = z_k + h * f(z_k, u_k)
# Jacobians:
#   A_k = ∂z_{k+1}/∂z_k = I + h * ∂f/∂z
#   B_k = ∂z_{k+1}/∂u_k =     h * ∂f/∂u
#
# Constraints:
#   - Dynamics equalities: z_{k+1} - (z_k + h f(z_k,u_k)) = 0
#   - Thrust magnitude box: T_min^2 ≤ ||u_k||^2 ≤ T_max^2
#   - Pointing cone about +y: cosθ·||u|| − n·u ≤ 0  (n=[0,1])
#
# Objective:
#   minimize  J = Σ ||u_k||^2  +  wN * || (px,py,vx,vy)_N - xT ||^2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from cyipopt import Problem

# --------------------------
# Problem data
# --------------------------
N      = 60              # intervals (N+1 knots). Increase for accuracy with Euler.
h      = 0.10            # step size [s]
g      = 9.81            # gravity [m/s^2]
alpha  = 3e-4            # mass flow coefficient [s/N]  (ṁ = -alpha * ||T||)

# Thrust limits and pointing cone (around +y)
T_min  = 6.0e2
T_max  = 2.5e4
theta_deg = 70.0
theta  = np.deg2rad(theta_deg)
n_vec  = np.array([0.0, 1.0])   # "up"

# Start and desired terminal state (position [m], velocity [m/s])
x0 = np.array([ 200.0, 200.0, -20.0, -20.0])   # [px, py, vx, vy]
xT = np.array([   0.0,   0.0,   0.0,   0.0])   # desired final state (soft landing)
wN = 1e2                                        # weight for terminal deviation

# Mass bounds
m0    = 2000.0
m_min =  300.0

# Small eps to avoid division by zero in norms
EPS = 1e-9

# --------------------------
# Dynamics and Jacobians
# --------------------------
def f_cont(z, u):
    x, y, vx, vy, m = z
    Tx, Ty = u
    Tn = np.hypot(Tx, Ty)
    return np.array([
        vx,
        vy,
        Tx / m,
        Ty / m - g,
        -alpha * Tn
    ], dtype=float)

def jac_f(z, u):
    """Analytic ∂f/∂z (5x5) and ∂f/∂u (5x2)."""
    x, y, vx, vy, m = z
    Tx, Ty = u
    Tn = np.hypot(Tx, Ty)

    Jz = np.zeros((5,5), dtype=float)
    # df1/dvx = 1, df2/dvy = 1
    Jz[0,2] = 1.0
    Jz[1,3] = 1.0
    # df3/dm = -Tx/m^2, df4/dm = -Ty/m^2
    Jz[2,4] = -Tx/(m*m)
    Jz[3,4] = -Ty/(m*m)

    Ju = np.zeros((5,2), dtype=float)
    Ju[2,0] = 1.0/m
    Ju[3,1] = 1.0/m
    Ju[4,0] = -alpha * Tx / Tn
    Ju[4,1] = -alpha * Ty / Tn
    return Jz, Ju

def euler_step(z, u, h):
    return z + h * f_cont(z, u)

def euler_jacobians(z, u, h):
    Jz, Ju = jac_f(z, u)
    A = np.eye(5) + h * Jz
    B = h * Ju
    return A, B

# --------------------------
# Variable layout
# [ z0(5), u0(2), z1(5), u1(2), ..., u_{N-1}(2), zN(5) ]
# --------------------------
NZ, NU = 5, 2
ZDIM = (N+1)*NZ + N*NU

def idx_z(k):
    if k == 0: return slice(0, NZ)
    return slice(NZ + (k-1)*(NU+NZ) + NU, NZ + (k-1)*(NU+NZ) + NU + NZ)

def idx_u(k):
    return slice(NZ + k*(NU+NZ), NZ + k*(NU+NZ) + NU)

def unpack_vec(x):
    Z = np.zeros((N+1, NZ))
    U = np.zeros((N,   NU))
    Z[0] = x[idx_z(0)]
    for k in range(N):
        U[k]   = x[idx_u(k)]
        Z[k+1] = x[idx_z(k+1)]
    return Z, U

def pack_vec(Z, U):
    x = np.zeros(ZDIM)
    x[idx_z(0)] = Z[0]
    for k in range(N):
        x[idx_u(k)]   = U[k]
        x[idx_z(k+1)] = Z[k+1]
    return x

# --------------------------
# Initial guess
# --------------------------
def initial_guess():
    U = np.tile(T_min * n_vec, (N,1))       # modest up-thrust
    Z = np.zeros((N+1, NZ))
    Z[0] = np.array([*x0, m0])
    for k in range(N):
        Z[k+1] = euler_step(Z[k], U[k], h)
        Z[k+1, 4] = np.clip(Z[k+1, 4], m_min, m0)   # mass bounds
        Z[k+1, 1] = max(Z[k+1, 1], 0.0)             # ground
    return pack_vec(Z, U)

# --------------------------
# IPOPT NLP with analytic Jacobian
# --------------------------
class RocketLandingNLP:
    def __init__(self):
        # Variable bounds
        lb = -np.inf*np.ones(ZDIM)
        ub =  np.inf*np.ones(ZDIM)

        # Mass bounds and ground (py≥0)
        for k in range(N+1):
            zs = idx_z(k)
            lb[zs][1] = 0.0
            lb[zs][4] = m_min
            ub[zs][4] = m0

        # Control box bounds (optional; magnitude is enforced via constraints)
        for k in range(N):
            us = idx_u(k)
            lb[us] = [-T_max, -T_max]
            ub[us] = [ T_max,  T_max]

        # Fix initial state
        lb[idx_z(0)] = [x0[0], x0[1], x0[2], x0[3], m0]
        ub[idx_z(0)] = [x0[0], x0[1], x0[2], x0[3], m0]

        self.lb, self.ub = lb, ub

        # Constraints count (NOTE: NO terminal equalities now)
        self.m_dyn   = 5*N
        self.m_norm  = N
        self.m_point = N
        self.mcon    = self.m_dyn + self.m_norm + self.m_point

        cl = np.zeros(self.mcon)
        cu = np.zeros(self.mcon)

        # Thrust-norm bounds: [T_min^2, T_max^2]
        off = self.m_dyn
        cl[off:off+self.m_norm] = T_min**2
        cu[off:off+self.m_norm] = T_max**2
        off += self.m_norm

        # Pointing: ≤ 0
        cu[off:off+self.m_point] = 0.0

        self.cl, self.cu = cl, cu

        # Dense Jacobian structure (simple & robust)
        rr, cc = np.nonzero(np.ones((self.mcon, ZDIM)))
        self.jr, self.jc = rr.astype(int), cc.astype(int)

    # ---- Objective and gradient ----
    def objective(self, x):
        Z, U = unpack_vec(x)
        # control effort
        J = np.sum(U*U)
        # terminal deviation (px,py,vx,vy at N)
        pxN, pyN, vxN, vyN = Z[-1,0], Z[-1,1], Z[-1,2], Z[-1,3]
        e = np.array([pxN - xT[0], pyN - xT[1], vxN - xT[2], vyN - xT[3]])
        J += wN * (e @ e)
        return J

    def gradient(self, x):
        Z, U = unpack_vec(x)
        g = np.zeros_like(x)

        # d/dx Σ ||u||^2 = 2 u_k
        for k in range(N):
            us = idx_u(k)
            g[us] = 2.0 * x[us]

        # d/dx wN * ||zN_xyv - xT||^2
        zN = idx_z(N)
        pxN, pyN, vxN, vyN = Z[-1,0], Z[-1,1], Z[-1,2], Z[-1,3]
        e = np.array([pxN - xT[0], pyN - xT[1], vxN - xT[2], vyN - xT[3]])
        # gradient w.r.t. [pxN, pyN, vxN, vyN]; mass has no contribution
        g[zN.start + 0] += 2.0 * wN * e[0]
        g[zN.start + 1] += 2.0 * wN * e[1]
        g[zN.start + 2] += 2.0 * wN * e[2]
        g[zN.start + 3] += 2.0 * wN * e[3]
        return g

    # ---- Constraints and analytic Jacobian ----
    def constraints(self, x):
        Z, U = unpack_vec(x)
        vals = []

        # (1) Dynamics: z_{k+1} - (z_k + h f(z_k,u_k)) = 0
        for k in range(N):
            vals.extend(Z[k+1] - euler_step(Z[k], U[k], h))

        # (2) Thrust-norm box: ||u||^2
        for k in range(N):
            Tx, Ty = U[k]
            vals.append(Tx*Tx + Ty*Ty)

        # (3) Pointing: cosθ||u|| - n·u ≤ 0
        cth = np.cos(theta)
        for k in range(N):
            Tx, Ty = U[k]
            Tn = max(np.hypot(Tx, Ty), EPS)
            vals.append(cth*Tn - (n_vec[0]*Tx + n_vec[1]*Ty))

        return np.asarray(vals, dtype=float)

    def jacobian(self, x):
        Z, U = unpack_vec(x)
        J = np.zeros((self.mcon, ZDIM), dtype=float)
        row = 0

        # --- (1) Dynamics Jacobian ---
        for k in range(N):
            A, B = euler_jacobians(Z[k], U[k], h)
            # g_k = z_{k+1} - (z_k + h f(z_k,u_k))
            # ∂g/∂z_{k+1} = I
            J[row:row+5, idx_z(k+1)] += np.eye(5)
            # ∂g/∂z_k    = -A
            J[row:row+5, idx_z(k)]   -= A
            # ∂g/∂u_k    = -B
            J[row:row+5, idx_u(k)]   -= B
            row += 5

        # --- (2) Thrust-norm: g = Tx^2 + Ty^2 ---
        for k in range(N):
            us = idx_u(k)
            Tx, Ty = U[k]
            J[row, us.start+0] = 2.0*Tx
            J[row, us.start+1] = 2.0*Ty
            row += 1

        # --- (3) Pointing: cosθ||u|| - n·u ---
        cth = np.cos(theta)
        for k in range(N):
            us = idx_u(k)
            Tx, Ty = U[k]
            Tn = max(np.hypot(Tx, Ty), EPS)
            dTn_dTx = Tx / Tn
            dTn_dTy = Ty / Tn
            J[row, us.start+0] = cth*dTn_dTx - n_vec[0]
            J[row, us.start+1] = cth*dTn_dTy - n_vec[1]
            row += 1

        return J[self.jr, self.jc]

    def jacobianstructure(self):
        return (self.jr, self.jc)

    # Hessian by L-BFGS
    def hessian(self, x, lagrange, obj_factor):
        return np.array([])

    def hessianstructure(self):
        return ([], [])

# --------------------------
# Helper: draw a rotated box (rocket)
# --------------------------
def draw_rocket_box(ax, center, angle_rad, length, width, color='tab:orange', alpha=0.7, lw=0.6):
    """
    Draw a rectangle centered at 'center' rotated by 'angle_rad'.
    The rectangle's long axis points along 'angle_rad'.
    """
    cx, cy = center
    L = length
    W = width
    # Rectangle corners in local coords (centered), long axis along +y_local
    # We choose +y_local as the "body axis", so set corners accordingly:
    local = np.array([
        [-W/2, -L/2],
        [ W/2, -L/2],
        [ W/2,  L/2],
        [-W/2,  L/2],
    ])
    # Rotate by angle around origin, where angle is measured from +x toward +y
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s],
                  [s,  c]])
    world = (local @ R.T) + np.array([cx, cy])
    poly = Polygon(world, closed=True, facecolor=color, edgecolor='k', alpha=alpha, linewidth=lw)
    ax.add_patch(poly)

# --------------------------
# Solve with IPOPT
# --------------------------
if __name__ == "__main__":
    x0_vec = initial_guess()
    nlp = RocketLandingNLP()

    prob = Problem(
        n=ZDIM, m=nlp.mcon,
        problem_obj=nlp,
        lb=nlp.lb, ub=nlp.ub,
        cl=nlp.cl, cu=nlp.cu
    )

    # IPOPT options
    prob.add_option('print_level', 5)
    prob.add_option('max_iter', 1500)
    prob.add_option('tol', 1e-6)
    prob.add_option('acceptable_tol', 1e-4)
    prob.add_option('mu_strategy', 'adaptive')
    prob.add_option('hessian_approximation', 'limited-memory')
    prob.add_option('linear_solver', 'mumps')
    prob.add_option('sb', 'yes')

    x_opt, info = prob.solve(x0_vec)
    Zopt, Uopt = unpack_vec(x_opt)

    # Report
    err_final = Zopt[-1,:4] - xT
    print("\nStatus:", info.get('status_msg', ''))
    print("Final objective:", info.get('obj_val', np.nan))
    print("Final state [px,py,vx,vy] =", Zopt[-1,:4])
    print("Terminal deviation norm =", np.linalg.norm(err_final))
    print(f"Final mass: {Zopt[-1,4]:.3f}  (fuel used {m0 - Zopt[-1,4]:.3f})")

    print(Zopt)
    print(nlp.lb)
    print(nlp.ub)


    # ------------- Plots -------------
    t = np.arange(N+1)*h

    # Trajectory with rocket boxes
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.plot(Zopt[:,0], Zopt[:,1], '-o', ms=2.5, color='tab:blue', label='trajectory')
    ax.axhline(0, color='k', lw=1)
    ax.scatter([x0[0]],[x0[1]], c='g', label='start')
    ax.scatter([xT[0]],[xT[1]], c='r', label='target')

    # Determine a visually reasonable rocket size relative to the scene
    xr = float(np.ptp(Zopt[:, 0]))  # <-- was Zopt[:,0].ptp()
    yr = float(np.ptp(Zopt[:, 1]))  # <-- was Zopt[:,1].ptp()
    scale = max(xr, yr) if max(xr, yr) > 0 else 1.0
    L = 0.06 * scale   # rocket length
    W = 0.25 * L       # rocket width


    # Place boxes at a subset of knots
    skip = max(1, N // 8)
    for k in range(0, N, skip):
        px, py = Zopt[k,0], Zopt[k,1]
        Tx, Ty = Uopt[k]
        if np.hypot(Tx, Ty) < 1e-6:
            # If almost zero thrust, default to pointing up
            angle = np.pi/2
        else:
            # Orient the rocket along the thrust direction
            # angle measured from +x toward +y
            angle = np.arctan2(Ty, Tx)
        draw_rocket_box(ax, (px, py), angle, L, W, color='tab:orange', alpha=0.8, lw=0.7)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Planar soft landing (Euler + analytic Jacobian, cyipopt)\nwith rocket boxes')
    ax.grid(True); ax.legend()

    # Thrust magnitude
    plt.figure(figsize=(10,4))
    Tmag = np.linalg.norm(Uopt, axis=1)
    plt.plot(t[:-1], Tmag, label='||T||', color='tab:purple')
    plt.axhline(T_min, ls='--', c='k', lw=1)
    plt.axhline(T_max, ls='--', c='k', lw=1)
    plt.xlabel('time [s]'); plt.ylabel('thrust [N]'); plt.grid(True); plt.legend()

    # Mass profile
    plt.figure(figsize=(10,4))
    plt.plot(t, Zopt[:,4], color='tab:green')
    plt.axhline(m_min, ls='--', c='k', lw=1)
    plt.xlabel('time [s]'); plt.ylabel('mass [kg]'); plt.grid(True); plt.title('Mass profile')

    plt.tight_layout()
    plt.show()
