# Unicycle Trajectory Optimization with cyipopt (Ipopt)
# ----------------------------------------------------------
# Drive a unicycle from A to B while avoiding circular obstacles:
#   x = [px, py, theta],  u = [v, omega]
# Discrete dynamics (Euler, step h):
#   px_{k+1}    = px_k + h * v_k * cos(theta_k)
#   py_{k+1}    = py_k + h * v_k * sin(theta_k)
#   theta_{k+1} = theta_k + h * omega_k
#
# Minimize (terminal goal error) + (control effort) + (control smoothness)
# Subject to:
#   - initial state equality
#   - dynamics equalities
#   - obstacle inequalities: (r+margin)^2 - ((px-cx)^2 + (py-cy)^2) <= 0
#
# Requires: cyipopt, numpy, matplotlib

from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import cyipopt

@dataclass
class Params:
    N: int = 60
    h: float = 0.10
    A: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))
    B: np.ndarray = field(default_factory=lambda: np.array([6.0, 5.0, 0.0], dtype=float))
    obstacles: np.ndarray = field(default_factory=lambda: np.array([
        [2.0, 1.5, 0.8],
        [3.5, 3.5, 0.9],
        [5.2, 3.8, 0.7],
    ], dtype=float))
    safety_margin: float = 0
    v_min: float = -1.5
    v_max: float =  1.5
    w_min: float = -2.0
    w_max: float =  2.0
    w_goal_pos: float = 400.0
    w_goal_theta: float = 20.0
    w_u: float = 0.05
    w_du: float = 0.20
    n_x: int = 3
    n_u: int = 2

    def __post_init__(self):
        self.Z_DIM = (self.n_x + self.n_u) * self.N + self.n_x
        self.n_ceq = 3 + 3 * self.N
        self.nObs = int(self.obstacles.shape[0])
        self.n_cineq = (self.N + 1) * self.nObs
        self.M = self.n_ceq + self.n_cineq



# --- Packing helpers ---
def idx_x(k, P: Params):
    """Indices in z for state at time k (0..N)."""
    if k < P.N:
        base = (P.n_x + P.n_u) * k
    else:
        base = (P.n_x + P.n_u) * P.N
    return slice(base, base + P.n_x)

def idx_u(k, P: Params):
    """Indices in z for control at time k (0..N-1)."""
    base = (P.n_x + P.n_u) * k + P.n_x
    return slice(base, base + P.n_u)

def unpack(z, P: Params):
    X = np.zeros((P.N + 1, P.n_x))
    U = np.zeros((P.N, P.n_u))
    for k in range(P.N):
        X[k, :] = z[idx_x(k, P)]
        U[k, :] = z[idx_u(k, P)]
    X[P.N, :] = z[idx_x(P.N, P)]
    return X, U

def f_disc(xk, uk, h):
    px, py, th = xk
    v, w = uk
    return np.array([
        px + h * v * np.cos(th),
        py + h * v * np.sin(th),
        th + h * w
    ])


# --- Initial guess: straight line in (px,py), constant heading and v ---
def initial_guess(P: Params):
    X = np.zeros((P.N + 1, P.n_x))
    U = np.zeros((P.N, P.n_u))
    dx = P.B[0] - P.A[0]
    dy = P.B[1] - P.A[1]
    path_ang = np.arctan2(dy, dx)
    for k in range(P.N + 1):
        alpha = k / P.N
        X[k, 0] = P.A[0] + alpha * dx
        X[k, 1] = P.A[1] + alpha * dy
        X[k, 2] = path_ang
    L = np.hypot(dx, dy)
    v_guess = np.clip(L / (P.N * P.h), P.v_min, P.v_max)
    U[:, 0] = v_guess
    U[:, 1] = 0.0
    # pack
    z0 = np.random.randn(P.Z_DIM) # random initialization
    # z0 = np.zeros(P.Z_DIM) # all-zero initialization
    # straight-line initialization
    # for k in range(P.N):
    #     z0[idx_x(k, P)] = X[k, :]
    #     z0[idx_u(k, P)] = U[k, :]
    # z0[idx_x(P.N, P)] = X[P.N, :]
    return z0


class UnicycleTO:
    def __init__(self, P: Params):
        self.P = P
        # --- Build bounds on variables ---
        lb = -np.inf * np.ones(P.Z_DIM)
        ub =  np.inf * np.ones(P.Z_DIM)
        for k in range(P.N):
            iu = idx_u(k, P)
            lb[iu.start + 0] = P.v_min
            ub[iu.start + 0] = P.v_max
            lb[iu.start + 1] = P.w_min
            ub[iu.start + 1] = P.w_max
        self.lb = lb
        self.ub = ub

        # --- Build bounds on constraints ---
        cl = np.zeros(P.M)
        cu = np.zeros(P.M)
        # equalities: exactly 0
        cl[:P.n_ceq] = 0.0
        cu[:P.n_ceq] = 0.0
        # inequalities: c <= 0
        cl[P.n_ceq:] = -np.inf
        cu[P.n_ceq:] = 0.0
        self.cl = cl
        self.cu = cu

        # --- Precompute Jacobian sparsity (row, col) ---
        self.jac_rows, self.jac_cols = self._build_jacobian_structure()

    # Objective
    def objective(self, z):
        P = self.P
        X, U = unpack(z, P)

        # Terminal goal tracking
        pos_err = X[-1, 0:2] - P.B[0:2]
        th_err  = X[-1, 2]   - P.B[2]
        J_goal = P.w_goal_pos * np.dot(pos_err, pos_err) + P.w_goal_theta * (th_err**2)

        # Control effort
        J_u = P.w_u * np.sum(U * U)

        # Control smoothness
        dU = U[1:, :] - U[:-1, :]
        J_du = P.w_du * np.sum(dU * dU)

        return 0.5 * (J_goal + J_u + J_du)

    # Gradient of objective
    def gradient(self, z):
        P = self.P
        X, U = unpack(z, P)
        grad = np.zeros(P.Z_DIM)

        # Terminal contributions (no 0.5 after derivative: cancels 2)
        pos_err = X[-1, 0:2] - P.B[0:2]
        th_err  = X[-1, 2]   - P.B[2]
        gN = np.array([P.w_goal_pos * pos_err[0],
                       P.w_goal_pos * pos_err[1],
                       P.w_goal_theta * th_err])
        grad[idx_x(P.N, P)] += gN

        # Control effort: 0.5 * 2 * w_u * u = w_u * u
        for k in range(P.N):
            iu = idx_u(k, P)
            grad[iu] += P.w_u * U[k, :]

        # Control smoothness: 0.5 * w_du * sum ||u_{k+1}-u_k||^2
        # d/d u_k:   -w_du*(u_{k+1}-u_k)  from (k,k+1)
        # d/d u_{k+1}: +w_du*(u_{k+1}-u_k)
        for k in range(P.N - 1):
            du = U[k + 1, :] - U[k, :]
            grad[idx_u(k, P)]     += -P.w_du * du
            grad[idx_u(k + 1, P)] +=  P.w_du * du

        return grad

    # Constraints g(z)
    def constraints(self, z):
        P = self.P
        X, U = unpack(z, P)
        g = np.zeros(P.M)
        r = 0

        # Initial equality: X0 - A = 0
        g[r:r+3] = X[0, :] - P.A
        r += 3

        # Dynamics equalities
        for k in range(P.N):
            xk = X[k, :]
            uk = U[k, :]
            xnext = f_disc(xk, uk, P.h)
            g[r:r+3] = X[k + 1, :] - xnext
            r += 3

        # Obstacle inequalities: (r+margin)^2 - ((px-cx)^2 + (py-cy)^2) <= 0
        for k in range(P.N + 1):
            px, py = X[k, 0], X[k, 1]
            for j in range(P.nObs):
                cx, cy, r0 = P.obstacles[j]
                r_eff = r0 + P.safety_margin
                g[r] = (r_eff ** 2) - ((px - cx) ** 2 + (py - cy) ** 2)
                r += 1

        return g

    # Jacobian sparsity
    def jacobianstructure(self):
        return (np.array(self.jac_rows, dtype=int),
                np.array(self.jac_cols, dtype=int))

    # Jacobian values (in the same order as jacobianstructure)
    def jacobian(self, z):
        P = self.P
        X, U = unpack(z, P)
        vals = []
        r = 0

        # Initial eq: d/dX0 is identity (one per row)
        # rows r..r+2 with columns X0(px,py,th)
        for i in range(3):
            vals.append(1.0)
        r += 3

        # Dynamics eqs
        for k in range(P.N):
            xk = X[k, :]
            uk = U[k, :]
            th = xk[2]
            v  = uk[0]

            # Row r: g1 = x_{k+1,px} - [px_k + h v cos(th_k)]
            # d wrt X_{k+1,px}
            vals.append(1.0)
            # d wrt X_k,px
            vals.append(-1.0)
            # d wrt X_k,theta ( + h v sin(th) )
            vals.append(P.h * v * np.sin(th))
            # d wrt U_k,v ( - h cos(th) )
            vals.append(-P.h * np.cos(th))

            # Row r+1: g2 = x_{k+1,py} - [py_k + h v sin(th_k)]
            vals.append(1.0)                  # d wrt X_{k+1,py}
            vals.append(-1.0)                 # d wrt X_k,py
            vals.append(-P.h * v * np.cos(th))# d wrt X_k,theta
            vals.append(-P.h * np.sin(th))    # d wrt U_k,v

            # Row r+2: g3 = x_{k+1,th} - [th_k + h w_k]
            vals.append(1.0)   # d wrt X_{k+1,theta}
            vals.append(-1.0)  # d wrt X_k,theta
            vals.append(-P.h)  # d wrt U_k,omega

            r += 3

        # Obstacle inequalities
        for k in range(P.N + 1):
            px, py = X[k, 0], X[k, 1]
            for j in range(P.nObs):
                cx, cy, _ = P.obstacles[j]
                # d/d px_k: -2(px - cx)
                vals.append(-2.0 * (px - cx))
                # d/d py_k: -2(py - cy)
                vals.append(-2.0 * (py - cy))
                r += 1

        return np.array(vals, dtype=float)

    # --- Internal: build Jacobian sparsity pattern once ---
    def _build_jacobian_structure(self):
        P = self.P
        rows = []
        cols = []
        r = 0

        # Initial equality: g0..g2 depend on X0(px,py,th) diagonally
        for i in range(3):
            rows.append(r + i)
            cols.append(idx_x(0, P).start + i)
        r += 3

        # Dynamics equalities
        for k in range(P.N):
            ixk   = idx_x(k, P)
            ixkp1 = idx_x(k + 1, P)
            iuk   = idx_u(k, P)

            # Row r (px)
            rows.extend([r, r, r, r])
            cols.extend([
                ixkp1.start + 0,   # X_{k+1,px}
                ixk.start   + 0,   # X_k,px
                ixk.start   + 2,   # X_k,theta
                iuk.start   + 0    # U_k,v
            ])
            # Row r+1 (py)
            rows.extend([r + 1, r + 1, r + 1, r + 1])
            cols.extend([
                ixkp1.start + 1,   # X_{k+1,py}
                ixk.start   + 1,   # X_k,py
                ixk.start   + 2,   # X_k,theta
                iuk.start   + 0    # U_k,v
            ])
            # Row r+2 (theta)
            rows.extend([r + 2, r + 2, r + 2])
            cols.extend([
                ixkp1.start + 2,   # X_{k+1,theta}
                ixk.start   + 2,   # X_k,theta
                iuk.start   + 1    # U_k,omega
            ])
            r += 3

        # Obstacle inequalities: each depends only on px_k, py_k
        for k in range(P.N + 1):
            ixk = idx_x(k, P)
            for _ in range(P.nObs):
                rows.extend([r, r])
                cols.extend([ixk.start + 0, ixk.start + 1])
                r += 1

        assert r == P.M, "Jacobian structure row count mismatch"
        return rows, cols


def solve_and_plot():
    P = Params()

    # Initial guess
    z0 = initial_guess(P)

    # Problem + IPOPT
    problem = UnicycleTO(P)
    nlp = cyipopt.Problem(
        n=P.Z_DIM, m=P.M,
        problem_obj=problem,
        lb=problem.lb, ub=problem.ub,
        cl=problem.cl, cu=problem.cu
    )

    # Options (tweak as desired)
    nlp.add_option("tol", 1e-6)
    nlp.add_option("dual_inf_tol", 1e-6)
    nlp.add_option("constr_viol_tol", 1e-6)
    nlp.add_option("compl_inf_tol", 1e-6)
    nlp.add_option("max_iter", 2000)
    nlp.add_option("hessian_approximation", "limited-memory")
    nlp.add_option("print_level", 5)

    z_star, info = nlp.solve(z0)

    # Report
    X_star, U_star = unpack(z_star, P)

    # Robust status/metrics extraction across cyipopt versions
    status      = info.get("status", "unknown")
    status_msg  = info.get("status_msg", "")
    obj_val     = info.get("obj_val", problem.objective(z_star))

    # Different cyipopt builds use different keys for iteration count
    iter_keys = ("iterations", "niter", "iter_count", "iter", "iteration_count")
    iters = next((info[k] for k in iter_keys if k in info), None)

    print("\n=== Ipopt status ===")
    if status_msg:
        print(f"Status: {status} ({status_msg})")
    else:
        print(f"Status: {status}")
    print(f"Iterations: {iters if iters is not None else 'N/A'}")
    print(f"Final objective: {obj_val:.6f}")
    print("Terminal state [px py theta] =",
        np.round(X_star[-1, :], 3))


    # Plot environment and trajectories
    blue = (0/255, 114/255, 178/255)
    red = (200/255, 10/255, 10/255)
    fig = plt.figure(figsize=(8, 6.5), dpi=120)
    ax = fig.add_subplot(111)
    ax.set_title("Unicycle TO with Obstacle Avoidance (cyipopt / Ipopt)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    # Obstacles + safety margins
    ang = np.linspace(0, 2*np.pi, 200)
    for j in range(P.nObs):
        cx, cy, r0 = P.obstacles[j]
        r_eff = r0 + P.safety_margin
        xobs = cx + r_eff * np.cos(ang)
        yobs = cy + r_eff * np.sin(ang)
        ax.plot(xobs, yobs, "r-", lw=1.2)
        ax.fill(xobs, yobs, color="red", alpha=0.12)

    # Start/Goal
    ax.plot(P.A[0], P.A[1], "go", label="start")
    ax.plot(P.B[0], P.B[1], "b*", ms=10, label="goal")

    # Initial guess path
    X_init, _ = unpack(z0, P)
    ax.plot(X_init[:, 0], X_init[:, 1], "k--", lw=1.2, label="initial guess")

    # Optimized path
    ax.plot(X_star[:, 0], X_star[:, 1], color=red, lw=2.0, label="optimized")

    # Heading arrows (sparse)
    skip = max(1, P.N // 20)
    for k in range(0, P.N + 1, skip):
        px, py, th = X_star[k, :]
        ax.quiver(px, py, 0.4*np.cos(th), 0.4*np.sin(th), angles='xy', scale_units='xy', scale=1.0, color=blue, width=0.003)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Plot controls
    t = np.arange(P.N) * P.h
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5), dpi=120)
    ax1.plot(t, unpack(z0, P)[1][:, 0], "k--", label="v init")
    ax1.plot(t, U_star[:, 0], color=blue, label="v*")
    ax1.axhline(P.v_min, linestyle=":", color="k")
    ax1.axhline(P.v_max, linestyle=":", color="k")
    ax1.set_ylabel("v [m/s]")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(t, unpack(z0, P)[1][:, 1], "k--", label=r"$\omega$ init")
    ax2.plot(t, U_star[:, 1], color=blue, label=r"$\omega^*$")
    ax2.axhline(P.w_min, linestyle=":", color="k")
    ax2.axhline(P.w_max, linestyle=":", color="k")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel(r"$\omega$ [rad/s]")
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    solve_and_plot()
