# Minimal dense QP with CVXPY
#   minimize    (1/2) x^T P x + q^T x
#   subject to  Ax <= b, 1^T x = 1
#
# pip install cvxpy

import numpy as np
import cvxpy as cp

# ----- QP data (dense) -----
P = np.array([
    [4.0, 1.0, 0.5],
    [1.0, 2.0, 0.3],
    [0.5, 0.3, 1.5]
], dtype=float)
# Make sure P is symmetric positive definite
P = 0.5 * (P + P.T) + 1e-9 * np.eye(3)

q = np.array([-1.0, -2.0, -3.0])

A = np.array([
    [1.0, -2.0, 1.0],   # linear inequality:  x1 - 2 x2 + x3 ≤ 2
    [-1.0, 0.0, 0.0],   # x1 ≥ 0  ->  -x1 ≤ 0
    [0.0, -1.0, 0.0],   # x2 ≥ 0  ->  -x2 ≤ 0
    [0.0, 0.0, -1.0],   # x3 ≥ 0  ->  -x3 ≤ 0
    [1.0, 0.0, 0.0],    # x1 ≤ 1.5
    [0.0, 1.0, 0.0],    # x2 ≤ 1.5
    [0.0, 0.0, 1.0],    # x3 ≤ 1.5
])
b = np.array([2.0, 0.0, 0.0, 0.0, 1.5, 1.5, 1.5])

# Equality: sum(x) = 1
e = np.ones((1, 3))
d = np.array([1.0])

# ----- CVXPY problem -----
x = cp.Variable(3)

objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x)
constraints = [
    A @ x <= b,
    e @ x == d
]

prob = cp.Problem(objective, constraints)
# You can choose a solver; OSQP is common for QPs. ECOS/SCS also work.
prob.solve(solver=cp.MOSEK, verbose=True)

print("Status:", prob.status)
print("Optimal value:", prob.value)
print("x* =", x.value.round(6))

# (Optional) check constraints
ineq_res = (A @ x.value - b)
eq_res = (e @ x.value - d)
print("Max inequality residual (<=0):", np.max(ineq_res))
print("Equality residual (≈0):", eq_res.item())
