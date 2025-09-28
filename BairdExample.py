# Off-policy policy evaluation with semi-gradient TD(0) on Baird's counterexample.
# This script defines the MDP, features, and runs TD(0) off-policy with and without
# importance sampling (IS). It then plots the weight norm over time to illustrate divergence.

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Baird's "star" MDP
# ----------------------------
# States: 0..5 = "upper" states, 6 = "lower" state
# Actions: 0 = dashed, 1 = solid
# Transitions:
#   dashed (0): to a random upper state uniformly
#   solid  (1): to the lower state deterministically
# Rewards: all zero
class BairdMDP:
    def __init__(self, gamma=0.99):
        self.nS = 7
        self.upper = list(range(6))
        self.lower = 6
        self.gamma = gamma
        # Transition matrices P[a][s, s']
        self.P = np.zeros((2, self.nS, self.nS))
        self.P[0, :, self.upper] = 1.0 / 6.0  # dashed -> uniform on uppers
        self.P[1, :, self.lower] = 1.0        # solid  -> to lower
        self.R = np.zeros((2, self.nS))       # zero rewards

    def step(self, s, a, rng):
        p = self.P[a, s]
        s_next = rng.choice(self.nS, p=p)
        r = self.R[a, s]
        return s_next, r

# Feature map used in Baird's example (linear function approximation for V^pi)
# 8-dimensional features for 7 states:
#  - Upper state i (0..5): 2*e_i + 1*e_8
#  - Lower state (6):      1*e_7 + 2*e_8
def baird_features():
    Phi = np.zeros((7, 8))
    Phi[:6, :6] = 2 * np.eye(6)  # 2*e_i in the first 6 coords
    Phi[:, -1] = 1.0             # +1 in the last coord for all states
    Phi[6, -2] = 1.0             # lower state: +1 in the 7th coord
    Phi[6, -1] = 2.0             # and +2 (not +1) in the last coord
    return Phi

# ----------------------------
# Off-policy semi-gradient TD(0) for policy evaluation
# ----------------------------
# We evaluate the target policy π while sampling from behavior μ.
# v_w(s) = phi(s)^T w
# δ_t = r_t + γ v_w(s_{t+1}) - v_w(s_t)
# w <- w + α * ρ_t * δ_t * phi(s_t),  where ρ_t = π(a_t|s_t)/μ(a_t|s_t)  (optionally ρ_t=1)
def run_offpolicy_td(steps=100_000, alpha=0.01, gamma=0.99, use_is=True, seed=0, w_init=None):
    rng = np.random.default_rng(seed)
    mdp = BairdMDP(gamma=gamma)
    Phi = baird_features()

    # Behavior μ: prefer dashed (common choice: dashed with prob 6/7, solid 1/7)
    p_mu = np.array([6/7, 1/7])  # [P(a=dashed), P(a=solid)]

    # Target π: always take SOLID (this is one standard version of Baird's example)
    # (Other expositions flip dashed/solid for π; the divergence phenomenon remains.)
    p_pi = np.array([0.0, 1.0])

    # Initialize parameters (nonzero so we can see dynamics)
    if w_init is None:
        w = np.ones(Phi.shape[1])
        w[-2] = 10.0  # classic initialization that accentuates the instability
    else:
        w = w_init.copy()

    # Start from a random upper state
    s = rng.integers(0, 6)

    norms = []
    for t in range(steps):
        # Sample action from behavior policy
        a = rng.choice(2, p=p_mu)

        # Transition
        s_next, r = mdp.step(s, a, rng)

        # Semi-gradient TD(0) target and error
        v  = Phi[s] @ w
        vp = Phi[s_next] @ w
        delta = r + gamma * vp - v

        # Importance sampling ratio (ρ=1 corresponds to the "ordinary" off-policy TD(0) used in many demos)
        rho = (p_pi[a] / p_mu[a]) if use_is else 1.0

        # Update
        w += alpha * rho * delta * Phi[s]
        norms.append(np.linalg.norm(w))

        s = s_next

    return np.array(norms), w

# ----------------------------
# Run two variants: without IS and with IS
# ----------------------------
norms_noIS, w_noIS = run_offpolicy_td(use_is=False, steps=80_000, alpha=0.01, seed=0)
norms_IS,  w_IS  = run_offpolicy_td(use_is=True,  steps=80_000, alpha=0.01, seed=0)

# ----------------------------
# Plot the weight norms
# ----------------------------
plt.figure()
plt.plot(norms_noIS)
plt.xlabel("updates")
plt.ylabel("||w||_2")
plt.title("Off-policy TD(0) on Baird (no IS): weight norm")
plt.show()

plt.figure()
plt.plot(norms_IS)
plt.xlabel("updates")
plt.ylabel("||w||_2")
plt.title("Off-policy TD(0) on Baird (with importance sampling): weight norm")
plt.show()

# Print final norms so users can compare numerically
print("Final ||w|| without IS:", float(norms_noIS[-1]))
print("Final ||w|| with IS   :", float(norms_IS[-1]))

