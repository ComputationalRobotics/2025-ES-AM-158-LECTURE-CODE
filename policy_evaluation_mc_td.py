#!/usr/bin/env python3
"""
Compare policy-evaluation methods in a tabular MDP (Random Walk):
- First-visit Monte Carlo (sample-average by default)
- TD(0)
- n-step TD
- TD(lambda) with accumulating eligibility traces

Usage:
    python policy_evaluation_mc_td.py --episodes 200 --runs 50 --alpha-td 0.1 --alpha-nstep 0.1 --alpha-lambda 0.1 --lam 0.9 --n-step 3 --mc-alpha 0.1

Outputs:
    - mc_td_comparison.png  : plot of mean-squared value error vs episodes
"""

import argparse
import math
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Environment: 5-state Random Walk ------------------
class RandomWalk:
    """
    States: 0 and 6 are terminal. Nonterminal: 1..5.
    Start at state 3.
    Actions: Left (-1), Right (+1). Policy: equiprobable.
    Reward: +1 when transitioning into state 6, else 0.
    Gamma = 1 for the episodic task.
    """
    def __init__(self, start_state: int = 3, n_states: int = 5, seed: int = 0):
        self.n_states = n_states
        self.start_state = start_state
        self.terminal_left = 0
        self.terminal_right = n_states + 1  # = 6 when n_states=5
        self.state = self.start_state
        self.rng = np.random.default_rng(seed)
        self.gamma = 1.0

    def reset(self) -> int:
        self.state = self.start_state
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool]:
        # action: -1 (left) or +1 (right)
        s_next = self.state + action
        reward = 1.0 if s_next == self.terminal_right else 0.0
        done = (s_next == self.terminal_left) or (s_next == self.terminal_right)
        self.state = s_next
        return s_next, reward, done

    def sample_action(self) -> int:
        return -1 if self.rng.random() < 0.5 else +1

def true_values(n_states: int = 5) -> np.ndarray:
    """True V(s) for states 1..n_states under equiprobable policy, gamma=1."""
    s_vals = np.arange(1, n_states + 1)
    return s_vals / (n_states + 1)

def mse(V: np.ndarray, V_true: np.ndarray) -> float:
    """Mean squared error over nonterminal states 1..n_states."""
    return float(np.mean((V[1:-1] - V_true) ** 2))

# ------------------ Algorithms ------------------
def mc_first_visit(env_seed: int, episodes: int, alpha: float = 0.0) -> np.ndarray:
    """
    First-visit Monte Carlo policy evaluation.
    If alpha <= 0: sample-average updates (count-based).
    If alpha  > 0: constant step size alpha.
    """
    env = RandomWalk(seed=env_seed)
    V = np.zeros(env.n_states + 2)      # include terminals 0..6
    counts = np.zeros_like(V)
    errors = np.zeros(episodes)
    V_true = true_values(env.n_states)

    for ep in range(episodes):
        s = env.reset()
        trajectory = []  # (s_t, r_t)
        done = False
        while not done:
            a = env.sample_action()
            s_next, r, done = env.step(a)
            trajectory.append((s, r))
            s = s_next

        # Compute returns g_t backward; update first-visit only
        G = 0.0
        G_list = [0.0] * len(trajectory)   # trajectory is a list of (state, reward) pairs
        for t in reversed(range(len(trajectory))):
            s_t, r_t = trajectory[t]
            G = r_t + env.gamma * G        # discounted return
            G_list[t] = G                  # store return starting at time t

        # Forward pass: update V only at the first time each state appears in the episode
        visited = set()
        for t, (s_t, _) in enumerate(trajectory):
            # if s_t in visited:             # skip if we've already seen this state earlier
            #     continue
            visited.add(s_t)

            G = G_list[t]                  # use the return from this first occurrence
            if alpha > 0:                  # constant step-size update
                V[s_t] += alpha * (G - V[s_t])
            else:                          # sample-average update (1 / visit count)
                counts[s_t] += 1.0
                V[s_t] += (G - V[s_t]) / counts[s_t]

        errors[ep] = mse(V, V_true)

    return errors

def td0(env_seed: int, episodes: int, alpha: float = 0.1) -> np.ndarray:
    env = RandomWalk(seed=env_seed)
    V = np.zeros(env.n_states + 2)
    errors = np.zeros(episodes)
    V_true = true_values(env.n_states)

    for ep in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a = env.sample_action()
            s_next, r, done = env.step(a)
            V[s] += alpha * (r + env.gamma * V[s_next] - V[s])
            s = s_next
        errors[ep] = mse(V, V_true)
    return errors

def n_step_td(env_seed: int, episodes: int, n: int = 3, alpha: float = 0.1) -> np.ndarray:
    """Sutton & Barto n-step TD (episodic)."""
    env = RandomWalk(seed=env_seed)
    V = np.zeros(env.n_states + 2)
    errors = np.zeros(episodes)
    V_true = true_values(env.n_states)

    for ep in range(episodes):
        s = env.reset()
        states = [s]
        rewards = []
        T = math.inf
        t = 0
        while True:
            if t < T:
                a = env.sample_action()
                s_next, r, done = env.step(a)
                states.append(s_next)
                rewards.append(r)
                if done:
                    T = t + 1
            tau = t - n + 1
            if tau >= 0:
                # Compute n-step return
                G = 0.0
                if T is math.inf:
                    last = tau + n
                else:
                    last = min(tau + n, int(T))
                for k in range(tau, last):
                    G += (env.gamma ** (k - tau)) * rewards[k]
                if tau + n < T:
                    G += (env.gamma ** n) * V[states[tau + n]]
                s_tau = states[tau]
                V[s_tau] += alpha * (G - V[s_tau])
            if tau == T - 1:
                break
            t += 1
        errors[ep] = mse(V, V_true)

    return errors

def td_lambda(env_seed: int, episodes: int, lam: float = 0.9, alpha: float = 0.1) -> np.ndarray:
    env = RandomWalk(seed=env_seed)
    V = np.zeros(env.n_states + 2)
    errors = np.zeros(episodes)
    V_true = true_values(env.n_states)

    for ep in range(episodes):
        z = np.zeros_like(V)  # eligibility traces
        s = env.reset()
        done = False
        while not done:
            a = env.sample_action()
            s_next, r, done = env.step(a)
            delta = r + env.gamma * V[s_next] - V[s]
            z *= env.gamma * lam
            z[s] += 1.0
            V += alpha * delta * z
            s = s_next
        errors[ep] = mse(V, V_true)
    return errors

# ------------------ Per-state decaying step-size helpers ------------------

def _alpha_per_state(counts, c=0.7, t0=10.0, p=1.0):
    """Elementwise α(s) = c / (counts(s)+t0)^p."""
    return c / np.power(counts + t0, p)

# ------------------ TD(0) with per-state decaying α ------------------

def td0_per_state_decay(env_seed: int, episodes: int,
                        c: float = 0.7, t0: float = 10.0, p: float = 1.0) -> np.ndarray:
    env = RandomWalk(seed=env_seed)
    V = np.zeros(env.n_states + 2)
    counts = np.zeros_like(V)  # N(s): how many times V(s) updated
    errors = np.zeros(episodes)
    V_true = true_values(env.n_states)

    for ep in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a = env.sample_action()
            s_next, r, done = env.step(a)
            delta = r + env.gamma * V[s_next] - V[s]
            alpha_s = _alpha_per_state(counts[s], c, t0, p)  # scalar for this s
            V[s] += alpha_s * delta
            counts[s] += 1.0
            s = s_next
        errors[ep] = mse(V, V_true)
    return errors

# ------------------ n-step TD with per-state decaying α ------------------

def n_step_td_per_state_decay(env_seed: int, episodes: int, n: int = 3,
                              c: float = 0.7, t0: float = 10.0, p: float = 1.0) -> np.ndarray:
    env = RandomWalk(seed=env_seed)
    V = np.zeros(env.n_states + 2)
    counts = np.zeros_like(V)
    errors = np.zeros(episodes)
    V_true = true_values(env.n_states)

    for ep in range(episodes):
        s = env.reset()
        states = [s]
        rewards = []
        T = np.inf
        t = 0
        while True:
            if t < T:
                a = env.sample_action()
                s_next, r, done = env.step(a)
                states.append(s_next)
                rewards.append(r)
                if done:
                    T = t + 1
            tau = t - n + 1
            if tau >= 0:
                # n-step return
                G = 0.0
                last = tau + n if T == np.inf else min(tau + n, int(T))
                for k in range(tau, last):
                    G += (env.gamma ** (k - tau)) * rewards[k]
                if tau + n < T:
                    G += (env.gamma ** n) * V[states[tau + n]]

                s_tau = states[tau]
                delta = G - V[s_tau]
                alpha_s = _alpha_per_state(counts[s_tau], c, t0, p)
                V[s_tau] += alpha_s * delta
                counts[s_tau] += 1.0
            if tau == T - 1:
                break
            t += 1
        errors[ep] = mse(V, V_true)
    return errors

# ------------------ TD(lambda) with per-state decaying α ------------------

def td_lambda_per_state_decay(env_seed: int, episodes: int, lam: float = 0.9,
                              c: float = 0.5, t0: float = 10.0, p: float = 1.0) -> np.ndarray:
    """
    Accumulating eligibility traces. Each state uses α(s)=c/(N(s)+t0)^p.
    We increment N(s) when that state's value gets a nonzero update at a step (z_t(s) != 0).
    """
    env = RandomWalk(seed=env_seed)
    V = np.zeros(env.n_states + 2)
    counts = np.zeros_like(V)
    errors = np.zeros(episodes)
    V_true = true_values(env.n_states)

    for ep in range(episodes):
        z = np.zeros_like(V)
        s = env.reset()
        done = False
        while not done:
            a = env.sample_action()
            s_next, r, done = env.step(a)

            delta = r + env.gamma * V[s_next] - V[s]
            z *= env.gamma * lam
            z[s] += 1.0

            # elementwise step sizes for all states (vectorized)
            alphas = _alpha_per_state(counts, c, t0, p)
            update = alphas * delta * z
            V += update

            # increment counts for states that actually changed this step
            counts += (np.abs(update) > 0).astype(float)

            s = s_next
        errors[ep] = mse(V, V_true)
    return errors


# ------------------ Experiment runner ------------------
def run_experiment(
    episodes=200, runs=50, alpha_td=0.1, alpha_nstep=0.1, alpha_lambda=0.1,
    lam=0.9, n_step=3, mc_alpha=0.0
):
    curves = {
        "MC (first-visit, sample-avg)" if mc_alpha <= 0 else f"MC (alpha={mc_alpha})": np.zeros(episodes),
        "TD(0)": np.zeros(episodes),
        f"{n_step}-step TD": np.zeros(episodes),
        f"TD(lambda={lam})": np.zeros(episodes),
    }

    for run in range(runs):
        seed = 1000 + run
        curves[list(curves.keys())[0]] += mc_first_visit(seed, episodes, alpha=mc_alpha)
        # curves["TD(0)"] += td0(seed, episodes, alpha=alpha_td)
        # curves[f"{n_step}-step TD"] += n_step_td(seed, episodes, n=n_step, alpha=alpha_nstep)
        # curves[f"TD(lambda={lam})"] += td_lambda(seed, episodes, lam=lam, alpha=alpha_lambda)
        curves["TD(0)"] += td0_per_state_decay(seed, episodes, c=4, t0=3, p=1.0)
        curves[f"{n_step}-step TD"] += n_step_td_per_state_decay(seed, episodes, n=n_step, c=2, t0=5.0, p=1.0)
        curves[f"TD(lambda={lam})"] += td_lambda_per_state_decay(seed, episodes, lam=lam, c=2, t0=5.0, p=1.0)


    for k in curves:
        curves[k] /= runs
    return curves

# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser(description="MC vs TD(0) vs n-step TD vs TD(lambda) on Random Walk")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--alpha-td", type=float, default=0.1)
    parser.add_argument("--alpha-nstep", type=float, default=0.1)
    parser.add_argument("--alpha-lambda", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=0.9)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--mc-alpha", type=float, default=0.0, help="<=0 for sample-average MC; >0 for constant step-size")
    parser.add_argument("--png", type=str, default="mc_td_comparison.png")
    args = parser.parse_args()

    curves = run_experiment(
        episodes=args.episodes, runs=args.runs,
        alpha_td=args.alpha_td, alpha_nstep=args.alpha_nstep,
        alpha_lambda=args.alpha_lambda, lam=args.lam,
        n_step=args.n_step, mc_alpha=args.mc_alpha
    )

    # Plot
    palette = {
        f"MC (alpha={args.mc_alpha})"   : "#D62728",  # red
        "MC (first-visit, sample-avg)"  : "#1f77b4",  # blue
        "TD(0)"                         : "#ff7f0e",  # orange
        f"{args.n_step}-step TD"        : "#2ca02c",  # green
    }
    plt.figure(figsize=(10, 6))
    x = np.arange(1, args.episodes + 1)
    for name, values in curves.items():
        plt.plot(x, values, label=name, color=palette.get(name, "#333333"), linewidth=2)
    plt.xlabel("Episodes")
    plt.ylabel("Mean Squared Value Error (states 1..5)")
    plt.title("Policy Evaluation: MC vs TD(0) vs n-step TD vs TD(λ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.png, dpi=150)
    print(f"Saved figure to: {args.png}")
    plt.show()

if __name__ == "__main__":
    main()
