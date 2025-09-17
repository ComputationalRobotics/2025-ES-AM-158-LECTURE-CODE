#!/usr/bin/env python3
"""
Policy evaluation on the Hangover MDP:
- Monte Carlo (first-visit, sample-average)
- TD(0) with per-state diminishing step sizes
- n-step TD with per-state diminishing step sizes
- TD(lambda) with per-state diminishing step sizes (accumulating traces)

Saves:
  - hangover_mc_td.png   : plot of MSE vs episodes
  - hangover_mc_td.csv   : averaged curves (episodes, methods)

Run:
  python policy_evaluation_mc_td_hangover.py --episodes 400 --runs 40 --alpha 0.4 --gamma 0.95 --n-step 3 --lam 0.9
"""

import argparse
import csv
import math
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ---------------- MDP spec (from your prompt) ----------------

State = str
Action = str

S: List[State] = [
    "Hangover", "Sleep", "More Sleep", "Visit Lecture", "Study", "Pass Exam"
]
A: List[Action] = ["Lazy", "Productive"]

P: Dict[Tuple[State, Action], List[Tuple[State, float]]] = {
    ("Hangover", "Lazy"):       [("Sleep", 1.0)],
    ("Hangover", "Productive"): [("Visit Lecture", 0.3), ("Hangover", 0.7)],

    ("Sleep", "Lazy"):          [("More Sleep", 1.0)],
    ("Sleep", "Productive"):    [("Visit Lecture", 0.6), ("More Sleep", 0.4)],

    ("More Sleep", "Lazy"):       [("More Sleep", 1.0)],
    ("More Sleep", "Productive"): [("Study", 0.5), ("More Sleep", 0.5)],

    ("Visit Lecture", "Lazy"):       [("Study", 0.8), ("Pass Exam", 0.2)],
    ("Visit Lecture", "Productive"): [("Study", 1.0)],

    ("Study", "Lazy"):         [("More Sleep", 1.0)],
    ("Study", "Productive"):   [("Pass Exam", 0.9), ("Study", 0.1)],

    ("Pass Exam", "Lazy"):       [("Pass Exam", 1.0)],
    ("Pass Exam", "Productive"): [("Pass Exam", 1.0)],
}

def R_state(s: State, a: Action) -> float:
    """Original: +1 in Pass Exam (when *current* state is Pass Exam), else -1."""
    return 1.0 if s == "Pass Exam" else -1.0

# --- Policy: π(a|s) = Lazy with prob α, Productive with prob 1-α
def pi(a: Action, s: State, alpha: float) -> float:
    return alpha if a == "Lazy" else (1.0 - alpha)

# ---------------- Simulation choices ----------------

# If True: reward on transition uses next-state convention r=+1 iff s_next == Pass Exam, else -1,
# and the episode terminates upon first entering Pass Exam (MC-friendly).
# If False: use the original R_state(s,a) each step and never force termination here (continuing).
REWARD_NEXT_STATE = True

START_STATE = "Hangover"
NONTERM = [s for s in S if s != "Pass Exam"]

# ---------------- Helpers ----------------

IDX = {s: i for i, s in enumerate(S)}

def sample_next(s: State, a: Action, rng: np.random.Generator) -> State:
    rows = P[(s, a)]
    probs = np.array([p for _, p in rows], dtype=float)
    nxt = rng.choice(len(rows), p=probs)
    return rows[nxt][0]

def expected_reward_next_state(s: State, a: Action) -> float:
    """E[r | s,a] for the next-state reward convention."""
    return sum((1.0 if s_next == "Pass Exam" else -1.0) * p
               for s_next, p in P[(s, a)])

def build_true_value(alpha: float, gamma: float) -> np.ndarray:
    """
    Solve (I - gamma P_pi) V = r_pi.
    r_pi uses next-state reward if REWARD_NEXT_STATE else R_state.
    """
    n = len(S)
    Ppi = np.zeros((n, n))
    rpi = np.zeros(n)
    for s in S:
        i = IDX[s]
        for a in A:
            w = pi(a, s, alpha)
            for s_next, p in P[(s, a)]:
                Ppi[i, IDX[s_next]] += w * p
            if REWARD_NEXT_STATE:
                rpi[i] += w * expected_reward_next_state(s, a)
            else:
                rpi[i] += w * R_state(s, a)

    I = np.eye(n)
    V = np.linalg.solve(I - gamma * Ppi, rpi)
    return V  # order matches S

# ---------------- Algorithms ----------------

def mse(V_est: np.ndarray, V_true: np.ndarray) -> float:
    idxs = [IDX[s] for s in NONTERM]
    return float(np.mean((V_est[idxs] - V_true[idxs]) ** 2))

def mc_first_visit(episodes: int, alpha: float, gamma: float,
                   runs: int = 1, seed0: int = 0, max_steps: int = 1000) -> np.ndarray:
    """Episodic, terminate on Pass Exam if REWARD_NEXT_STATE, else truncate at max_steps."""
    V = np.zeros(len(S))
    counts = np.zeros(len(S))
    errors = np.zeros(episodes)

    rng = np.random.default_rng(seed0)
    V_true = build_true_value(alpha, gamma)

    for ep in range(episodes):
        s = START_STATE
        traj = []  # (s_t, r_t)
        for t in range(max_steps):
            # sample action and next state
            a = "Lazy" if rng.random() < alpha else "Productive"
            s_next = sample_next(s, a, rng)
            if REWARD_NEXT_STATE:
                r = 1.0 if s_next == "Pass Exam" else -1.0
            else:
                r = R_state(s, a)
            traj.append((s, r))
            # termination
            if REWARD_NEXT_STATE and s_next == "Pass Exam":
                s = s_next
                break
            s = s_next

        # first-visit returns
        G = 0.0
        seen = set()
        for t in reversed(range(len(traj))):
            s_t, r_t = traj[t]
            G = r_t + gamma * G
            if s_t not in seen:
                seen.add(s_t)
                i = IDX[s_t]
                counts[i] += 1.0
                V[i] += (G - V[i]) / counts[i]

        errors[ep] = mse(V, V_true)

    return errors

def _alpha(count: float, c=0.7, t0=10.0, p=1.0) -> float:
    return float(c / ((count + t0) ** p))

def td0_decay(episodes: int, alpha_prob: float, gamma: float,
              c=0.7, t0=10.0, p=1.0, seed0: int = 0, max_steps: int = 1000) -> np.ndarray:
    V = np.zeros(len(S))
    counts = np.zeros(len(S))
    errors = np.zeros(episodes)
    V_true = build_true_value(alpha_prob, gamma)
    rng = np.random.default_rng(seed0)

    for ep in range(episodes):
        s = START_STATE
        for t in range(max_steps):
            a = "Lazy" if rng.random() < alpha_prob else "Productive"
            s_next = sample_next(s, a, rng)
            r = 1.0 if (REWARD_NEXT_STATE and s_next == "Pass Exam") else (-1.0 if REWARD_NEXT_STATE else R_state(s, a))
            i = IDX[s]; j = IDX[s_next]
            delta = r + gamma * V[j] - V[i]
            V[i] += _alpha(counts[i], c, t0, p) * delta
            counts[i] += 1.0
            if REWARD_NEXT_STATE and s_next == "Pass Exam":
                s = s_next
                break
            s = s_next
        errors[ep] = mse(V, V_true)
    return errors

def n_step_td_decay(episodes: int, alpha_prob: float, gamma: float, n: int = 3,
                    c=0.7, t0=10.0, p=1.0, seed0: int = 0, max_steps: int = 1000) -> np.ndarray:
    V = np.zeros(len(S))
    counts = np.zeros(len(S))
    errors = np.zeros(episodes)
    V_true = build_true_value(alpha_prob, gamma)
    rng = np.random.default_rng(seed0)

    for ep in range(episodes):
        s = START_STATE
        states = [s]
        rewards: List[float] = []
        T = math.inf
        t = 0
        while True:
            if t < T:
                a = "Lazy" if rng.random() < alpha_prob else "Productive"
                s_next = sample_next(s, a, rng)
                r = 1.0 if (REWARD_NEXT_STATE and s_next == "Pass Exam") else (-1.0 if REWARD_NEXT_STATE else R_state(s, a))
                states.append(s_next)
                rewards.append(r)
                if (REWARD_NEXT_STATE and s_next == "Pass Exam") or len(rewards) >= max_steps:
                    T = t + 1
                s = s_next
            tau = t - n + 1
            if tau >= 0:
                # n-step return
                G = 0.0
                last = tau + n if T is math.inf else min(tau + n, int(T))
                for k in range(tau, last):
                    G += (gamma ** (k - tau)) * rewards[k]
                if tau + n < T:
                    G += (gamma ** n) * V[IDX[states[tau + n]]]
                i_tau = IDX[states[tau]]
                V[i_tau] += _alpha(counts[i_tau], c, t0, p) * (G - V[i_tau])
                counts[i_tau] += 1.0
            if tau == T - 1:
                break
            t += 1

        errors[ep] = mse(V, V_true)
    return errors

def td_lambda_decay(episodes: int, alpha_prob: float, gamma: float, lam: float = 0.9,
                    c=0.5, t0=10.0, p=1.0, seed0: int = 0, max_steps: int = 1000) -> np.ndarray:
    V = np.zeros(len(S))
    counts = np.zeros(len(S))
    errors = np.zeros(episodes)
    V_true = build_true_value(alpha_prob, gamma)
    rng = np.random.default_rng(seed0)

    for ep in range(episodes):
        z = np.zeros(len(S))
        s = START_STATE
        for t in range(max_steps):
            a = "Lazy" if rng.random() < alpha_prob else "Productive"
            s_next = sample_next(s, a, rng)
            r = 1.0 if (REWARD_NEXT_STATE and s_next == "Pass Exam") else (-1.0 if REWARD_NEXT_STATE else R_state(s, a))
            i = IDX[s]; j = IDX[s_next]
            delta = r + gamma * V[j] - V[i]
            # traces
            z *= gamma * lam
            z[i] += 1.0
            # elementwise per-state step sizes
            alphas = np.array([_alpha(counts[k], c, t0, p) for k in range(len(S))])
            upd = alphas * delta * z
            V += upd
            counts += (np.abs(upd) > 0).astype(float)
            if REWARD_NEXT_STATE and s_next == "Pass Exam":
                break
            s = s_next
        errors[ep] = mse(V, V_true)

    return errors

# ---------------- Experiment harness ----------------

def run(episodes=400, runs=40, alpha=0.4, gamma=0.95, n_step=3, lam=0.9):
    curves = {
        "MC (first-visit)": np.zeros(episodes),
        "TD(0) (decay)": np.zeros(episodes),
        f"{n_step}-step TD (decay)": np.zeros(episodes),
        f"TD(lambda={lam}) (decay)": np.zeros(episodes),
    }
    for r in range(runs):
        seed = 1234 + r
        curves["MC (first-visit)"] += mc_first_visit(episodes, alpha, gamma, seed0=seed)
        curves["TD(0) (decay)"] += td0_decay(episodes, alpha, gamma, seed0=seed)
        curves[f"{n_step}-step TD (decay)"] += n_step_td_decay(episodes, alpha, gamma, n=n_step, seed0=seed)
        curves[f"TD(lambda={lam}) (decay)"] += td_lambda_decay(episodes, alpha, gamma, lam=lam, seed0=seed)
    for k in curves:
        curves[k] /= runs
    return curves

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.4, help="π(Lazy|s)=alpha")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--png", type=str, default="hangover_mc_td.png")
    ap.add_argument("--csv", type=str, default="hangover_mc_td.csv")
    args = ap.parse_args()

    curves = run(episodes=args.episodes, runs=args.runs,
                 alpha=args.alpha, gamma=args.gamma,
                 n_step=args.n_step, lam=args.lam)

    # plot
    x = np.arange(1, args.episodes + 1)
    plt.figure(figsize=(10, 6))
    for name, values in curves.items():
        plt.plot(x, values, label=name, linewidth=2)
    plt.xlabel("Episodes")
    plt.ylabel("MSE vs true V^π (nonterminal states)")
    plt.title("Hangover MDP: MC vs TD-family (per-state diminishing step sizes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.png, dpi=150)
    print(f"Saved figure to: {args.png}")
    plt.show()

    # CSV
    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["episode"] + list(curves.keys())
        writer.writerow(header)
        for i in range(args.episodes):
            writer.writerow([i + 1] + [curves[k][i] for k in curves])
    print(f"Saved CSV to: {args.csv}")

if __name__ == "__main__":
    main()
