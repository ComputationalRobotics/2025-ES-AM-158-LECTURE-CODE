# Finite-horizon policy evaluation for the Hangover MDP

from collections import defaultdict
from typing import Dict, List, Tuple

State = str
Action = str

# --- MDP spec ---------------------------------------------------------------

S: List[State] = [
    "Hangover", "Sleep", "More Sleep", "Visit Lecture", "Study", "Pass Exam"
]
A: List[Action] = ["Lazy", "Productive"]

# P[s, a] -> list of (s_next, prob)
P: Dict[Tuple[State, Action], List[Tuple[State, float]]] = {
    # Hangover
    ("Hangover", "Lazy"):       [("Sleep", 1.0)],
    ("Hangover", "Productive"): [("Visit Lecture", 0.3), ("Hangover", 0.7)],

    # Sleep
    ("Sleep", "Lazy"):          [("More Sleep", 1.0)],
    ("Sleep", "Productive"):    [("Visit Lecture", 0.6), ("More Sleep", 0.4)],

    # More Sleep
    ("More Sleep", "Lazy"):       [("More Sleep", 1.0)],
    ("More Sleep", "Productive"): [("Study", 0.5), ("More Sleep", 0.5)],

    # Visit Lecture
    ("Visit Lecture", "Lazy"):       [("Study", 0.8), ("Pass Exam", 0.2)],
    ("Visit Lecture", "Productive"): [("Study", 1.0)],

    # Study
    ("Study", "Lazy"):         [("More Sleep", 1.0)],
    ("Study", "Productive"):   [("Pass Exam", 0.9), ("Study", 0.1)],

    # Pass Exam (absorbing)
    ("Pass Exam", "Lazy"):       [("Pass Exam", 1.0)],
    ("Pass Exam", "Productive"): [("Pass Exam", 1.0)],
}

def R(s: State, a: Action) -> float:
    """Reward: +1 in Pass Exam, -1 otherwise."""
    return 1.0 if s == "Pass Exam" else -1.0

# --- Policy: time-invariant, state-independent ------------------------------

def pi(a: Action, s: State, alpha: float) -> float:
    """π(a|s): Lazy with prob α, Productive with prob 1-α."""
    return alpha if a == "Lazy" else (1.0 - alpha)

# --- Policy evaluation -------------------------------------------------------

def policy_evaluation(T: int, alpha: float):
    """
    Compute {V_t(s)} and {Q_t(s,a)} for t=0..T with terminal condition V_T = Q_T = 0.
    Returns:
        V: Dict[int, Dict[State, float]]
        Q: Dict[int, Dict[Tuple[State, Action], float]]
    """
    assert T >= 0
    # sanity: probabilities sum to 1 for each (s,a)
    for key, rows in P.items():
        total = sum(p for _, p in rows)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Probabilities for {key} sum to {total}, not 1.")

    V: Dict[int, Dict[State, float]] = defaultdict(dict)
    Q: Dict[int, Dict[Tuple[State, Action], float]] = defaultdict(dict)

    # Terminal boundary
    for s in S:
        V[T][s] = 0.0
        for a in A:
            Q[T][(s, a)] = 0.0

    # Backward recursion
    for t in range(T - 1, -1, -1):
        for s in S:
            # First compute Q_t(s,a)
            for a in A:
                exp_next = sum(p * V[t + 1][s_next] for s_next, p in P[(s, a)])
                Q[t][(s, a)] = R(s, a) + exp_next
            # Then V_t(s) = E_{a~π}[Q_t(s,a)]
            V[t][s] = sum(pi(a, s, alpha) * Q[t][(s, a)] for a in A)

    return V, Q

# --- Example run -------------------------------------------------------------

if __name__ == "__main__":
    T = 10        # horizon
    alpha = 0.4  # probability of choosing Lazy
    V, Q = policy_evaluation(T=T, alpha=alpha)

    # Print V_0
    print(f"V_0(s) with T={T}, alpha={alpha}:")
    for s in S:
        print(f"  {s:13s}: {V[0][s]: .3f}")
