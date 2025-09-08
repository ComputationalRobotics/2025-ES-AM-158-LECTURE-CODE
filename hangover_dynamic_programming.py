# Dynamic programming (finite-horizon optimal control) for the Hangover MDP

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

# --- Dynamic programming (Bellman optimality) -------------------------------

def dynamic_programming(T: int):
    """
    Compute optimal finite-horizon tables:
      - V[t][s] = V_t^*(s)
      - Q[t][(s,a)] = Q_t^*(s,a)
      - PI[t][s] = optimal action at (t,s)
    with terminal condition V_T^* = 0.
    """
    assert T >= 0

    # sanity: probabilities sum to 1 for each (s,a)
    for key, rows in P.items():
        total = sum(p for _, p in rows)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Probabilities for {key} sum to {total}, not 1.")

    V: Dict[int, Dict[State, float]] = defaultdict(dict)
    Q: Dict[int, Dict[Tuple[State, Action], float]] = defaultdict(dict)
    PI: Dict[int, Dict[State, Action]] = defaultdict(dict)

    # Terminal boundary
    for s in S:
        V[T][s] = 0.0
        for a in A:
            Q[T][(s, a)] = 0.0

    # Backward recursion (Bellman optimality)
    for t in range(T - 1, -1, -1):
        for s in S:
            # compute Q*_t(s,a)
            for a in A:
                exp_next = sum(p * V[t + 1][s_next] for s_next, p in P[(s, a)])
                Q[t][(s, a)] = R(s, a) + exp_next

            # greedy action and optimal value
            # tie-breaking is deterministic by the order in A
            best_a = max(A, key=lambda a: Q[t][(s, a)])
            PI[t][s] = best_a
            V[t][s] = Q[t][(s, best_a)]

    return V, Q, PI

# --- Example run -------------------------------------------------------------

if __name__ == "__main__":
    T = 10  # horizon
    V, Q, PI = dynamic_programming(T=T)

    print(f"Optimal V_0(s) with T={T}:")
    for s in S:
        print(f"  {s:13s}: {V[0][s]: .3f}")

    print("\nGreedy policy at t=0:")
    for s in S:
        print(f"  {s:13s}: {PI[0][s]}")

    print("\nAction value at t=0:")
    for s in S:
        print(f"  {s:13s}: {Q[0][s, A[0]]: .3f}, {Q[0][s, A[1]]: .3f}")
