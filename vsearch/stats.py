"""Small statistics for comparing retrieval configs on per-query scores."""
import numpy as np

from vsearch.config import SEED


def paired_ci(d: np.ndarray) -> tuple[float, float]:
    """Mean of paired per-query differences and its 95% half-width (normal approximation)."""
    d = np.asarray(d, dtype=float)
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))


def draw_power(M: np.ndarray, n: int = 50, draws: int = 20_000, seed: int = SEED) -> dict:
    """M: queries x configs. How often does a random n-query subset pick the full-set winner,
    and the full-set order? winner_share[k] = share of draws that config k wins."""
    M = np.asarray(M, dtype=float)
    order = tuple(np.argsort(-M.mean(0), kind="stable"))
    rng = np.random.default_rng(seed)
    wins = np.zeros(M.shape[1])
    same = 0
    for _ in range(draws):
        o = tuple(np.argsort(-M[rng.choice(len(M), n, replace=False)].mean(0), kind="stable"))
        wins[o[0]] += 1
        same += o == order
    return {"winner": int(order[0]), "winner_right": float(wins[order[0]] / draws),
            "order_right": float(same / draws), "winner_share": [float(w / draws) for w in wins]}
