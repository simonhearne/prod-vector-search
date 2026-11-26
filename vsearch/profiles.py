"""Turn a purchase history into ranking signals (B6): brand affinity, price band, user embedding."""
from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd

from vsearch.config import K
from vsearch.filters import escape

JUNK_BRANDS = {"", "unknown", "generic", "n/a", "none"}
TOP_BRANDS = 5
MIN_PRICED = 2


def fold(brand) -> str:
    """Spelling-insensitive brand key; "" when there is no real brand."""
    b = brand.strip().casefold() if isinstance(brand, str) else ""
    return "" if b in JUNK_BRANDS else b


def brand_variants(brands: pd.Series) -> dict[str, list[str]]:
    """Folded brand -> every raw spelling in the catalogue (filters match raw values)."""
    out = defaultdict(set)
    for b in brands.unique():
        if key := fold(b):
            out[key].add(b)
    return {k: sorted(v) for k, v in out.items()}


def brand_weights(history: Iterable[str], brand_of: dict[str, str]) -> dict[str, float]:
    """The user's top brands by purchase count, as shares of their branded purchases."""
    n = Counter(b for p in history if (b := fold(brand_of.get(p))))
    total = sum(n.values())
    return {b: c / total for b, c in sorted(n.items(), key=lambda x: (-x[1], x[0]))[:TOP_BRANDS]}


def brand_filter(weights: dict[str, float], variants: dict[str, list[str]]) -> str | None:
    raw = sorted({v for b in weights for v in variants.get(b, [])})
    return "brand in [" + ",".join(f'"{escape(b)}"' for b in raw) + "]" if raw else None


def price_filter(history: Iterable[str], price_of: dict[str, float]) -> str | None:
    """Half to twice the user's median price. Unpriced items (missing, or the stored -1) never count or match."""
    prices = [p for i in history if (p := price_of.get(i, -1.0)) >= 0]  # NaN fails the comparison too
    if len(prices) < MIN_PRICED:
        return None
    m = float(np.median(prices))
    return f"price >= {m / 2:.2f} and price <= {2 * m:.2f}"


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n if n else v).astype(np.float32)


def user_vec(history_rows: list[int], tv: np.ndarray) -> np.ndarray:
    return _unit(np.asarray(tv[sorted(history_rows)], dtype=np.float64).mean(0))


def blend(qv: np.ndarray, u: np.ndarray, alpha: float) -> np.ndarray:
    return _unit(np.asarray(qv, np.float64) + alpha * np.asarray(u, np.float64))


def rerank(ids: list[str], scores: list[float], bonus: np.ndarray, weight: float) -> list[str]:
    """score + weight * bonus, keeping the engine's order on ties."""
    s = np.asarray(scores, float) + weight * np.asarray(bonus, float)
    return [ids[i] for i in np.argsort(-s, kind="stable")]


def rr(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def moved(a: list[str], b: list[str], k: int = K) -> float:
    """Share of the top-k page that differs: how strongly an arm acted, whatever its score scale."""
    return 1.0 - len(set(a[:k]) & set(b[:k])) / max(len(a[:k]), 1)
