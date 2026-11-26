"""Brute-force top-k by inner product (vectors are unit-normalised, so = cosine)."""
import numpy as np

from vsearch.config import K


def exact_topk(docs: np.ndarray, queries: np.ndarray, k: int = K,
               mask: np.ndarray | None = None, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Row indexes and scores, shape (nq, k). Slots beyond the mask's size hold -1 / -inf."""
    nq = len(queries)
    idx_out = np.full((nq, k), -1, dtype=np.int64)
    sc_out = np.full((nq, k), -np.inf, dtype=np.float32)
    if mask is not None:
        rows = np.flatnonzero(mask)
        sub = docs[rows]
    else:
        rows, sub = None, docs
    n = len(sub)
    if n == 0:
        return idx_out, sc_out
    kk = min(k, n)
    for s in range(0, nq, batch):
        scores = queries[s:s + batch] @ sub.T
        part = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
        ps = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-ps, axis=1, kind="stable")
        top = np.take_along_axis(part, order, axis=1)
        idx_out[s:s + batch, :kk] = top if rows is None else rows[top]
        sc_out[s:s + batch, :kk] = np.take_along_axis(ps, order, axis=1)
    return idx_out, sc_out
