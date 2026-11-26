"""Relevance and retrieval-fidelity metrics.

nDCG uses ESCI gains (E=1, S=0.1, C=0.01, I=0); unjudged results score 0;
IDCG comes from all of the query's judged labels.
"""
import numpy as np
import pandas as pd

from vsearch.config import GAINS, K, SEED


def _dcg(gains) -> float:
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains)))


def ndcg_at_k(ranked: list[str], labels: dict[str, str], k: int = K) -> float:
    ideal = _dcg(sorted((GAINS[l] for l in labels.values()), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return _dcg(GAINS.get(labels.get(p, ""), 0.0) for p in ranked[:k]) / ideal


def judged_at_k(ranked: list[str], labels: dict[str, str], k: int = K) -> float:
    return sum(p in labels for p in ranked[:k]) / k


def recall_at_k(ranked: list[str], truth: list[str], k: int = K) -> float:
    truth = list(truth)[:k]
    if not truth:
        return 1.0
    return len(set(ranked[:k]) & set(truth)) / min(k, len(truth))


def summarise_quality(rankings: dict[str, list[str]], qrels: pd.DataFrame, k: int = K) -> dict:
    lab = {qid: dict(zip(g.product_id, g.label)) for qid, g in qrels.groupby("query_id")}
    qids = list(rankings)
    nd = np.array([ndcg_at_k(rankings[q], lab.get(q, {}), k) for q in qids])
    jd = np.array([judged_at_k(rankings[q], lab.get(q, {}), k) for q in qids])
    rng = np.random.default_rng(SEED)
    boots = [nd[rng.integers(0, len(nd), len(nd))].mean() for _ in range(1000)] if len(nd) else [0.0]
    return {"ndcg@10": float(nd.mean()) if len(nd) else 0.0,
            "ndcg@10_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "judged@10": float(jd.mean()) if len(jd) else 0.0,
            "n_queries": len(qids)}
