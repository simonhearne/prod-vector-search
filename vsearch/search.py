"""Sequential, individually-timed search runs. Latency is client-side wall time."""
import time
from dataclasses import dataclass, field

import numpy as np
from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

from vsearch.config import K

CANDIDATES = 100  # per arm: with only k each, RRF cannot see agreement below the page and alternates disjoint lists


@dataclass
class RunResult:
    ids: list[list[str]]
    latencies_ms: np.ndarray
    scores: list[list[float]] = field(default_factory=list)  # aligned to the hits, before any page cut


def _ids(hits) -> list[str]:
    return [h.id for h in hits]


def _vec(v: np.ndarray) -> list[float]:
    return np.asarray(v, dtype=np.float32).tolist()


def run_queries(client, collection: str, anns_field: str, qvecs: np.ndarray, *, k: int = K,
                filters: list[str | None] | None = None, search_params: dict | None = None,
                warmup: int = 20, extra: dict | list[dict | None] | None = None,
                page: int | None = None) -> RunResult:
    """extra: more client.search arguments (e.g. ranker), or one dict per query. page: keep only the first `page` ids of the k fetched."""
    params = {"params": dict(search_params or {})}
    filters = filters if filters is not None else [None] * len(qvecs)
    extras = extra if isinstance(extra, list) else [extra] * len(qvecs)

    def one(i: int):
        return client.search(collection, data=[_vec(qvecs[i])], anns_field=anns_field,
                             filter=filters[i] or "", search_params=params, limit=k, **(extras[i] or {}))[0]

    for w in range(warmup):
        one(w % len(qvecs))
    ids, lat, scores = [], [], []
    for i in range(len(qvecs)):
        t0 = time.perf_counter()
        hits = one(i)
        lat.append((time.perf_counter() - t0) * 1000)
        ids.append(_ids(hits)[:page])
        scores.append([h.distance for h in hits])
    return RunResult(ids, np.array(lat), scores)


def run_hybrid(client, collection: str, qvecs: np.ndarray, qtexts: list[str], *, k: int = K,
               filters: list[str | None] | None = None, rrf_k: int = 60, warmup: int = 20,
               dense_params: dict | None = None, candidates: int = CANDIDATES) -> RunResult:
    arm_limit = max(k, candidates)
    filters = filters if filters is not None else [None] * len(qvecs)

    def one(i: int):
        expr = filters[i] or ""
        reqs = [AnnSearchRequest(data=[_vec(qvecs[i])], anns_field="text_vec",
                                 param={"params": dict(dense_params or {})}, limit=arm_limit, expr=expr),
                AnnSearchRequest(data=[qtexts[i]], anns_field="text_sparse", param={}, limit=arm_limit, expr=expr)]
        return client.hybrid_search(collection, reqs=reqs, ranker=RRFRanker(rrf_k), limit=k)[0]

    for w in range(warmup):
        one(w % len(qvecs))
    ids, lat, scores = [], [], []
    for i in range(len(qvecs)):
        t0 = time.perf_counter()
        hits = one(i)
        lat.append((time.perf_counter() - t0) * 1000)
        ids.append(_ids(hits))
        scores.append([h.distance for h in hits])
    return RunResult(ids, np.array(lat), scores)


def run_brand_hybrid(client, collection: str, qvecs: np.ndarray, sparse: list[dict[int, float] | None], *, k: int,
                     dense_weight: float, dense_params: dict | None = None, warmup: int = 20) -> RunResult:
    """Dense text plus the user's sparse brand vector, weighted on normalised scores. No profile -> dense only."""
    params = {"params": dict(dense_params or {})}

    def one(i: int):
        if not sparse[i]:
            return client.search(collection, data=[_vec(qvecs[i])], anns_field="text_vec", search_params=params, limit=k)[0]
        reqs = [AnnSearchRequest(data=[_vec(qvecs[i])], anns_field="text_vec", param=params, limit=k),
                AnnSearchRequest(data=[sparse[i]], anns_field="brand_sparse", param={}, limit=k)]
        ranker = WeightedRanker(dense_weight, 1 - dense_weight, norm_score=True)
        return client.hybrid_search(collection, reqs=reqs, ranker=ranker, limit=k)[0]

    for w in range(warmup):
        one(w % len(qvecs))
    ids, lat, scores = [], [], []
    for i in range(len(qvecs)):
        t0 = time.perf_counter()
        hits = one(i)
        lat.append((time.perf_counter() - t0) * 1000)
        ids.append(_ids(hits))
        scores.append([h.distance for h in hits])
    return RunResult(ids, np.array(lat), scores)


def summarise_latency(lat: np.ndarray) -> dict:
    lat = np.asarray(lat, dtype=float)
    return {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
            "p99": float(np.percentile(lat, 99)), "mean": float(lat.mean()), "n": int(len(lat))}
