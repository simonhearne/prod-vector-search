import numpy as np
import pytest

from vsearch.search import summarise_latency


def test_summarise_latency_known():
    s = summarise_latency(np.arange(1, 101, dtype=float))
    assert s["p50"] == pytest.approx(50.5) and s["p99"] == pytest.approx(99.01)
    assert s["n"] == 100 and s["mean"] == pytest.approx(50.5)


class _Hit:
    def __init__(self, i):
        self.id, self.distance = f"p{i}", 1.0 - i / 1000


class _SearchClient:
    """Returns `limit` descending hits and records the keyword arguments of every call."""
    def __init__(self):
        self.calls = []

    def search(self, collection, **kw):
        self.calls.append(kw)
        return [[_Hit(i) for i in range(kw["limit"])]]

    def hybrid_search(self, collection, **kw):
        return self.search(collection, **kw)


def test_run_queries_passes_extra_and_cuts_the_page_but_keeps_scores():
    from vsearch.search import run_queries
    c, ranker = _SearchClient(), object()
    r = run_queries(c, "t", "text_vec", np.zeros((2, 4)), k=200, extra={"ranker": ranker}, page=10, warmup=1)
    assert all(kw["ranker"] is ranker and kw["limit"] == 200 for kw in c.calls)
    assert [len(x) for x in r.ids] == [10, 10] and [len(s) for s in r.scores] == [200, 200]
    assert r.scores[0][:2] == [1.0, 0.999]


def test_run_queries_default_is_unchanged():
    from vsearch.search import run_queries
    c = _SearchClient()
    r = run_queries(c, "t", "text_vec", np.zeros((1, 4)), warmup=0)
    assert "ranker" not in c.calls[0] and "function_chains" not in c.calls[0]
    assert len(r.ids[0]) == 10


def test_run_hybrid_returns_scores():
    from vsearch.search import run_hybrid
    r = run_hybrid(_SearchClient(), "t", np.zeros((1, 4)), ["q"], k=200, warmup=0)
    assert len(r.ids[0]) == len(r.scores[0]) == 200


def test_run_queries_takes_one_extra_per_query():
    from vsearch.search import run_queries
    c = _SearchClient()
    run_queries(c, "t", "text_vec", np.zeros((3, 4)), extra=[{"ranker": "a"}, None, {"ranker": "c"}], warmup=0)
    assert [kw.get("ranker") for kw in c.calls] == ["a", None, "c"]


class _HybridClient(_SearchClient):
    def hybrid_search(self, collection, **kw):
        self.calls.append({"hybrid": True, **kw})
        return [[_Hit(i) for i in range(kw["limit"])]]


def test_run_hybrid_gives_each_arm_more_candidates_than_it_returns():
    from vsearch.search import run_hybrid
    c = _HybridClient()
    r = run_hybrid(c, "t", np.zeros((1, 4)), ["q"], warmup=0)
    assert [req.limit for req in c.calls[0]["reqs"]] == [100, 100]
    assert c.calls[0]["limit"] == 10 and len(r.ids[0]) == 10


def test_run_hybrid_never_asks_an_arm_for_fewer_than_k():
    from vsearch.search import run_hybrid
    c = _HybridClient()
    run_hybrid(c, "t", np.zeros((1, 4)), ["q"], k=200, warmup=0)
    assert [req.limit for req in c.calls[0]["reqs"]] == [200, 200] and c.calls[0]["limit"] == 200


def test_brand_hybrid_falls_back_to_dense_without_a_profile():
    from vsearch.search import run_brand_hybrid
    c = _HybridClient()
    r = run_brand_hybrid(c, "t", np.zeros((2, 4)), [{3: 0.5}, None], k=200, dense_weight=0.9,
                         dense_params={"ef": 512}, warmup=0)
    hybrid, dense = c.calls
    assert hybrid["hybrid"] and hybrid["limit"] == 200
    assert [q.anns_field for q in hybrid["reqs"]] == ["text_vec", "brand_sparse"]
    assert hybrid["ranker"].dict()["params"]["weights"] == pytest.approx([0.9, 0.1])
    assert "hybrid" not in dense and dense["anns_field"] == "text_vec" and dense["limit"] == 200
    assert len(r.ids) == len(r.scores) == 2
