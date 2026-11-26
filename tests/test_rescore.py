"""Critical 1 (final review): filled nDCG depends on the judgment pool at scoring time, so every canonical
result is re-scored at one snapshot. Rescoring must change only quality fields and record the snapshot."""
import pandas as pd
import pytest

from bench import rescore
from vsearch import judge

QR = pd.DataFrame({"query_id": ["q1", "q2"], "product_id": ["e1", "e2"], "label": ["E", "E"]})


@pytest.fixture(autouse=True)
def pool(monkeypatch):
    monkeypatch.setattr(judge, "cached_judgments", lambda: pd.DataFrame(
        {"query_id": ["q1"], "product_id": ["x"], "label": ["S"], "model": [judge.JUDGE_MODEL]}))


def test_rescore_points_keeps_measurements_and_replaces_quality():
    points = [{"ef": 10, "recall@10": 0.8, "latency": {"p50": 1.0}, "ndcg@10": 0.99, "filled": {"ndcg@10": 0.99}}]
    out = rescore.rescore_points(points, {"10": {"q1": ["e1", "x"], "q2": ["y"]}}, QR, key="ef")
    p = out[0]
    assert p["recall@10"] == 0.8 and p["latency"] == {"p50": 1.0}
    assert p["ndcg@10"] == pytest.approx(0.5) and p["filled"]["judged@10"] == pytest.approx(0.1)


def test_rescore_b4c_recomputes_censored_stats():
    ev = [{"pid": "a", "t_write": 0.0, "t_visible": None, "missing_s": 900.0}]
    out = rescore.rescore_b4c({"timeout_s": 900, "variants": {"ig": {"n": 1, "events": ev, "timed_out": 0}}})
    assert out["variants"]["ig"]["timed_out"] == 1 and out["variants"]["ig"]["visible_p99_censored"] is True
    assert out["variants"]["ig"]["events"] == ev


def test_rescore_showcase_updates_labels_and_ndcg_but_not_measurements():
    card = lambda p, lab, src: {"product_id": p, "title": p, "label": lab, "label_source": src}
    side = lambda *c: {"ndcg@10": 0.9, "judged@10": 0.0, "returned": len(c), "results": list(c)}
    row = {"query_id": "q1", "top_category": "Electronics", "ndcg_gap": 0.0,
           "poc": side(card("x", None, None)), "prod": side(card("e1", "E", "esci"))}
    payload = {"latency_split": {"embed": {"p50": 33.0}}, "narrow_filter": {"short_rate": 0.365},
               "poc_vs_prod": {"poc_category": "Electronics", "candidates": [], "candidates_in_category": [], "rows": [row]}}
    out = rescore.rescore_showcase(payload, QR)
    r = out["poc_vs_prod"]["rows"][0]
    assert r["poc"]["results"][0]["label"] == "S" and r["poc"]["results"][0]["label_source"] == "judge"
    assert r["prod"]["ndcg@10"] == pytest.approx(1 / (1 + 0.1 / 1.5849625007211562))
    assert r["ndcg_gap"] == pytest.approx(r["prod"]["ndcg@10"] - r["poc"]["ndcg@10"])
    assert out["latency_split"] == payload["latency_split"] and out["narrow_filter"] == payload["narrow_filter"]
