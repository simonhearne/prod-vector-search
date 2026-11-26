"""B5 scoring is offline: cached rankings in, the m1-m4 blocks out. No Milvus here."""
import pandas as pd
import pytest

from bench import b5_merch as b5
from bench import rescore
from vsearch import judge

QIDS = ["q1", "q2", "q3"]
QR = pd.DataFrame({"query_id": QIDS, "product_id": ["e1", "e2", "e3"], "label": ["E"] * 3})
MERCH = pd.DataFrame({"margin": [0.1, 0.2, 0.3, 0.5, 0.6, 0.1], "campaigns": [[], [], [], ["bf26"], [], []],
                      "hidden": [False, False, False, False, False, True]},
                     index=["e1", "e2", "e3", "c", "m", "h"])


def _run(a, b, c, p50=2.0, p99=4.0):
    return {"ids": dict(zip(QIDS, [a, b, c])), "latency": {"p50": p50, "p99": p99, "n": 3}}


@pytest.fixture(autouse=True)
def pool(monkeypatch):
    monkeypatch.setattr(judge, "cached_judgments", lambda: pd.DataFrame(columns=judge.COLUMNS))


@pytest.fixture
def cache():
    return {"n": 3, "query_ids": QIDS, "segments": {"prod_merch": 4}, "level": {"prod_merch": True}, "runs": {
        "base|prod_merch": _run(["e1", "h"], ["e2"], ["e3"]),
        "base500|prod_merch": _run(["e1", "h", "c"], ["e2", "c"], ["e3", "h"]),
        "m1|prod_merch|1.5|10": _run(["e1", "h"], ["e2"], ["e3"], p50=3.0),
        "m1|prod_merch|1.5|200": _run(["c", "e1"], ["e2"], ["e3"], p50=4.0),
        "m2|Multiply|1.1": _run(["e1", "m"], ["e2"], ["e3"]),
        "m3|bury|0.5|10": _run(["e1", "h"], ["e2"], ["e3"]),
        "m3|hide|flag": _run(["e1"], ["e2"], ["e3"], p50=6.0),
        "m3|hide|ids10": _run(["e1"], ["e2"], ["e3"]),
        "m4|dense_text|off": _run(["e1"], ["e2"], ["e3"]),
        "m4|dense_text|on": _run(["c", "e1"], ["c", "e2"], ["c", "e3"]),
        "m4|hybrid_rrf|off": _run(["m", "e1"], ["e2"], ["e3"]),
        "m4|hybrid_rrf|on": _run(["m", "e1"], ["e2"], ["e3"]),
    }}


def test_m1_reach_counts_only_queries_with_a_campaign_product_in_reach(cache):
    m1 = {r["limit"]: r for r in b5.score(cache, QR, MERCH, QIDS)["m1"]["rows"]}
    assert m1[10]["reach"] == 0.0 and m1[200]["reach"] == 0.5  # q3 has none in its top 500
    assert m1[10]["p50_rel"] == pytest.approx(1.5) and m1[200]["p50_rel"] == pytest.approx(2.0)
    assert m1[10]["ndcg@10"] == pytest.approx(1.0)  # limits 10 and 200 on prod_merch are judge-filled
    block = b5.score(cache, QR, MERCH, QIDS)["m1"]
    assert block["base_reach"]["prod_merch"] == 0.0 and block["n_eligible"]["prod_merch"] == 2


def test_m1_cells_outside_the_fill_report_no_ndcg(cache):
    cache["runs"]["m1|prod_merch|1.5|500"] = cache["runs"]["m1|prod_merch|1.5|200"]
    row = next(r for r in b5.score(cache, QR, MERCH, QIDS)["m1"]["rows"] if r["limit"] == 500)
    assert "ndcg@10" not in row
    assert "m1|prod_merch|1.5|500" not in b5.fill_keys(cache) and "m1|prod_merch|1.5|200" in b5.fill_keys(cache)


def test_m2_margin_against_relevance(cache):
    m2 = b5.score(cache, QR, MERCH, QIDS)["m2"]
    assert m2["base"]["margin@10"] == pytest.approx(((0.1 + 0.1) / 2 + 0.2 + 0.3) / 3)
    assert m2["rows"][0]["margin@10"] == pytest.approx(((0.1 + 0.6) / 2 + 0.2 + 0.3) / 3)
    assert m2["rows"][0]["mode"] == "Multiply" and m2["rows"][0]["weight"] == 1.1


def test_m3_leak_and_page_fill(cache):
    m3 = b5.score(cache, QR, MERCH, QIDS)["m3"]
    assert m3["base"]["leak"] == pytest.approx(1 / 3)
    bury, hide = m3["bury"][0], {r["kind"]: r for r in m3["hide"]}
    assert bury["leak"] == pytest.approx(1 / 3) and bury["weight"] == 0.5 and bury["limit"] == 10
    assert hide["flag"]["leak"] == 0.0 and hide["flag"]["p50_rel"] == pytest.approx(3.0)
    assert hide["flag"]["page_fill"] == pytest.approx(0.1)  # 1 of 10 slots


def test_m4_reports_the_order_flip(cache):
    m4 = b5.score(cache, QR, MERCH, QIDS)["m4"]
    assert m4["order_off"] == ["dense_text", "hybrid_rrf"] and m4["order_on"] == ["hybrid_rrf", "dense_text"]
    assert m4["flip"] is True
    assert m4["on_minus_off"]["hybrid_rrf"][0] == pytest.approx(0.0) and m4["on_minus_off"]["dense_text"][0] < 0


def test_score_refuses_a_cache_from_other_queries(cache):
    with pytest.raises(ValueError, match="query_ids"):
        b5.score(cache, QR, MERCH, ["q1", "q2", "other"])


def test_hide_ids_puts_the_most_seen_hidden_products_first():
    base = {"q1": ["h2", "a"], "q2": ["h2", "h1"]}
    assert b5.hide_ids(base, {"h0", "h1", "h2"}, 2) == ["h2", "h1"]
    assert b5.hide_ids(base, {"h0", "h1", "h2"}, 10) == ["h2", "h1", "h0"]


def test_rescore_b5_changes_quality_and_keeps_latency(cache, monkeypatch):
    before = {"ef": 512, "n": 3, **b5.score(cache, QR, MERCH, QIDS)}
    monkeypatch.setattr(judge, "cached_judgments", lambda: pd.DataFrame(
        {"query_id": ["q1"], "product_id": ["c"], "label": ["E"], "model": [judge.JUDGE_MODEL], "prompt": ["x"]}))
    after = rescore.rescore_b5(before, cache, QR, MERCH, QIDS)
    assert after["ef"] == 512
    b, a = ({r["limit"]: r for r in p["m1"]["rows"]} for p in (before, after))
    assert a[10]["ndcg@10"] < b[10]["ndcg@10"]  # q1 now has a second Exact product that the page misses
    assert a[10]["p50_rel"] == b[10]["p50_rel"] and after["m3"] == before["m3"]


def test_campaign_slot_share_is_reported_for_m1_and_m4(cache):
    out = b5.score(cache, QR, MERCH, QIDS)
    row = next(r for r in out["m1"]["rows"] if r["limit"] == 200)
    assert row["campaign_slots"] == pytest.approx(1 / 4)  # c among c, e1, e2, e3
    assert out["m1"]["base_campaign_slots"]["prod_merch"] == 0.0
    dense, hybrid = out["m4"]["configs"]["dense_text"], out["m4"]["configs"]["hybrid_rrf"]
    assert dense["on"]["campaign_slots"] == pytest.approx(0.5) and dense["on"]["reach"] == 1.0
    assert hybrid["on"]["campaign_slots"] == 0.0 and hybrid["on"]["reach"] == 0.0


def test_rescore_b5_refuses_a_cache_from_another_run(cache):
    payload = {"n": 1000, **b5.score(cache, QR, MERCH, QIDS)}
    with pytest.raises(ValueError, match="n=3"):
        rescore.rescore_b5(payload, cache, QR, MERCH, QIDS)
    with pytest.raises(ValueError, match="query_ids"):
        rescore.rescore_b5({**payload, "n": 3}, cache, QR, MERCH, ["q1", "q2", "other"])


def test_sample_runs_cache_elsewhere():
    assert b5.rankings_path(sample=False) == b5.RANKINGS
    assert b5.rankings_path(sample=True) != b5.RANKINGS
