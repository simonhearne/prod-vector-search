from pathlib import Path

import pandas as pd
import pytest

from bench import showcase as sc
from vsearch.config import ROOT


class FakeClient:
    def __init__(self, exists=True, count=10):
        self.exists, self.count, self.calls = exists, count, []

    def list_collections(self):
        return []

    def has_collection(self, name):
        return self.exists

    def load_collection(self, name):
        self.calls.append("load")

    def query(self, name, filter, output_fields):
        return [{"count(*)": self.count}]

    def __getattr__(self, attr):  # create_collection, insert, ... must never be reached
        raise AssertionError(f"unexpected call {attr}")


def test_open_existing_refuses_missing_and_short(monkeypatch):
    monkeypatch.setattr(sc, "wait_loaded", lambda c, n, rows: None)
    with pytest.raises(sc.NotReady, match="missing"):
        sc.open_existing(FakeClient(exists=False), "poc_20k", 10)
    with pytest.raises(sc.NotReady, match="9 rows"):
        sc.open_existing(FakeClient(count=9), "poc_20k", 10)
    assert sc.open_existing(FakeClient(), "poc_20k", 10) == "poc_20k"


def _info():
    return pd.DataFrame({"product_id": ["a", "b"], "title": ["A", "B"], "brand": ["X", float("nan")],
                         "category": ["Electronics"] * 2, "price": [9.991, -1.0]}).set_index("product_id")


def test_label_maps_marks_source_esci_wins():
    qr = pd.DataFrame({"query_id": ["1", "1"], "product_id": ["a", "b"], "label": ["E", "S"]})
    filled = pd.concat([qr, pd.DataFrame({"query_id": ["1", "2"], "product_id": ["c", "a"], "label": ["I", "C"]})])
    m = sc.label_maps(qr, filled, ["1"])
    assert m == {"1": {"a": ("E", "esci"), "b": ("S", "esci"), "c": ("I", "judge")}}


def test_cards_price_brand_label_and_image(tmp_path):
    img = ROOT / "data" / "work" / "images"
    paths = {"a": img / "a.jpg", "b": tmp_path / "missing.jpg"}
    real_exists = Path.exists
    c = sc.cards(["a", "b"], _info(), {"a": ("E", "judge")},
                 img=lambda p: paths[p])
    assert c[0]["price"] == 9.99 and c[1]["price"] is None
    assert c[0]["brand"] == "X" and c[1]["brand"] == ""
    assert (c[0]["label"], c[0]["label_source"]) == ("E", "judge")
    assert (c[1]["label"], c[1]["label_source"]) == (None, None)
    assert c[1]["image"] is None
    assert c[0]["image"] == (str(paths["a"].relative_to(ROOT)) if real_exists(paths["a"]) else None)


def test_side_short_page_and_judged():
    s = sc.side(["a"], {"a": ("E", "esci")}, _info())
    assert s["returned"] == 1 and s["judged@10"] == pytest.approx(0.1) and s["ndcg@10"] == pytest.approx(1.0)


def _row(qid, poc, prod, jp=1.0, jq=1.0):
    return {"query_id": qid, "poc": {"ndcg@10": poc, "judged@10": jp}, "prod": {"ndcg@10": prod, "judged@10": jq}}


def test_rank_poc_vs_prod_needs_judged_and_orders_by_gap():
    rows = [_row("small", 0.5, 0.55), _row("big", 0.9, 0.2), _row("mid", 0.2, 0.5), _row("holes", 0.0, 0.9, jp=0.3)]
    assert sc.rank_poc_vs_prod(rows) == ["big", "mid", "small"]


def test_rank_narrow_short_rabitq_full_hnsw():
    r = lambda key, h, q, rec=1.0: {"key": key, "hnsw": {"returned": h, "recall@10": rec}, "rabitq": {"returned": q}}
    rows = [r("both_full", 10, 10), r("six", 10, 6), r("two", 10, 2), r("hnsw_short", 8, 3)]
    assert sc.rank_narrow(rows) == ["two", "six"]


def test_latency_split_times_each_stage():
    import time

    class C:
        def search(self, *a, **k):
            time.sleep(0.002)

    def enc(texts):
        time.sleep(0.004)
        return [[0.0] * 4]

    s = sc.latency_split(C(), "prod_naive", ["q1", "q2", "q3"], 192, enc)
    assert s["n"] == 3 and len(s["per_query_ms"]) == 3
    assert s["embed"]["p50"] >= 4 and s["search"]["p50"] >= 2
    assert s["total"]["p50"] >= s["embed"]["p50"] + s["search"]["p50"] - 0.5
    assert 0 < s["embed_share_p50"] < 1
