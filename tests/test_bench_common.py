import json

import numpy as np
import pandas as pd
import pytest

import bench._common as bc


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "DATA_WORK", tmp_path)
    monkeypatch.setattr(bc, "GT_CACHE", tmp_path / "gt")
    monkeypatch.setattr(bc, "CURATED", tmp_path / "cur.jsonl")
    monkeypatch.setattr(bc, "POC_SETS", tmp_path / "poc_sets.json")
    (tmp_path / "poc_sets.json").write_text(json.dumps({"poc50": ["10", "11"], "head50": ["11"]}))
    np.save(tmp_path / "q_ids.npy", np.array(["10", "11", "curated:10", "curated:11"]))
    np.save(tmp_path / "q_text_vec.npy", np.arange(8, dtype=np.float32).reshape(4, 2))
    np.save(tmp_path / "q_siglip_vec.npy", -np.arange(8, dtype=np.float32).reshape(4, 2))
    pd.DataFrame({"query_id": ["10", "11"], "query": ["a", "b"], "in_eval5k": [True, False]}).to_parquet(tmp_path / "queries.parquet")
    (tmp_path / "cur.jsonl").write_text(
        json.dumps({"query_id": "10", "curated_query": "A!", "approved": True}) + "\n"
        + json.dumps({"query_id": "11", "curated_query": "B!", "approved": False}) + "\n")
    return tmp_path


def test_load_eval_curated_uses_all_drafted_rows_aligned(work):
    q, tv, sv = bc.load_eval("curated")
    assert q.to_dict("records") == [{"key": "curated:10", "query_id": "10", "query": "A!"},
                                    {"key": "curated:11", "query_id": "11", "query": "B!"}]
    np.testing.assert_array_equal(tv, [[4, 5], [6, 7]]) and np.testing.assert_array_equal(sv, [[-4, -5], [-6, -7]])


def test_load_eval_eval5k(work):
    q, tv, _ = bc.load_eval("eval5k")
    assert q.query_id.tolist() == ["10"] and tv.tolist() == [[0, 1]]


def test_exact_ids_maps_rows_and_caches(work):
    docs = np.eye(3, dtype=np.float32)
    prods = pd.DataFrame({"row_id": [0, 1, 2], "product_id": ["a", "b", "c"]})
    out = bc.exact_ids("t", docs, np.array([1, 2]), docs[[2]], prods, k=1)
    assert out == [["c"]] and len(list((work / "gt").glob("t_*.json"))) == 1
    assert bc.exact_ids("t", docs, np.array([1, 2]), docs[[2]], prods, k=1) == [["c"]]


def test_load_eval_poc50_uses_source_esci_queries(work):
    q, tv, _ = bc.load_eval("poc50")
    assert q.to_dict("records") == [{"key": "10", "query_id": "10", "query": "a"},
                                    {"key": "11", "query_id": "11", "query": "b"}]
    np.testing.assert_array_equal(tv, [[0, 1], [2, 3]])


def test_load_eval_head50_from_poc_sets(work):
    q, tv, _ = bc.load_eval("head50")
    assert q.to_dict("records") == [{"key": "11", "query_id": "11", "query": "b"}]
    assert tv.tolist() == [[2, 3]]


def test_load_corpus_without_image_vectors_derives_has_image_from_manifest(work, monkeypatch):
    monkeypatch.setattr(bc, "DATA_DATASET", work / "absent")
    pd.DataFrame({"row_id": [0, 1, 2], "product_id": ["a", "b", "c"]}).to_parquet(work / "products.parquet")
    np.save(work / "text_vec.npy", np.zeros((3, 2), np.float32))
    pd.DataFrame({"product_id": ["a", "c"], "ok": [True, False]}).to_parquet(work / "images_manifest.parquet")
    products, tv, iv, has = bc.load_corpus()
    assert iv is None and has.tolist() == [True, False, False] and tv.shape == (3, 2)


class _CountClient:
    def __init__(self, count):
        self.count, self.loaded = count, []

    def has_collection(self, name):
        return True

    def query(self, name, filter, output_fields):
        return [{"count(*)": self.count}]

    def load_collection(self, name):
        self.loaded.append(name)

    def list_loaded_segments(self, name):
        from types import SimpleNamespace
        from pymilvus.client.types import SegmentState
        return [SimpleNamespace(num_rows=self.count, state=SegmentState.Sealed)]


@pytest.mark.parametrize("count,rebuilt", [(3, False), (2, True)])
def test_ensure_collection_rebuilds_partial_collection(monkeypatch, count, rebuilt):
    """A build killed mid-insert leaves a short collection; reusing it silently capped PoC recall (3 Nov)."""
    from vsearch.collections import CollectionSpec
    products = pd.DataFrame({"row_id": [0, 1, 2], "product_id": ["a", "b", "c"], "category": ["x"] * 3})
    monkeypatch.setattr(bc, "load_corpus", lambda: (products, None, None, None))
    built = []
    monkeypatch.setattr(bc, "build_collection", lambda c, spec, *a, **k: built.append(spec.name))
    c = _CountClient(count)
    assert bc.ensure_collection(c, CollectionSpec("t", rows="full")) == "t"
    assert built == (["t"] if rebuilt else []) and c.loaded == ["t"]


def test_with_query_meta_adds_columns_keeping_row_order(work):
    """make_filter_mix needs top_category, which load_eval does not return (b2 crash, 3 Nov)."""
    pd.DataFrame({"query_id": ["10", "11"], "query": ["a", "b"], "in_eval5k": [True, True],
                  "top_category": ["X", "Y"]}).to_parquet(work / "queries.parquet")
    q = pd.DataFrame({"key": ["11", "10"], "query_id": ["11", "10"], "query": ["b", "a"]})
    out = bc.with_query_meta(q, ["top_category"])
    assert out.query_id.tolist() == ["11", "10"] and out.top_category.tolist() == ["Y", "X"]


def test_quality_adds_filled(monkeypatch):
    import pandas as pd
    from bench import _common
    from vsearch import judge
    monkeypatch.setattr(judge, "cached_judgments", lambda: pd.DataFrame(
        {"query_id": ["q"], "product_id": ["b"], "label": ["E"], "model": [judge.JUDGE_MODEL]}))
    qr = pd.DataFrame({"query_id": ["q"], "product_id": ["a"], "label": ["E"]})
    out = _common.quality({"q": ["b", "a"]}, qr)
    assert out["judged@10"] == 0.1 and out["filled"]["judged@10"] == 0.2
    assert out["filled"]["ndcg@10"] > out["ndcg@10"]
    assert _common.CANONICAL_N == 1000
