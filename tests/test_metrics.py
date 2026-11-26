import numpy as np
import pandas as pd
import pytest

from vsearch.metrics import judged_at_k, ndcg_at_k, recall_at_k, summarise_quality


def test_ndcg_perfect_and_unjudged_zero():
    labels = {"a": "E", "b": "S", "c": "I"}
    assert ndcg_at_k(["a", "b", "c"], labels) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], labels) == 0.0


def test_ndcg_known_value():
    labels = {"a": "E", "b": "E"}
    assert ndcg_at_k(["x", "b"], labels) == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))


def test_ndcg_uses_esci_gains():
    labels = {"a": "E", "s": "S"}
    # ranked s then a: DCG = 0.1 + 1/log2(3); IDCG = 1 + 0.1/log2(3)
    assert ndcg_at_k(["s", "a"], labels) == pytest.approx((0.1 + 1 / np.log2(3)) / (1 + 0.1 / np.log2(3)))


def test_judged_at_k():
    assert judged_at_k(["a", "x", "b", "y"], {"a": "E", "b": "I"}, k=4) == 0.5


def test_recall_fewer_than_k_matches():
    assert recall_at_k(["a", "b", "z"], ["a", "b"], k=10) == 1.0


def test_recall_empty_truth_is_one():
    assert recall_at_k(["a"], [], k=10) == 1.0


def test_summarise_quality_keys_and_ci():
    qrels = pd.DataFrame({"query_id": ["1", "2"], "product_id": ["a", "b"], "label": ["E", "E"]})
    s = summarise_quality({"1": ["a"], "2": ["x"]}, qrels)
    assert s["n_queries"] == 2 and s["ndcg@10"] == pytest.approx(0.5)
    assert s["judged@10"] == pytest.approx(0.05)
    lo, hi = s["ndcg@10_ci95"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
