import pandas as pd
import pytest

from vsearch import judge
from vsearch.judge import JUDGE_MODEL


def _cache(rows):
    return lambda: pd.DataFrame(rows, columns=["query_id", "product_id", "label", "model"])


def test_filled_qrels_esci_wins(monkeypatch):
    qr = pd.DataFrame({"query_id": ["q"], "product_id": ["p"], "label": ["E"]})
    monkeypatch.setattr(judge, "cached_judgments", _cache([("q", "p", "I", JUDGE_MODEL), ("q", "x", "S", JUDGE_MODEL)]))
    out = judge.filled_qrels(qr).set_index("product_id").label.to_dict()
    assert out == {"p": "E", "x": "S"}


def test_filled_qrels_ignores_other_models(monkeypatch):
    qr = pd.DataFrame(columns=["query_id", "product_id", "label"])
    monkeypatch.setattr(judge, "cached_judgments", _cache([("q", "x", "S", "other-model")]))
    assert judge.filled_qrels(qr).empty


def test_unjudged_pairs_dedup_and_empty():
    qr = pd.DataFrame({"query_id": ["q"], "product_id": ["a"], "label": ["E"]})
    r1, r2 = {"q": ["a", "b"], "r": []}, {"q": ["b", "c"]}
    assert judge.unjudged_pairs([r1, r2], qr).values.tolist() == [["q", "b"], ["q", "c"]]


def test_fill_guard_refuses_before_submit(monkeypatch):
    monkeypatch.setattr(judge, "cached_judgments", _cache([]))
    monkeypatch.setattr(judge, "submit", lambda *a, **k: pytest.fail("submitted"))
    qr = pd.DataFrame(columns=["query_id", "product_id", "label"])
    with pytest.raises(ValueError, match="max_pairs=1"):
        judge.fill(None, [{"q": ["a", "b"]}], qr, pd.DataFrame({"query_id": ["q"], "query": ["x"]}), None, max_pairs=1)


def test_fill_skips_cached(monkeypatch):
    qr = pd.DataFrame(columns=["query_id", "product_id", "label"])
    monkeypatch.setattr(judge, "cached_judgments", _cache([("q", "a", "E", JUDGE_MODEL)]))
    monkeypatch.setattr(judge, "submit", lambda *a, **k: pytest.fail("submitted"))
    out = judge.fill(None, [{"q": ["a"]}], qr, pd.DataFrame({"query_id": ["q"], "query": ["x"]}), None)
    assert out["pairs"] == 1 and out["submitted"] == 0 and out["batch"] is None


def test_fingerprint_changes_with_the_pool(monkeypatch):
    rows = [("q", "a", "E", JUDGE_MODEL)]
    monkeypatch.setattr(judge, "cached_judgments", lambda: _cache(rows)())
    f1 = judge.fingerprint()
    rows.append(("q", "b", "I", JUDGE_MODEL))
    f2 = judge.fingerprint()
    assert f1["rows"] == 1 and f2["rows"] == 2 and f1["sha"] != f2["sha"] and len(f1["sha"]) == 12
