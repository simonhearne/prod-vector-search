from types import SimpleNamespace

import numpy as np
import pandas as pd

from vsearch.collections import drifted_ids, price_drift, restore_prices

PRODS = pd.DataFrame({"product_id": ["a", "b", "c"], "price": [1.0, np.nan, 3.0]})


class FakeIter:
    def __init__(self, rows):
        self.batches = [rows, []]

    def next(self):
        return self.batches.pop(0)

    def close(self):
        pass


def test_restore_prices_batches_and_nan():
    calls = []
    c = SimpleNamespace(upsert=lambda name, data, partial_update=False: calls.append((name, data, partial_update)),
                        flush=lambda name: calls.append(("flush", name)))
    n = restore_prices(c, "t", PRODS, batch=2)
    assert n == 3 and [len(x[1]) for x in calls[:2]] == [2, 1] and all(x[2] for x in calls[:2])
    assert calls[0][1][1] == {"product_id": "b", "price": -1.0} and calls[-1] == ("flush", "t")


def test_restore_prices_subset():
    calls = []
    c = SimpleNamespace(upsert=lambda name, data, partial_update=False: calls.append(data), flush=lambda name: None)
    assert restore_prices(c, "t", PRODS, ids={"c"}) == 1 and calls == [[{"product_id": "c", "price": 3.0}]]


def test_price_drift_counts_mismatches_and_missing():
    rows = [{"product_id": "a", "price": 1.0}, {"product_id": "b", "price": -1.0}]  # c missing
    c = SimpleNamespace(query_iterator=lambda *a, **k: FakeIter(rows))
    assert price_drift(c, "t", PRODS) == 1
    rows[0]["price"] = 1.1
    assert price_drift(c, "t", PRODS) == 2


def test_drifted_ids():
    rows = [{"product_id": "a", "price": 1.5}, {"product_id": "b", "price": -1.0}]  # a changed, c missing
    c = SimpleNamespace(query_iterator=lambda *a, **k: FakeIter(rows))
    assert drifted_ids(c, "t", PRODS) == {"a", "c"}


def test_ensure_prices_restores_only_drifted_rows(monkeypatch):
    # a full restore rewrites every row (partial updates write whole rows) and can push Milvus into write throttling
    from bench import _common
    drift = iter([{"a", "c"}, set()])
    calls = []
    monkeypatch.setattr(_common, "drifted_ids", lambda c, n, p: next(drift))
    monkeypatch.setattr(_common, "restore_prices", lambda c, n, p, ids=None: calls.append(ids) or len(ids))
    assert _common.ensure_prices(None, "t", PRODS) == 2 and calls == [{"a", "c"}]


def test_ensure_prices_raises_if_restore_did_not_take(monkeypatch):
    import pytest
    from bench import _common
    monkeypatch.setattr(_common, "drifted_ids", lambda c, n, p: {"a", "b", "c", "d", "e"})
    monkeypatch.setattr(_common, "restore_prices", lambda c, n, p, ids=None: len(ids))
    with pytest.raises(RuntimeError, match="still 5"):
        _common.ensure_prices(None, "t", PRODS)
