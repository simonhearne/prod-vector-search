import pytest

from bench.b2_filters import short_rate


def test_short_rate():
    assert short_rate([list("abcdefghij"), list("abc"), []]) == pytest.approx(2 / 3)


def test_short_rate_full_pages():
    assert short_rate([list("abcdefghij")] * 3) == 0.0


def test_open_accumulates_price_drift(monkeypatch):
    from bench import b2_filters as b2
    drifts = iter([152_588, 0])
    monkeypatch.setattr(b2, "release_all", lambda c: None)
    monkeypatch.setattr(b2, "ensure_collection", lambda c, s: s)
    monkeypatch.setattr(b2, "level_start", lambda c, n, rows: (0.0, True))
    monkeypatch.setattr(b2, "ensure_prices", lambda c, n, p: next(drifts))
    monkeypatch.setattr(b2, "DRIFT", {})
    b2._open(None, "prod_tuned", 10, products=object())
    b2._open(None, "prod_tuned", 10, products=object())
    assert b2.DRIFT == {"prod_tuned": 152_588}


def test_unfiltered_tax_order_is_configurable():
    import inspect
    from bench import b2_filters as b2
    assert inspect.signature(b2.unfiltered_tax).parameters["order"].default == ("naive", "tuned")
