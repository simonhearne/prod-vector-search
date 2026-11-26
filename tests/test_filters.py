import numpy as np
import pandas as pd
import pytest

from vsearch.filters import BUCKETS, Filter, bucket_of, escape, make_filter_mix


def test_to_expr_full():
    f = Filter(category="Books", price_min=5, price_max=20, rating_min=4.0)
    assert f.to_expr() == 'category == "Books" and price >= 0 and price >= 5 and price <= 20 and average_rating >= 4.0'


def test_escape_and_mask_agree():
    prods = pd.DataFrame({"category": ['Kid"s', "Books"], "brand": ["O'Neill", "X"],
                          "price": [1.0, 2.0], "average_rating": [4.5, 3.0]})
    f = Filter(category='Kid"s', brand="O'Neill")
    assert f.to_expr() == 'category == "Kid\\"s" and brand == "O\'Neill"'
    assert f.mask(prods).tolist() == [True, False]


def test_escape_backslash():
    assert escape('a\\b"c') == 'a\\\\b\\"c'


def test_empty_filter_is_none():
    assert Filter().to_expr() is None and Filter().mask(pd.DataFrame({"category": ["a"]})).all()


def test_mask_nan_price_and_rating_never_match():
    prods = pd.DataFrame({"category": ["a", "a"], "brand": ["b", "b"],
                          "price": [np.nan, 3.0], "average_rating": [np.nan, 4.5]})
    assert Filter(price_min=1).mask(prods).tolist() == [False, True]
    assert Filter(rating_min=4.0).mask(prods).tolist() == [False, True]


def test_bucket_of():
    assert bucket_of(1.0, filtered=False) == "unfiltered"
    assert bucket_of(0.2, filtered=True) == "10-50%"
    assert bucket_of(0.0005, filtered=True) == "<0.1%"
    assert bucket_of(0.7, filtered=True) is None


@pytest.fixture
def tiny():
    rng = np.random.default_rng(27)
    n = 20000
    cats = np.where(np.arange(n) < 6000, "Books", np.where(np.arange(n) < 6300, "Tiny", "Other"))
    prods = pd.DataFrame({"product_id": [f"p{i}" for i in range(n)], "category": cats,
                          "brand": rng.choice(["A", "B", "C", "D"], n),
                          "price": rng.uniform(1, 100, n), "average_rating": rng.uniform(1, 5, n)})
    queries = pd.DataFrame({"query_id": [str(i) for i in range(60)],
                            "top_category": ["Books"] * 30 + ["Tiny"] * 30})
    exact = [(str(i), f"p{(i * 7) % 6000 if i < 30 else 6000 + i}") for i in range(60) for _ in range(3)]
    qrels = pd.DataFrame(exact, columns=["query_id", "product_id"]).assign(label="E")
    return queries, prods, qrels


def test_mix_min_matches(tiny):
    mix = make_filter_mix(*tiny, per_bucket=50)
    assert len(mix) and (mix.n_matches >= 10).all()
    assert set(mix.bucket) <= {b[0] for b in BUCKETS}
    row = mix.iloc[0]
    assert row.expr == (row["filter"].to_expr() or "")
    assert np.isclose(row.selectivity, row.n_matches / 20000)


def test_mix_deterministic_and_capped(tiny):
    a, b = make_filter_mix(*tiny, per_bucket=5), make_filter_mix(*tiny, per_bucket=5)
    assert a[["query_id", "bucket", "expr"]].equals(b[["query_id", "bucket", "expr"]])
    assert a.groupby("bucket").size().max() <= 5
