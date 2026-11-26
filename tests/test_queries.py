import pandas as pd
import pytest

from data.queries import build_qrels, build_queries, sample_eval


@pytest.fixture
def examples():
    rows = []
    # q1: 3 covered exact (2 Books, 1 Toys), q2: 2 covered exact, q3: es locale
    for pid, lab in [("a", "E"), ("b", "E"), ("c", "E"), ("x", "E"), ("d", "S")]:
        rows.append({"query_id": 1, "query": "q one", "product_id": pid, "product_locale": "us", "esci_label": lab, "small_version": 1})
    for pid, lab in [("a", "E"), ("b", "E"), ("c", "I")]:
        rows.append({"query_id": 2, "query": "q two", "product_id": pid, "product_locale": "us", "esci_label": lab, "small_version": 0})
    rows.append({"query_id": 3, "query": "q es", "product_id": "a", "product_locale": "es", "esci_label": "E", "small_version": 0})
    return pd.DataFrame(rows)


@pytest.fixture
def products():
    return pd.DataFrame({"product_id": ["a", "b", "c", "d"],
                         "category": ["Books", "Books", "Toys and Games", "Books"]})


def test_qrels_filters_to_covered_us(examples, products):
    q = build_qrels(examples, set(products.product_id))
    assert "x" not in set(q.product_id)
    assert "3" not in set(q.query_id)
    assert set(q.columns) == {"query_id", "product_id", "label"}
    assert q.query_id.map(type).eq(str).all()


def test_eligible_requires_three_exact(examples, products):
    qr = build_qrels(examples, set(products.product_id))
    qs = build_queries(examples, qr, products).set_index("query_id")
    assert qs.loc["1", "n_exact"] == 3 and qs.loc["1", "eligible"]
    assert qs.loc["2", "n_exact"] == 2 and not qs.loc["2", "eligible"]


def test_top_category_is_mode_of_exact(examples, products):
    qr = build_qrels(examples, set(products.product_id))
    qs = build_queries(examples, qr, products).set_index("query_id")
    assert qs.loc["1", "top_category"] == "Books"


def test_eval_sample_deterministic():
    qs = pd.DataFrame({"query_id": [str(i) for i in range(12000)],
                       "eligible": [i % 2 == 0 for i in range(12000)]})
    a, b = sample_eval(qs), sample_eval(qs)
    assert list(a) == list(b) and len(a) == 5000
    assert qs.set_index("query_id").loc[list(a), "eligible"].all()


def test_hard_flag_from_small_version(examples, products):
    qr = build_qrels(examples, set(products.product_id))
    qs = build_queries(examples, qr, products).set_index("query_id")
    assert bool(qs.loc["1", "hard"]) and not bool(qs.loc["2", "hard"])
