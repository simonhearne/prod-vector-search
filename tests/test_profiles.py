import numpy as np
import pandas as pd
import pytest

from vsearch.profiles import (blend, brand_filter, brand_variants, brand_weights, fold, moved, price_filter, rerank, rr,
                              user_vec)


def test_brand_filter_matches_every_spelling_and_escapes():
    v = brand_variants(pd.Series(["Apple", "apple ", 'Bob"s', "Unknown", None, "Apple"]))
    assert brand_filter({"apple": 1.0}, v) == 'brand in ["Apple","apple "]'
    assert '\\"' in brand_filter({'bob"s': 1.0}, v)
    assert brand_filter({}, v) is None
    assert "" not in v  # junk and missing brands have no variants to match


@pytest.mark.parametrize("junk", [None, "", "  ", "Unknown", "GENERIC", "N/A", "none", float("nan")])
def test_junk_brands_fold_to_empty(junk):
    assert fold(junk) == ""


def test_junk_brands_carry_no_affinity():
    assert brand_weights(["a", "b"], {"a": "Unknown", "b": ""}) == {}


def test_brand_weights_are_shares_of_the_top_five_by_count_then_name():
    brand_of = {f"p{i}": b for i, b in enumerate(["Sony", "sony", "f", "e", "d", "c", "b"])}
    w = brand_weights(list(brand_of), brand_of)
    assert list(w) == ["sony", "b", "c", "d", "e"]  # f is sixth
    assert w["sony"] == pytest.approx(2 / 7) and w["b"] == pytest.approx(1 / 7)


def test_price_filter_needs_two_priced_items_and_excludes_the_sentinel():
    assert price_filter(["a"], {"a": 10.0}) is None
    assert price_filter(["a", "b"], {"a": 10.0, "b": float("nan")}) is None
    assert price_filter(["a", "b", "c"], {"a": 10.0, "b": 20.0, "c": -1.0}) == "price >= 7.50 and price <= 30.00"


def test_user_vec_is_the_normalised_mean():
    tv = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    u = user_vec([0, 1], tv)
    np.testing.assert_allclose(u, [2 ** -0.5] * 2, rtol=1e-6)
    assert u.dtype == np.float32


def test_blend_zero_is_the_query_and_result_is_unit():
    q, u = np.array([1, 0], np.float32), np.array([0, 1], np.float32)
    np.testing.assert_allclose(blend(q, u, 0.0), q, atol=1e-6)
    np.testing.assert_allclose(blend(q, u, 1.0), [2 ** -0.5] * 2, atol=1e-6)


def test_rerank_zero_weight_keeps_order_and_bonus_moves_items():
    ids, scores, bonus = ["a", "b", "c"], [0.9, 0.8, 0.8], np.array([0.0, 0.0, 1.0])
    assert rerank(ids, scores, bonus, 0.0) == ["a", "b", "c"]
    assert rerank(ids, scores, bonus, 0.2) == ["c", "a", "b"]


def test_rr_and_moved():
    assert rr(None) == 0 and rr(4) == 0.25
    a, b = [f"a{i}" for i in range(12)], [f"b{i}" for i in range(12)]
    assert moved(a, a) == 0 and moved(a, b) == 1
    assert moved(a, a[:5] + b[:5]) == 0.5
    assert moved(a, list(reversed(a[:10]))) == 0  # same page, other order
