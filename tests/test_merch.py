"""Synthetic merchandising fields (B5): seeded, independent of row order, never matching an empty brand."""
import numpy as np
import pandas as pd
import pytest

from vsearch.merch import CAMPAIGN, merch_fields


@pytest.fixture
def prods():
    rng = np.random.default_rng(0)
    brand = np.array([f"brand{i:03d}" for i in rng.integers(0, 200, 5000)], dtype=object)
    brand[rng.random(5000) < 0.05] = ""
    return pd.DataFrame({"product_id": [f"p{i:05d}" for i in range(5000)], "brand": brand,
                         "category": [f"cat{i}" for i in rng.integers(0, 5, 5000)]})


def test_deterministic_and_row_order_independent(prods):
    a = merch_fields(prods)
    b = merch_fields(prods.sample(frac=1, random_state=1)).reindex(prods.index)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["margin", "campaigns", "hidden"]


def test_margin_range_and_category_structure(prods):
    m = merch_fields(prods).margin
    assert m.between(0.02, 0.70).all()
    assert m.groupby(prods.category).mean().std() > 0.02


def test_campaign_is_whole_brands_and_never_the_empty_brand(prods):
    camp = merch_fields(prods).campaigns
    on = camp.map(bool)
    assert 0.03 <= on.mean() <= 0.07
    assert not on[prods.brand == ""].any()
    assert on[prods.brand != ""].groupby(prods.brand).nunique().eq(1).all()
    assert camp[on].map(lambda c: c == [CAMPAIGN]).all()


def test_missing_brand_is_treated_as_empty(prods):
    prods.loc[prods.brand == "", "brand"] = None
    assert not merch_fields(prods).campaigns.map(bool)[prods.brand.isna()].any()


def test_a_dominant_brand_is_skipped_not_overshot(prods):
    prods.loc[prods.index[:2000], "brand"] = "giant"
    on = merch_fields(prods).campaigns.map(bool)
    assert not on[prods.brand == "giant"].any() and on.mean() <= 0.07


def test_hidden_share(prods):
    assert 0.04 <= merch_fields(prods).hidden.mean() <= 0.06
