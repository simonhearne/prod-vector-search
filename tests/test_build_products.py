import numpy as np
import pandas as pd
import pytest

from data.build_products import build_products


@pytest.fixture
def inputs():
    esci = pd.DataFrame({
        "product_id": ["B2", "A1", "C3", "D4", "A1"],
        "product_title": ["Title", "Alpha", "", "Delta", "Alpha ES"],
        "product_brand": ["B", "BrandA", "X", None, "BrandA"],
        "product_color": ["red", None, None, None, None],
        "product_bullet_point": ["bullet1", "x" * 5000, "y", None, None],
        "product_description": ["desc", None, None, None, None],
        "product_locale": ["us", "us", "us", "us", "es"]})
    covered = pd.DataFrame({"product_id": ["A1", "B2", "C3", "D4"],
                            "category": ["Books", "Electronics", "Books", "Toys and Games"]})
    meta = pd.DataFrame({"parent_asin": ["A1", "B2", "C3"], "main_category": ["m", "m", "m"],
                         "price": [5.0, 9.5, 1.0], "average_rating": [4.0, 3.5, 5.0],
                         "rating_number": [10, 2, 1], "store": ["s", "s", "s"],
                         "image_url": ["a.jpg", None, "c.jpg"]})
    return esci, covered, meta


def test_embed_text_format(inputs):
    df = build_products(*inputs).set_index("product_id")
    assert df.loc["B2", "embed_text"] == "Title\nBrand: B\nbullet1"


def test_embed_text_truncated(inputs):
    df = build_products(*inputs)
    assert df.embed_text.str.len().max() <= 2000


def test_drops_rows_without_title(inputs):
    assert "C3" not in set(build_products(*inputs).product_id)


def test_row_id_sorted_dense(inputs):
    df = build_products(*inputs)
    assert df.row_id.tolist() == list(range(len(df)))
    assert df.product_id.is_monotonic_increasing


def test_missing_meta_keeps_product_with_null_price(inputs):
    df = build_products(*inputs).set_index("product_id")
    assert "D4" in df.index and np.isnan(df.loc["D4", "price"])
    assert df.loc["D4", "category"] == "Toys and Games"


def test_us_rows_only(inputs):
    df = build_products(*inputs).set_index("product_id")
    assert df.loc["A1", "title"] == "Alpha"
