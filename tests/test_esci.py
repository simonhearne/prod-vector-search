import pandas as pd

from data.esci import covered_ids


def test_covered_ids_us_only_and_joined():
    prods = pd.DataFrame({"product_id": ["A", "B", "C", "A"],
                          "product_locale": ["us", "us", "us", "es"]})
    out = covered_ids(prods, {"A": "Books", "C": "Toys and Games", "Z": "Books"})
    assert sorted(out.product_id) == ["A", "C"]
    assert dict(zip(out.product_id, out.category)) == {"A": "Books", "C": "Toys and Games"}
