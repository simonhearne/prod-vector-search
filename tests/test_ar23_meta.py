import json

import pytest

from data.ar23_meta import filter_meta_lines, meta_url, parse_price


@pytest.mark.parametrize("raw,expected", [
    ("$12.99", 12.99), ("12.99", 12.99), (12.99, 12.99), ("None", None),
    ("", None), (None, None), ("$5.99 - $9.99", 5.99), ("—", None)])
def test_parse_price(raw, expected):
    assert parse_price(raw) == expected


def test_meta_url_spaces():
    assert meta_url("Clothing Shoes and Jewelry").endswith("meta_Clothing_Shoes_and_Jewelry.jsonl")


def test_filter_meta_lines_keeps_only_ids_and_picks_image():
    lines = [json.dumps({"parent_asin": "A", "price": "$3", "average_rating": 4.5,
                         "rating_number": 10, "store": "S", "main_category": "M",
                         "images": [{"hi_res": None, "large": "L.jpg"}]}).encode(),
             json.dumps({"parent_asin": "B"}).encode()]
    out = list(filter_meta_lines(lines, {"A"}))
    assert out == [{"parent_asin": "A", "main_category": "M", "price": 3.0,
                    "average_rating": 4.5, "rating_number": 10, "store": "S",
                    "image_url": "L.jpg"}]


def test_filter_meta_lines_prefers_hi_res_and_tolerates_missing_images():
    lines = [json.dumps({"parent_asin": "A", "images": [{"hi_res": "H.jpg", "large": "L.jpg"}]}).encode(),
             json.dumps({"parent_asin": "C", "images": []}).encode()]
    out = {r["parent_asin"]: r["image_url"] for r in filter_meta_lines(lines, {"A", "C"})}
    assert out == {"A": "H.jpg", "C": None}
