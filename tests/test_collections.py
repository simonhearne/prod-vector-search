import numpy as np
import pandas as pd
import pytest
from pymilvus import DataType

from vsearch.collections import SPECS, CollectionSpec, IndexSpec, build_index_params, build_schema, frontier_specs, poc_slice


def _fields(schema):
    return {f.name: f for f in schema.fields}


def test_poc_slice_single_category_size_deterministic():
    prods = pd.DataFrame({"row_id": range(100), "product_id": [f"p{i:03d}" for i in range(100)],
                          "category": ["Electronics" if i % 2 else "Books" for i in range(100)]})
    a, b = poc_slice(prods, n=20), poc_slice(prods, n=20)
    assert a.product_id.tolist() == b.product_id.tolist() and len(a) == 20
    assert set(a.category) == {"Electronics"}


def test_frontier_specs_grid():
    specs = frontier_specs()
    assert len(specs) == 12 and len({s.name for s in specs}) == 12
    assert all(s.rows == "full" and not s.image and not s.bm25 for s in specs)
    assert {s.text_dim for s in specs} == {1024, 512, 256}


@pytest.mark.parametrize("name", ["poc_20k", "prod_naive", "prod_tuned"])
def test_schema_fields(name):
    spec = SPECS[name]
    f = _fields(build_schema(spec))
    assert f["product_id"].is_primary and f["text_vec"].dtype == DataType.FLOAT_VECTOR
    assert f["image_vec"].dtype == DataType.FLOAT_VECTOR and f["has_image"].dtype == DataType.BOOL
    assert f["text_sparse"].dtype == DataType.SPARSE_FLOAT_VECTOR
    assert f["price"].dtype == DataType.FLOAT and f["rating_number"].dtype == DataType.INT64
    assert bool(f["category"].is_partition_key) == (name == "prod_tuned")


def test_schema_fp16_and_no_image():
    spec = next(s for s in frontier_specs() if s.text_dtype == "fp16" and s.text_dim == 512)
    f = _fields(build_schema(spec))
    assert f["text_vec"].dtype == DataType.FLOAT16_VECTOR and f["text_vec"].params["dim"] == 512
    assert "image_vec" not in f and "text_sparse" not in f


def test_index_params_scalar_only_when_tuned():
    def idx(spec):
        return {i.field_name: i.index_type for i in build_index_params(spec)}
    naive, tuned = idx(SPECS["prod_naive"]), idx(SPECS["prod_tuned"])
    assert "category" not in naive and naive["text_vec"] == "HNSW"
    assert tuned["category"] == "INVERTED" and tuned["price"] == "STL_SORT"
    assert tuned["text_sparse"] == "SPARSE_INVERTED_INDEX"


def test_clip_counts_utf8_bytes():
    from vsearch.collections import _clip
    assert _clip("é" * 10, 5) == "éé" and _clip(float("nan"), 5) == "" and _clip(None, 5) == ""


class _IndexClient:
    """describe_index reports progress: pending on the first poll, finished on the second."""
    def __init__(self, polls_until_done=1):
        self.calls, self.left = 0, polls_until_done

    def list_indexes(self, name):
        return ["text_vec", "text_sparse"]

    def describe_index(self, name, index_name):
        self.calls += 1
        done = self.calls > self.left * 2
        return {"state": "Finished" if done else "InProgress", "total_rows": 5,
                "indexed_rows": 5 if done else 2, "pending_index_rows": 0 if done else 3}


def test_wait_index_blocks_until_every_index_covers_all_rows():
    """Searching before the index lands gave flat, wrong recall in tune_poc (3 Nov)."""
    from vsearch.collections import wait_index
    c = _IndexClient(polls_until_done=1)
    wait_index(c, "t", 5, poll_s=0)
    assert c.calls == 4  # one unfinished round over both indexes, then one finished round


def test_wait_index_times_out_loudly():
    from vsearch.collections import wait_index
    with pytest.raises(TimeoutError):
        wait_index(_IndexClient(polls_until_done=10**6), "t", 5, timeout_s=0.01, poll_s=0)


class _Seg:
    """Mirrors pymilvus LoadedSegmentInfo: state is a SegmentState enum, not a str."""
    def __init__(self, num_rows, state):
        from pymilvus.client.types import SegmentState
        self.num_rows, self.state = num_rows, SegmentState[state]


class _LoadClient:
    """Query node first serves growing data, then sealed segments for part, then all rows."""
    def __init__(self):
        self.views = [[], [_Seg(3, "Sealed"), _Seg(2, "Growing")], [_Seg(3, "Sealed"), _Seg(2, "Sealed")]]
        self.calls = 0

    def list_loaded_segments(self, name):
        v = self.views[min(self.calls, len(self.views) - 1)]
        self.calls += 1
        return v


def test_wait_loaded_waits_for_sealed_segments_covering_all_rows():
    """describe_index said Finished at once, yet search ignored ef for ~20 s after load (3 Nov)."""
    from vsearch.collections import wait_loaded
    c = _LoadClient()
    wait_loaded(c, "t", 5, timeout_s=1, poll_s=0)  # bounded so a never-matching state fails, not hangs
    assert c.calls == 3


def test_wait_loaded_times_out_loudly():
    from vsearch.collections import wait_loaded
    c = _LoadClient()
    c.views = [[]]
    with pytest.raises(TimeoutError):
        wait_loaded(c, "t", 5, timeout_s=0.01, poll_s=0)


@pytest.mark.parametrize("name,pk", [("prod_merch", False), ("prod_merch_pk", True)])
def test_merch_specs_add_rule_fields_to_a_text_only_copy(name, pk):
    f = _fields(build_schema(SPECS[name]))
    assert f["margin"].dtype == DataType.FLOAT and f["hidden"].dtype == DataType.BOOL
    assert f["campaigns"].dtype == DataType.ARRAY and f["campaigns"].element_type == DataType.VARCHAR
    assert "image_vec" not in f and "text_sparse" in f
    assert SPECS[name].rows == "full" and bool(f["category"].is_partition_key) == pk


def test_merch_fields_absent_by_default():
    assert not {"margin", "campaigns", "hidden"} & set(_fields(build_schema(SPECS["prod_naive"])))


def test_rows_carry_merch_values_aligned_by_product():
    from vsearch.collections import _rows
    from vsearch.merch import merch_fields
    prods = pd.DataFrame({"product_id": ["b", "a"], "category": ["x", "y"], "brand": ["B", "A"], "title": ["t", "t"],
                          "image_url": ["", ""], "price": [1.0, np.nan], "average_rating": [4.0, np.nan],
                          "rating_number": [1, 2], "row_id": [0, 1], "embed_text": ["e", "e"]})
    m = merch_fields(prods)
    out = list(_rows(SPECS["prod_merch"], prods, np.zeros((2, 4)), None, [False, False], m))
    assert [r["margin"] for r in out] == pytest.approx(m.margin.tolist())
    assert [r["campaigns"] for r in out] == m.campaigns.tolist()
    assert [r["hidden"] for r in out] == m.hidden.tolist()


def test_prod_perso_has_a_brand_sparse_field_and_nothing_synthetic():
    f = _fields(build_schema(SPECS["prod_perso"]))
    assert f["brand_sparse"].dtype == DataType.SPARSE_FLOAT_VECTOR
    assert not {"image_vec", "text_sparse", "margin", "campaigns", "hidden"} & set(f)
    idx = {i.field_name: (i.index_type, i.get_index_configs().get("metric_type")) for i in build_index_params(SPECS["prod_perso"])}
    assert idx["brand_sparse"] == ("SPARSE_INVERTED_INDEX", "IP")


def test_other_specs_have_no_brand_sparse_field():
    for s in [*(v for k, v in SPECS.items() if k != "prod_perso"), *frontier_specs()]:
        assert "brand_sparse" not in _fields(build_schema(s))


def test_brand_ids_fold_spellings_reserve_zero_and_ignore_row_order():
    from vsearch.collections import brand_ids
    p = pd.DataFrame({"brand": ["Sony", "sony ", "Unknown", "", None, "Apple"]})
    assert brand_ids(p) == {"apple": 1, "sony": 2}
    assert brand_ids(p.iloc[::-1]) == brand_ids(p)


def test_rows_give_junk_brands_the_reserved_dimension():
    from vsearch.collections import _rows
    p = pd.DataFrame({"product_id": ["a", "b"], "category": "c", "brand": ["Sony", "Unknown"], "title": "t", "image_url": "",
                      "price": 1.0, "average_rating": 1.0, "rating_number": 1, "row_id": [0, 1]})
    rows = list(_rows(SPECS["prod_perso"], p, np.zeros((2, 4)), None, [False, False], brands={"sony": 7}))
    assert [r["brand_sparse"] for r in rows] == [{7: 1.0}, {0: 1.0}]
