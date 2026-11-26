import pytest

from bench.b3a_frontier import parse_mem
from bench.b3b_cost_table import cost_table
from bench.b3c_reembed import reembed_bill


def test_cost_table_scales():
    rows = cost_table([{"name": "a", "bytes_per_vector": 1000}], n_products=(1000,), vectors=(2,), replicas=(3,))
    assert rows == [{"name": "a", "n_products": 1000, "vectors": 2, "replicas": 3, "gb": 6000 * 1000 / 1e9}]


def test_cost_table_grid_size():
    rows = cost_table([{"name": "a", "bytes_per_vector": 1}, {"name": "b", "bytes_per_vector": 2}])
    assert len(rows) == 2 * 3 * 2 * 3


def test_reembed_bill():
    assert reembed_bill(100.0, 360_000, 2.0) == {"hours": 1.0, "usd": 2.0}


@pytest.mark.parametrize("s,expected", [("1.5GiB / 12GiB", 1.5 * 2**30), ("512MiB / 12GiB", 512 * 2**20),
                                         ("900KiB / 1GiB", 900 * 2**10), ("2.0GB / 12GB", 2e9)])
def test_parse_mem(s, expected):
    assert parse_mem(s) == pytest.approx(expected)


def test_loaded_bytes_sums_cache_series():
    from bench.b3a_frontier import loaded_bytes
    text = ('# HELP internal_cache_loaded_bytes x\n'
            'internal_cache_loaded_bytes{cell_type="vector"} 700\n'
            'internal_cache_loaded_bytes{cell_type="scalar"} 57\n'
            'internal_cache_loaded_count 3\n')
    assert loaded_bytes(text)["total"] == 757


def test_loaded_bytes_splits_ram_from_mmap_by_location():
    """3.0.3 labels each series with location; "disk" is mmap'd (raw vector field), not RAM.
    Summing it in made HNSW_SQ8+refine look as big as fp32."""
    from bench.b3a_frontier import loaded_bytes
    text = ('internal_cache_loaded_bytes{data_type="vector_index",location="memory"} 1000\n'
            'internal_cache_loaded_bytes{data_type="scalar_field",location="memory"} 100\n'
            'internal_cache_loaded_bytes{data_type="vector_field",location="disk"} 4000\n'
            'internal_cache_loaded_bytes{data_type="vector_field",location="mixed"} 0\n')
    assert loaded_bytes(text) == {"memory": 1100, "disk": 4000, "mixed": 0, "total": 5100}
