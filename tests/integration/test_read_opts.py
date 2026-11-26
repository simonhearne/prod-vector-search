"""Does the server honour B4a/B4c's read options? A misplaced ignore_growing is silently dropped, so
the variant would measure nothing. Scratch collection: 1,000 flushed rows plus one unflushed row."""
import numpy as np
import pytest
from pymilvus import DataType

from bench.b4a_write_storm import ReadOpts

NAME = "t_read_opts"


@pytest.fixture(scope="module")
def scratch(client):
    if client.has_collection(NAME):
        client.drop_collection(NAME)
    schema = client.create_schema(auto_id=False)
    schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=16)
    schema.add_field("text_vec", DataType.FLOAT_VECTOR, dim=4)
    idx = client.prepare_index_params()
    idx.add_index("text_vec", index_type="HNSW", metric_type="COSINE", params={"M": 8, "efConstruction": 32})
    client.create_collection(NAME, schema=schema, index_params=idx)
    rng = np.random.default_rng(27)
    base = rng.normal(size=(1000, 4)) * [1, 1, 1, 0.01] + [0, 0, 0, -1]  # all far from +w
    client.insert(NAME, [{"pk": f"r{i}", "text_vec": v.tolist()} for i, v in enumerate(base)])
    client.flush(NAME)
    client.load_collection(NAME)
    vec = [0.0, 0.0, 0.0, 1.0]
    client.insert(NAME, [{"pk": "new", "text_vec": vec}])  # growing: not flushed
    yield client, NAME, vec
    client.drop_collection(NAME)


def _hit(c, name, vec, o):
    return [h.id for h in c.search(name, data=[vec], anns_field="text_vec", limit=1, **o.search_kwargs(ef=16))[0]]


@pytest.mark.integration
def test_read_opts_ignore_growing_hides_unflushed_row(scratch):
    c, name, vec = scratch
    assert _hit(c, name, vec, ReadOpts("Strong")) == ["new"]
    assert _hit(c, name, vec, ReadOpts("Strong", ignore_growing=True)) != ["new"]


@pytest.mark.integration
def test_read_opts_query_honours_ignore_growing(scratch):  # B4c polls with client.query
    c, name, _ = scratch
    ids = lambda o: [r["pk"] for r in c.query(name, filter='pk == "new"', output_fields=["pk"], **o.query_kwargs())]
    assert ids(ReadOpts("Strong")) == ["new"] and ids(ReadOpts("Strong", ignore_growing=True)) == []


@pytest.mark.integration
def test_read_opts_eventually_and_bounded_accepted(scratch):
    c, name, vec = scratch
    for level in ("Eventually", "Bounded"):
        assert len(_hit(c, name, vec, ReadOpts(level))) == 1
