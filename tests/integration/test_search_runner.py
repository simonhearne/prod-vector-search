import pytest

from vsearch.search import run_hybrid, run_queries


@pytest.mark.integration
def test_run_queries_shapes(client, small):
    name, (prods, text_vec, _, _) = small
    r = run_queries(client, name, "text_vec", text_vec[:5], search_params={"ef": 32}, warmup=2)
    assert len(r.ids) == 5 and all(0 < len(x) <= 10 for x in r.ids) and len(r.latencies_ms) == 5
    assert r.ids[0][0] == prods.product_id[0]


@pytest.mark.integration
def test_run_queries_filtered(client, small):
    name, (prods, text_vec, _, _) = small
    books = set(prods[prods.category == "Books"].product_id)
    r = run_queries(client, name, "text_vec", text_vec[:5], filters=['category == "Books"'] * 5, warmup=0)
    assert all(pid in books for ids in r.ids for pid in ids)


@pytest.mark.integration
def test_run_hybrid_nonempty(client, small):
    name, (_, text_vec, _, _) = small
    r = run_hybrid(client, name, text_vec[:3], ["usb cable", "garden hose", "charger"], warmup=1)
    assert len(r.ids) == 3 and all(len(x) > 0 for x in r.ids)
