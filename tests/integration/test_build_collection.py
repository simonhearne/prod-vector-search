import pytest


@pytest.mark.integration
def test_build_small(client, small):
    name, (prods, text_vec, image_vec, has) = small
    n = client.query(name, filter="", output_fields=["count(*)"])[0]["count(*)"]
    assert n == len(prods)
    res = client.search(name, data=[image_vec[0].tolist()], anns_field="image_vec",
                        filter="has_image == true", limit=20, output_fields=["has_image"])
    assert res[0] and all(h["entity"]["has_image"] for h in res[0])
    idx = client.describe_index(name, index_name="category")
    assert idx["index_type"] == "INVERTED"
