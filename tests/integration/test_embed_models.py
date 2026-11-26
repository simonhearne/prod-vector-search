import numpy as np
import pytest
from PIL import Image

from vsearch.embed import ImageEmbedder, TextEmbedder


@pytest.mark.slow
def test_text_embedder_shapes_and_ranking():
    te = TextEmbedder()
    docs = te.encode_docs(["USB-C charging cable 2m", "garden hose"])
    q = te.encode_queries(["usb c cable"])
    assert docs.shape == (2, 1024) and q.shape == (1, 1024)
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-3)
    s = (q @ docs.T)[0]
    assert s[0] > s[1]


@pytest.mark.slow
def test_image_embedder_shapes(tmp_path):
    p = tmp_path / "x.jpg"; Image.new("RGB", (64, 64), "blue").save(p)
    ie = ImageEmbedder()
    iv, tv = ie.encode_images([p]), ie.encode_texts(["a blue square"])
    assert iv.shape == (1, 1024) and tv.shape == (1, 1024)
    np.testing.assert_allclose(np.linalg.norm(iv, axis=1), 1.0, rtol=1e-3)
