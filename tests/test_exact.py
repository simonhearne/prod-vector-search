import numpy as np

from vsearch.exact import exact_topk


def test_exact_topk_order():
    docs = np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32)
    idx, scores = exact_topk(docs, docs[[0]], k=2)
    assert list(idx[0]) == [0, 1] and scores[0, 0] >= scores[0, 1]


def test_exact_topk_with_mask():
    docs = np.eye(4, dtype=np.float32); q = docs[[0]]
    idx, _ = exact_topk(docs, q, k=2, mask=np.array([False, True, True, False]))
    assert set(idx[0]) <= {1, 2}


def test_exact_topk_pads_when_mask_small():
    docs = np.eye(4, dtype=np.float32)
    idx, _ = exact_topk(docs, docs[[0]], k=3, mask=np.array([True, False, False, False]))
    assert list(idx[0]) == [0, -1, -1]


def test_exact_topk_batches_consistently():
    rng = np.random.default_rng(27)
    docs = rng.normal(size=(500, 8)).astype(np.float32); qs = rng.normal(size=(9, 8)).astype(np.float32)
    a, _ = exact_topk(docs, qs, k=5, batch=2)
    b, _ = exact_topk(docs, qs, k=5, batch=100)
    np.testing.assert_array_equal(a, b)
