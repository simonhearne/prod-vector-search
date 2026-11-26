import numpy as np

from vsearch.embed import truncate_mrl


def test_truncate_mrl_shape_and_norm():
    v = np.random.default_rng(27).normal(size=(5, 1024)).astype(np.float32)
    t = truncate_mrl(v, 256)
    assert t.shape == (5, 256) and t.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, rtol=1e-5)


def test_truncate_mrl_full_dim_is_renormalised_copy():
    v = np.full((2, 4), 2.0, dtype=np.float32)
    t = truncate_mrl(v, 4)
    np.testing.assert_allclose(t, 0.5)
    assert v[0, 0] == 2.0


def test_fill_missing_images_uses_unit_placeholder():
    from data.embed import fill_missing
    vecs = np.ones((2, 4), dtype=np.float32) / 2
    out = fill_missing(4, np.array([1, 3]), vecs)
    assert out.shape == (4, 4)
    np.testing.assert_array_equal(out[0], [1, 0, 0, 0])
    np.testing.assert_array_equal(out[1], vecs[0])
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0)


def test_chunked_resumes(tmp_path):
    from data.embed import chunked
    calls = []
    def enc(items):
        calls.append(len(items)); return np.ones((len(items), 3), dtype=np.float32)
    a = chunked("x", list(range(7)), enc, chunk=3, workdir=tmp_path)
    assert a.shape == (7, 3) and calls == [3, 3, 1]
    calls.clear()
    chunked("x", list(range(7)), enc, chunk=3, workdir=tmp_path)
    assert calls == []


def test_chunked_stats_count_only_freshly_encoded_items(tmp_path):
    """Throughput after a resume must count only chunks encoded in this run (b3c bug, 3 Nov)."""
    from data.embed import chunked
    enc = lambda items: np.ones((len(items), 3), dtype=np.float32)
    (tmp_path / "x_00000000.npy").parent.mkdir(exist_ok=True)
    np.save(tmp_path / "x_00000000.npy", np.ones((3, 3), dtype=np.float32))  # first chunk cached
    stats = {}
    a = chunked("x", list(range(7)), enc, chunk=3, workdir=tmp_path, stats=stats)
    assert a.shape == (7, 3)
    assert stats["n_encoded"] == 4 and stats["encode_s"] > 0
    stats = {}
    chunked("x", list(range(7)), enc, chunk=3, workdir=tmp_path, stats=stats)
    assert stats == {"n_encoded": 0, "encode_s": 0.0}
