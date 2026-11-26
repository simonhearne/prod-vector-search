import io

import pandas as pd
from PIL import Image

from data.images import download_all, image_path


def _jpeg(size=(800, 600)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "red").save(buf, "JPEG")
    return buf.getvalue()


def test_image_path_sharded(tmp_path):
    assert image_path("B00ABC12", tmp_path) == tmp_path / "12" / "B00ABC12.jpg"


def test_downloads_and_resizes(tmp_path):
    prods = pd.DataFrame({"product_id": ["P1"], "image_url": ["u1"]})
    m = download_all(prods, fetch=lambda url: _jpeg(), threads=2, root=tmp_path)
    assert m.ok.tolist() == [True]
    assert max(Image.open(image_path("P1", tmp_path)).size) <= 384


def test_skips_existing(tmp_path):
    p = image_path("P1", tmp_path); p.parent.mkdir(parents=True); p.write_bytes(_jpeg((10, 10)))
    calls = []
    m = download_all(pd.DataFrame({"product_id": ["P1"], "image_url": ["u1"]}),
                     fetch=lambda url: calls.append(url) or _jpeg(), threads=1, root=tmp_path)
    assert calls == [] and m.ok.tolist() == [True]


def test_missing_url_marked_not_ok(tmp_path):
    m = download_all(pd.DataFrame({"product_id": ["P1"], "image_url": [None]}),
                     fetch=lambda url: _jpeg(), threads=1, root=tmp_path)
    assert m.iloc[0].ok is False or m.iloc[0].ok == False  # noqa: E712
    assert m.iloc[0].error == "no_url"


def test_http_error_marked_not_ok(tmp_path):
    def boom(url):
        raise RuntimeError("404")
    prods = pd.DataFrame({"product_id": ["P1", "P2"], "image_url": ["bad", "good"]})
    m = download_all(prods, fetch=lambda u: boom(u) if u == "bad" else _jpeg(), threads=2, root=tmp_path)
    by = m.set_index("product_id")
    assert not by.loc["P1", "ok"] and "404" in by.loc["P1", "error"]
    assert by.loc["P2", "ok"]


def test_nan_url_marked_no_url(tmp_path):
    import numpy as np
    calls = []
    m = download_all(pd.DataFrame({"product_id": ["P1"], "image_url": [np.nan]}),
                     fetch=lambda url: calls.append(url) or _jpeg(), threads=1, root=tmp_path)
    assert calls == [] and m.iloc[0].error == "no_url" and not m.iloc[0].ok
