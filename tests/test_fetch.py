import hashlib
import http.server
import threading
from functools import partial

import pytest

from data.fetch import fetch


@pytest.fixture
def server(tmp_path):
    src = tmp_path / "remote"; src.mkdir()
    (src / "a.parquet").write_bytes(b"alpha")
    (src / "b.npy").write_bytes(b"bravo" * 1000)
    sums = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sorted(src.iterdir()))
    (src / "SHA256SUMS").write_text(sums)
    hits = []

    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            hits.append(self.path)
            return super().do_GET()

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(H, directory=str(src)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", src, hits
    httpd.shutdown()


def test_fetch_downloads_and_verifies(server, tmp_path):
    url, src, _ = server
    out = fetch(tmp_path / "dest", base_url=url)
    assert sorted(p.name for p in out) == ["a.parquet", "b.npy"]
    assert (tmp_path / "dest" / "b.npy").read_bytes() == (src / "b.npy").read_bytes()


def test_fetch_verifies_sha(server, tmp_path):
    url, src, _ = server
    (src / "a.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="a.parquet"):
        fetch(tmp_path / "dest", base_url=url)
    assert not (tmp_path / "dest" / "a.parquet").exists()


def test_fetch_skips_verified(server, tmp_path):
    url, _, hits = server
    fetch(tmp_path / "dest", base_url=url)
    hits.clear()
    fetch(tmp_path / "dest", base_url=url)
    assert hits == ["/SHA256SUMS"]
