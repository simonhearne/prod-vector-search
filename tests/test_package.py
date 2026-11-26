import pandas as pd

from data import package


def _work(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    for name in package.COPY:
        (work / name).write_bytes(b"x")
    return work


def test_card_records_pool_fingerprint_and_licence_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "DATA_WORK", _work(tmp_path))
    monkeypatch.setattr(package, "fingerprint", lambda: {"rows": 3, "sha": "abc123def456", "prompt": "esci-v1"})
    dest = tmp_path / "dataset"
    package.package(dest)
    card = (dest / "README.md").read_text()
    assert "abc123def456" in card and "esci-v1" in card  # which judgments a filled nDCG was scored at
    assert "26 Nov 2026" in card and "27 Nov" not in card
    assert "Licence decision" in card
    sums = (dest / "SHA256SUMS").read_text()
    assert "README.md" in sums and "curated_queries.parquet" in sums
