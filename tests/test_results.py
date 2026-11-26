import pytest

from vsearch import results


@pytest.fixture(autouse=True)
def tmp_results(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS", tmp_path)
    return tmp_path


def test_write_read_roundtrip(tmp_results):
    p = results.write_result("b0", {"x": 1, "sample": True})
    assert p == tmp_results / "b0.json"
    r = results.read_result("b0")
    assert r["payload"] == {"x": 1, "sample": True}
    assert r["meta"]["sample"] is True


def test_meta_has_git_and_timestamp():
    r = results.read_result(results.write_result("b1", {"y": 2}).stem)
    m = r["meta"]
    assert len(m["git_sha"]) >= 7 and "git_dirty" in m
    assert m["timestamp_utc"].endswith("Z") and m["sample"] is False
    assert m["mem_gb"] > 0 and m["milvus_version"] is None


def test_write_result_subdir(tmp_results):
    p = results.write_result("monitor/x", {})
    assert p.parent.name == "monitor" and results.read_result("monitor/x")["payload"] == {}


def test_paths_under_the_repo_are_stored_relative(tmp_results):
    from vsearch.config import ROOT
    p = results.write_result("b2", {"raw": str(ROOT / "data" / "work" / "x.npz"), "nested": [{"m": ROOT / "a.gz"}],
                                    "other": "/elsewhere/y.npz", "n": 3})
    got = results.read_result(p.stem)["payload"]
    assert got["raw"] == "data/work/x.npz" and got["nested"][0]["m"] == "a.gz"
    assert got["other"] == "/elsewhere/y.npz" and got["n"] == 3
