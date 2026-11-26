import json
from types import SimpleNamespace

import pandas as pd
import pytest

import re

from vsearch.judge import build_request, cohen_kappa, collect, custom_id


def _res(cid, kind, label=None, stop="end_turn", text=True):
    if kind == "succeeded":
        blocks = [SimpleNamespace(type="thinking", thinking="...")]
        if text:
            blocks.append(SimpleNamespace(type="text", text=json.dumps({"label": label})))
        msg = SimpleNamespace(content=blocks, stop_reason=stop)
        return SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="succeeded", message=msg))
    return SimpleNamespace(custom_id=cid, result=SimpleNamespace(type=kind))


@pytest.fixture
def fake_batch_client():
    results = [_res("q1__p2", "succeeded", "S"), _res("q1__p1", "errored"), _res("q2__p3", "succeeded", "E")]
    batches = SimpleNamespace(results=lambda batch_id: iter(results))
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def test_build_request_schema_and_id():
    r = build_request("q1|p1", "usb cable", {"title": "T", "brand": "B", "bullets": "x"})
    assert r["custom_id"] == "q1|p1"
    assert r["params"]["model"] == "claude-haiku-5-5"
    assert r["params"]["output_config"]["format"]["schema"]["properties"]["label"]["enum"] == ["E", "S", "C", "I"]
    content = r["params"]["messages"][0]["content"]
    assert "usb cable" in content and "T" in content


def test_collect_out_of_order_and_errors(fake_batch_client):
    df, counts = collect(fake_batch_client, "b1")
    assert dict(zip(df.product_id, df.label)) == {"p2": "S", "p3": "E"}
    assert dict(zip(df.product_id, df.query_id)) == {"p2": "q1", "p3": "q2"}
    assert counts["errored"] == 1 and counts["succeeded"] == 2 and "p1" not in set(df.product_id)


def test_kappa():
    assert cohen_kappa(list("EESI"), list("EESI")) == pytest.approx(1.0)
    assert cohen_kappa(list("EEII"), list("IIEE")) < 0


def test_custom_id_matches_batch_api_pattern():
    cid = custom_id("12345", "B07XYZ1234")
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cid) and cid == "12345__B07XYZ1234"


def test_collect_skips_truncated_thinking_only_results():
    results = [_res("q1__p1", "succeeded", stop="max_tokens", text=False),
               _res("q1__p2", "succeeded", "E", stop="max_tokens"),
               _res("q1__p3", "succeeded", "I")]
    c = SimpleNamespace(messages=SimpleNamespace(batches=SimpleNamespace(results=lambda batch_id: iter(results))))
    df, counts = collect(c, "b1")
    assert df.product_id.tolist() == ["p3"] and counts["truncated"] == 2 and counts["succeeded"] == 1


def test_build_request_leaves_room_for_thinking():
    r = build_request("q1__p1", "usb cable", {"title": "T", "brand": "B", "bullets": "x"})
    assert r["params"]["max_tokens"] >= 2048


def test_build_request_sends_system_prompt():
    # 5699a45 moved system=SYSTEM_PROMPT into a comment; fills on 3 Nov ran without the ESCI definitions
    from vsearch.judge import SYSTEM_PROMPT
    r = build_request("q1__p1", "usb cable", {"title": "T", "brand": "B", "bullets": "x"})
    assert r["params"]["system"] == SYSTEM_PROMPT


def test_cached_judgments_only_current_prompt(tmp_path, monkeypatch):
    from vsearch import judge
    monkeypatch.setattr(judge, "CACHE", tmp_path / "j.parquet")
    pd.DataFrame({"query_id": ["q", "q", "q"], "product_id": ["a", "b", "c"], "label": ["E", "I", "S"],
                  "model": ["m"] * 3, "prompt": [judge.PROMPT_VERSION, "nosystem-5699a45", None]}
                 ).to_parquet(judge.CACHE, index=False)
    assert judge.cached_judgments().product_id.tolist() == ["a"]
    judge.append_judgments(pd.DataFrame({"query_id": ["q"], "product_id": ["d"], "label": ["E"], "model": ["m"]}))
    assert judge.cached_judgments().product_id.tolist() == ["a", "d"]
    assert len(pd.read_parquet(judge.CACHE)) == 4  # old rows kept, never deleted


def test_cached_judgments_fall_back_to_published_dataset(tmp_path, monkeypatch):
    from vsearch import judge
    # a fresh clone has data/dataset/ (fetched) but no data/work/: filled nDCG must use the published pool
    monkeypatch.setattr(judge, "CACHE", tmp_path / "work" / "j.parquet")
    monkeypatch.setattr(judge, "PUBLISHED", tmp_path / "dataset" / "j.parquet")
    judge.PUBLISHED.parent.mkdir()
    pd.DataFrame({"query_id": ["1", "2"], "product_id": ["a", "b"], "label": ["E", "I"],
                  "model": ["m", "m"], "prompt": [judge.PROMPT_VERSION, "old"]}).to_parquet(judge.PUBLISHED, index=False)
    assert judge.cached_judgments().product_id.tolist() == ["a"]
    judge.append_judgments(pd.DataFrame({"query_id": ["3"], "product_id": ["c"], "label": ["S"], "model": ["m"]}))
    assert sorted(judge.cached_judgments().product_id) == ["a", "c"]  # new labels extend, not replace, the pool
    assert len(pd.read_parquet(judge.PUBLISHED)) == 2  # the fetched file is never written


def test_use_pool_isolates_and_restores(tmp_path):
    from vsearch import judge
    before = (judge.CACHE, judge.PUBLISHED)
    with pytest.raises(RuntimeError), judge.use_pool(tmp_path / "x.parquet"):
        assert judge.fingerprint()["rows"] == 0  # never falls back to the main or the published pool
        judge.append_judgments(pd.DataFrame([("q", "p", "E", "m")], columns=["query_id", "product_id", "label", "model"]))
        assert judge.fingerprint()["rows"] == 1 and (tmp_path / "x.parquet").exists()
        raise RuntimeError
    assert (judge.CACHE, judge.PUBLISHED) == before
