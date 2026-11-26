import json
from types import SimpleNamespace

import pandas as pd

from data.paraphrase import Paraphrase, pick_sources, run


def _queries(n_per=(30, 20, 10, 5, 5, 5, 5, 5, 5, 5, 3, 2)):
    rows = []
    for c, n in enumerate(n_per):
        for i in range(n):
            rows.append({"query_id": f"{c}_{i}", "query": f"q {c} {i}", "top_category": f"cat{c}",
                         "eligible": True, "in_eval5k": True})
    return pd.DataFrame(rows)


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kw):
        self.calls += 1
        assert kw["model"] == "claude-opus-5-5" and kw["output_format"] is Paraphrase
        return SimpleNamespace(parsed_output=Paraphrase(curated_query="x"), stop_reason="end_turn")


def test_pick_sources_stratified_deterministic():
    q = _queries()
    a, b = pick_sources(q, n=50), pick_sources(q, n=50)
    assert a.query_id.tolist() == b.query_id.tolist() and len(a) == 50
    top10 = q.top_category.value_counts().index[:10]
    assert set(top10) <= set(a.top_category)


def test_cli_preserves_approved_rows(tmp_path):
    path = tmp_path / "c.jsonl"
    q = _queries()
    src = pick_sources(q, n=50)
    first = src.query_id.iloc[0]
    path.write_text(json.dumps({"query_id": first, "esci_query": "orig", "curated_query": "mine",
                                "approved": True}) + "\n")
    fc = FakeClient()
    out = run(fc, q, path, regenerate=True)
    by = out.set_index("query_id")
    assert by.loc[first, "curated_query"] == "mine" and bool(by.loc[first, "approved"])
    assert len(out) == 50 and fc.calls == 49
    assert not by.drop(first).approved.any()
