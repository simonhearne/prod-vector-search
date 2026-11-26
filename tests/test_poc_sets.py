import json

import pandas as pd

from data.poc_sets import is_broad, is_head_like, pick_head, title_df, write_poc_sets


def _queries():
    rows = []
    for c, n in enumerate([40, 30, 20, 10, 10, 10, 10, 10, 10, 10, 6, 4]):
        for i in range(n):
            rows.append({"query_id": f"{c}_{i}", "top_category": f"cat{c}", "eligible": True, "in_eval5k": i < 5,
                         "hard": i % 4 == 0, "query": "usb c cable" if i % 3 else "usb c cable 2m long braided"})
    return pd.DataFrame(rows)


def test_is_head_like():
    assert is_head_like("glass jars") and is_head_like("earbuds")
    assert not is_head_like("airpods pro 2") and not is_head_like("gifts for twenty year old women")


def test_pick_head_filters_and_is_deterministic():
    q = _queries()
    a, b = pick_head(q, n=50), pick_head(q, n=50)
    assert a.query_id.tolist() == b.query_id.tolist() and len(a) == 50
    assert not a.hard.any() and a["query"].map(is_head_like).all() and a.eligible.all()
    assert set(q.top_category.value_counts().index[:10]) <= set(a.top_category)


def test_write_poc_sets(tmp_path):
    cur = tmp_path / "cur.jsonl"
    cur.write_text("".join(json.dumps({"query_id": f"0_{i}", "curated_query": "x", "approved": False}) + "\n"
                           for i in range(3)))
    out = write_poc_sets(_queries(), cur, tmp_path / "poc_sets.json", n=50)
    saved = json.loads((tmp_path / "poc_sets.json").read_text())
    assert saved == out and saved["poc50"] == ["0_0", "0_1", "0_2"] and len(saved["head50"]) == 50


def test_title_df_counts_documents_not_occurrences():
    df = title_df(pd.Series(["Glass Jars glass", "glass bowl", "Daniel Merriam print"]))
    assert df["glass"] == 2 and df["jars"] == 1 and df["merriam"] == 1


def test_is_broad_requires_every_token_common():
    df = {"glass": 900, "jars": 600, "daniel": 40, "merriam": 2}
    assert is_broad("glass jars", df, min_df=500)
    assert not is_broad("daniel merriam", df, min_df=500) and not is_broad("glass unknownword", df, min_df=500)


def test_pick_head_with_df_excludes_niche():
    q = _queries()
    q.loc[q.index[::2], "query"] = "sharon cooper"
    out = pick_head(q, n=20, df={"usb": 1000, "c": 1000, "cable": 1000}, min_df=500)
    assert len(out) == 20 and (out["query"] == "usb c cable").all()
