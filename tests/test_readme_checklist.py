"""The README is the take-home checklist (spec "The checklist"). Pin its shape and its evidence links."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
TALK_COLS = ["#", "Check", "How to check by hand", "Red flag", "Fix", "Our evidence"]
EXTRA_COLS = ["#", "Stage", "Check", "How to check by hand", "Red flag", "Fix", "Our evidence"]
TALK_IDS = [f"{g}.{i}" for g, n in [(1, 4), (2, 4), (3, 4), (4, 4), (5, 6)] for i in range(1, n + 1)]


def parse_tables(text: str, heading: str) -> list[list[dict]]:
    """Markdown tables (in order) between `heading` and the next heading of the same level."""
    level = heading.split(" ")[0]
    body = text.split(heading, 1)[1]
    body = re.split(rf"\n{level} ", body, maxsplit=1)[0]
    tables, cur = [], None
    for line in body.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cur is None:
                cur = {"cols": cells, "rows": []}
            elif not set("".join(cells)) <= set("-: "):
                cur["rows"].append(dict(zip(cur["cols"], cells)))
        elif cur is not None:
            tables.append(cur)
            cur = None
    if cur is not None:
        tables.append(cur)
    return tables


def test_checklist_tables():
    talk, extra = parse_tables(README, "## Checklist")
    assert talk["cols"] == TALK_COLS and extra["cols"] == EXTRA_COLS
    assert [r["#"] for r in talk["rows"]] == TALK_IDS
    assert [r["#"] for r in extra["rows"]] == [f"R{i}" for i in range(1, 12) if i != 3]  # R3 became 2.4
    for r in talk["rows"]:
        if r["#"].startswith("5."):
            assert r["Our evidence"].startswith("field pattern") and r["How to check by hand"].rstrip().endswith("?")
        else:
            links = re.findall(r"results/[\w/.-]+\.json", r["Our evidence"])
            assert links, r["#"]
            for p in links:
                assert (ROOT / p).exists() and not json.loads((ROOT / p).read_text())["meta"]["sample"], p
    five_one = next(r for r in talk["rows"] if r["#"] == "5.1")["How to check by hand"].lower()
    assert "restore" in five_one and "rbac" in five_one


def test_checklist_is_engine_neutral_outside_fix():
    talk, extra = parse_tables(README, "## Checklist")
    for r in talk["rows"] + extra["rows"]:
        for col in ("Check", "How to check by hand", "Red flag"):
            assert "milvus" not in r[col].lower(), (r["#"], col)


def _b6_row():
    return next(r for r in parse_tables(README, "## Results")[0]["rows"] if "b6_lift" in r["Chart"])["Result"]


def test_b6_row_states_how_hard_each_arm_acted():
    """The stranger control barely acts on the brand arms, so the row must say so, with the measured influence."""
    arms = json.loads((ROOT / "results/b6.json").read_text())["payload"]["arms"]
    row = _b6_row()
    for a in arms.values():
        assert f"{a['own']['moved@10']:.1%}" in row
    for name in ("brand_boost", "brand_sparse"):
        assert f"{arms[name]['stranger']['moved@10']:.1%}" in row and f"{arms[name]['peer']['moved@10']:.1%}" in row


def test_b6_row_claims_only_what_was_measured():
    row = _b6_row()
    assert "isn't personal" not in row and "not shown to be personal" in row
    assert "specific" in row and "broad" in row
