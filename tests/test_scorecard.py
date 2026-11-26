import pytest

from bench.scorecard import scorecard


def test_scorecard_rows():
    poc = {"ndcg@10": 0.8, "judged@10": 0.9, "recall@10": 0.97, "latency": {"p99": 4.0}}
    prod = {"ndcg@10": 0.5, "judged@10": 0.6, "recall@10": 0.80, "latency": {"p99": 9.0}}
    rows = scorecard(poc, prod)["rows"]
    assert [r["metric"] for r in rows] == ["ndcg@10", "judged@10", "recall@10", "p99_ms"]
    by = {r["metric"]: r for r in rows}
    assert by["ndcg@10"]["delta"] == pytest.approx(-0.3) and by["p99_ms"]["delta"] == pytest.approx(5.0)
    assert by["p99_ms"]["poc"] == 4.0 and by["p99_ms"]["prod"] == 9.0
