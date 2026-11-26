"""A fresh clone has no data/work/ (data.fetch fills data/dataset/), but benchmarks still write
caches there: every such write must create the directory first."""
import re
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "bench"
# Module-level paths that point into data/work/ (DATA_WORK / ...).
WORK_TARGET = re.compile(r"\b(DATA_WORK|RANKINGS|STATE)\b")
WRITE = re.compile(r"\.(write_text|to_parquet)\(")


def test_ensure_parent_creates_a_missing_work_dir(tmp_path):
    from bench._common import ensure_parent
    p = tmp_path / "work" / "filter_mix.parquet"
    assert ensure_parent(p) == p
    assert p.parent.is_dir()


def test_bench_writes_into_data_work_create_the_dir_first():
    unguarded = [f"{f.name}:{i}: {line.strip()}"
                 for f in sorted(BENCH.glob("*.py"))
                 for i, line in enumerate(f.read_text().splitlines(), 1)
                 if WRITE.search(line) and WORK_TARGET.search(line) and "ensure_parent(" not in line]
    assert unguarded == []
