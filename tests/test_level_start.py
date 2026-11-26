import pytest

from vsearch import load


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(load.time, "sleep", lambda s: None)


def test_level_start_order(monkeypatch):
    calls = []
    monkeypatch.setattr(load, "compact_and_wait", lambda c, n: calls.append("compact"))
    monkeypatch.setattr(load, "wait_index", lambda c, n, rows: calls.append(("index", rows)))
    s, quiet = load.level_start(None, "x", 10, fetch=lambda: "", settle_timeout_s=5)  # no compaction lines = idle
    assert calls == ["compact", ("index", 10)] and s >= 0 and quiet is True


def test_level_start_propagates_timeout(monkeypatch):
    busy = 'milvus_datacoord_compaction_task_num{status="executing"} 1'
    monkeypatch.setattr(load, "compact_and_wait", lambda c, n: None)
    monkeypatch.setattr(load, "wait_index", lambda c, n, rows: None)
    with pytest.raises(TimeoutError):
        load.level_start(None, "x", 10, fetch=lambda: busy, settle_timeout_s=0)


def _metrics(active, done):
    return (f'milvus_datacoord_compaction_task_num{{status="executing"}} {active}\n'
            f'milvus_datacoord_compaction_task_num{{status="done"}} {done}')


def test_level_start_repeats_when_compaction_starts_after_settle(monkeypatch):
    # 17 Nov: a background compaction began after settle() and ran through a pass's baseline
    calls = []
    monkeypatch.setattr(load, "compact_and_wait", lambda c, n: calls.append("compact"))
    monkeypatch.setattr(load, "wait_index", lambda c, n, rows: None)
    seq = iter([_metrics(0, 5)] * 4 + [_metrics(1, 5)] + [_metrics(0, 6)] * 60)
    load.level_start(None, "x", 10, fetch=lambda: next(seq), settle_timeout_s=5, quiet_polls=4)
    assert calls == ["compact", "compact"]


def test_level_start_quiet_window_counts_finished_compactions(monkeypatch):
    calls = []
    monkeypatch.setattr(load, "compact_and_wait", lambda c, n: calls.append("compact"))
    monkeypatch.setattr(load, "wait_index", lambda c, n, rows: None)
    seq = iter([_metrics(0, 5)] * 4 + [_metrics(0, 7)] + [_metrics(0, 7)] * 60)  # one finished between polls
    load.level_start(None, "x", 10, fetch=lambda: next(seq), settle_timeout_s=5, quiet_polls=4)
    assert calls == ["compact", "compact"]


def test_level_start_reports_when_it_never_gets_quiet(monkeypatch):
    # review: after max_rounds it used to return silently; callers must be able to record it
    monkeypatch.setattr(load, "compact_and_wait", lambda c, n: None)
    monkeypatch.setattr(load, "wait_index", lambda c, n, rows: None)
    done = iter(range(10_000))
    fetch = lambda: _metrics(0, next(done))  # idle, but a compaction finishes between every poll
    s, quiet = load.level_start(None, "x", 10, fetch=fetch, settle_timeout_s=5, quiet_polls=2, max_rounds=2)
    assert quiet is False
