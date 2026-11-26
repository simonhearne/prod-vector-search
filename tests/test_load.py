import time

import pytest

from vsearch.load import Sample, run_load, scrape_metrics_text, windowed


def test_run_load_rate():
    s = run_load(lambda i: time.sleep(0.001), qps=50, duration_s=1.0, threads=4)
    assert 45 <= len(s) <= 55 and all(x.ok for x in s)


def test_run_load_records_errors():
    def fn(i):
        if i % 2:
            raise ValueError("boom")
    s = run_load(fn, qps=40, duration_s=1.0, threads=2)
    bad = [x for x in s if not x.ok]
    assert 15 <= len(bad) <= 25 and all("boom" in x.error for x in bad)


def test_windowed():
    samples = [Sample(t=0.5, latency_ms=10, ok=True, error=None), Sample(t=1.0, latency_ms=30, ok=True, error=None),
               Sample(t=6.0, latency_ms=5, ok=False, error="x")]
    w = windowed(samples, window_s=5.0)
    assert w[0]["t0"] == 0.0 and w[0]["n"] == 2 and w[0]["errors"] == 0
    assert w[0]["p99"] == pytest.approx(29.8) and w[1]["errors"] == 1


def test_scrape_metrics_text_filters_and_sums():
    text = "# HELP x\nmilvus_querynode_segment_num{a=\"1\"} 3\nmilvus_querynode_segment_num{a=\"2\"} 4\nother 9\n"
    assert scrape_metrics_text(text, r"milvus_.*segment.*") == {"milvus_querynode_segment_num": 7.0}


COMPACTION_TEXT = (
    'milvus_datacoord_compaction_task_num{compaction_type="MixCompaction",node_id="-1",status="pending"} 2\n'
    'milvus_datacoord_compaction_task_num{compaction_type="MixCompaction",node_id="0",status="executing"} 12\n'
    'milvus_datacoord_compaction_task_num{compaction_type="MixCompaction",node_id="2",status="done"} 12\n'
    'milvus_datacoord_compaction_task_num{compaction_type="MixCompaction",node_id="2",status="executing"} -11\n'
    'milvus_datacoord_compaction_task_num{compaction_type="SortCompaction",node_id="2",status="done"} 60\n')


def test_compaction_tasks_splits_active_from_done():
    """3.0.3 reports executing as +n on the scheduling node and -n on the worker; done is cumulative.
    Summing every series (as scrape_metrics_text does) plots finished work, not activity."""
    from vsearch.load import compaction_tasks
    assert compaction_tasks(COMPACTION_TEXT) == {"active": 3.0, "done": 72.0}


def test_settle_waits_for_consecutive_idle_polls():
    from vsearch.load import settle
    busy = COMPACTION_TEXT
    idle = COMPACTION_TEXT.replace("} 2\n", "} 0\n").replace("} -11\n", "} -12\n")
    seq = iter([busy, idle, busy, idle, idle, idle])
    polls = settle(lambda: next(seq), idle_polls=3, poll_s=0, timeout_s=5)
    assert polls == 6


def test_settle_times_out():
    from vsearch.load import settle
    with pytest.raises(TimeoutError):
        settle(lambda: COMPACTION_TEXT, idle_polls=2, poll_s=0.01, timeout_s=0.05)


def test_phase_summary_uses_raw_samples_per_phase():
    from vsearch.load import phase_summary
    s = [Sample(t=t, latency_ms=lat, ok=ok, error=None)
         for t, lat, ok in [(1, 10, True), (2, 20, True), (11, 100, True), (12, 300, True), (13, 0, False), (21, 5, True)]]
    out = phase_summary(s, {"baseline": [0, 10], "storm": [10, 20], "cooldown": [20, 30]})
    assert out["baseline"]["n"] == 2 and out["baseline"]["p50"] == pytest.approx(15)
    assert out["storm"]["n"] == 3 and out["storm"]["errors"] == 1 and out["storm"]["p99"] == pytest.approx(298)
    assert out["cooldown"]["p99"] == pytest.approx(5)


def test_querynode_requests_split_by_query_type():
    """Partial update reads the old row on the query node (query_type="upsert_query") before
    writing, so the two paths load the query node differently (B4a paths: write-phase p99 106–112 ms partial, 92–111 ms full)."""
    from vsearch.load import querynode_requests
    text = ('# HELP milvus_querynode_sq_core_latency x\n'
            'milvus_querynode_sq_core_latency_sum{node_id="2",query_type="search"} 1500.5\n'
            'milvus_querynode_sq_core_latency_count{node_id="2",query_type="search"} 300\n'
            'milvus_querynode_sq_core_latency_sum{node_id="2",query_type="upsert_query"} 50\n'
            'milvus_querynode_sq_core_latency_count{node_id="2",query_type="upsert_query"} 10\n'
            'milvus_querynode_sq_core_latency_bucket{node_id="2",query_type="search",le="1"} 7\n'
            'milvus_querynode_sq_queue_latency_count{db_name="default",node_id="2",query_type="search"} 99\n')
    assert querynode_requests(text) == {"qn_search_count": 300, "qn_search_ms": 1500.5,
                                        "qn_upsert_query_count": 10, "qn_upsert_query_ms": 50}


def test_run_load_counts_queueing_and_stamps_due_time():
    # review: latency was timed from execution start, so a saturated pool hid its queue (coordinated omission)
    import time as _t
    from vsearch.load import run_load
    samples = run_load(lambda i: _t.sleep(0.05), qps=50, duration_s=0.4, threads=1)
    assert len(samples) == 20
    assert [round(s.t, 3) for s in samples] == [round(i / 50, 3) for i in range(20)]   # due times
    assert samples[-1].latency_ms > 400   # ~20 x 50 ms of work through one thread, offered every 20 ms


@pytest.mark.parametrize("uptime_s", [1_000.0, 98_425.314, 600_000.4])
def test_run_load_offers_the_same_queries_at_any_uptime(monkeypatch, uptime_s):
    # (start + i/qps) - start rounds to just under duration_s at some uptimes (perf_counter's size), offering one query too many
    import time as _t
    from types import SimpleNamespace
    from vsearch import load
    t0 = _t.perf_counter()
    clock = SimpleNamespace(perf_counter=lambda: uptime_s + (_t.perf_counter() - t0), sleep=_t.sleep)
    monkeypatch.setattr(load, "time", clock)
    assert len(load.run_load(lambda i: None, qps=50, duration_s=0.4)) == 20
