"""B4d: one write operation per pass. Offline checks of the design and the bookkeeping."""
import json

import numpy as np
import pandas as pd
import pytest

from bench import b4d_single_ops as b4d
from vsearch.load import Sample

N = 447_880


# ---------- design ----------

def test_design_is_forward_then_reversed_with_controls_at_both_ends():
    labels = [p.op.label for p in b4d.design()]
    fwd = ["control", "trickle", "promo", "promo_flush", "delta", "delist", "storm10"]
    assert labels == fwd + fwd[::-1]
    assert [p.repeat for p in b4d.design()] == [0] * 7 + [1] * 7
    assert [p.slot for p in b4d.design()] == list(range(14))


def test_volumes_match_the_brief():
    o = {op.label: op for op in b4d.OPS}
    assert o["trickle"].rows == 3_000 and o["trickle"].rate == 5
    assert o["promo"].rows == round(N * 0.01) == 4_479 and o["promo"].rate == 150 and o["promo"].batch == 50
    assert o["promo_flush"].flush and not o["promo"].flush
    assert o["promo_flush"].rows == o["promo"].rows
    assert o["delta"].rows == 4_479 and o["delta"].kind == "full" and o["delta"].batch == 500
    assert o["delta"].rate is None
    assert o["delist"].kind == "delete" and o["delist"].rows == 4_479 and o["delist"].batch == 500
    assert o["storm10"].rows == 45_000 and o["storm10"].rate == 150 and o["storm10"].batch == 50
    assert o["control"].kind == "none"


def test_control_lasts_as_long_as_the_longest_write():
    """Controls have no writes, so they sit idle for the longest op's planned write time."""
    assert b4d.control_s(b4d.OPS) == max(op.rows / op.rate for op in b4d.OPS if op.rate)
    assert b4d.control_s(b4d.OPS) == 600


def test_promo_and_promo_flush_share_a_row_set_and_differ_only_in_flush():
    products = pd.DataFrame({"row_id": range(N), "product_id": [f"p{i}" for i in range(N)]})
    o = {op.label: op for op in b4d.OPS}
    for repeat in (0, 1):
        a = b4d.op_rows(products, o["promo"], repeat)
        b = b4d.op_rows(products, o["promo_flush"], repeat)
        assert list(a.product_id) == list(b.product_id)
    r0 = set(b4d.op_rows(products, o["promo"], 0).product_id)
    r1 = set(b4d.op_rows(products, o["promo"], 1).product_id)
    assert r0 != r1 and len(r0) == 4_479
    assert list(b4d.op_rows(products, o["promo"], 0).row_id) == sorted(b4d.op_rows(products, o["promo"], 0).row_id)


def test_op_rows_is_deterministic():
    products = pd.DataFrame({"row_id": range(N), "product_id": [f"p{i}" for i in range(N)]})
    o = {op.label: op for op in b4d.OPS}["delist"]
    assert b4d.op_rows(products, o, 0).product_id.tolist() == b4d.op_rows(products, o, 0).product_id.tolist()


def test_smoke_scales_rows_and_durations_but_keeps_every_op():
    ops = b4d.scaled_ops(b4d.OPS, 0.1)
    assert [op.label for op in ops] == [op.label for op in b4d.OPS]
    o = {op.label: op for op in ops}
    assert o["promo"].rows == 448 and o["trickle"].rows == 300
    assert o["control"].rows == 0


# ---------- phases ----------

def test_phases_are_anchored_to_the_last_write():
    tl = {"runin_s": 300.0, "write_start": 300.0, "write_end": 330.0, "runoff_start": 330.0}
    ph = b4d.phases(tl, runoff_s=1800)
    assert ph["runin"] == [0, 300.0] and ph["write"] == [300.0, 330.0]
    assert ph["post_0_5m"] == [330.0, 630.0] and ph["post_5_10m"] == [630.0, 930.0]
    assert ph["post_10_15m"] == [930.0, 1230.0] and ph["post_15_30m"] == [1230.0, 2130.0]


def test_phases_run_off_starts_at_flush_return():
    tl = {"runin_s": 300.0, "write_start": 300.0, "write_end": 330.0, "runoff_start": 350.5,
          "flush_start": 330.2, "flush_return": 350.5}
    ph = b4d.phases(tl, runoff_s=1800)
    assert ph["post_0_5m"] == [350.5, 650.5]


# ---------- recovery ----------

def _series(values, window=30, per=600):
    """One sample per (window/per) seconds, constant latency per window."""
    out = []
    for w, v in enumerate(values):
        for j in range(per // 10):
            out.append(Sample(t=w * window + j * (window / (per // 10)), latency_ms=v, ok=True, error=None))
    return out


def test_recovery_is_first_start_of_three_clean_minutes():
    # 30 s windows: 2 slow, then clean from 60 s. ref 10, limit 15.
    s = _series([50, 50, 10, 10, 10, 10, 10, 10, 10, 10])
    assert b4d.recovery_s(s, t0=0, ref_p99=10, window=30, hold=180) == 60


def test_recovery_requires_three_consecutive_minutes():
    s = _series([50, 10, 10, 10, 50, 10, 10, 10, 10, 10, 10, 10])
    # window 0 bad, 1-3 clean, 4 bad, 5.. clean: 6 clean windows from 5 => t=150
    assert b4d.recovery_s(s, t0=0, ref_p99=10, window=30, hold=180) == 150


def test_recovery_none_when_not_reached():
    assert b4d.recovery_s(_series([50] * 12), t0=0, ref_p99=10, window=30, hold=180) is None


def test_recovery_offset_is_from_t0():
    s = [Sample(t=1000 + x.t, latency_ms=x.latency_ms, ok=x.ok, error=x.error) for x in _series([50, 10] + [10] * 8)]
    assert b4d.recovery_s(s, t0=1000, ref_p99=10, window=30, hold=180) == 30


def test_recovery_breach_at_start():
    """The control check: does its own p99 already sit above the 1.5x limit at t0?"""
    s = _series([10] * 12)
    assert b4d.recovery_s(s, t0=0, ref_p99=10, window=30, hold=180) == 0
    assert b4d.breached(s, t0=0, ref_p99=10, duration=300) is False
    assert b4d.breached(_series([20] * 12), t0=0, ref_p99=10, duration=300) is True


# ---------- levelness ----------

def test_level_verdict():
    ok, reasons = b4d.level_verdict(runin_p99=30, ref=27, gate_ok=True, quiet=True)
    assert ok and reasons == []
    ok, reasons = b4d.level_verdict(runin_p99=50, ref=27, gate_ok=True, quiet=True)
    assert not ok and any("p99" in r for r in reasons)
    ok, reasons = b4d.level_verdict(runin_p99=27, ref=27, gate_ok=False, quiet=True)
    assert not ok and any("segment" in r for r in reasons)
    ok, reasons = b4d.level_verdict(runin_p99=27, ref=27, gate_ok=True, quiet=False)
    assert not ok and any("quiet" in r for r in reasons)


def test_level_ref_is_the_lowest_control_runin_p99():
    passes = [{"label": "promo", "runin_p99": 10}, {"label": "control", "runin_p99": 30},
              {"label": "control", "runin_p99": 26}]
    assert b4d.level_ref(passes) == 26


def test_level_ref_without_controls_uses_the_lowest():
    assert b4d.level_ref([{"label": "promo", "runin_p99": 40}, {"label": "delta", "runin_p99": 33}]) == 33


# ---------- segment gate ----------

class _Seg:
    def __init__(self, segment_id, state="Flushed", level="L1", num_rows=10):
        self.segment_id, self.state, self.level, self.num_rows = segment_id, state, level, num_rows


def test_segments_ready_needs_loaded_to_match_persistent():
    persistent = [_Seg(1), _Seg(2)]
    ok, why = b4d.segments_ready(persistent, [_Seg(1, "Sealed"), _Seg(2, "Sealed")], idx_active=0)
    assert ok and why == []
    ok, why = b4d.segments_ready(persistent, [_Seg(1, "Sealed")], idx_active=0)
    assert not ok and any("load" in w for w in why)
    ok, why = b4d.segments_ready([_Seg(1), _Seg(3, "Growing")], [_Seg(1, "Sealed"), _Seg(3, "Growing")], idx_active=0)
    assert not ok and any("Growing" in w for w in why)
    ok, why = b4d.segments_ready([_Seg(1, "Flushed", "L0")], [], idx_active=0)
    assert not ok and any("L0" in w for w in why)
    ok, why = b4d.segments_ready(persistent, [_Seg(1, "Sealed"), _Seg(2, "Sealed")], idx_active=1)
    assert not ok and any("index" in w for w in why)


# ---------- metrics ----------

METRICS_TEXT = (
    'milvus_querynode_segment_num{collection_id="7",node_id="8",segment_level="L1",segment_state="Sealed"} 3\n'
    'milvus_querynode_segment_num{collection_id="7",node_id="8",segment_level="L0",segment_state="Growing"} 1\n'
    'milvus_querynode_segment_num{collection_id="9",node_id="8",segment_level="L1",segment_state="Sealed"} 5\n'
    'milvus_datacoord_segment_num{segment_level="L1",segment_state="Flushed",segment_format="unknown"} 9\n'
    'milvus_datacoord_compaction_task_num{compaction_type="MixCompaction",node_id="8",status="done"} 4\n'
    'milvus_datacoord_compaction_task_num{compaction_type="L0DeleteCompaction",node_id="8",status="executing"} -1\n'
    'milvus_datacoord_task_count{task_state="JobStateInProgress",task_type="JobTypeIndexJob"} 2\n'
    'milvus_wal_segment_assign_flushed_segment_total{channel_name="dml_11",node_id="8",policy="idle"} 3\n'
)


def test_labelled_sums_groups_by_labels():
    from vsearch.load import labelled_sums
    got = labelled_sums(METRICS_TEXT, "milvus_querynode_segment_num", ["segment_level", "segment_state"],
                        where={"collection_id": "7"})
    assert got == {("L1", "Sealed"): 3.0, ("L0", "Growing"): 1.0}


def test_state_metrics_row_has_real_metric_groups():
    row = b4d.state_metrics(METRICS_TEXT, collection_id="7")
    assert row["qn_L1_Sealed"] == 3 and row["qn_L0_Growing"] == 1
    assert row["dc_L1_Flushed"] == 9
    assert row["comp_MixCompaction_done"] == 4 and row["comp_L0DeleteCompaction_executing"] == -1
    assert row["idx_JobStateInProgress"] == 2
    assert row["seal_idle"] == 3


# ---------- write timing ----------

class _FakeMilvus:
    def __init__(self):
        self.calls = []

    def upsert(self, name, data, **kw):
        self.calls.append(("upsert", name, len(data), kw))
        if data and data[0].get("fail"):
            raise RuntimeError("boom")
        return {"upsert_count": len(data)}

    def delete(self, name, ids):
        self.calls.append(("delete", name, len(ids)))

    def flush(self, name):
        self.calls.append(("flush", name))

    def describe(self):
        return "x"


def test_timed_client_records_each_batch_and_delegates():
    f = _FakeMilvus()
    log = []
    tc = b4d.TimedClient(f, log)
    tc.upsert("n", [{"product_id": "a"}], partial_update=True)
    tc.delete("n", ids=["a", "b"])
    tc.flush("n")
    assert [e["op"] for e in log] == ["upsert", "delete", "flush"]
    assert log[0]["rows"] == 1 and log[1]["rows"] == 2 and all(e["ok"] for e in log)
    assert all(e["ms"] >= 0 and e["t"] > 0 for e in log)
    assert tc.describe() == "x"  # anything else passes straight through
    assert f.calls[0][3] == {"partial_update": True}


def test_timed_client_logs_and_reraises_errors():
    log = []
    with pytest.raises(RuntimeError):
        b4d.TimedClient(_FakeMilvus(), log).upsert("n", [{"fail": True}])
    assert log[0]["ok"] is False and "boom" in log[0]["error"]


def test_write_stats_is_measured_not_planned():
    log = [{"t": 10.0, "op": "upsert", "rows": 50, "ms": 100.0, "ok": True, "error": None},
           {"t": 10.5, "op": "upsert", "rows": 50, "ms": 100.0, "ok": True, "error": None},
           {"t": 11.0, "op": "flush", "rows": 0, "ms": 5.0, "ok": True, "error": None}]
    s = b4d.write_stats(log)
    assert s["rows"] == 100 and s["batches"] == 2 and s["errors"] == 0
    # last batch returns at 10.6; first starts at 10.0 -> 0.6 s
    assert s["seconds"] == pytest.approx(0.6) and s["rows_per_s"] == pytest.approx(100 / 0.6)
    assert s["batch_ms_p50"] == pytest.approx(100.0)


def test_write_stats_empty():
    assert b4d.write_stats([])["rows"] == 0


# ---------- raw samples ----------

def test_raw_samples_round_trip(tmp_path):
    s = [Sample(0.0, 5.0, True, None), Sample(0.05, 9.5, False, "X: y")]
    p = b4d.save_raw(tmp_path / "a.npz", s)
    assert p.exists()
    back = b4d.load_raw(p)
    assert [x.t for x in back] == pytest.approx([0.0, 0.05])
    assert [x.ok for x in back] == [True, False] and back[0].latency_ms == 5.0


# ---------- resume ----------

def test_resume_keeps_passes_before_and_reports_todo():
    passes = [{"slot": 0, "label": "control", "attempt": 0}, {"slot": 1, "label": "trickle", "attempt": 0}]
    kept, todo = b4d.resume(passes, at=1, n_slots=14)
    assert [p["slot"] for p in kept] == [0] and todo[0] == 1 and len(todo) == 13


def test_resume_from_start_keeps_nothing():
    kept, todo = b4d.resume([], at=0, n_slots=14)
    assert kept == [] and todo == list(range(14))


def test_unlevel_slots_are_due_for_one_rerun():
    passes = [{"slot": 0, "attempt": 0, "level_ok": True}, {"slot": 1, "attempt": 0, "level_ok": False},
              {"slot": 2, "attempt": 0, "level_ok": False}, {"slot": 2, "attempt": 1, "level_ok": True}]
    assert b4d.reruns_due(passes, max_attempts=2) == [1]


# ---------- restore ----------

class _Store:
    """A tiny collection: product_id -> price, with just what the restore code calls."""

    def __init__(self, rows):
        self.rows = dict(rows)
        self.log = []

    def query_iterator(self, name, batch_size, filter, output_fields, **kw):
        data = [{"product_id": k, "price": v} for k, v in self.rows.items()]
        it = iter([data])

        class It:
            def next(self):
                return next(it, [])

            def close(self):
                pass
        return It()

    def query(self, name, filter, output_fields, **kw):
        return [{"count(*)": len(self.rows)}]

    def upsert(self, name, data, partial_update=False, **kw):
        self.log.append((len(data), partial_update))
        for d in data:
            if partial_update and d["product_id"] not in self.rows:
                raise RuntimeError("partial update of a missing row")
            self.rows[d["product_id"]] = d["price"]
        return {}

    def flush(self, name):
        pass

    def get_collection_stats(self, name):
        return {"row_count": len(self.rows)}


def _fixture_products():
    n = 5
    products = pd.DataFrame({"product_id": [f"p{i}" for i in range(n)], "row_id": range(n),
                             "price": [1.0, 2.0, 3.0, 4.0, None], "category": "c", "brand": "b", "title": "t",
                             "image_url": "", "average_rating": 4.0, "rating_number": 3, "embed_text": "x"})
    tv = np.zeros((n, 4), dtype=np.float32)
    iv = np.zeros((n, 4), dtype=np.float32)
    has = np.array([False] * n)
    return products, tv, iv, has


def test_restore_reinserts_missing_then_fixes_prices():
    products, tv, iv, has = _fixture_products()
    store = _Store({"p0": 1.0, "p1": 9.9, "p3": 4.0, "p4": -1.0})  # p2 deleted, p1 price drifted, p4 = NaN price
    r = b4d.restore_state(store, "prod_tuned", products, tv, iv, has)
    assert r["missing"] == 1 and r["drifted"] == 1
    assert store.rows == {"p0": 1.0, "p1": 2.0, "p2": 3.0, "p3": 4.0, "p4": -1.0}
    # the missing row went back as a full row (not partial), and before any partial update
    assert any(not partial for _, partial in store.log) and any(partial for _, partial in store.log)


def test_restore_is_a_noop_when_clean():
    products, tv, iv, has = _fixture_products()
    store = _Store({"p0": 1.0, "p1": 2.0, "p2": 3.0, "p3": 4.0, "p4": -1.0})
    r = b4d.restore_state(store, "prod_tuned", products, tv, iv, has)
    assert r["missing"] == 0 and r["drifted"] == 0 and store.log == []


def test_restore_raises_if_it_does_not_converge():
    products, tv, iv, has = _fixture_products()

    class Stuck(_Store):
        def upsert(self, *a, **k):
            return {}  # writes are lost
    with pytest.raises(RuntimeError):
        b4d.restore_state(Stuck({"p0": 1.0}), "prod_tuned", products, tv, iv, has)


def test_verify_fails_on_row_count():
    products, *_ = _fixture_products()
    store = _Store({"p0": 1.0})
    with pytest.raises(RuntimeError):
        b4d.verify_state(store, "prod_tuned", products)


# ---------- guard ----------

class _Other:
    def __init__(self, loaded_rows):
        self._l = loaded_rows

    def list_collections(self):
        return list(self._l)

    def get_load_state(self, name):
        return {"state": "Loaded"}

    def get_collection_stats(self, name):
        return {"row_count": self._l[name]}


def test_refuses_to_start_with_another_big_collection_loaded():
    with pytest.raises(RuntimeError, match="prod_naive"):
        b4d.guard_loaded(_Other({"prod_tuned": 447_880, "prod_naive": 447_880}), "prod_tuned")
    b4d.guard_loaded(_Other({"prod_tuned": 447_880, "t_small": 2_000}), "prod_tuned")


# ---------- writer reuse ----------

def test_writer_accepts_batch_and_rate():
    """B4a's writer is reused for the paced ops; its defaults must not change for B4a and B4c."""
    import inspect
    from bench.b4a_write_storm import BATCH, BATCHES_PER_S, writer
    sig = inspect.signature(writer)
    assert sig.parameters["batch"].default == BATCH and sig.parameters["batches_per_s"].default == BATCHES_PER_S


def test_results_schema_is_json(tmp_path):
    json.dumps({"ops": [op.__dict__ for op in b4d.OPS]})


# ---------- safety: never rebuild, strong reads, interrupts, failures ----------

class _Guarded(_Store):
    """Fails the test if anything tries to drop, create or rename; records read consistency."""

    def __init__(self, rows):
        super().__init__(rows)
        self.reads = []

    def load_collection(self, name):
        pass

    def has_collection(self, name):
        return True

    def drop_collection(self, name):
        raise AssertionError("drop_collection called")

    def create_collection(self, *a, **k):
        raise AssertionError("create_collection called")

    def rename_collection(self, *a, **k):
        raise AssertionError("rename_collection called")

    def query_iterator(self, name, batch_size, filter, output_fields, **kw):
        self.reads.append(kw.get("consistency_level"))
        return super().query_iterator(name, batch_size, filter, output_fields)

    def query(self, name, filter, output_fields, **kw):
        self.reads.append(kw.get("consistency_level"))
        return super().query(name, filter, output_fields)


def test_opening_a_short_collection_restores_it_and_never_rebuilds_it():
    """bench._common.ensure_collection drops and rebuilds on a short row count, which after an interrupted
    delist would drop prod_tuned. B4d must not use it."""
    products, tv, iv, has = _fixture_products()
    store = _Guarded({"p0": 1.0, "p1": 9.9, "p3": 4.0})
    b4d.open_collection(store, "prod_tuned", products, tv, iv, has)
    assert store.rows["p2"] == 3.0 and store.rows["p1"] == 2.0 and len(store.rows) == 5
    assert not hasattr(b4d, "ensure_collection")


def test_restore_and_verify_read_at_strong_consistency():
    products, tv, iv, has = _fixture_products()
    store = _Guarded({"p0": 1.0, "p1": 9.9, "p3": 4.0, "p4": -1.0})
    b4d.restore_state(store, "prod_tuned", products, tv, iv, has)
    b4d.verify_state(store, "prod_tuned", products)
    assert store.reads and set(store.reads) == {"Strong"}


def test_write_end_is_the_last_batch_return_not_the_pacing_sleep():
    log = [{"t": 10.0, "op": "upsert", "rows": 50, "ms": 100.0, "ok": True, "error": None},
           {"t": 10.5, "op": "upsert", "rows": 50, "ms": 200.0, "ok": True, "error": None},
           {"t": 11.0, "op": "flush", "rows": 0, "ms": 5.0, "ok": True, "error": None}]
    assert b4d.last_write_at(log) == pytest.approx(10.7)


def test_run_op_stops_when_told_to():
    """An interrupted pass must stop writing before the restore starts."""
    import threading
    products = pd.DataFrame({"product_id": [f"p{i}" for i in range(100)], "row_id": range(100), "price": 1.0})
    op = b4d.Op("t", "price", rows=100, batch=5, rate=5)
    stop, log = threading.Event(), []
    stop.set()
    b4d.run_op(b4d.TimedClient(_FakeMilvus(), log), "n", op, products, None, None, None, log, stop)
    assert log == []


def test_failed_passes_are_due_for_rerun_too():
    passes = [{"slot": 0, "level_ok": True}]
    failures = [{"slot": 1, "level_ok": False}]
    assert b4d.reruns_due(passes + failures, max_attempts=2) == [1]


def test_missing_runin_p99_is_not_level():
    ok, reasons = b4d.level_verdict(runin_p99=None, ref=27, gate_ok=True, quiet=True)
    assert not ok and reasons
    assert b4d.level_ref([{"label": "control", "runin_p99": None}, {"label": "control", "runin_p99": 26}]) == 26


# ---------- same layout at every start ----------

def _fake_prep(monkeypatch, counts):
    """level_start/wait_gate stand-ins: the n-th round ends with counts[n] loaded segments."""
    calls = []
    monkeypatch.setattr(b4d, "level_start", lambda c, name, n: (calls.append(n) or (100.0, True)))
    monkeypatch.setattr(b4d, "wait_gate", lambda c, name: {
        "ok": True, "waited_s": 10.0, "reasons": [], "n_persistent": counts[len(calls) - 1],
        "n_loaded": counts[len(calls) - 1]})
    return calls


def test_bring_level_takes_the_expected_layout_first_time(monkeypatch):
    calls = _fake_prep(monkeypatch, [b4d.EXPECT_SEGMENTS])
    prep_s, quiet, gate = b4d.bring_level(None, "x", 5)
    assert len(calls) == 1 and gate["ok"] and gate["rounds"] == 1 and quiet and prep_s == 110.0


def test_bring_level_compacts_again_when_there_are_too_many_segments(monkeypatch):
    calls = _fake_prep(monkeypatch, [b4d.EXPECT_SEGMENTS + 4, b4d.EXPECT_SEGMENTS])
    prep_s, quiet, gate = b4d.bring_level(None, "x", 5)
    assert len(calls) == 2 and gate["ok"] and gate["rounds"] == 2 and prep_s == 220.0


def test_bring_level_gives_up_and_marks_the_gate_failed(monkeypatch):
    calls = _fake_prep(monkeypatch, [b4d.EXPECT_SEGMENTS + 4] * 9)
    _, _, gate = b4d.bring_level(None, "x", 5)
    assert len(calls) == b4d.LAYOUT_ROUNDS and not gate["ok"]
    assert any("segments" in r for r in gate["reasons"])
    ok, _ = b4d.level_verdict(runin_p99=27, ref=27, gate_ok=gate["ok"], quiet=True)
    assert not ok


def test_bring_level_does_not_recompact_for_too_few_segments(monkeypatch):
    calls = _fake_prep(monkeypatch, [b4d.EXPECT_SEGMENTS - 2] * 9)
    _, _, gate = b4d.bring_level(None, "x", 5)
    assert len(calls) == 1 and not gate["ok"]
