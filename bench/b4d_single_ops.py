"""B4d: search p99 around ONE write operation at a time (items 4.1 / 4.2).

B4a ran a price storm, a bulk feed and a second storm back to back, so its p99 can't be assigned to any
one of them. Here every pass makes exactly one kind of write to prod_tuned, 20 QPS dense search runs
throughout, and the run-off is 30 minutes. Passes, 7 ops forward then the same 7 reversed (control first
and last, so drift over the night shows up between the two controls):

  control      no write, idle for as long as the longest op
  trickle      price-only partial upsert, 5 rows/s, 3,000 rows (10 min)
  promo        price-only, 4,479 rows (1%), batch 50, 150 rows/s (~30 s)
  promo_flush  the same rows as promo, then one flush straight after the last write
  delta        full-row upsert, 4,479 rows, batch 500, unpaced, one client. Vectors are rewritten with the
               values they already hold (only the price changes), so exact ground truth still applies.
  delist       delete 4,479 rows by primary key, batch 500
  storm10      price-only, 45,000 rows (10%), batch 50, 150 rows/s (5 min)

Each pass: level start, 30 s warm-up (discarded), 5 min run-in, the write, 30 min run-off from the last
write (from the flush's return for promo_flush; both times are logged). Then the collection is put back
(prices, deleted rows) and verified (row count and price drift) before the next level start.

Raw samples (due time, latency, ok) go to data/work/b4d/*.npz and 5 s state metrics to *.metrics.jsonl.gz;
results/b4d_single_ops.json keeps summaries and their paths and is rewritten after every pass. The load
generator counts queueing time, which the old b4a.json runs did not, so their numbers are not comparable.

  uv run python -m bench.b4d_single_ops                       # full run (14 passes, overnight)
  uv run python -m bench.b4d_single_ops --resume-at 6         # continue at pass 6
  uv run python -m bench.b4d_single_ops --smoke               # short version of every op, once
  uv run python -m bench.b4d_single_ops --only promo control  # those ops once, full length
  uv run python -m bench.b4d_single_ops --restore-only        # put prod_tuned back, then exit
"""
import gzip
import json
import os
import shutil
import signal
import sys
import threading
import time
import zlib
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from bench._common import client, load_corpus, load_eval, parser
from bench.b4a_write_storm import QPS, THREADS, feed, writer
from vsearch.collections import SPECS, _rows, drifted_ids, restore_prices
from vsearch.config import DATA_WORK, RESULTS, SEED
from vsearch.load import (Sample, labelled_sums, level_start, metrics_text, phase_summary, querynode_requests,
                          run_load, windowed)
from vsearch.results import read_result, write_result

# ---------- constants: volumes and times, in one place ----------

N_ROWS = 447_880
ONE_PCT = 4_479
RUNIN_S, RUNOFF_S, WARM_S = 300.0, 1800.0, 30.0
SMOKE = {"runin_s": 60.0, "runoff_s": 120.0, "warm_s": 10.0, "scale": 0.1}
METRIC_EVERY_S = 5.0
LEVEL_TOL = 1.5          # a pass is level if its run-in p99 <= LEVEL_TOL x the reference
RECOVERY_TOL = 1.5       # recovered when every window's p99 <= RECOVERY_TOL x control p99 ...
RECOVERY_WINDOW_S, RECOVERY_HOLD_S = 30.0, 180.0  # ... for this long, in windows this wide
MIN_QUIET_S = 0.0        # minimum time between the last write and level start (0: not used)
EXPECT_SEGMENTS = 14     # loaded segments after a level start on a clean prod_tuned (6 of 7 smoke passes)
LAYOUT_ROUNDS = 2        # level starts tried when there are more than that
GATE_TIMEOUT_S, GATE_STABLE_POLLS, GATE_POLL_S = 900.0, 3, 5.0
WRITE_CAP = 3.0          # a paced write is stopped after this multiple of its planned time (recorded)
MAX_ATTEMPTS = 2         # an unlevel pass is rerun once, at the end
BIG_COLLECTION = 100_000
NAME = "prod_tuned"
OUT_ID = "b4d_single_ops"
RAW_DIR = DATA_WORK / "b4d"
STRONG = {"consistency_level": "Strong"}  # restore and verify must not read a stale snapshot
MAX_CONSECUTIVE_FAILURES = 2


@dataclass(frozen=True)
class Op:
    label: str
    kind: Literal["none", "price", "full", "delete"]
    rows: int = 0
    batch: int = 0
    rate: float | None = None      # rows/s; None is unpaced (as fast as one client goes)
    flush: bool = False            # one flush straight after the last write
    rowset: int | None = None      # ops with the same rowset touch the same rows (promo, promo_flush)


OPS: list[Op] = [
    Op("control", "none"),
    Op("trickle", "price", rows=3_000, batch=5, rate=5, rowset=1),
    Op("promo", "price", rows=ONE_PCT, batch=50, rate=150, rowset=2),
    Op("promo_flush", "price", rows=ONE_PCT, batch=50, rate=150, flush=True, rowset=2),
    Op("delta", "full", rows=ONE_PCT, batch=500, rowset=3),
    Op("delist", "delete", rows=ONE_PCT, batch=500, rowset=4),
    Op("storm10", "price", rows=45_000, batch=50, rate=150, rowset=5),
]


@dataclass(frozen=True)
class Pass:
    slot: int
    repeat: int
    op: Op


def design(ops: list[Op] = OPS) -> list[Pass]:
    """All ops forward, then all reversed: two repeats, control first and last."""
    fwd = [Pass(i, 0, op) for i, op in enumerate(ops)]
    return fwd + [Pass(len(ops) + i, 1, op) for i, op in enumerate(reversed(ops))]


def control_s(ops: list[Op]) -> float:
    return max((op.rows / op.rate for op in ops if op.rate), default=0.0)


def scaled_ops(ops: list[Op], scale: float) -> list[Op]:
    return [replace(op, rows=int(round(op.rows * scale))) for op in ops]


def op_rows(products: pd.DataFrame, op: Op, repeat: int) -> pd.DataFrame:
    """The rows an op writes: random, fixed by (rowset, repeat), so promo and promo_flush match exactly and
    the two repeats use different rows."""
    return products.sample(n=op.rows, random_state=SEED + 1_000 * repeat + op.rowset).sort_values("row_id")


# ---------- phases, recovery, levelness (pure) ----------

def phases(tl: dict, runoff_s: float) -> dict[str, list[float]]:
    """Phase boundaries (s since load start). Run-off starts at the last write (or the flush's return)."""
    r0 = tl["runoff_start"]
    ph = {"runin": [0.0, tl["runin_s"]], "write": [tl["write_start"], tl["write_end"]]}
    if "flush_start" in tl:
        ph["flush"] = [tl["flush_start"], tl["flush_return"]]
    for lo, hi, n in [(0, 300, "post_0_5m"), (300, 600, "post_5_10m"), (600, 900, "post_10_15m"),
                      (900, 1800, "post_15_30m")]:
        if lo < runoff_s:
            ph[n] = [r0 + lo, r0 + min(hi, runoff_s)]
    return ph


def _window_p99(samples: list[Sample], lo: float, hi: float) -> float | None:
    lat = [s.latency_ms for s in samples if lo <= s.t < hi and s.ok]
    return float(np.percentile(lat, 99)) if lat else None


def recovery_s(samples: list[Sample], t0: float, ref_p99: float, window: float = RECOVERY_WINDOW_S,
               hold: float = RECOVERY_HOLD_S, tol: float = RECOVERY_TOL) -> float | None:
    """Seconds after t0 until window p99s first stay within tol x ref for `hold` seconds (None: not reached)."""
    end = max((s.t for s in samples), default=t0)
    k = int(round(hold / window))
    n = int((end - t0) // window)
    p99 = [_window_p99(samples, t0 + i * window, t0 + (i + 1) * window) for i in range(n)]
    limit = tol * ref_p99
    for i in range(n - k + 1):
        if all(v is not None and v <= limit for v in p99[i:i + k]):
            return i * window
    return None


def breached(samples: list[Sample], t0: float, ref_p99: float, duration: float,
             window: float = RECOVERY_WINDOW_S, tol: float = RECOVERY_TOL) -> bool:
    """Does any window in [t0, t0 + duration) have p99 > tol x ref?"""
    vals = [_window_p99(samples, t0 + i * window, t0 + (i + 1) * window) for i in range(int(duration // window))]
    return any(v is None or v > tol * ref_p99 for v in vals)  # None: nothing succeeded in that window


def level_verdict(runin_p99: float | None, ref: float, gate_ok: bool, quiet: bool,
                  tol: float = LEVEL_TOL) -> tuple[bool, list[str]]:
    reasons = []
    if runin_p99 is None:
        reasons.append("no successful search in the run-in")
    elif runin_p99 > tol * ref:
        reasons.append(f"run-in p99 {runin_p99:.0f} ms > {tol:g} x {ref:.0f} ms")
    if not gate_ok:
        reasons.append("segment/index gate not reached before the run-in")
    if not quiet:
        reasons.append("no compaction-quiet window before the run-in")
    return not reasons, reasons


def level_ref(passes: list[dict]) -> float:
    """Lowest control run-in p99 so far (lowest of any pass when there is no control yet)."""
    got = [p for p in passes if p["runin_p99"] is not None]
    ctrl = [p["runin_p99"] for p in got if p["label"] == "control"]
    return min(ctrl or [p["runin_p99"] for p in got], default=float("nan"))


# ---------- segment state ----------

def _state(seg) -> str:
    return getattr(seg.state, "name", str(seg.state))


def segments_ready(persistent, loaded, idx_active: float) -> tuple[bool, list[str]]:
    """Level state: every persistent segment flushed at L1+, every one loaded sealed, no index jobs."""
    why = []
    bad = [s for s in persistent if _state(s) != "Flushed"]
    if bad:
        why.append(f"{len(bad)} persistent segments not Flushed ({sorted({_state(s) for s in bad})})")
    l0 = [s for s in persistent if str(getattr(s, "level", "")) == "L0"]
    if l0:
        why.append(f"{len(l0)} L0 (delete) segments not compacted")
    sealed = {s.segment_id for s in loaded if _state(s) == "Sealed"}
    growing = [s for s in loaded if _state(s) == "Growing" and s.num_rows > 0]
    if growing:
        why.append(f"{len(growing)} loaded Growing segments")
    want = {s.segment_id for s in persistent if str(getattr(s, "level", "")) != "L0"}
    if sealed != want:
        why.append(f"loaded sealed segments {len(sealed)} != persistent {len(want)}")
    if idx_active:
        why.append(f"{idx_active:.0f} index jobs active")
    return not why, why


def index_jobs_active(text: str) -> float:
    s = labelled_sums(text, "milvus_datacoord_task_count", ["task_state", "task_type"])
    return sum(v for (state, ttype), v in s.items() if ttype == "JobTypeIndexJob"
               and state in ("JobStateInit", "JobStateInProgress", "JobStateRetry"))


def wait_gate(c, name: str) -> dict:
    """Poll until segments_ready holds for GATE_STABLE_POLLS polls with an unchanged loaded-segment count."""
    t0, streak, last = time.monotonic(), 0, None
    while True:
        persistent = [s for s in c.list_persistent_segments(name)]
        loaded = c.list_loaded_segments(name)
        ok, why = segments_ready(persistent, loaded, index_jobs_active(metrics_text()))
        key = tuple(sorted(s.segment_id for s in loaded))
        streak = streak + 1 if ok and key == last else (1 if ok else 0)
        last = key
        if streak >= GATE_STABLE_POLLS or time.monotonic() - t0 > GATE_TIMEOUT_S:
            return {"ok": streak >= GATE_STABLE_POLLS, "waited_s": time.monotonic() - t0, "reasons": why,
                    "n_persistent": len(persistent), "n_loaded": len(loaded)}
        time.sleep(GATE_POLL_S)


def bring_level(c, name: str, n_rows: int) -> tuple[float, bool, dict]:
    """level_start + wait_gate, and require the same number of loaded segments every time: compact again when
    there are more than EXPECT_SEGMENTS, and fail the gate (the pass is then unlevel) if that doesn't fix it."""
    total_s = 0.0
    for r in range(1, LAYOUT_ROUNDS + 1):
        prep_s, quiet = level_start(c, name, n_rows)
        gate = {**wait_gate(c, name), "rounds": r}
        total_s += prep_s + gate["waited_s"]
        if gate["n_loaded"] <= EXPECT_SEGMENTS:
            break
    if gate["n_loaded"] != EXPECT_SEGMENTS:
        gate["ok"] = False
        gate["reasons"] = gate["reasons"] + [f"{gate['n_loaded']} loaded segments, expected {EXPECT_SEGMENTS}"]
    return total_s, quiet, gate


def state_metrics(text: str, collection_id: str) -> dict[str, float]:
    """One 5 s row of segment / compaction / index / flush state, using the metric names on 3.0.3's /metrics.
    Datacoord's segment_num is instance-wide; querynode's is per collection."""
    row: dict[str, float] = {}
    for (lvl, st), v in labelled_sums(text, "milvus_querynode_segment_num", ["segment_level", "segment_state"],
                                      {"collection_id": collection_id}).items():
        row[f"qn_{lvl}_{st}"] = v
    for (lvl, st), v in labelled_sums(text, "milvus_datacoord_segment_num",
                                      ["segment_level", "segment_state"]).items():
        row[f"dc_{lvl}_{st}"] = v
    for (ct, st), v in labelled_sums(text, "milvus_datacoord_compaction_task_num",
                                     ["compaction_type", "status"]).items():
        row[f"comp_{ct}_{st}"] = row.get(f"comp_{ct}_{st}", 0.0) + v
    for (st,), v in labelled_sums(text, "milvus_datacoord_task_count", ["task_state"],
                                  {"task_type": "JobTypeIndexJob"}).items():
        row[f"idx_{st}"] = v
    for (pol,), v in labelled_sums(text, "milvus_wal_segment_assign_flushed_segment_total", ["policy"]).items():
        row[f"seal_{pol}"] = v  # why the WAL sealed a growing segment (capacity, idle, fenced, ...)
    for (lvl,), v in labelled_sums(text, "milvus_datanode_flush_buffer_op_count", ["segment_level"]).items():
        row[f"flush_{lvl}"] = v
    for (lvl,), v in labelled_sums(text, "milvus_datanode_autoflush_buffer_op_count", ["segment_level"]).items():
        row[f"autoflush_{lvl}"] = v
    row["load1"] = os.getloadavg()[0]
    return row | querynode_requests(text)


# ---------- timed writes ----------

class TimedClient:
    """Wraps a MilvusClient: logs every write call's time, rows, latency and error; everything else passes
    through. Writers and feeders are used unchanged by handing them this instead of the client."""

    def __init__(self, c, log: list):
        self._c, self._log = c, log

    def __getattr__(self, k):
        return getattr(self._c, k)

    def _timed(self, op: str, rows: int, fn):
        t = time.perf_counter()
        ok, err = True, None
        try:
            return fn()
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {e}"[:200]
            raise
        finally:
            self._log.append({"t": t, "op": op, "rows": rows, "ms": (time.perf_counter() - t) * 1000,
                              "ok": ok, "error": err})

    def upsert(self, name, data, **kw):
        return self._timed("upsert", len(data), lambda: self._c.upsert(name, data, **kw))

    def delete(self, name, ids, **kw):
        return self._timed("delete", len(ids), lambda: self._c.delete(name, ids=ids, **kw))

    def flush(self, name, **kw):
        return self._timed("flush", 0, lambda: self._c.flush(name, **kw))


def write_stats(log: list[dict]) -> dict:
    """Achieved write rate, measured from the first batch's start to the last batch's return."""
    b = [e for e in log if e["op"] != "flush"]
    if not b:
        return {"rows": 0, "batches": 0, "errors": 0, "seconds": 0.0, "rows_per_s": None, "batch_ms_p50": None,
                "batch_ms_p99": None}
    ms = np.array([e["ms"] for e in b])
    secs = max(e["t"] + e["ms"] / 1000 for e in b) - min(e["t"] for e in b)
    rows = sum(e["rows"] for e in b if e["ok"])
    return {"rows": rows, "batches": len(b), "errors": sum(not e["ok"] for e in b), "seconds": secs,
            "rows_per_s": rows / secs if secs > 0 else None, "batch_ms_p50": float(np.percentile(ms, 50)),
            "batch_ms_p99": float(np.percentile(ms, 99))}


def last_write_at(log: list[dict]) -> float | None:
    """When the last write batch returned (the writer then sleeps out its pacing slot; that isn't writing)."""
    return max((e["t"] + e["ms"] / 1000 for e in log if e["op"] != "flush"), default=None)


def delete_rows(c, name, prods, batch: int) -> None:
    for s in range(0, len(prods), batch):
        c.delete(name, ids=prods.product_id.iloc[s:s + batch].tolist())


def run_op(tc, name, op: Op, rows: pd.DataFrame, tv, iv, has, log: list, stop: threading.Event) -> dict:
    """Run one op's writes through the timed client; returns timeline facts (absolute perf_counter times).
    `stop` ends a paced write early (an interrupt); a paced write also stops at WRITE_CAP x its planned time."""
    facts: dict = {"truncated": False}
    if op.kind == "price":
        def hit_cap():
            facts["truncated"] = True
            stop.set()
        cap = threading.Timer(WRITE_CAP * op.rows / op.rate, hit_cap)
        cap.daemon = True
        cap.start()
        try:
            writer(tc, name, "partial_update", rows, tv, iv, has, stop, [], batch=op.batch,
                   batches_per_s=op.rate / op.batch)
        finally:
            cap.cancel()
    elif op.kind == "full":
        feed(tc, name, rows, tv, iv, has, [])
    elif op.kind == "delete":
        delete_rows(tc, name, rows, op.batch)
    facts["write_end"] = last_write_at(log) or time.perf_counter()
    if op.flush:
        facts["flush_start"] = time.perf_counter()
        tc.flush(name)
        facts["flush_return"] = time.perf_counter()
    facts["runoff_start"] = facts.get("flush_return", facts["write_end"])
    return facts


# ---------- raw samples and logs ----------

def save_raw(path: Path, samples: list[Sample]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=np.array([s.t for s in samples], dtype=np.float64),
                        latency_ms=np.array([s.latency_ms for s in samples], dtype=np.float32),
                        ok=np.array([s.ok for s in samples], dtype=bool))
    return path


def load_raw(path: Path) -> list[Sample]:
    z = np.load(path)
    return [Sample(float(t), float(l), bool(o), None) for t, l, o in zip(z["t"], z["latency_ms"], z["ok"])]


def save_jsonl_gz(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


# ---------- resume / reruns ----------

def resume(passes: list[dict], at: int, n_slots: int) -> tuple[list[dict], list[int]]:
    """Keep saved passes before slot `at`; return (kept, slots still to run in order)."""
    kept = [p for p in passes if p["slot"] < at]
    return kept, list(range(at, n_slots))


def reruns_due(passes: list[dict], max_attempts: int) -> list[int]:
    """Slots whose attempts so far are all unlevel, and which have attempts left."""
    out = []
    for slot in sorted({p["slot"] for p in passes}):
        at = [p for p in passes if p["slot"] == slot]
        if not any(p.get("level_ok") for p in at) and len(at) < max_attempts:
            out.append(slot)
    return out


# ---------- collection state ----------

def _stored_ids(c, name: str) -> set[str]:
    it = c.query_iterator(name, batch_size=10_000, filter="", output_fields=["product_id"], **STRONG)
    got = set()
    while rows := it.next():
        got.update(r["product_id"] for r in rows)
    it.close()
    return got


def reinsert(c, name, prods: pd.DataFrame, tv, iv, has, batch: int = 500) -> None:
    """Put whole rows back from products.parquet (deleted rows)."""
    spec = SPECS[NAME]
    for s in range(0, len(prods), batch):
        ch = prods.iloc[s:s + batch]
        r = ch.row_id.to_numpy()
        c.upsert(name, list(_rows(spec, ch, np.asarray(tv[r]), np.asarray(iv[r]), has[r])))


def verify_state(c, name: str, products: pd.DataFrame) -> None:
    n = c.query(name, filter="", output_fields=["count(*)"], **STRONG)[0]["count(*)"]
    if n != len(products):
        raise RuntimeError(f"{name}: {n:,} rows, expected {len(products):,}")
    if (left := drifted_ids(c, name, products, **STRONG)):
        raise RuntimeError(f"{name}: {len(left):,} rows differ from products.parquet")


def restore_state(c, name: str, products: pd.DataFrame, tv, iv, has) -> dict:
    """Bring the collection back to products.parquet: re-insert missing rows, reset drifted prices, flush,
    then verify. Safe to call on a clean collection (writes nothing) and after an interrupted run."""
    missing = products[~products.product_id.isin(_stored_ids(c, name))]
    if len(missing):
        reinsert(c, name, missing, tv, iv, has)
        c.flush(name)
    drift = drifted_ids(c, name, products, **STRONG)
    if drift:
        restore_prices(c, name, products, ids=drift)  # flushes
    if len(missing) or drift:
        verify_state(c, name, products)
    return {"missing": len(missing), "drifted": len(drift)}


def open_collection(c, name: str, products: pd.DataFrame, tv, iv, has) -> dict:
    """Load the collection and put it back to products.parquet. Deliberately not _common.ensure_collection,
    which drops and rebuilds a collection with a short row count, i.e. after an interrupted delist."""
    if not c.has_collection(name):
        raise RuntimeError(f"{name} does not exist")
    c.load_collection(name)
    return restore_state(c, name, products, tv, iv, has)


def guard_loaded(c, name: str) -> None:
    """Another full-size loaded collection competes for the 12 GB cap: stop rather than measure with it."""
    big = [n for n in c.list_collections() if n != name
           and "Loaded" in str(c.get_load_state(n).get("state", ""))
           and int(c.get_collection_stats(n)["row_count"]) >= BIG_COLLECTION]
    if big:
        raise RuntimeError(f"other large collections are loaded, release them first: {big}")


# ---------- one pass ----------

def run_pass(c, name, p: Pass, products, tv, iv, has, vecs, ef, times: dict, out_stem: str,
             last_write: list[float]) -> dict:
    op, runin_s, runoff_s, warm_s = p.op, times["runin_s"], times["runoff_s"], times["warm_s"]
    rows = op_rows(products, op, p.repeat) if op.kind != "none" else products.iloc[:0]
    if MIN_QUIET_S and last_write and (gap := MIN_QUIET_S - (time.time() - last_write[0])) > 0:
        time.sleep(gap)
    prep_s, quiet, gate = bring_level(c, name, len(products))
    cid = str(c.describe_collection(name)["collection_id"])
    kw = {"search_params": {"params": {"ef": ef}}, "consistency_level": "Bounded"}

    def search(i):
        c.search(name, data=[vecs[i % len(vecs)]], anns_field="text_vec", limit=10, **kw)

    run_load(search, QPS, warm_s, threads=THREADS)  # warm-up, discarded
    wlog: list[dict] = []
    tc = TimedClient(c, wlog)
    stop_load, stop_m, mlog, facts, write_error = threading.Event(), threading.Event(), [], {}, []
    abort, stop_w = threading.Event(), threading.Event()
    since_last_write = time.time() - last_write[0] if last_write else None
    started = datetime.now().isoformat(timespec="seconds")
    load_t0 = time.perf_counter()

    def metrics_loop():
        while not stop_m.is_set():
            t = time.perf_counter() - load_t0
            try:
                mlog.append({"t": t, **state_metrics(metrics_text(), cid)})
            except Exception as e:  # noqa: BLE001
                mlog.append({"t": t, "error": f"{type(e).__name__}: {e}"[:200]})
            stop_m.wait(METRIC_EVERY_S)

    def schedule():
        try:
            if abort.wait(runin_s):
                return
            facts["write_start"] = time.perf_counter()
            if op.kind == "none":
                abort.wait(times["control_s"])
                facts["write_end"] = facts["runoff_start"] = time.perf_counter()
            else:
                facts.update(run_op(tc, name, op, rows, tv, iv, has, wlog, stop_w))
            abort.wait(max(0.0, facts["runoff_start"] + runoff_s - time.perf_counter()))
        except Exception as e:  # noqa: BLE001 - re-raised below, after the threads have stopped
            write_error.append(e)
        finally:
            stop_load.set()

    mt, st = threading.Thread(target=metrics_loop), threading.Thread(target=schedule)
    mt.start(); st.start()
    try:
        samples = run_load(search, QPS, float("inf"), threads=THREADS, stop=stop_load)
    finally:  # no writer may outlive the pass: the caller restores the collection next
        abort.set(); stop_w.set(); stop_m.set()
        st.join(); mt.join()
    if write_error:
        save_raw(RAW_DIR / f"{out_stem}.failed.npz", samples)
        raise RuntimeError(f"write failed: {write_error[0]}") from write_error[0]

    rel = {k: (v - load_t0) for k, v in facts.items() if k != "truncated"}
    tl = {"runin_s": runin_s, **rel}
    ph = phases(tl, runoff_s)
    raw = save_raw(RAW_DIR / f"{out_stem}.npz", samples)
    met = save_jsonl_gz(RAW_DIR / f"{out_stem}.metrics.jsonl.gz", mlog)
    wlg = save_jsonl_gz(RAW_DIR / f"{out_stem}.writes.jsonl.gz", [{**e, "t": e["t"] - load_t0} for e in wlog])
    summary = phase_summary(samples, ph)
    out = {"slot": p.slot, "repeat": p.repeat, "label": op.label, "op": asdict(op),
           "rows_sha": f"{zlib.crc32(','.join(rows.product_id).encode()):08x}" if len(rows) else None,
           "started": started, "prep_s": prep_s, "quiet": quiet, "gate": gate,
           "since_last_write_s": since_last_write,
           "timeline": tl, "write_stats": write_stats(wlog), "write_truncated": facts.get("truncated", False),
           "phases": ph, "phase_summary": summary,
           "windows_30s": windowed(samples, 30.0), "errors_total": sum(not s.ok for s in samples),
           "error_kinds": sorted({s.error for s in samples if s.error})[:5],
           "runin_p99": summary["runin"]["p99"], "runin_p50": summary["runin"]["p50"],
           "segments_start": {k: v for k, v in mlog[0].items() if k.startswith(("qn_L", "dc_L"))} if mlog else {},
           "raw": str(raw), "metrics": str(met), "writes": str(wlg),
           "free_gb": shutil.disk_usage("/").free / 1e9}
    print(f"{time.strftime('%T')} pass {p.slot} {op.label}: runin p99={out['runin_p99'] or float('nan'):.0f} "
          + " ".join(f"{k}={v['p99']:.0f}" for k, v in summary.items() if v["p99"] is not None and k != "runin")
          + f" errors={out['errors_total']}", flush=True)
    return out


# ---------- main ----------

def finish(passes: list[dict]) -> None:
    ref = level_ref(passes)
    for q in passes:
        q["level_ref_p99"] = ref
        q["level_ok"], q["level_reasons"] = level_verdict(q["runin_p99"], ref, q["gate"]["ok"], q["quiet"])


def main(argv=None):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # so a plain kill still runs the restore
    ap = parser(__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true", help="short run of every op, once")
    ap.add_argument("--only", nargs="+", default=None, help="run these ops once, full length")
    ap.add_argument("--resume-at", type=int, default=0, help="continue at this slot (earlier passes kept)")
    ap.add_argument("--restore-only", action="store_true")
    a = ap.parse_args(argv)

    c = client()
    products, tv, iv, has = load_corpus()
    if len(products) != N_ROWS:
        raise RuntimeError(f"products.parquet has {len(products):,} rows, expected {N_ROWS:,}")
    guard_loaded(c, NAME)
    print(f"restore at start: {open_collection(c, NAME, products, tv, iv, has)}", flush=True)
    if a.restore_only:
        verify_state(c, NAME, products)
        return
    ef = read_result("poc_config")["payload"]["ef"]
    _, qv, _ = load_eval("eval5k")
    vecs = [np.asarray(v, dtype=np.float32).tolist() for v in qv]

    name = NAME
    if a.smoke:
        times = {"runin_s": SMOKE["runin_s"], "runoff_s": SMOKE["runoff_s"], "warm_s": SMOKE["warm_s"]}
        ops = scaled_ops(OPS, SMOKE["scale"])
        plan = [Pass(i, 0, op) for i, op in enumerate(ops)]
        out_id = OUT_ID + "_smoke"
    elif a.only:
        times = {"runin_s": RUNIN_S, "runoff_s": RUNOFF_S, "warm_s": WARM_S}
        by = {op.label: op for op in OPS}
        ops = OPS
        plan = [Pass(i, 0, by[l]) for i, l in enumerate(a.only)]
        out_id = OUT_ID + "_only"
    else:
        times = {"runin_s": RUNIN_S, "runoff_s": RUNOFF_S, "warm_s": WARM_S}
        ops, plan, out_id = OPS, design(), OUT_ID
    times["control_s"] = control_s(ops)

    out_path = RESULTS / f"{out_id}.json"
    prior: dict = {}
    if a.resume_at:
        prior = read_result(out_id)["payload"]
        if prior["times_s"] != times or prior["ops"] != [asdict(o) for o in ops]:
            raise RuntimeError("the saved passes used other times or volumes; don't mix them")
        shutil.copy(out_path, RAW_DIR / f"{out_id}.before_resume_{int(time.time())}.json")
    elif out_path.exists() and not (a.smoke or a.only):
        raise RuntimeError(f"results/{out_id}.json exists: use --resume-at, or move it away first")
    passes, todo = resume(prior.get("passes", []), a.resume_at, len(plan))
    failures: list[dict] = prior.get("failures", [])

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    last_write: list[float] = []

    def save():
        finish(passes)
        write_result(out_id, {"design": "single_ops", "qps": QPS, "threads": THREADS, "ef": ef,
                              "read": "dense text_vec top-10, Bounded, no filter", "times_s": times,
                              "level_tol": LEVEL_TOL, "recovery_tol": RECOVERY_TOL, "min_quiet_s": MIN_QUIET_S,
                              "ops": [asdict(o) for o in ops], "n_rows": N_ROWS, "smoke": bool(a.smoke),
                              "passes": passes, "failures": failures, "sample": bool(a.smoke)}, client=c)

    def attempts(slot: int) -> int:
        return sum(q["slot"] == slot for q in passes + failures)

    def one(p: Pass):
        attempt = attempts(p.slot)
        stem = f"{stamp}_s{p.slot:02d}_a{attempt}_{p.op.label}"
        r, err, rs = None, None, {}
        try:
            r = run_pass(c, name, p, products, tv, iv, has, vecs, ef, times, stem, last_write)
        except Exception as e:  # noqa: BLE001 - recorded, rerun at the end; a crash must not cost the night
            err = f"{type(e).__name__}: {e}"[:500]
            print(f"{time.strftime('%T')} pass {p.slot} {p.op.label} FAILED: {err}", flush=True)
        finally:  # whatever happened (Ctrl-C too), put the collection back before anything else
            t0 = time.perf_counter()
            rs = restore_state(c, name, products, tv, iv, has)
            if rs["missing"] or rs["drifted"]:
                last_write[:] = [time.time()]
            verify_state(c, name, products)
            print(f"{time.strftime('%T')} restored {rs} in {time.perf_counter() - t0:.0f}s", flush=True)
        if r is None:
            failures.append({"slot": p.slot, "label": p.op.label, "attempt": attempt, "error": err,
                             "level_ok": False, "at": datetime.now().isoformat(timespec="seconds")})
        else:
            passes.append({**r, "attempt": attempt, "restore": rs})
        if passes:
            save()
        if r is None:
            consecutive.append(p.slot)
            if len(consecutive) >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(f"{len(consecutive)} passes failed in a row {consecutive}: stopping")
        else:
            consecutive.clear()
            print(f"{time.strftime('%T')} level: {passes[-1]['level_ok']} {passes[-1]['level_reasons']}", flush=True)

    consecutive: list[int] = []
    for slot in todo:
        one(plan[slot])
    while not (a.smoke or a.only) and (due := reruns_due(passes + failures, MAX_ATTEMPTS)):
        print(f"{time.strftime('%T')} rerunning unlevel or failed passes {due}", flush=True)
        for slot in due:
            one(plan[slot])
    verify_state(c, name, products)
    print(f"{time.strftime('%T')} done; unlevel: {[q['label'] for q in passes if not q['level_ok']]}", flush=True)


if __name__ == "__main__":
    main()
