"""B4b isolated: the alias swap with the load generator and the build in separate processes, so the
harness can't be what slows search. One pass: 5 min of search through the alias, then build
prod_tuned_green (HNSW_SQ8 re-index of the same vectors) in its own process, timing every step, switch
the alias, and search for 15 min more. Then the alias goes back to prod_tuned and green is dropped.
prod_tuned is never written, dropped or renamed.  -> results/b4b_isolated.json

  uv run python -m bench.b4b_isolated
"""
import json
import multiprocessing as mp
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime

import numpy as np

from bench._common import client, load_corpus, load_eval, parser
from bench.b4a_write_storm import QPS, THREADS
from bench.b4d_bluegreen import GIB, GREEN, MemWatch, check_alias, cleanup, rss_bytes
from bench.b4d_single_ops import (EXPECT_SEGMENTS, NAME, bring_level, guard_loaded, save_jsonl_gz, state_metrics,
                                  verify_state, wait_gate)
from vsearch.collections import (HNSW_SQ8, SPECS, IndexSpec, _rows, build_index_params, build_schema,
                                 compact_and_wait, truncate_mrl, wait_index, wait_loaded)
from vsearch.config import DATA_WORK
from vsearch.load import Sample, metrics_text, phase_summary, run_load, windowed
from vsearch.results import read_result, write_result

V2, ALIAS, OUT_ID = GREEN, "products", "b4b_isolated"
RUNIN_S, AFTER_S, WARM_S = 300.0, 900.0, 30.0
BUILD_TIMEOUT_S = 2400.0
RAW_DIR = DATA_WORK / "b4b_isolated"


def v2_spec():
    return replace(SPECS[NAME], name=V2, text_index=IndexSpec("HNSW_SQ", HNSW_SQ8))


def ps_seconds(s: str) -> float:
    """ps -o time= on macOS: [h:]m:ss.cc"""
    parts = [float(x) for x in s.strip().split(":")]
    out = 0.0
    for x in parts:
        out = out * 60 + x
    return out


def serving(got, ref: dict) -> list:
    """'v1' when a response's distances equal prod_tuned's for the same query, else 'v2' (None: no result)."""
    out = []
    for i, d in got:
        if d is None:
            out.append(None)
        else:
            r = ref.get(i)
            out.append("v1" if r is not None and len(r) == len(d) and np.allclose(r, d, rtol=0, atol=1e-6) else "v2")
    return out


def harness_lag_ms(sends: dict, qps: float) -> dict:
    """How late the load generator sent each query: send time minus due time (start + i / qps)."""
    start = min(t - i / qps for i, t in sends.items())
    return {i: (t - (start + i / qps)) * 1000 for i, t in sends.items()}


# ---------- child processes ----------

def load_proc(stop, ready, out_path: str, ef: int) -> None:
    c = client()
    _, qv, _ = load_eval("eval5k")
    vecs = [np.asarray(v, dtype=np.float32).tolist() for v in qv]
    kw = {"search_params": {"params": {"ef": ef}}, "consistency_level": "Bounded"}
    sends, recv, dists = {}, {}, {}

    def fn(i):
        sends[i] = time.perf_counter()
        r = c.search(ALIAS, data=[vecs[i % len(vecs)]], anns_field="text_vec", limit=10, **kw)
        recv[i] = time.perf_counter()
        dists[i] = [h["distance"] for h in r[0]]

    run_load(fn, QPS, WARM_S, threads=THREADS)  # warm-up, discarded
    sends.clear(); recv.clear(); dists.clear()
    ready.set()
    samples = run_load(fn, QPS, float("inf"), threads=THREADS, stop=stop)
    n = len(samples)
    idx = np.array([round(s.t * QPS) for s in samples])
    d = np.full((n, 10), np.nan)
    for k, i in enumerate(idx):
        if i in dists and len(dists[i]) == 10:
            d[k] = dists[i]
    np.savez_compressed(out_path, i=idx, t=np.array([s.t for s in samples]),
                        latency_ms=np.array([s.latency_ms for s in samples]), ok=np.array([s.ok for s in samples]),
                        error=np.array([s.error or "" for s in samples]),
                        send=np.array([sends.get(i, np.nan) for i in idx]),
                        recv=np.array([recv.get(i, np.nan) for i in idx]), dist=d, n_vecs=len(vecs))


def build_proc(out_path: str) -> None:
    marks = []

    def m(k):
        marks.append([k, time.perf_counter()])
        with open(out_path, "w") as f:  # rewritten at every step, so a crash keeps the steps so far
            json.dump(marks, f)

    m("proc_start")
    c = client()
    if c.has_collection(V2):
        raise RuntimeError(f"{V2} exists")
    products, tv, iv, has = load_corpus()
    spec = v2_spec()
    prods = products.sort_values("row_id")
    rows = prods.row_id.to_numpy()
    m("create")
    c.create_collection(V2, schema=build_schema(spec), index_params=build_index_params(spec))
    m("insert")
    for s in range(0, len(prods), 1000):
        r = rows[s:s + 1000]
        c.insert(V2, list(_rows(spec, prods.iloc[s:s + 1000], truncate_mrl(tv[r], spec.text_dim),
                                iv[r] if spec.image else None, has[r])))
    m("flush")
    c.flush(V2)
    m("compact")
    compact_and_wait(c, V2)
    m("index")
    wait_index(c, V2, len(prods))
    m("load")
    c.load_collection(V2)
    wait_loaded(c, V2, len(prods))
    m("switch")
    c.alter_alias(V2, ALIAS)
    m("done")


# ---------- coordinator ----------

def ps_cpu(pids: list[int]) -> dict:
    try:
        out = subprocess.run(["ps", "-o", "pid=,time=", "-p", ",".join(map(str, pids))],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return {}
    return {int(p): ps_seconds(t) for p, t in (ln.split() for ln in out.splitlines() if ln.strip())}


def main(argv=None):  # pragma: no cover - needs the live stack
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    parser(__doc__.split("\n\n")[0]).parse_args(argv)
    c = client()
    if c.has_collection(V2):
        raise SystemExit(f"{V2} already exists: inspect it before running")
    guard_loaded(c, NAME)
    check_alias(c)
    products, _, _, _ = load_corpus()
    verify_state(c, NAME, products)
    gate = wait_gate(c, NAME)
    prep_s = gate["waited_s"]
    if not gate["ok"] or gate["n_loaded"] != EXPECT_SEGMENTS:
        prep_s, _, gate = bring_level(c, NAME, len(products))
    print(f"{time.strftime('%T')} level: {gate}", flush=True)
    ef = read_result("poc_config")["payload"]["ef"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stem = RAW_DIR / datetime.now().strftime("%Y%m%dT%H%M%S")
    load_npz, marks_json = f"{stem}_load.npz", f"{stem}_build.json"

    ctx = mp.get_context("spawn")
    stop, ready = ctx.Event(), ctx.Event()
    lp = ctx.Process(target=load_proc, args=(stop, ready, load_npz, ef), name="b4b-load")
    bp = ctx.Process(target=build_proc, args=(marks_json,), name="b4b-build")
    mon_stop, mlog, tripped = threading.Event(), [], []
    blue_cid = str(c.describe_collection(NAME)["collection_id"])

    def on_trip():
        tripped.append(time.perf_counter())
        cleanup(client())  # dropping green also stops the build
    mem = MemWatch(on_trip=on_trip)

    def monitor():
        mc, green_cid, last_m = client(), None, 0.0
        while not mon_stop.is_set():
            now = time.perf_counter()
            row = {"t": now}
            try:
                row["alias"] = mc.describe_alias(ALIAS)["collection_name"]
            except Exception as e:  # noqa: BLE001
                row["alias_error"] = str(e)[:100]
            if now - last_m >= 5:
                last_m = now
                try:
                    if green_cid is None and mc.has_collection(V2):
                        green_cid = str(mc.describe_collection(V2)["collection_id"])
                    text = metrics_text()
                    row["rss"] = rss_bytes(text)
                    row.update({f"blue_{k}": v for k, v in state_metrics(text, blue_cid).items()})
                    if green_cid:
                        row.update({f"green_{k}": v for k, v in state_metrics(text, green_cid).items()})
                    pids = [p.pid for p in (lp, bp) if p.pid]
                    row["cpu_s"] = {("load" if p == lp.pid else "build"): v for p, v in ps_cpu(pids).items()}
                except Exception as e:  # noqa: BLE001
                    row["metrics_error"] = f"{type(e).__name__}: {e}"[:200]
            mlog.append(row)
            mon_stop.wait(1.0)

    mt = threading.Thread(target=monitor, daemon=True)
    started = datetime.now().isoformat(timespec="seconds")
    err = None
    try:
        lp.start()
        if not ready.wait(120):
            raise RuntimeError("load process not ready after 120 s")
        mem.start(); mt.start()
        t_ready = time.perf_counter()
        print(f"{time.strftime('%T')} load running; build in {RUNIN_S:.0f} s", flush=True)
        time.sleep(RUNIN_S)
        bp.start()
        bp.join(BUILD_TIMEOUT_S)
        if bp.is_alive() or bp.exitcode != 0:
            raise RuntimeError(f"build process failed (exit {bp.exitcode}, alive {bp.is_alive()})")
        marks = dict(json.load(open(marks_json)))
        print(f"{time.strftime('%T')} switched after {marks['done'] - marks['create']:.0f} s of build; "
              f"{AFTER_S:.0f} s more", flush=True)
        time.sleep(max(0.0, marks["done"] + AFTER_S - time.perf_counter()))
    except BaseException as e:  # noqa: BLE001 - always clean up; recorded below
        err = f"{type(e).__name__}: {e}"[:300]
        print(f"{time.strftime('%T')} FAILED: {err}", flush=True)
    finally:
        stop.set()
        lp.join(120)
        if bp.is_alive():
            bp.terminate(); bp.join(30)
        mon_stop.set(); mem.stop()
        cleanup(c)
        check_alias(c)
        verify_state(c, NAME, products)
        print(f"{time.strftime('%T')} cleaned up: alias on {NAME}, {V2} dropped, {NAME} verified", flush=True)

    z = np.load(load_npz)
    samples = [Sample(float(t), float(l), bool(o), str(e) or None)
               for t, l, o, e in zip(z["t"], z["latency_ms"], z["ok"], z["error"])]
    sends = {int(i): float(s) for i, s in zip(z["i"], z["send"]) if not np.isnan(s)}
    lag = harness_lag_ms(sends, QPS)
    t0 = min(s - i / QPS for i, s in sends.items())  # perf_counter time of the load's t = 0
    marks = dict(json.load(open(marks_json))) if bp.pid else {}
    rel = {k: v - t0 for k, v in marks.items()}

    # serving collection: prod_tuned's own distances for every query vector used, read after cleanup
    _, qv, _ = load_eval("eval5k")
    vecs = [np.asarray(v, dtype=np.float32).tolist() for v in qv]
    used = sorted({int(i) % int(z["n_vecs"]) for i in z["i"]})
    ref_by_vec = {}
    for s in range(0, len(used), 50):
        part = used[s:s + 50]
        res = c.search(NAME, data=[vecs[k] for k in part], anns_field="text_vec", limit=10,
                       search_params={"params": {"ef": ef}}, consistency_level="Bounded")
        ref_by_vec.update({k: np.array([h["distance"] for h in hits]) for k, hits in zip(part, res)})
    got = [(int(i), None if np.isnan(d).any() else d) for i, d in zip(z["i"], z["dist"])]
    ref = {i: ref_by_vec[i % int(z["n_vecs"])] for i, _ in got}
    who = serving(got, ref)

    steps = ["create", "insert", "flush", "compact", "index", "load", "switch", "done"]
    ph = {"baseline": [0.0, rel.get("create", RUNIN_S)]}
    for a, b in zip(steps, steps[1:]):
        if a in rel and b in rel:
            ph[f"build_{a}"] = [rel[a], rel[b]]
    if "done" in rel:
        ph["build"] = [rel["create"], rel["switch"]]
        for k, (lo, hi) in {"after_0_5m": (0, 300), "after_5_10m": (300, 600), "after_10_15m": (600, 900)}.items():
            ph[k] = [rel["done"] + lo, rel["done"] + hi]
        ph["after"] = [rel["done"], rel["done"] + AFTER_S]

    def share(lo, hi):
        w = [x for s, x in zip(samples, who) if lo <= s.t < hi and x]
        return {"n": len(w), "v1": sum(x == "v1" for x in w), "v2": sum(x == "v2" for x in w)}

    def lag_stats(lo, hi):
        v = np.array([lag[i] for i, s in zip(z["i"], samples) if lo <= s.t < hi and int(i) in lag])
        return {"p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99)), "max": float(v.max())} if len(v) else None

    service = {int(i): (r - s) * 1000 for i, s, r in zip(z["i"], z["send"], z["recv"]) if not np.isnan(r)}

    def service_stats(lo, hi):  # send to response: Milvus plus the client, without any wait to be sent
        v = np.array([service[i] for i, s in zip(z["i"], samples) if lo <= s.t < hi and int(i) in service])
        return {"p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99))} if len(v) else None

    first_v2 = next((s.t for s, x in zip(samples, who) if x == "v2"), None)
    met = save_jsonl_gz(RAW_DIR / f"{stem.name}_monitor.jsonl.gz", [{**r, "t": r["t"] - t0} for r in mlog])
    alias_flip = next((r["t"] - t0 for r in mlog if r.get("alias") == V2), None)
    out = {"design": "one pass: 5 min search, isolated build + switch, 15 min search", "started": started,
           "qps": QPS, "threads": THREADS, "ef": ef, "v2_index": "HNSW_SQ8 refine FP16", "error": err,
           "level_gate": gate, "prep_s": prep_s, "marks_s": rel,
           "step_s": {a: rel[b] - rel[a] for a, b in zip(steps, steps[1:]) if a in rel and b in rel},
           "phases": ph, "phase_summary": phase_summary(samples, ph),
           "serving_by_phase": {k: share(*v) for k, v in ph.items()},
           "first_v2_response_t": first_v2, "alias_seen_on_v2_t": alias_flip,
           "harness_lag_ms_by_phase": {k: lag_stats(*v) for k, v in ph.items()},
           "service_ms_by_phase": {k: service_stats(*v) for k, v in ph.items()},
           "errors_total": sum(not s.ok for s in samples),
           "error_kinds": sorted({s.error for s in samples if s.error})[:5],
           "n": len(samples), "windows_5s": windowed(samples, 5.0), "windows_30s": windowed(samples, 30.0),
           "mem_peak_gib": round(mem.peak / GIB, 2), "mem_tripped": bool(tripped),
           "raw": {"load": load_npz, "build": marks_json, "monitor": str(met)}}
    path = write_result(OUT_ID, out, client=c)
    print(json.dumps({k: out[k] for k in ("error", "step_s", "first_v2_response_t", "alias_seen_on_v2_t",
                                          "errors_total", "mem_peak_gib")}, indent=1, default=str))
    for k, v in out["phase_summary"].items():
        print(f"  {k:16s} p50 {v['p50'] or float('nan'):7.1f}  p99 {v['p99'] or float('nan'):7.1f}  n {v['n']:6d}  "
              f"lag p99 {(out['harness_lag_ms_by_phase'][k] or {}).get('p99', float('nan')):6.1f}  "
              f"serving {out['serving_by_phase'][k]}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
