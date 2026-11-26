"""B4d blue/green: build a second full copy of prod_tuned while search runs through an alias, switch the alias,
then keep both loaded for a 30-minute run-off (the model-change case).

The green copy holds the same vectors as blue: a stand-in for a new embedding model, which would also have to
switch the query encoder at the same instant as the alias. Green is a new collection name and is the only thing
this script ever drops. prod_tuned is never written, dropped or renamed here.

Order: control, bluegreen, bluegreen, control. Controls search the alias with no build, for as long as the
expected build (BUILD_EST_S).  -> results/b4d_bluegreen.json

  uv run python -m bench.b4d_bluegreen --fit-check   # build green, measure memory, drop it, stop
  uv run python -m bench.b4d_bluegreen               # the run (refuses unless the fit check passed)

Not run on the 12 GiB laptop: both copies fit at rest (9.7 GiB) but the build peaked at 11.2 GiB, over
Milvus's 90% line (results/b4d_bluegreen_fit.json).
"""
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np

from bench._common import client, load_corpus, load_eval, parser
from bench.b4a_write_storm import QPS, THREADS
from bench.b4d_single_ops import (MAX_ATTEMPTS, METRIC_EVERY_S, NAME, RAW_DIR, bring_level, finish, guard_loaded,
                                  phases as op_phases, reruns_due, save_jsonl_gz, save_raw, state_metrics,
                                  verify_state)
from vsearch.collections import SPECS, build_collection
from vsearch.config import RESULTS
from vsearch.load import metrics_text, phase_summary, run_load, windowed
from vsearch.results import read_result, write_result

GREEN, ALIAS, OUT_ID = NAME + "_green", "products", "b4d_bluegreen"
BUILD_EST_S = 365.0  # b4d_bluegreen_fit.json: green built in 365 s; controls last this long
RUNIN_S, RUNOFF_S, WARM_S = 300.0, 1800.0, 30.0
CAP_BYTES = 12 * 2 ** 30  # the container's memory limit (docker inspect: 12884901888)
FIT_LINE = 0.90  # Milvus's own line: queryNode.overloadedMemoryThresholdPercentage
TRIP_LINE = 0.95  # the fit check drops green if the process gets this close to the cap
GIB = 2 ** 30


@dataclass(frozen=True)
class Pass:
    slot: int
    repeat: int
    label: str


def design() -> list[Pass]:
    return [Pass(0, 0, "control"), Pass(1, 0, "bluegreen"), Pass(2, 1, "bluegreen"), Pass(3, 1, "control")]


def green_spec():
    return replace(SPECS[NAME], name=GREEN)


def cleanup(c) -> None:
    """Point the alias back at blue, then drop green (if it exists). Nothing else, ever."""
    assert GREEN != NAME
    c.alter_alias(NAME, ALIAS)
    if c.has_collection(GREEN):
        c.drop_collection(GREEN)


def fit_verdict(rss: float, cap_bytes: float = CAP_BYTES, headroom: float = FIT_LINE) -> tuple[bool, str]:
    ok = rss <= headroom * cap_bytes
    return ok, (f"{rss / GIB:.1f} GiB used with both copies loaded; {headroom:.0%} of the {cap_bytes / GIB:.1f} GiB "
                f"cap is {headroom * cap_bytes / GIB:.1f} GiB")


def phases(tl: dict, runoff_s: float) -> dict[str, list[float]]:
    ph = {"runin": [0.0, tl["runin_s"]], "build": [tl["build_start"], tl["build_end"]],
          "switch": [tl["switch_start"], tl["switch_return"]]}
    ph.update({k: v for k, v in op_phases({"runin_s": tl["runin_s"], "write_start": tl["build_start"],
                                           "write_end": tl["build_end"], "runoff_start": tl["switch_return"]},
                                          runoff_s).items() if k.startswith("post")})
    return ph


def around_switch(samples, switch_t: float, half_width: float = 10) -> dict:
    w = [s for s in samples if abs(s.t - switch_t) <= half_width]
    return {"n": len(w), "errors": sum(not s.ok for s in w), "max_ms": max((s.latency_ms for s in w), default=None),
            "p99": float(np.percentile([s.latency_ms for s in w], 99)) if w else None}


def rss_bytes(text: str) -> float:
    """process_resident_memory_bytes from a Prometheus scrape (an unlabelled gauge)."""
    for line in text.splitlines():
        if line.startswith("process_resident_memory_bytes "):
            return float(line.rsplit(" ", 1)[1])
    raise KeyError("process_resident_memory_bytes")


def read_rss() -> float:
    return rss_bytes(metrics_text())


class MemWatch:
    """Polls RSS in a thread, keeps the peak, and calls on_trip once if it passes limit_bytes."""

    def __init__(self, read=read_rss, limit_bytes: float = TRIP_LINE * CAP_BYTES, on_trip=None, poll_s: float = 2.0):
        self.read, self.limit, self.on_trip, self.poll_s = read, limit_bytes, on_trip, poll_s
        self.peak, self.tripped, self.errors = 0.0, False, 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                v = self.read()
            except Exception:
                self.errors += 1
            else:
                self.peak = max(self.peak, v)
                if v > self.limit and not self.tripped:
                    self.tripped = True
                    if self.on_trip:
                        self.on_trip()
            self._stop.wait(self.poll_s)

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t.is_alive():
            self._t.join(timeout=10)


def fit_check(c, products, tv, iv, has, settle_s: float = 120, samples: int = 15) -> dict:
    """Build and load green next to blue, measure RSS with both loaded, then drop green. A load Milvus
    refuses, or a peak over the line, is a fail. Never touches the memory cap."""
    rss_one = read_rss()
    w = MemWatch(on_trip=lambda: cleanup(client()))
    error, t0 = None, time.time()
    w.start()
    try:
        build_collection(c, green_spec(), products, tv, iv, has)
        time.sleep(settle_s)
        both = []
        for _ in range(samples):
            both.append(read_rss())
            time.sleep(2)
        rss_both = float(np.median(both))
    except Exception as e:  # a refused load ("memory not enough") is a result, not a crash
        error, rss_both = f"{type(e).__name__}: {e}", None
    finally:
        w.stop()
        cleanup(c)
    peak = max(w.peak, rss_both or 0.0)
    ok_both, msg = fit_verdict(rss_both) if rss_both is not None else (False, "green did not load")
    ok_peak = peak <= FIT_LINE * CAP_BYTES
    return {"rss_one_gib": round(rss_one / GIB, 2), "rss_both_gib": None if rss_both is None else round(rss_both / GIB, 2),
            "peak_gib": round(peak / GIB, 2), "cap_gib": CAP_BYTES / GIB, "fit_line_gib": FIT_LINE * CAP_BYTES / GIB,
            "tripped": w.tripped, "error": error, "build_s": round(time.time() - t0),
            "fits": error is None and not w.tripped and ok_both and ok_peak,
            "verdict": msg + ("" if ok_peak else f"; peak {peak / GIB:.1f} GiB during the build")}


def check_alias(c) -> None:
    got = c.describe_alias(ALIAS)["collection_name"]
    if got != NAME:
        raise RuntimeError(f"alias {ALIAS} points at {got}, expected {NAME}")


def alias_search(c, vecs, ef: int):
    kw = {"search_params": {"params": {"ef": ef}}, "consistency_level": "Bounded"}

    def search(i):
        c.search(ALIAS, data=[vecs[i % len(vecs)]], anns_field="text_vec", limit=10, **kw)
    return search


def build_and_switch(c, products, tv, iv, has, clock=time.perf_counter) -> dict:
    """Build green beside blue (blue stays loaded and keeps serving), then point the alias at green."""
    f = {"build_start": clock()}
    build_collection(c, green_spec(), products, tv, iv, has)
    f["build_end"] = f["switch_start"] = clock()
    c.alter_alias(GREEN, ALIAS)
    f["switch_return"] = f["runoff_start"] = clock()
    return f


def control_wait(c, wait, build_s: float = BUILD_EST_S, clock=time.perf_counter) -> dict:
    f = {"build_start": clock()}
    wait(build_s)
    f["build_end"] = f["switch_start"] = f["switch_return"] = f["runoff_start"] = clock()
    return f


def bg_metrics(text: str, cids: dict[str, str]) -> dict:
    out = {"rss": rss_bytes(text)}
    for tag, cid in cids.items():
        out.update({f"{tag}_{k}": v for k, v in state_metrics(text, cid).items()})
    return out


def run_pass(c, p: Pass, products, tv, iv, has, vecs, ef, times: dict, out_stem: str) -> dict:
    cleanup(c)  # alias on blue, no green left over
    check_alias(c)
    prep_s, quiet, gate = bring_level(c, NAME, len(products))
    cids = {"blue": str(c.describe_collection(NAME)["collection_id"])}
    search = alias_search(c, vecs, ef)
    builder = client()  # the build gets its own connection; search keeps c
    run_load(search, QPS, times["warm_s"], threads=THREADS)  # warm-up, discarded
    stop_load, stop_m, abort = threading.Event(), threading.Event(), threading.Event()
    mlog, facts, err = [], {}, []
    mem = MemWatch(on_trip=lambda: (abort.set(), cleanup(client())))  # dropping green also stops its build
    started = datetime.now().isoformat(timespec="seconds")
    load_t0 = time.perf_counter()

    def metrics_loop():
        while not stop_m.is_set():
            t = time.perf_counter() - load_t0
            try:
                if "green" not in cids and builder.has_collection(GREEN):
                    cids["green"] = str(builder.describe_collection(GREEN)["collection_id"])
                mlog.append({"t": t, **bg_metrics(metrics_text(), cids)})
            except Exception as e:  # noqa: BLE001
                mlog.append({"t": t, "error": f"{type(e).__name__}: {e}"[:200]})
            stop_m.wait(METRIC_EVERY_S)

    def schedule():
        try:
            if abort.wait(times["runin_s"]):
                return
            if p.label == "bluegreen":
                facts.update(build_and_switch(builder, products, tv, iv, has))
            else:
                facts.update(control_wait(builder, abort.wait, times["build_s"]))
            abort.wait(max(0.0, facts["runoff_start"] + times["runoff_s"] - time.perf_counter()))
        except Exception as e:  # noqa: BLE001 - re-raised below, after the threads have stopped
            err.append(e)
        finally:
            stop_load.set()

    mt, st = threading.Thread(target=metrics_loop), threading.Thread(target=schedule)
    mem.start(); mt.start(); st.start()
    try:
        samples = run_load(search, QPS, float("inf"), threads=THREADS, stop=stop_load)
    finally:  # blue stays loaded until here: the run-off has ended (or the pass failed)
        abort.set(); stop_m.set()
        st.join(); mt.join(); mem.stop()
    if err or mem.tripped:
        save_raw(RAW_DIR / f"{out_stem}.failed.npz", samples)
    if err:
        raise RuntimeError(f"build or switch failed: {err[0]}") from err[0]
    if mem.tripped:
        raise RuntimeError(f"memory passed {TRIP_LINE:.0%} of the cap (peak {mem.peak / GIB:.1f} GiB)")

    tl = {"runin_s": times["runin_s"], **{k: v - load_t0 for k, v in facts.items()}}
    ph = phases(tl, times["runoff_s"])
    raw = save_raw(RAW_DIR / f"{out_stem}.npz", samples)
    met = save_jsonl_gz(RAW_DIR / f"{out_stem}.metrics.jsonl.gz", mlog)
    summary = phase_summary(samples, ph)
    out = {"slot": p.slot, "repeat": p.repeat, "label": p.label, "started": started, "prep_s": prep_s,
           "quiet": quiet, "gate": gate, "timeline": tl, "build_s": tl["build_end"] - tl["build_start"],
           "switch_ms": 1000 * (tl["switch_return"] - tl["switch_start"]),
           "around_switch": around_switch(samples, tl["switch_start"]),
           "phases": ph, "phase_summary": summary, "windows_30s": windowed(samples, 30.0),
           "errors_total": sum(not s.ok for s in samples),
           "error_kinds": sorted({s.error for s in samples if s.error})[:5],
           "runin_p99": summary["runin"]["p99"], "runin_p50": summary["runin"]["p50"],
           "mem_peak_gib": round(mem.peak / GIB, 2), "raw": str(raw), "metrics": str(met)}
    print(f"{time.strftime('%T')} pass {p.slot} {p.label}: runin p99={out['runin_p99'] or float('nan'):.0f} "
          + " ".join(f"{k}={v['p99']:.0f}" for k, v in summary.items() if v["p99"] is not None and k != "runin")
          + f" errors={out['errors_total']} build={out['build_s']:.0f}s peak={out['mem_peak_gib']} GiB", flush=True)
    return out


def main(argv=None):  # pragma: no cover - needs the live stack
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # so a plain kill still runs the cleanup
    ap = parser(__doc__.split("\n\n")[0])
    ap.add_argument("--fit-check", action="store_true")
    args = ap.parse_args(argv)
    c = client()
    if c.has_collection(GREEN):
        raise SystemExit(f"{GREEN} already exists: inspect it before running")
    guard_loaded(c, NAME)
    check_alias(c)
    products, tv, iv, has = load_corpus()
    verify_state(c, NAME, products)
    if args.fit_check:
        r = fit_check(c, products, tv, iv, has)
        verify_state(c, NAME, products)  # raises unless prod_tuned has every row at its parquet price
        r["after"] = {"prod_tuned": f"{len(products):,} rows, matches products.parquet",
                      "load_state": str(c.get_load_state(NAME)["state"]),
                      "alias": c.describe_alias(ALIAS)["collection_name"], "green_exists": c.has_collection(GREEN)}
        print(json.dumps(r, indent=2, default=str), flush=True)
        write_result(OUT_ID + "_fit", r, client=c)
        return
    fit = read_result(OUT_ID + "_fit")["payload"]
    if not fit["fits"]:
        raise SystemExit(f"the fit check failed: {fit['verdict']}")
    if (RESULTS / f"{OUT_ID}.json").exists():
        raise SystemExit(f"results/{OUT_ID}.json exists: move it away first")
    ef = read_result("poc_config")["payload"]["ef"]
    _, qv, _ = load_eval("eval5k")
    vecs = [np.asarray(v, dtype=np.float32).tolist() for v in qv]
    times = {"runin_s": RUNIN_S, "runoff_s": RUNOFF_S, "warm_s": WARM_S, "build_s": BUILD_EST_S}
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    passes, failures, plan = [], [], design()

    def save():
        finish(passes)
        write_result(OUT_ID, {"design": "bluegreen", "qps": QPS, "threads": THREADS, "ef": ef, "alias": ALIAS,
                              "green": GREEN, "read": "dense text_vec top-10 through the alias, Bounded, no filter",
                              "times_s": times, "fit": fit, "passes": passes, "failures": failures}, client=c)

    def one(p: Pass):
        attempt = sum(q["slot"] == p.slot for q in passes + failures)
        r, e = None, None
        try:
            r = run_pass(c, p, products, tv, iv, has, vecs, ef, times, f"{stamp}_bg{p.slot:02d}_a{attempt}_{p.label}")
        except Exception as ex:  # noqa: BLE001 - recorded, rerun at the end
            e = f"{type(ex).__name__}: {ex}"[:500]
            print(f"{time.strftime('%T')} pass {p.slot} {p.label} FAILED: {e}", flush=True)
        finally:  # alias back on blue, green dropped, blue checked: before anything else
            cleanup(c)
            check_alias(c)
            verify_state(c, NAME, products)
            print(f"{time.strftime('%T')} cleaned up: alias on {NAME}, {GREEN} dropped, {NAME} verified", flush=True)
        if r is None:
            failures.append({"slot": p.slot, "label": p.label, "attempt": attempt, "error": e, "level_ok": False})
        else:
            passes.append({**r, "attempt": attempt})
        if passes:
            save()
        if r is not None:
            print(f"{time.strftime('%T')} level: {passes[-1]['level_ok']} {passes[-1]['level_reasons']}", flush=True)

    for p in plan:
        one(p)
    while due := reruns_due(passes + failures, MAX_ATTEMPTS):
        print(f"{time.strftime('%T')} rerunning unlevel or failed passes {due}", flush=True)
        for slot in due:
            one(plan[slot])
    print(f"{time.strftime('%T')} done; unlevel: {[q['label'] for q in passes if not q['level_ok']]}", flush=True)


if __name__ == "__main__":
    main()
