"""B4a: search p99 while the catalogue churns (a pre-launch check: measure reads under your write load).

20 QPS dense search on prod_tuned. Every pass starts level (compact, re-index, settle), warms up, then
runs baseline 120 s, its write phases back to back, and a 1,200 s no-write tail. Write phases:
  partial  300 s storm of partial price updates, 150 rows/s (fresh random 10% of products)
  full     the same storm as full-row upserts (vectors rewritten)
  feed     a 45,000-row full upsert, batch 500, unpaced (the "nightly feed")
Designs (balanced order):
  stacking  control, single, stacked, stacked_strong, stacked_ignore_growing, stacked, single, control
            stacked = partial, feed, partial with only background compaction between (item 4.1)
  paths     single_partial, single_full, single_full, single_partial (item 4.2)
  -> results/b4a.json (stacking) or results/b4a_paths.json, rewritten after every pass"""
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from bench._common import client, ensure_collection, ensure_prices, load_corpus, load_eval, parser
from vsearch.collections import SPECS, _rows, release_all, restore_prices
from vsearch.config import SEED
from vsearch.load import (compaction_tasks, level_start, metrics_text, phase_summary, querynode_requests,
                          run_load, scrape_metrics_text, windowed)
from vsearch.results import read_result, write_result

QPS, THREADS = 20, 4
BATCH, BATCHES_PER_S = 50, 3
FEED_ROWS, FEED_BATCH = 45_000, 500
METRICS = r"milvus_(datacoord_segment_num|querynode_segment_num|datanode_compaction_latency_count)"
LEVEL_TOL = 1.5


@dataclass(frozen=True)
class ReadOpts:
    """Read-side protection under writes. ignore_growing skips not-yet-sealed segments; its placement
    (a top-level kwarg) is pinned by tests/integration/test_read_opts.py, since a misplaced one is ignored."""
    consistency: Literal["Strong", "Bounded", "Eventually"] = "Bounded"
    ignore_growing: bool = False

    def search_kwargs(self, ef: int) -> dict:
        return {"search_params": {"params": {"ef": ef}}, "consistency_level": self.consistency,
                "ignore_growing": self.ignore_growing}

    def query_kwargs(self) -> dict:
        return {"consistency_level": self.consistency, "ignore_growing": self.ignore_growing}


@dataclass(frozen=True)
class PassSpec:
    label: str
    writes: tuple[str, ...]
    read: ReadOpts = field(default_factory=ReadOpts)


def design(name: Literal["stacking", "paths"]) -> list[PassSpec]:
    if name == "paths":
        return [PassSpec("single_partial", ("partial",)), PassSpec("single_full", ("full",)),
                PassSpec("single_full", ("full",)), PassSpec("single_partial", ("partial",))]
    control, single = PassSpec("control", ()), PassSpec("single", ("partial",))
    stacked = ("partial", "feed", "partial")
    return [control, single, PassSpec("stacked", stacked),
            PassSpec("stacked_strong", stacked, ReadOpts("Strong")),
            PassSpec("stacked_ignore_growing", stacked, ReadOpts("Bounded", ignore_growing=True)),
            PassSpec("stacked", stacked), single, control]


def level_flags(baseline_p99: list[float], ref: float, tol: float = LEVEL_TOL) -> list[bool]:
    return [p <= tol * ref for p in baseline_p99]


def phase_windows(base: float, writes: list[tuple[str, float]], tail: float) -> dict[str, list[float]]:
    """Phase boundaries (s since load start) from the write phases' measured durations."""
    ph, t = {"baseline": [0, base]}, base
    for i, (kind, dur) in enumerate(writes):
        ph[f"w{i}_{kind}"] = [t, t + dur]
        t += dur
    for name, lo, hi in [("tail_0_5m", 0, 300), ("tail_5_10m", 300, 600), ("tail_10_20m", 600, tail)]:
        ph[name] = [t + lo, t + hi]
    return ph


def new_prices(p: pd.Series) -> np.ndarray:
    return np.where(p.isna(), 9.99, (p.fillna(0) * 1.1).round(2)).astype(float)


def writer(c, name, mode, prods, tv, iv, has, stop, log, batch=BATCH, batches_per_s=BATCHES_PER_S):
    """Paced storm: `batch` rows `batches_per_s` times a second until prods run out or stop is set.
    mode: "partial_update" (price only) or "full_upsert" (whole rows, vectors included)."""
    spec = SPECS["prod_tuned"]
    prices = new_prices(prods.price)
    t_next = time.perf_counter()
    for s in range(0, len(prods), batch):
        if stop.is_set():
            break
        chunk = prods.iloc[s:s + batch]
        if mode == "partial_update":
            data = [{"product_id": pid, "price": float(pr)} for pid, pr in zip(chunk.product_id, prices[s:s + batch])]
            c.upsert(name, data, partial_update=True)
        else:
            r = chunk.row_id.to_numpy()
            data = list(_rows(spec, chunk, np.asarray(tv[r]), np.asarray(iv[r]), has[r]))
            for d, pr in zip(data, prices[s:s + batch]):
                d["price"] = float(pr)
            c.upsert(name, data)
        log.append((time.perf_counter(), len(chunk)))
        t_next += 1 / batches_per_s
        time.sleep(max(0.0, t_next - time.perf_counter()))


def feed(c, name, prods, tv, iv, has, log):
    """Unpaced full-row upsert of prods in FEED_BATCH batches (new vectors, as a nightly feed)."""
    spec = SPECS["prod_tuned"]
    prices = new_prices(prods.price)
    for s in range(0, len(prods), FEED_BATCH):
        chunk = prods.iloc[s:s + FEED_BATCH]
        r = chunk.row_id.to_numpy()
        data = list(_rows(spec, chunk, np.asarray(tv[r]), np.asarray(iv[r]), has[r]))
        for d, pr in zip(data, prices[s:s + FEED_BATCH]):
            d["price"] = float(pr)
        c.upsert(name, data)
        log.append((time.perf_counter(), len(chunk)))


def run_pass(c, name, k, spec, products, tv, iv, has, vecs, ef, ph) -> dict:
    base, storm_s, tail, warm, feed_rows = ph
    prep_s, level_quiet = level_start(c, name, len(products))
    kw = spec.read.search_kwargs(ef)

    def search(i):
        c.search(name, data=[vecs[i % len(vecs)]], anns_field="text_vec", limit=10, **kw)

    run_load(search, QPS, warm, threads=THREADS)  # warm-up, discarded
    stop_load, stop_m, mlog, durations, rows, touched = threading.Event(), threading.Event(), [], [], [], set()
    t0 = time.perf_counter()

    def metrics_loop():
        while not stop_m.is_set():
            try:
                text = metrics_text()
                mlog.append({"t": time.perf_counter() - t0, **scrape_metrics_text(text, METRICS),
                             **{f"compaction_{x}": v for x, v in compaction_tasks(text).items()},
                             **querynode_requests(text)})
            except Exception as e:  # noqa: BLE001
                mlog.append({"t": time.perf_counter() - t0, "error": str(e)})
            stop_m.wait(5)

    def schedule():
        time.sleep(base)
        for j, kind in enumerate(spec.writes):
            n = feed_rows if kind == "feed" else int(BATCH * BATCHES_PER_S * storm_s)
            targets = products.sample(n=n, random_state=SEED + 100 * k + j).sort_values("row_id")
            touched.update(targets.product_id)
            wlog, t_w = [], time.perf_counter()
            if kind == "feed":
                feed(c, name, targets, tv, iv, has, wlog)
            else:
                stop_w = threading.Event()
                mode = {"partial": "partial_update", "full": "full_upsert"}[kind]
                w = threading.Thread(target=writer, args=(c, name, mode, targets, tv, iv, has, stop_w, wlog))
                w.start()
                time.sleep(max(0.0, storm_s - (time.perf_counter() - t_w)))
                stop_w.set()
                w.join()
            durations.append((kind, time.perf_counter() - t_w))
            rows.append(sum(n for _, n in wlog))
        time.sleep(tail)
        stop_load.set()

    mt, st = threading.Thread(target=metrics_loop), threading.Thread(target=schedule)
    mt.start(); st.start()
    samples = run_load(search, QPS, float("inf"), threads=THREADS, stop=stop_load)
    st.join(); stop_m.set(); mt.join()
    phases = phase_windows(base, durations, tail)
    if touched:  # back to parquet prices; the next pass's level start compacts these rewrites away
        restore_prices(c, name, products, ids=touched)
    out = {"label": spec.label, "writes": list(spec.writes), "read": asdict(spec.read), "prep_s": prep_s,
           "level_quiet": level_quiet,
           "rows_written": rows, "write_s": [d for _, d in durations], "phases": phases,
           "windows": windowed(samples, 30.0), "phase_summary": phase_summary(samples, phases), "metrics": mlog}
    print(f"{time.strftime('%T')} pass {k} {spec.label}: " +
          " ".join(f"{p}={v['p99']:.0f}" for p, v in out["phase_summary"].items() if v["p99"] is not None), flush=True)
    return out


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--design", choices=["stacking", "paths"], default="stacking")
    ap.add_argument("--restore-only", action="store_true", help="reset prod_tuned prices to products.parquet, then exit")
    a = ap.parse_args(argv)
    if a.restore_only:
        c = client()
        print(f"restored {ensure_prices(c, ensure_collection(c, 'prod_tuned'), load_corpus()[0]):,} drifted prices")
        return
    ph = (10, 20, 30, 5, 5_000) if a.sample else (120, 300, 1200, 30, FEED_ROWS)
    out_id = "b4a" if a.design == "stacking" else "b4a_paths"
    c = client()
    ef = read_result("poc_config")["payload"]["ef"]
    products, tv, iv, has = load_corpus()
    _, qv, _ = load_eval("eval5k")
    release_all(c)
    name = ensure_collection(c, "prod_tuned")
    ensure_prices(c, name, products)
    vecs = [np.asarray(v, dtype=np.float32).tolist() for v in qv]
    passes = []
    for k, spec in enumerate(design(a.design)):
        passes.append(run_pass(c, name, k, spec, products, tv, iv, has, vecs, ef, ph))
        base = [p["phase_summary"]["baseline"]["p99"] for p in passes]
        ctrl = [b for p, b in zip(passes, base) if p["label"] == "control"]
        ref = ctrl[0] if ctrl else min(base)
        for p, ok in zip(passes, level_flags(base, ref)):
            p["level_ok"] = ok
        write_result(out_id, {"design": a.design, "passes": passes, "qps": QPS, "ef": ef,
                              "ref_baseline_p99": ref, "level_tol": LEVEL_TOL, "phases_s": ph,
                              "sample": a.sample is not None}, client=c)
    ensure_prices(c, name, products)  # B2/B2c score filters against products.parquet prices


if __name__ == "__main__":
    main()
