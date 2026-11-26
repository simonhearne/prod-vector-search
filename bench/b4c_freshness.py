"""B4c (item 4.4): how long does a write take to become searchable, per read setting, during a storm?

One pass per variant (strong, bounded, eventually, bounded + ignore_growing), each from a level start.
A background 300 s partial-update storm (150 rows/s) runs on prod_tuned; meanwhile a marker product
gets a sentinel price every 2 s. A poller queries the outstanding markers every 0.25 s with the
variant's read options: a marker is visible once its sentinel price comes back, and "missing" while the
product is absent from results altogether (with ignore_growing, an upserted row vanishes until its
segment is sealed). Polling continues after the storm until every marker is visible or times out;
timed-out markers count at the timeout (censored, never dropped).  -> results/b4c.json"""
import threading
import time
from dataclasses import asdict

import numpy as np

from bench._common import client, ensure_collection, ensure_prices, load_corpus, parser
from bench.b4a_write_storm import BATCH, BATCHES_PER_S, ReadOpts, writer
from vsearch.collections import release_all, restore_prices
from vsearch.config import SEED
from vsearch.load import level_start
from vsearch.results import read_result, write_result

VARIANTS = {"strong": ReadOpts("Strong"), "bounded": ReadOpts("Bounded"),
            "eventually": ReadOpts("Eventually"), "ignore_growing": ReadOpts("Bounded", ignore_growing=True)}
MARKER_EVERY_S, POLL_S, MARKER_TIMEOUT_S = 2.0, 0.25, 900
PRICE_TOL = 0.05  # price is stored as float32; sentinels are 1.0 apart


def sentinel(i: int) -> float:
    return 100_000 + i + 0.01


def poll_update(events: dict[str, dict], rows: list[dict], now: float, dt: float) -> None:
    """Apply one poll's rows to the outstanding markers: visible if the sentinel price is back, missing
    (+dt) if the product is absent, unchanged if the old price is still served."""
    got = {r["product_id"]: r["price"] for r in rows}
    for pid, e in events.items():
        if e["t_visible"] is not None:
            continue
        if pid not in got:
            e["missing_s"] += dt
        elif abs(got[pid] - e["price"]) < PRICE_TOL:
            e["t_visible"] = now


def marker_stats(events: list[dict], timeout_s: float = MARKER_TIMEOUT_S) -> dict:
    """Timed-out markers count at the timeout (censored). A percentile q is flagged censored when the
    timed-out share exceeds 1 - q: its true value is then only known to be >= the timeout."""
    delay = np.array([(e["t_visible"] - e["t_write"]) if e["t_visible"] is not None else timeout_s for e in events])
    miss = np.array([e["missing_s"] for e in events])
    timed_out = sum(e["t_visible"] is None for e in events)
    share = timed_out / len(events) if events else 0.0
    pct = lambda a, q: float(np.percentile(a, q)) if len(a) else None
    return {"n": len(events), "timed_out": timed_out,
            "visible_p50": pct(delay, 50), "visible_p99": pct(delay, 99),
            "visible_p50_censored": share > 0.5, "visible_p99_censored": share > 0.01,
            "missing_share": float(np.mean(miss > 0)) if len(miss) else 0.0,
            "missing_p50": pct(miss, 50), "missing_p99": pct(miss, 99),
            "missing_p50_censored": share > 0.5, "missing_p99_censored": share > 0.01}


def storm_timeline(storm_s: float, hooks: dict[str, float]) -> list[tuple[float, str]]:
    """When each storm event fires: hooks at their offset (s from storm start), the writer's "stop" at
    storm_s; a hook at exactly storm_s runs after the writer has stopped (e.g. flush after the storm)."""
    return sorted([(storm_s, "stop"), *((t, k) for k, t in hooks.items())], key=lambda e: (e[0], e[1] != "stop"))


def poll_once(c, name: str, out: dict[str, dict], kw: dict, prev_send: float) -> float:
    """One poll over the outstanding markers, stamped at send time; the time since the previous send is
    credited as missing to markers absent from the results. Returns this poll's send time."""
    sent = time.perf_counter()
    expr = "product_id in [" + ", ".join(f'"{p}"' for p in out) + "]"
    rows = c.query(name, filter=expr, output_fields=["product_id", "price"], **kw)
    poll_update(out, rows, sent, sent - prev_send)
    return sent


def run_variant(c, name, k, opts, products, tv, iv, has, storm_s, timeout_s, ef, hooks=None) -> dict:
    """hooks: {label: (offset_s, fn(c, name))} run in the storm thread (probes: flush / compact)."""
    hooks = hooks or {}
    prep_s, level_quiet = level_start(c, name, len(products))
    n_markers = int(storm_s / MARKER_EVERY_S)
    rng_seed = SEED + 1000 + k
    markers = products.sample(n=n_markers, random_state=rng_seed)
    rest = products.drop(markers.index)
    targets = rest.sample(n=int(BATCH * BATCHES_PER_S * storm_s), random_state=rng_seed).sort_values("row_id")
    events: dict[str, dict] = {}
    lock, storm_done, wlog = threading.Lock(), threading.Event(), []

    hook_t: dict[str, float] = {}

    def storm():
        stop = threading.Event()
        w = threading.Thread(target=writer, args=(c, name, "partial_update", targets, tv, iv, has, stop, wlog))
        t0 = time.perf_counter()
        w.start()
        for at, label in storm_timeline(storm_s, {k: v[0] for k, v in hooks.items()}):
            time.sleep(max(0.0, t0 + at - time.perf_counter()))
            if label == "stop":
                stop.set()
                w.join()
            else:
                hook_t[label] = time.perf_counter()
                hooks[label][1](c, name)
                hook_t[label + "_done"] = time.perf_counter()
        storm_done.set()

    def mark():
        t_next = time.perf_counter()
        for i, pid in enumerate(markers.product_id):
            c.upsert(name, [{"product_id": pid, "price": sentinel(i)}], partial_update=True)
            with lock:
                events[pid] = {"pid": pid, "t_write": time.perf_counter(), "t_visible": None, "missing_s": 0.0,
                               "price": sentinel(i)}
            t_next += MARKER_EVERY_S
            time.sleep(max(0.0, t_next - time.perf_counter()))

    st, mt = threading.Thread(target=storm), threading.Thread(target=mark)
    st.start(); mt.start()
    kw = opts.query_kwargs()
    prev = time.perf_counter()
    while True:
        now = time.perf_counter()
        with lock:
            out = {p: e for p, e in events.items() if e["t_visible"] is None and now - e["t_write"] < timeout_s}
        if out:  # only this thread mutates event dicts; the lock guards the events map the marker thread adds to
            prev = poll_once(c, name, out, kw, prev)
        elif storm_done.is_set() and not mt.is_alive():
            break
        time.sleep(POLL_S)
    st.join(); mt.join()
    ev = list(events.values())
    for e in ev:  # timed out: the poller stopped counting at the timeout
        if e["t_visible"] is None:
            e["missing_s"] = min(e["missing_s"], timeout_s)
    agree = check_search_agrees(c, name, ev[:20], markers, tv, opts, ef)
    restore_prices(c, name, products, ids=set(markers.product_id) | set(targets.product_id))
    return {**marker_stats(ev, timeout_s), "read": asdict(opts),
            "prep_s": prep_s, "level_quiet": level_quiet, "storm_rows": sum(n for _, n in wlog), "search_query_agree": agree, "hooks": hook_t, "events": ev}


def check_search_agrees(c, name, ev, markers, tv, opts, ef) -> bool:
    """Visible-by-query markers must also come back, with the sentinel, from a filtered vector search."""
    rows = markers.set_index("product_id").row_id
    for e in ev:
        if e["t_visible"] is None:
            continue
        vec = np.asarray(tv[rows[e["pid"]]], dtype=np.float32).tolist()
        hits = c.search(name, data=[vec], anns_field="text_vec", filter=f'product_id == "{e["pid"]}"', limit=1,
                        output_fields=["price"], **opts.search_kwargs(ef))[0]
        if not hits or abs(hits[0]["entity"]["price"] - e["price"]) >= PRICE_TOL:
            return False
    return True


def main(argv=None):
    a = parser(__doc__).parse_args(argv)
    storm_s, timeout_s = (20, 60) if a.sample else (300, MARKER_TIMEOUT_S)
    c = client()
    ef = read_result("poc_config")["payload"]["ef"]
    products, tv, iv, has = load_corpus()
    release_all(c)
    name = ensure_collection(c, "prod_tuned")
    ensure_prices(c, name, products)
    out = {}
    for k, (label, opts) in enumerate(VARIANTS.items()):
        out[label] = run_variant(c, name, k, opts, products, tv, iv, has, storm_s, timeout_s, ef)
        print(f"{time.strftime('%T')} {label}: " + " ".join(f"{x}={out[label][x]}" for x in
              ("n", "timed_out", "visible_p50", "visible_p99", "missing_share", "search_query_agree")), flush=True)
        write_result("b4c", {"variants": out, "storm_rows_per_s": BATCH * BATCHES_PER_S, "storm_s": storm_s,
                             "marker_every_s": MARKER_EVERY_S, "poll_s": POLL_S, "timeout_s": timeout_s,
                             "search_query_agree": all(v["search_query_agree"] for v in out.values()),
                             "sample": a.sample is not None}, client=c)
    ensure_prices(c, name, products)  # markers carry sentinel prices; B2/B2c need parquet prices


if __name__ == "__main__":
    main()
