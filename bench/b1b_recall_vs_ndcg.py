"""B1b: recall != relevance. Sweep HNSW ef on prod_naive (450k), canonical eval5k subsample, unfiltered:
recall@10 vs exact, nDCG@10/judged@10 raw and judge-filled, latency.  -> results/b1b.json

    uv run python bench/b1b_recall_vs_ndcg.py              # full run: Milvus sweep, judge fill, write
    uv run python bench/b1b_recall_vs_ndcg.py --fill-only  # re-judge/re-score the cached rankings, no Milvus
    uv run python bench/b1b_recall_vs_ndcg.py --monitor    # item 1.4: one drift check -> results/monitor/

Monitor mode stands in for a scheduled production check: recall@10 against exact search on a fresh
sample of "live" queries (200 eval5k queries seeded from today's UTC date), and nDCG@10 on the fixed
golden set (the canonical 1,000) against b1b's baseline at the same ef. Unjudged hits count as 0, so
the nDCG verdict is "needs_judging" when judged@10 is below 0.9 rather than a false "drift"."""
import json
from datetime import datetime, timezone

import anthropic
import numpy as np

from bench._common import (CANONICAL_N, client, corpus_rows, ensure_collection, ensure_parent, exact_ids,
                           load_corpus, load_eval, parser, qrels, quality, rankings)
from vsearch import judge
from vsearch.collections import release_all
from vsearch.config import DATA_WORK, RESULTS
from vsearch.metrics import recall_at_k
from vsearch.results import read_result, write_result
from vsearch.search import run_queries, summarise_latency

EF_GRID = [10, 16, 24, 32, 48, 64, 128, 256]
RANKINGS = DATA_WORK / "b1b_rankings.json"


def monitor_report(recall: float, golden: dict, baseline_ndcg: float | None, *, baseline_judged: float | None = None,
                   min_recall: float = 0.95, max_ndcg_drop: float = 0.02, min_judged: float = 0.9) -> dict:
    """Unjudged hits score 0, so a judged@10 gap to the baseline alone can look like drift: require
    judged@10 >= max(min_judged, baseline_judged - 0.01) before judging nDCG."""
    f = golden["filled"]
    floor = max(min_judged, (baseline_judged - 0.01) if baseline_judged is not None else min_judged)
    if f["judged@10"] < floor:
        status = "needs_judging"
    elif baseline_ndcg is None:
        status = "no_baseline"
    else:
        status = "drift" if baseline_ndcg - f["ndcg@10"] > max_ndcg_drop else "ok"
    return {"recall@10": recall, "recall_ok": recall >= min_recall, "ndcg@10": f["ndcg@10"],
            "judged@10": f["judged@10"], "ndcg_status": status, "baseline_ndcg@10": baseline_ndcg,
            "baseline_judged@10": baseline_judged,
            "thresholds": {"min_recall": min_recall, "max_ndcg_drop": max_ndcg_drop, "min_judged": min_judged}}


def monitor_baseline(ef: int, b1b_rankings: dict[int, dict], prior: list[dict]) -> dict | None:
    """Baseline *rankings* for a monitor run, re-scored by the caller against today's labels (a stored
    nDCG drifts as the judgment pool grows). b1b's cached rankings if this ef was swept, else the earliest
    earlier monitor run at this ef that was ok / no_baseline: pinned, so slow drift can't ratchet it."""
    if ef in b1b_rankings:
        return b1b_rankings[ef]
    good = sorted((r for r in prior if r.get("ef") == ef and r.get("golden_rankings")
                   and r.get("ndcg_status") in ("ok", "no_baseline")), key=lambda r: r["timestamp_utc"])
    return good[0]["golden_rankings"] if good else None


def _prior_monitor_runs() -> list[dict]:
    out = []
    for f in sorted((RESULTS / "monitor").glob("b1b_*.json")):
        r = json.loads(f.read_text())
        out.append({**r["payload"], "timestamp_utc": r["meta"]["timestamp_utc"]})
    return out


def monitor(ef: int | None, live_n: int) -> None:
    c = client()
    ef = ef or read_result("b1a")["payload"]["ef"]
    products, tv, _, _ = load_corpus()
    release_all(c)  # one collection loaded at a time (12 GB container)
    name = ensure_collection(c, "prod_naive")
    now = datetime.now(timezone.utc)
    q_all, qv_all, _ = load_eval("eval5k")
    live = np.random.default_rng(int(now.strftime("%Y%m%d"))).choice(len(q_all), live_n, replace=False)
    lq = qv_all[np.sort(live)]
    truth = exact_ids("monitor_live", tv, corpus_rows("prod_naive", products), lq, products)
    r = run_queries(c, name, "text_vec", lq, search_params={"ef": ef})
    recall = float(np.mean([recall_at_k(i, t) for i, t in zip(r.ids, truth)]))
    gq, gv, _ = load_eval("eval5k", CANONICAL_N)
    qr = qrels()
    golden_ranks = rankings(gq, run_queries(c, name, "text_vec", gv, search_params={"ef": ef}).ids)
    golden = quality(golden_ranks, qr)
    b1b = {int(k): v for k, v in json.loads(RANKINGS.read_text())["rankings"].items()} if RANKINGS.exists() else {}
    base_ranks = monitor_baseline(ef, b1b, _prior_monitor_runs())
    base = quality(base_ranks, qr)["filled"] if base_ranks else None
    rep = monitor_report(recall, golden, base and base["ndcg@10"], baseline_judged=base and base["judged@10"])
    write_result(f"monitor/b1b_{now.strftime('%Y%m%dT%H%M%SZ')}",
                 {**rep, "ef": ef, "collection": name, "live_queries": live_n, "golden_queries": CANONICAL_N,
                  "live_seed": int(now.strftime("%Y%m%d")), "golden_rankings": golden_ranks,
                  "judgments": judge.fingerprint()}, client=c)
    print(json.dumps({k: rep[k] for k in ("recall@10", "recall_ok", "ndcg@10", "judged@10", "ndcg_status")}))


def sweep(c, name, q, qv, truth) -> tuple[dict, dict]:
    """Per ef: rankings keyed by query_id, and the recall/latency part of the point."""
    ranks, parts = {}, {}
    for ef in EF_GRID:
        r = run_queries(c, name, "text_vec", qv, search_params={"ef": ef})
        ranks[ef] = rankings(q, r.ids)
        parts[ef] = {"ef": ef, "recall@10": float(np.mean([recall_at_k(i, t) for i, t in zip(r.ids, truth)])),
                     "latency": summarise_latency(r.latencies_ms)}
        print(f"ef={ef} recall={parts[ef]['recall@10']:.3f}", flush=True)
    return ranks, parts


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--no-fill", action="store_true", help="skip the judge fill (score with cached labels only)")
    ap.add_argument("--fill-only", action="store_true", help="reuse data/work/b1b_rankings.json; no Milvus")
    ap.add_argument("--max-pairs", type=int, default=30_000, help="judge spend guard (each pair is a paid API call)")
    ap.add_argument("--monitor", action="store_true", help="item 1.4: one drift check, no sweep, no judging")
    ap.add_argument("--ef", type=int, default=None, help="monitor: production ef (default: b1a's)")
    ap.add_argument("--live-sample", type=int, default=200, help="monitor: live queries sampled")
    a = ap.parse_args(argv)
    if a.monitor:
        return monitor(a.ef, a.live_sample)
    n = a.sample or CANONICAL_N
    q, qv, _ = load_eval("eval5k", n)
    c = None
    if a.fill_only:
        saved = json.loads(RANKINGS.read_text())
        assert saved["n_queries"] == n, (saved["n_queries"], n)
        ranks = {int(ef): r for ef, r in saved["rankings"].items()}
        parts = {int(ef): p for ef, p in saved["parts"].items()}
        name = saved["collection"]
    else:
        c = client()
        products, tv, _, _ = load_corpus()
        name = ensure_collection(c, "prod_naive")
        truth = exact_ids("full_eval5k", tv, corpus_rows("prod_naive", products), qv, products)
        ranks, parts = sweep(c, name, q, qv, truth)
        ensure_parent(RANKINGS).write_text(
            json.dumps({"n_queries": n, "collection": name, "rankings": ranks, "parts": parts}))
    qr = qrels()
    fill = None
    if not a.no_fill:
        fill = judge.fill(anthropic.Anthropic(), ranks.values(), qr, q[["query_id", "query"]], load_corpus()[0],
                          max_pairs=a.max_pairs)
    points = []
    for ef in EF_GRID:
        qual = quality(ranks[ef], qr)
        points.append({**parts[ef], **qual})
        print(f"ef={ef} ndcg raw={qual['ndcg@10']:.4f} filled={qual['filled']['ndcg@10']:.4f} "
              f"judged={qual['filled']['judged@10']:.3f}", flush=True)
    write_result("b1b", {"collection": name, "points": points, "n_queries": n, "fill": fill,
                         "judgments": judge.fingerprint(),
                         "sample": a.sample is not None}, client=c)


if __name__ == "__main__":
    main()
