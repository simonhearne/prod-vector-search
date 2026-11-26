"""B2a/B2b: filtered search by selectivity bucket. PoC (20k, unfiltered) vs prod_naive vs prod_tuned
(scalar indexes + partition key) vs prod_tuned with per-bucket ef, plus IVF_RABITQ (§3's memory winner)
per nprobe. Recall is against exact *filtered* search; short_rate = share of result lists under K rows
(the mix guarantees >= K matches). unfiltered_tax (item 2.2): does the tuned layout slow unfiltered
queries, on level-started collections?  -> results/b2.json, results/scorecard_prod.json"""
import numpy as np
import pandas as pd

from bench._common import (DATA_WORK, client, corpus_rows, ensure_collection, ensure_parent, ensure_prices, exact_ids,
                           load_corpus, load_eval, parser, qrels, rankings, with_query_meta)
from vsearch.collections import frontier_specs, release_all
from vsearch.config import K
from vsearch.load import level_start
from vsearch.filters import BUCKETS, make_filter_mix
from vsearch.metrics import recall_at_k, summarise_quality
from vsearch.results import read_result, write_result
from vsearch.search import run_queries, summarise_latency

EF_GRID = [10, 16, 24, 32, 48, 64, 128, 256]
NPROBE_GRID = [8, 16, 32, 64]
RABITQ = "f_1024_rabitq"
TAX_EFS, TAX_QUERIES, TAX_PASSES = [32, 192], 300, 3


def short_rate(ids: list[list[str]], k: int = K) -> float:
    return float(np.mean([len(i) < k for i in ids])) if ids else 0.0


DRIFT: dict[str, int] = {}
LEVEL_QUIET: dict[str, list[bool]] = {}


def _open(c, spec, n_rows: int, products=None) -> str:
    """Load one collection alone, with parquet prices (prod_tuned is written by B4), from a level start."""
    release_all(c)
    name = ensure_collection(c, spec)
    if name == "prod_tuned" and products is not None:
        DRIFT[name] = DRIFT.get(name, 0) + ensure_prices(c, name, products)
    LEVEL_QUIET.setdefault(name, []).append(level_start(c, name, n_rows)[1])
    return name


def unfiltered_tax(c, products, docs, order: tuple[str, ...] = ("naive", "tuned")) -> dict:
    q, qv, _ = load_eval("eval5k", TAX_QUERIES)
    truth = exact_ids("tax_unf", docs, corpus_rows("prod_naive", products), qv, products)
    out = {f"ef={ef}": {"naive": [], "tuned": [], "recall": {}} for ef in TAX_EFS}
    for key in order:
        spec = {"naive": "prod_naive", "tuned": "prod_tuned"}[key]
        name = _open(c, spec, len(products), products)
        run_queries(c, name, "text_vec", qv, search_params={"ef": max(TAX_EFS)}, warmup=0)  # warm pass, discarded
        for ef in TAX_EFS:
            for _ in range(TAX_PASSES):
                r = run_queries(c, name, "text_vec", qv, search_params={"ef": ef})
                out[f"ef={ef}"][key].append(summarise_latency(r.latencies_ms))
            out[f"ef={ef}"]["recall"][key] = float(np.mean([recall_at_k(i, t) for i, t in zip(r.ids, truth)]))
            print(f"tax {key} ef={ef} p50={out[f'ef={ef}'][key][-1]['p50']:.1f}", flush=True)
    out["order"] = list(order)
    return out


def filtered_truth(mix, qv, docs, products):
    rows = corpus_rows("prod_naive", products)
    truth = [None] * len(mix)
    for expr, g in mix.groupby("expr", sort=False):
        mask = g["filter"].iloc[0].mask(products.set_index("row_id").loc[rows].reset_index())
        ids = exact_ids("filt", docs, rows, qv[g.index.to_numpy()], products, mask=mask)
        for i, t in zip(g.index, ids):
            truth[i] = t
    return truth


def measure(c, name, qv, filters, truth, ef=None, params=None):
    r = run_queries(c, name, "text_vec", qv, filters=filters, search_params=params or {"ef": ef})
    return r, {"recall@10": float(np.mean([recall_at_k(i, t) for i, t in zip(r.ids, truth)])),
               "short_rate": short_rate(r.ids), "latency": summarise_latency(r.latencies_ms), **(params or {"ef": ef})}


def main(argv=None):
    a = parser(__doc__).parse_args(argv)
    c = client()
    poc_ef = read_result("poc_config")["payload"]["ef"]
    products, docs, _, _ = load_corpus()
    q, qv, _ = load_eval("eval5k")
    q = with_query_meta(q, ["top_category"])
    qr = qrels()
    mix = make_filter_mix(q, products, qr, per_bucket=a.sample or 200)
    mix.drop(columns="filter").to_parquet(ensure_parent(DATA_WORK / "filter_mix.parquet"), index=False)
    pos = {qid: i for i, qid in enumerate(q.query_id)}
    mix = mix.assign(qpos=mix.query_id.map(pos)).reset_index(drop=True)
    mqv = qv[mix.qpos.to_numpy()]
    mq = q.iloc[mix.qpos.to_numpy()].reset_index(drop=True)
    truth = filtered_truth(mix, mqv, docs, products)
    poc_truth = exact_ids("poc_mix", docs, corpus_rows("poc_20k", products), mqv, products)

    out, naive_ids = {b[0]: {} for b in BUCKETS}, {}
    rabitq = next(sp for sp in frontier_specs() if sp.name == RABITQ)
    for spec in ["poc_20k", "prod_naive", "prod_tuned", rabitq]:
        name = _open(c, spec, len(products) if spec != "poc_20k" else len(corpus_rows("poc_20k", products)), products)
        for b, g in mix.groupby("bucket", sort=False):
            i = g.index.to_numpy()
            sub_qv, sub_truth = mqv[i], [truth[j] for j in i]
            o = out[b]
            o.update(n=len(g), selectivity_median=float(g.selectivity.median()))
            if spec == "poc_20k":
                _, o["poc_unfiltered_baseline"] = measure(c, name, sub_qv, None, [poc_truth[j] for j in i], poc_ef)
            elif spec is rabitq:
                o["rabitq"] = {f"nprobe={v}": measure(c, name, sub_qv, g.expr.tolist(), sub_truth,
                                                      params={"nprobe": v})[1] for v in NPROBE_GRID}
            elif spec == "prod_naive":
                r, o["naive"] = measure(c, name, sub_qv, g.expr.tolist(), sub_truth, poc_ef)
                naive_ids[b] = (mq.iloc[i], r)
            else:
                _, o["tuned"] = measure(c, name, sub_qv, g.expr.tolist(), sub_truth, poc_ef)
                for ef in EF_GRID:
                    _, m = measure(c, name, sub_qv, g.expr.tolist(), sub_truth, ef)
                    if m["recall@10"] >= 0.95 or ef == EF_GRID[-1]:
                        o["tuned_params"] = m
                        break
            print(f"{name} {b} done", flush=True)
    tax = unfiltered_tax(c, products, docs)
    write_result("b2", {"buckets": out, "poc_ef": poc_ef, "unfiltered_tax": tax, "rabitq_collection": RABITQ,
                        "nprobe_grid": NPROBE_GRID, "price_drift_restored": DRIFT, "level_quiet": LEVEL_QUIET, "sample": a.sample is not None}, client=c)

    # §0 prod conditions: naive prod, PoC ef, filter mix, equal weight per bucket
    n_min = min(len(r.ids) for _, r in naive_ids.values())
    quals = [summarise_quality(rankings(mqb, r.ids), qr) for mqb, r in naive_ids.values()]
    lat = np.concatenate([r.latencies_ms[:n_min] for _, r in naive_ids.values()])
    write_result("scorecard_prod", {
        "ndcg@10": float(np.mean([x["ndcg@10"] for x in quals])),
        "judged@10": float(np.mean([x["judged@10"] for x in quals])),
        "recall@10": float(np.mean([out[b]["naive"]["recall@10"] for b in naive_ids])),
        "latency": summarise_latency(lat), "ef": poc_ef, "collection": "prod_naive",
        "queries": "eval5k + filter mix (equal weight per bucket)", "sample": a.sample is not None}, client=c)


if __name__ == "__main__":
    main()
