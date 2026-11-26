"""Showcase: concrete per-query examples for the slides, in place of a live UI (decided 23 Nov 2026).

  poc_vs_prod (§1): the poc50 queries, dense_text at B1a's ef, on poc_20k (the PoC) and prod_naive.
      Per query and side: top-10 cards (label and its source: ESCI or judge), filled nDCG@10, judged@10.
      Unjudged results are judged first (--no-fill skips); after new judging run bench.rescore.
      poc_20k holds one category, so `candidates_in_category` lists the queries from that category.
  narrow_filter (2.1): B2's <0.1% bucket on prod_tuned (HNSW, B2's ef) and f_1024_rabitq (nprobe 8):
      returned count and recall against the exact filtered top 10; cards where the two differ.
  latency_split (R5): each poc50 query embedded live (Qwen3 on this machine), then searched on prod_naive:
      embed / search / total ms, as an application would time them.

Never builds a collection: a missing or short one stops the run. Candidates are ranked for picking the
slide examples; every query is kept so the choice is visible.  -> results/showcase.json"""
import time

import anthropic
import numpy as np
import pandas as pd

from bench._common import (_src, client, corpus_rows, ensure_prices, load_corpus, load_eval, parser, qrels,
                           rankings, with_query_meta)
from bench.b2_filters import RABITQ, filtered_truth
from data.images import image_path
from vsearch import judge
from vsearch.collections import release_all, wait_loaded
from vsearch.config import K, POC_CATEGORY, ROOT
from vsearch.filters import make_filter_mix
from vsearch.load import level_start
from vsearch.metrics import judged_at_k, ndcg_at_k, recall_at_k
from vsearch.results import read_result, write_result
from vsearch.search import run_queries, summarise_latency

NARROW = "<0.1%"
RABITQ_NPROBE = 8
SPLIT_WARMUP = 5


class NotReady(RuntimeError):
    """A collection the showcase needs is missing or short; building one is a separate, deliberate step."""


def open_existing(c, name: str, n_rows: int) -> str:
    """Load one collection alone, refusing (never building) when it's missing or short."""
    release_all(c)
    if not c.has_collection(name):
        raise NotReady(f"{name}: missing; build it with its benchmark first")
    c.load_collection(name)
    got = c.query(name, filter="", output_fields=["count(*)"])[0]["count(*)"]
    if got != n_rows:
        raise NotReady(f"{name}: {got:,} rows, expected {n_rows:,}")
    wait_loaded(c, name, n_rows)
    return name


def label_maps(qr: pd.DataFrame, filled: pd.DataFrame, qids) -> dict[str, dict[str, tuple[str, str]]]:
    """{query_id: {product_id: (label, "esci" | "judge")}}; ESCI wins, as in filled_qrels."""
    qids = set(qids)
    esci = set(zip(qr.query_id, qr.product_id))
    f = filled[filled.query_id.isin(qids)]
    out: dict[str, dict] = {q: {} for q in qids}
    for q, p, lab in zip(f.query_id, f.product_id, f.label):
        out[q][p] = (lab, "esci" if (q, p) in esci else "judge")
    return out


def _price(v) -> float | None:
    return None if v is None or pd.isna(v) or v < 0 else round(float(v), 2)


def cards(ids: list[str], info: pd.DataFrame, labels: dict[str, tuple[str, str]],
          img=image_path) -> list[dict]:
    """One display card per returned id; info is products indexed by product_id."""
    out = []
    for p in ids:
        r = info.loc[p]
        lab, src = labels.get(p, (None, None))
        path = img(p)
        out.append({"product_id": p, "title": str(r.title)[:200], "brand": "" if pd.isna(r.brand) else str(r.brand),
                    "category": str(r.category), "price": _price(r.price), "label": lab, "label_source": src,
                    "image": str(path.relative_to(ROOT)) if path.exists() else None})
    return out


def side(ids: list[str], labels: dict[str, tuple[str, str]], info: pd.DataFrame) -> dict:
    flat = {p: l for p, (l, _) in labels.items()}
    return {"ndcg@10": ndcg_at_k(ids, flat), "judged@10": judged_at_k(ids, flat), "returned": len(ids),
            "results": cards(ids, info, labels)}


def rank_poc_vs_prod(rows: list[dict], min_judged: float = 0.9) -> list[str]:
    """Query ids where both sides are judged (>= min_judged), largest nDCG gap (either way) first."""
    ok = [r for r in rows if r["poc"]["judged@10"] >= min_judged and r["prod"]["judged@10"] >= min_judged]
    return [r["query_id"] for r in sorted(ok, key=lambda r: -abs(r["prod"]["ndcg@10"] - r["poc"]["ndcg@10"]))]


def rank_narrow(rows: list[dict]) -> list[str]:
    """Filter-mix keys where RaBitQ comes back short and HNSW full; shortest RaBitQ page first."""
    ok = [r for r in rows if r["rabitq"]["returned"] < K <= r["hnsw"]["returned"]]
    return [r["key"] for r in sorted(ok, key=lambda r: (r["rabitq"]["returned"], -r["hnsw"]["recall@10"]))]


def latency_split(c, name: str, texts: list[str], ef: int, encode) -> dict:
    """Per query: embed (encode one query), search (one Milvus call), total (both, back to back)."""
    params = {"params": {"ef": ef}}

    def one(t: str) -> tuple[float, float, float]:
        t0 = time.perf_counter()
        v = encode([t])[0]
        t1 = time.perf_counter()
        c.search(name, data=[np.asarray(v, dtype=np.float32).tolist()], anns_field="text_vec",
                 search_params=params, limit=K)
        t2 = time.perf_counter()
        return (t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000

    for t in texts[:SPLIT_WARMUP]:
        one(t)
    e, s, tot = map(np.array, zip(*(one(t) for t in texts)))
    return {"embed": summarise_latency(e), "search": summarise_latency(s), "total": summarise_latency(tot),
            "embed_share_p50": float(np.median(e / tot)), "ef": ef, "collection": name, "n": len(texts),
            "per_query_ms": [{"embed": float(a), "search": float(b), "total": float(t)} for a, b, t in zip(e, s, tot)]}


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=2000, help="judge spend guard")
    a = ap.parse_args(argv)
    c = client()
    products, docs, _, _ = load_corpus()
    info = products.set_index("product_id")
    qr = qrels()
    ef1, ef2 = read_result("b1a")["payload"]["ef"], read_result("b2")["payload"]["poc_ef"]
    n_full, n_poc = len(products), len(corpus_rows("poc_20k", products))

    # §1 rankings, one collection at a time
    q, qv, _ = load_eval("poc50")
    q = with_query_meta(q, ["top_category"])
    if a.sample:
        q, qv = q.head(a.sample), qv[:a.sample]
    name = open_existing(c, "poc_20k", n_poc)
    poc = rankings(q, run_queries(c, name, "text_vec", qv, search_params={"ef": ef1}).ids)
    name = open_existing(c, "prod_naive", n_full)
    prod = rankings(q, run_queries(c, name, "text_vec", qv, search_params={"ef": ef1}).ids)
    print("§1 rankings done", flush=True)

    # R5 on prod_naive, from a level start
    level_start(c, name, n_full)
    from vsearch.embed import TextEmbedder
    split = latency_split(c, name, q["query"].tolist(), ef1, TextEmbedder().encode_queries)
    print(f"R5 embed p50 {split['embed']['p50']:.1f} ms, search p50 {split['search']['p50']:.1f} ms", flush=True)

    # 2.1: B2's mix (same queries and arguments), narrowest bucket only
    q5, qv5, _ = load_eval("eval5k")
    q5 = with_query_meta(q5, ["top_category"])
    mix = make_filter_mix(q5, products, qr, per_bucket=200)
    pos = {qid: i for i, qid in enumerate(q5.query_id)}
    mix = mix[mix.bucket == NARROW].assign(qpos=lambda d: d.query_id.map(pos)).reset_index(drop=True)
    if a.sample:
        mix = mix.head(a.sample)
    mqv = qv5[mix.qpos.to_numpy()]
    truth = filtered_truth(mix, mqv, docs, products)
    name = open_existing(c, "prod_tuned", n_full)
    drift = ensure_prices(c, name, products)
    hnsw = run_queries(c, name, "text_vec", mqv, filters=mix.expr.tolist(), search_params={"ef": ef2}).ids
    name = open_existing(c, RABITQ, n_full)
    rbq = run_queries(c, name, "text_vec", mqv, filters=mix.expr.tolist(),
                      search_params={"nprobe": RABITQ_NPROBE}).ids
    release_all(c)
    print("2.1 runs done", flush=True)

    fill = None
    if not a.no_fill:
        text = pd.read_parquet(_src("queries.parquet"), columns=["query_id", "query"])
        fill = judge.fill(anthropic.Anthropic(), [poc, prod], qr, text, products, max_pairs=a.max_pairs)
    labels = label_maps(qr, judge.filled_qrels(qr), q.query_id)

    rows1 = [{"query_id": qid, "query": txt, "top_category": cat, "poc": side(poc[qid], labels[qid], info),
              "prod": side(prod[qid], labels[qid], info)}
             for qid, txt, cat in zip(q.query_id, q["query"], q.top_category)]
    for r in rows1:
        r["ndcg_gap"] = r["prod"]["ndcg@10"] - r["poc"]["ndcg@10"]
    rows2 = []  # aligned with mix and truth
    for i, m in mix.iterrows():
        f = m["filter"]
        r = {"key": f"{NARROW}|{m.query_id}", "query_id": m.query_id, "query": q5["query"].iloc[m.qpos],
             "expr": m.expr, "filter": {k: v for k, v in vars(f).items() if v is not None},
             "n_matches": int(m.n_matches), "n_rows": n_full, "selectivity": float(m.selectivity),
             "exact": {"returned": len(truth[i])}}
        for key, ids in (("hnsw", hnsw[i]), ("rabitq", rbq[i])):
            r[key] = {"returned": len(ids), "recall@10": recall_at_k(ids, truth[i]), "ids": ids}
        rows2.append(r)
    picks2 = rank_narrow(rows2)
    top = set(picks2[:10])
    for r, t in zip(rows2, truth):  # cards only where they'd go on a slide
        if r["key"] in top:
            for key in ("hnsw", "rabitq"):
                r[key]["results"] = cards(r[key]["ids"], info, {})
            r["exact"]["results"] = cards(t, info, {})

    write_result("showcase", {
        "poc_vs_prod": {"ef": ef1, "collections": ["poc_20k", "prod_naive"], "queries": "poc50", "mode": "dense_text",
                        "poc_category": POC_CATEGORY, "candidates": rank_poc_vs_prod(rows1),
                        # the PoC slice holds one category: elsewhere the gap is mostly missing products
                        "candidates_in_category": rank_poc_vs_prod([r for r in rows1 if r["top_category"] == POC_CATEGORY]),
                        "rows": rows1},
        "narrow_filter": {"bucket": NARROW, "hnsw": {"collection": "prod_tuned", "ef": ef2},
                          "rabitq": {"collection": RABITQ, "nprobe": RABITQ_NPROBE},
                          "short_rate": {"hnsw": float(np.mean([r["hnsw"]["returned"] < K for r in rows2])),
                                         "rabitq": float(np.mean([r["rabitq"]["returned"] < K for r in rows2]))},
                          "price_drift_restored": drift, "candidates": picks2, "rows": rows2},
        "latency_split": split, "fill": fill, "judgments": judge.fingerprint(), "sample": a.sample is not None},
        client=c)


if __name__ == "__main__":
    main()
