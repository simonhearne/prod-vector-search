"""B1a: does the small PoC query set rank retrieval configs the way real queries do?
Configs on prod_naive (450k): dense_text, dense_image (SigLIP text->image), hybrid_rrf.
Canonical nDCG is judge-filled. Also: "power" (how often a random 50-query set picks the
1,000-query winner) and "wording" (curated paraphrases vs their own ESCI sources, paired). Also writes the §0 'PoC conditions' scorecard cells (raw, not on stage).
  -> results/b1a.json, results/scorecard_poc.json"""
import json

import anthropic
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bench._common import (ALL_POC_SETS, CANONICAL_N, POC_SET, _src, client, corpus_rows, ensure_collection,
                           ensure_parent, exact_ids, load_corpus, load_eval, parser, qrels, quality, rankings)
from vsearch import judge
from vsearch.config import DATA_WORK
from vsearch.metrics import ndcg_at_k, recall_at_k, summarise_quality
from vsearch.stats import draw_power, paired_ci
from vsearch.results import read_result, write_result
from vsearch.search import run_hybrid, run_queries, summarise_latency

CONFIGS = ["dense_text", "dense_image", "hybrid_rrf"]
RANKINGS = DATA_WORK / "b1a_rankings.json"
INTERACTIONS = [("dense_image", "dense_text"), ("hybrid_rrf", "dense_text"), ("dense_image", "hybrid_rrf")]


def per_query_ndcg(ranks: dict[str, dict[str, list[str]]], qr: pd.DataFrame, qids: list[str]) -> np.ndarray:
    """queries x CONFIGS matrix of nDCG@10."""
    lab = {q: dict(zip(g.product_id, g.label)) for q, g in qr.groupby("query_id")}
    return np.array([[ndcg_at_k(ranks[c][q], lab.get(q, {})) for c in CONFIGS] for q in qids])


def power_block(ranks: dict, qr: pd.DataFrame) -> dict:
    qids = sorted(ranks[CONFIGS[0]])
    out = {"n_queries": len(qids), "draw_size": 50, "draws": 20_000, "configs": CONFIGS}
    for label, labels in [("filled", judge.filled_qrels(qr)), ("raw", qr)]:
        M = per_query_ndcg(ranks, labels, qids)
        dp = draw_power(M, n=50, draws=20_000)
        w = dp["winner"]
        out[label] = {"full_ndcg": dict(zip(CONFIGS, map(float, M.mean(0)))), "draw_power": dp,
                      "paired_vs_winner": {CONFIGS[k]: list(paired_ci(M[:, w] - M[:, k]))
                                           for k in range(len(CONFIGS)) if k != w}}
    return out


def wording_block(orig: dict, para: dict, qr: pd.DataFrame) -> dict:
    """orig/para: per-config rankings for poc50 and curated, both keyed by the ESCI source query_id."""
    qids = sorted(set(orig[CONFIGS[0]]) & set(para[CONFIGS[0]]))
    out = {"n": len(qids)}
    for label, labels in [("filled", judge.filled_qrels(qr)), ("raw", qr)]:
        O, P = per_query_ndcg(orig, labels, qids), per_query_ndcg(para, labels, qids)
        ix = {c: i for i, c in enumerate(CONFIGS)}
        out[label] = {"wording_effect": {c: list(paired_ci(P[:, ix[c]] - O[:, ix[c]])) for c in CONFIGS},
                      "interaction": {f"{a}-{b}": list(paired_ci((P[:, ix[a]] - P[:, ix[b]]) - (O[:, ix[a]] - O[:, ix[b]])))
                                      for a, b in INTERACTIONS}}
    return out


def run_config(c, name, cfg, q, tv, sv, ef):
    if cfg == "dense_text":
        return run_queries(c, name, "text_vec", tv, search_params={"ef": ef})
    if cfg == "dense_image":
        return run_queries(c, name, "image_vec", sv, search_params={"ef": ef},
                           filters=["has_image == true"] * len(sv))
    if cfg == "hybrid_rrf":
        return run_hybrid(c, name, tv, q["query"].tolist(), dense_params={"ef": ef})
    raise ValueError(cfg)


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=30_000, help="judge spend guard (each pair is a paid API call)")
    a = ap.parse_args(argv)
    c = client()
    poc_ef = read_result("poc_config")["payload"]["ef"]
    ef = max(64, poc_ef)
    qr = qrels()
    name = ensure_collection(c, "prod_naive")
    n_eval = a.sample or CANONICAL_N
    ranks, lat = {}, {}
    for set_name, n in [*[(ps, None) for ps in ALL_POC_SETS], ("eval5k", n_eval)]:
        q, tv, sv = load_eval(set_name, n)
        ranks[set_name], lat[set_name] = {}, {}
        for cfg in CONFIGS:
            r = run_config(c, name, cfg, q, tv, sv, ef)
            ranks[set_name][cfg] = rankings(q, r.ids)
            lat[set_name][cfg] = summarise_latency(r.latencies_ms)
            print(f"{set_name} {cfg} done", flush=True)
    ensure_parent(RANKINGS).write_text(json.dumps(ranks))
    fill = None
    if not a.no_fill:  # curated rankings are keyed by their ESCI source query_id and judged against it
        text = pd.read_parquet(_src("queries.parquet"), columns=["query_id", "query"])
        fill = judge.fill(anthropic.Anthropic(), [r for s in ranks.values() for r in s.values()], qr, text,
                          load_corpus()[0], max_pairs=a.max_pairs)
    sets = {s: {cfg: {**quality(ranks[s][cfg], qr), "latency": lat[s][cfg]} for cfg in CONFIGS} for s in ranks}
    nd = lambda s, k: sets[s][k]["filled"]["ndcg@10"]
    rank = {s: sorted(CONFIGS, key=lambda k: -nd(s, k)) for s in sets}
    rho = {ps: float(spearmanr([nd(ps, k) for k in CONFIGS], [nd("eval5k", k) for k in CONFIGS]).statistic)
           for ps in ALL_POC_SETS}
    write_result("b1a", {"collection": name, "ef": ef, "sets": sets, "poc_sets": ALL_POC_SETS,
                         "poc_set": POC_SET, "rank": rank, "spearman_vs_eval5k": rho,
                         "n": {s: sets[s]["dense_text"]["n_queries"] for s in sets},
                         "power": power_block(ranks["eval5k"], qr),
                         "wording": wording_block(ranks["poc50"], ranks["curated"], qr), "fill": fill,
                         "judgments": judge.fingerprint(),
                         "head50_note": "head proxy: eligible, not ESCI-hard, <=3 words, no digits",
                         "sample": a.sample is not None}, client=c)

    # §0 PoC conditions: dense_text, poc_20k, PoC ef, each PoC query set, unfiltered
    products, docs, _, _ = load_corpus()
    poc = ensure_collection(c, "poc_20k")
    cells = {}
    for ps in ALL_POC_SETS:
        q_ps, tv_ps, _ = load_eval(ps)
        r = run_queries(c, poc, "text_vec", tv_ps, search_params={"ef": poc_ef})
        truth = exact_ids(f"poc_{ps}", docs, corpus_rows("poc_20k", products), tv_ps, products)
        cells[ps] = {**summarise_quality(rankings(q_ps, r.ids), qr),
                     "recall@10": float(np.mean([recall_at_k(i, t) for i, t in zip(r.ids, truth)])),
                     "latency": summarise_latency(r.latencies_ms)}
    write_result("scorecard_poc", {**cells[POC_SET], "by_set": cells, "ef": poc_ef, "collection": poc,
                                   "queries": POC_SET, "sample": a.sample is not None}, client=c)

if __name__ == "__main__":
    main()
