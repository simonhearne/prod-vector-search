"""B5: merchandising rules, on SYNTHETIC margins, campaigns and hide flags (vsearch.merch). README row 2.4.
  M1 does a campaign boost reach the page?   boost weight x request limit, few vs many segments
  M2 what does margin cost in relevance?     multiply / sum / proportional boosts, nDCG vs margin@10
  M3 do bury and hide hold?                  leak into the top 10, page fill, latency
  M4 do rules change the winning config?     dense vs hybrid, rules applied in the application
Every ranking is cached (data/work/b5_rankings.json); scoring is offline, so bench.rescore can redo it.
ef is fixed at 512 in every search: only `limit` and the rule vary. Latency is relative to the unboosted base.

    uv run python -m bench.b5_merch [--sample 20] [--n 300] [--no-fill] [--score-only]   -> results/b5.json"""
import json
from collections import Counter

import anthropic
import numpy as np
import pandas as pd

from bench._common import (CANONICAL_N, _src, client, ensure_collection, ensure_parent, load_corpus, load_eval,
                           parser, qrels, quality, rankings)
from vsearch import judge
from vsearch.collections import release_all
from vsearch.config import DATA_WORK, K
from vsearch.load import level_start
from vsearch.merch import CAMPAIGN, merch_fields
from vsearch.metrics import ndcg_at_k
from vsearch.results import write_result
from vsearch.rules import apply_app, boost, combined, leak, margin_chain, page_margin, reach, slot_share
from vsearch.search import run_hybrid, run_queries, summarise_latency
from vsearch.stats import paired_ci

RANKINGS = DATA_WORK / "b5_rankings.json"
MAIN, MANY = "prod_merch", "prod_merch_pk"  # few vs many segments
EF, DEEP, RULE_LIMIT = 512, 500, 200
M1_WEIGHTS, M1_LIMITS = [1.05, 1.1, 1.2, 1.5, 2.0], [10, 50, 100, 200, 500]
M1_FILLED = (10, 200)
MARGIN_CUT = 0.4
M2 = {"Multiply": [1.02, 1.05, 1.1, 1.2, 1.5], "Sum": [0.005, 0.01, 0.02, 0.05, 0.1],
      "proportional": [0.02, 0.05, 0.1, 0.2, 0.5]}
BURY_WEIGHTS, BURY_LIMITS, HIDE_LISTS = [0.9, 0.7, 0.5], [10, 100], [10, 1000, 10_000]
M4_CAMPAIGN, M4_MARGIN = 1.2, 1.1
IN_CAMPAIGN, HIGH_MARGIN = f'array_contains(campaigns, "{CAMPAIGN}")', f"margin >= {MARGIN_CUT}"


def hide_ids(base: dict[str, list[str]], hidden: set[str], n: int) -> list[str]:
    """The hidden products seen most often in the base top 100s, padded with the rest by id."""
    seen = Counter(p for ids in base.values() for p in ids[:100] if p in hidden)
    return sorted(hidden, key=lambda p: (-seen[p], p))[:n]


def rankings_path(sample: bool):
    """Sample runs cache elsewhere, so a smoke run can't replace the rankings the committed result is scored from."""
    return RANKINGS.with_name("b5_rankings.sample.json") if sample else RANKINGS


def _filled(key: str) -> bool:
    kind, *rest = key.split("|")
    if kind == "m1":
        return rest[0] == MAIN and int(rest[2]) in M1_FILLED
    return kind in ("m2", "m4") or key == f"base|{MAIN}"


def fill_keys(cache: dict) -> list[str]:
    return [k for k in cache["runs"] if _filled(k)]


def _run_queries_block(c, cache: dict, name: str, q, qv, hidden: set[str]) -> None:
    runs = cache["runs"]

    def go(key: str, limit: int = K, page: int = K, filt: str | None = None, **extra):
        r = run_queries(c, name, "text_vec", qv, k=limit, search_params={"ef": EF}, extra=extra, page=page,
                        filters=[filt] * len(qv) if filt else None)
        runs[key] = {"ids": rankings(q, r.ids), "latency": summarise_latency(r.latencies_ms)}
        print(f"{key}: p50 {runs[key]['latency']['p50']:.1f} ms", flush=True)
        return r

    go(f"base|{name}")
    go(f"base500|{name}", limit=DEEP, page=DEEP)
    for w in M1_WEIGHTS:
        for lim in M1_LIMITS:
            go(f"m1|{name}|{w}|{lim}", limit=lim, ranker=boost(IN_CAMPAIGN, w))
    if name == MAIN:
        for mode, weights in M2.items():
            for w in weights:
                rule = ({"function_chains": margin_chain(w)} if mode == "proportional"
                        else {"ranker": combined([boost(HIGH_MARGIN, w)], mode)})
                go(f"m2|{mode}|{w}", limit=RULE_LIMIT, **rule)
        for w in BURY_WEIGHTS:
            for lim in BURY_LIMITS:
                go(f"m3|bury|{w}|{lim}", limit=lim, ranker=boost("hidden == true", w))
        go("m3|hide|flag", filt="hidden == false")
        base = runs[f"base500|{name}"]["ids"]
        for n in HIDE_LISTS:
            ids = hide_ids(base, hidden, n)
            go(f"m3|hide|ids{n}", filt="product_id not in [" + ",".join(f'"{p}"' for p in ids) + "]")
    go(f"base_end|{name}")


def _run_m4(c, cache: dict, q, qv, merch: pd.DataFrame) -> None:
    on_campaign, high = merch.campaigns.map(bool), merch.margin >= MARGIN_CUT
    mult = {p: (M4_CAMPAIGN if a else 1.0) * (M4_MARGIN if b else 1.0)
            for p, a, b in zip(merch.index, on_campaign, high) if a or b}
    for cfg in ("dense_text", "hybrid_rrf"):
        r = (run_queries(c, MAIN, "text_vec", qv, k=RULE_LIMIT, search_params={"ef": EF}) if cfg == "dense_text"
             else run_hybrid(c, MAIN, qv, q["query"].tolist(), k=RULE_LIMIT, dense_params={"ef": EF}))
        lat = summarise_latency(r.latencies_ms)
        on = [apply_app(list(zip(i, s)), mult)[:K] for i, s in zip(r.ids, r.scores)]
        cache["runs"][f"m4|{cfg}|off"] = {"ids": rankings(q, [i[:K] for i in r.ids]), "latency": lat}
        cache["runs"][f"m4|{cfg}|on"] = {"ids": rankings(q, on), "latency": lat}
        # one weight is a different strength on each score scale
        cache.setdefault("score_spread", {})[cfg] = float(np.median([s[99] / s[0] for s in r.scores if len(s) >= 100]))
        print(f"m4 {cfg} done", flush=True)


def _rel(run: dict, base: dict) -> dict:
    return {f"{p}_rel": run["latency"][p] / base["latency"][p] for p in ("p50", "p99")}


def score(cache: dict, qr: pd.DataFrame, merch: pd.DataFrame, query_ids: list[str]) -> dict:
    """The m1-m4 blocks from cached rankings. merch: merch_fields indexed by product_id."""
    if list(cache["query_ids"]) != list(query_ids):
        raise ValueError("cached query_ids differ from the requested ones; rerun without --score-only")
    runs = cache["runs"]
    qids = list(query_ids)
    qr = qr[qr.query_id.isin(set(qids))]
    campaign = set(merch.index[merch.campaigns.map(bool)])
    hidden = set(merch.index[merch.hidden])
    margin = merch.margin.to_dict()
    pages = lambda key: [runs[key]["ids"][q][:K] for q in qids]

    def qual(key: str) -> dict:
        f = quality({q: runs[key]["ids"][q] for q in qids}, qr)["filled"]
        return {k: f[k] for k in ("ndcg@10", "ndcg@10_ci95", "judged@10")}

    def margin10(key: str) -> float:
        return float(np.mean([m for p in pages(key) if (m := page_margin(p, margin)) is not None]))

    names = [k.split("|")[1] for k in runs if k.startswith("base|")]
    eligible = {n: [any(p in campaign for p in runs[f"base500|{n}"]["ids"][q]) for q in qids] for n in names}
    drift = {n: runs[f"base_end|{n}"]["latency"]["p50"] / runs[f"base|{n}"]["latency"]["p50"]
             for n in names if f"base_end|{n}" in runs}
    m1_rows, m2_rows, bury, hide = [], [], [], []
    for key in runs:
        kind, *rest = key.split("|")
        if kind == "m1":
            name, w, lim = rest[0], float(rest[1]), int(rest[2])
            m1_rows.append({"collection": name, "weight": w, "limit": lim, "reach": reach(pages(key), campaign, eligible[name]),
                            "campaign_slots": slot_share(pages(key), campaign),
                            **_rel(runs[key], runs[f"base|{name}"]), **(qual(key) if _filled(key) else {})})
        elif kind == "m2":
            m2_rows.append({"mode": rest[0], "weight": float(rest[1]), "margin@10": margin10(key), **qual(key)})
        elif kind == "m3" and rest[0] == "bury":
            bury.append({"weight": float(rest[1]), "limit": int(rest[2]), "leak": leak(pages(key), hidden),
                         **_rel(runs[key], runs[f"base|{MAIN}"])})
        elif kind == "m3":
            target = hidden if rest[1] == "flag" else set(
                hide_ids(runs[f"base500|{MAIN}"]["ids"], hidden, int(rest[1].removeprefix("ids"))))
            hide.append({"kind": rest[1], "n_hidden": len(target), "leak": leak(pages(key), target),
                         "page_fill": float(np.mean([len(p) / K for p in pages(key)])),
                         **_rel(runs[key], runs[f"base|{MAIN}"])})
    base_key = f"base|{MAIN}"
    return {"m1": {"rows": m1_rows, "segments": cache.get("segments", {}), "level_quiet": cache.get("level", {}),
                   "n_eligible": {n: int(sum(e)) for n, e in eligible.items()},
                   "base_reach": {n: reach(pages(f"base|{n}"), campaign, eligible[n]) for n in names},
                   "base_campaign_slots": {n: slot_share(pages(f"base|{n}"), campaign) for n in names},
                   "base_drift_p50": drift},
            "m2": {"base": {"margin@10": margin10(base_key), **qual(base_key)}, "rows": m2_rows},
            "m3": {"base": {"leak": leak(pages(base_key), hidden)}, "bury": bury, "hide": hide},
            "m4": _score_m4(runs, qids, judge.filled_qrels(qr), qual, margin10, campaign, cache.get("score_spread", {}))}


def _score_m4(runs, qids, labels, qual, margin10, campaign, spread) -> dict:
    cfgs = sorted({k.split("|")[1] for k in runs if k.startswith("m4|")})
    lab = {q: dict(zip(g.product_id, g.label)) for q, g in labels.groupby("query_id")}
    nd = {(c, s): np.array([ndcg_at_k(runs[f"m4|{c}|{s}"]["ids"][q], lab.get(q, {})) for q in qids])
          for c in cfgs for s in ("off", "on")}
    order = {s: sorted(cfgs, key=lambda c: -nd[c, s].mean()) for s in ("off", "on")}

    def cell(key: str) -> dict:  # reach and slots say how strongly the same weights acted on each score scale
        pages = [runs[key]["ids"][q][:K] for q in qids]
        return {**qual(key), "margin@10": margin10(key), "reach": reach(pages, campaign, [True] * len(pages)),
                "campaign_slots": slot_share(pages, campaign)}

    return {"configs": {c: {s: cell(f"m4|{c}|{s}") for s in ("off", "on")} for c in cfgs},
            "on_minus_off": {c: list(paired_ci(nd[c, "on"] - nd[c, "off"])) for c in cfgs},
            "hybrid_minus_dense": {s: list(paired_ci(nd["hybrid_rrf", s] - nd["dense_text", s])) for s in ("off", "on")},
            "order_off": order["off"], "order_on": order["on"], "flip": order["off"] != order["on"],
            "score_rank100_over_rank1": spread, "rules": {"campaign": M4_CAMPAIGN, "high_margin": M4_MARGIN}}


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--n", type=int, default=CANONICAL_N, help="queries (300 if the judge guard refuses)")
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=30_000, help="judge spend guard")
    ap.add_argument("--score-only", action="store_true", help="re-score the cached rankings, no Milvus")
    a = ap.parse_args(argv)
    q, qv, _ = load_eval("eval5k", a.sample or a.n)
    qids = q.query_id.tolist()
    products = load_corpus()[0]
    merch = merch_fields(products).set_axis(products.product_id.to_numpy())
    qr, c = qrels(), None
    if a.score_only:
        cache = json.loads(rankings_path(a.sample is not None).read_text())
    else:
        c = client()
        cache = {"n": len(q), "query_ids": qids, "segments": {}, "level": {}, "runs": {}}
        for name in (MAIN, MANY):
            release_all(c)
            ensure_collection(c, name)
            cache["level"][name] = level_start(c, name, len(products))[1]
            cache["segments"][name] = sum(getattr(s.state, "name", s.state) == "Sealed" for s in c.list_loaded_segments(name))
            _run_queries_block(c, cache, name, q, qv, set(merch.index[merch.hidden]))
            if name == MAIN:
                _run_m4(c, cache, q, qv, merch)
        release_all(c)
        ensure_parent(rankings_path(a.sample is not None)).write_text(json.dumps(cache))
    fill = None
    if not a.no_fill:
        text = pd.read_parquet(_src("queries.parquet"), columns=["query_id", "query"])
        top = [{k: v[:K] for k, v in cache["runs"][key]["ids"].items()} for key in fill_keys(cache)]
        fill = judge.fill(anthropic.Anthropic(), top, qr, text, products, max_pairs=a.max_pairs)
    write_result("b5", {"synthetic": True, "ef": EF, "n": len(q), "campaign_share": float(merch.campaigns.map(bool).mean()),
                        "hidden_share": float(merch.hidden.mean()), "margin_cut": MARGIN_CUT,
                        **score(cache, qr, merch, qids), "fill": fill, "judgments": judge.fingerprint(),
                        "sample": a.sample is not None}, client=c)


if __name__ == "__main__":
    main()
