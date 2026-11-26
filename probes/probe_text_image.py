"""B1a follow-up: what if the candidates include text + image fused? B1a's hybrid is text + BM25, so the 50-query
"power" result rests on three close configs. Fuses text and image top 100s with RRF (k=60, 100 per arm, as run_hybrid
does) offline; the top 100s are cached in data/work/deep_rankings.json (fetched from prod_naive if missing, ef as B1a).
Exact RRF ties keep the text result first; Milvus may order them differently, so `ties_image_first` gives the other
order. Latency isn't measured.

    python -m probes.probe_text_image --fill   # judge the fused top 10s the pool lacks, then run bench.rescore
    python -m probes.probe_text_image          # score at B1a's pool"""
import json
import sys
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from pymilvus.client.types import LoadState

from bench._common import CANONICAL_N, _src, client, load_corpus, load_eval, qrels, rankings
from bench.b1a_eval_sets import CONFIGS, RANKINGS, per_query_ndcg
from vsearch import judge
from vsearch.config import DATA_WORK
from vsearch.results import read_result
from vsearch.search import CANDIDATES, run_queries
from vsearch.stats import draw_power, paired_ci

SETS = {"poc50": None, "curated": None, "eval5k": CANONICAL_N}
DEEP, MAX_PAIRS = DATA_WORK / "deep_rankings.json", 1_500
NEW, ALL = "text_image_rrf", [*CONFIGS, "text_image_rrf"]


def rrf(*lists: list[str], k: int = 60, n: int = 10) -> list[str]:
    s: dict[str, float] = {}
    for ids in lists:
        for i, p in enumerate(ids):
            s[p] = s.get(p, 0.0) + 1 / (k + i + 1)
    return sorted(s, key=lambda p: -s[p])[:n]


def fused(deep: dict, first="dense_text", second="dense_image") -> dict:
    return {q: rrf(deep[first][q], deep[second][q]) for q in deep["dense_text"]}


def deep_rankings(ef: int) -> dict:
    if DEEP.exists():
        return json.loads(DEEP.read_text())
    c, name = client(), "prod_naive"
    was_loaded = c.get_load_state(name)["state"] == LoadState.Loaded
    out = {}
    try:
        c.load_collection(name)
        for s, n in SETS.items():
            q, tv, sv = load_eval(s, n)
            kw = {"k": CANDIDATES, "search_params": {"ef": ef}, "warmup": 0}
            out[s] = {"dense_text": rankings(q, run_queries(c, name, "text_vec", tv, **kw).ids),
                      "dense_image": rankings(q, run_queries(c, name, "image_vec", sv, filters=["has_image == true"] * len(sv), **kw).ids)}
    finally:
        if not was_loaded:
            c.release_collection(name)
    DEEP.write_text(json.dumps(out))
    return out


def matrix(ranks: dict, qr, qids: list[str]) -> np.ndarray:
    """queries x ALL nDCG@10; per_query_ndcg scores B1a's three configs, the fused one swaps in for the first."""
    three = per_query_ndcg(ranks, qr, qids)
    new = per_query_ndcg({**ranks, CONFIGS[0]: ranks[NEW]}, qr, qids)[:, :1]
    return np.hstack([three, new])


published = read_result("b1a")["payload"]
D = deep_rankings(published["ef"])
R = json.loads(RANKINGS.read_text())
for name in SETS:
    R[name][NEW] = fused(D[name])
if "--fill" in sys.argv:
    text = pd.read_parquet(_src("queries.parquet"), columns=["query_id", "query"])
    print(judge.fill(anthropic.Anthropic(), [R[n][NEW] for n in SETS], qrels(), text, load_corpus()[0], max_pairs=MAX_PAIRS))
    sys.exit("filled: run bench.rescore, then this probe without --fill")
pool = judge.fingerprint()
assert pool == published["judgments"], f"pool {pool} is not B1a's {published['judgments']}"
qr = judge.filled_qrels(qrels())

E = R["eval5k"]
qids = sorted(E[NEW])
M = matrix(E, qr, qids)
three = draw_power(M[:, :3])
assert np.allclose(three["winner_share"], published["power"]["filled"]["draw_power"]["winner_share"]), "B1a not reproduced"
labelled = set(zip(qr.query_id.astype(str), qr.product_id.astype(str)))
unjudged = sum((q, p) not in labelled for q in qids for p in E[NEW][q])

wq = sorted(set(R["poc50"][NEW]) & set(R["curated"][NEW]))
O, P = matrix(R["poc50"], qr, wq), matrix(R["curated"], qr, wq)
M2 = matrix({**E, NEW: fused(D["eval5k"], "dense_image", "dense_text")}, qr, qids)
out = {"configs": ALL, "n_queries": len(qids), "judgments": pool, "rrf_k": 60, "per_arm": CANDIDATES,
       "unjudged_in_top10": unjudged, "ndcg@10": dict(zip(ALL, map(float, M.mean(0)))),
       "paired_new_minus": {c: list(paired_ci(M[:, 3] - M[:, i])) for i, c in enumerate(CONFIGS)},
       "draw_power_3": three,
       "draw_power_4": {str(n): draw_power(M, n=n) for n in (50, 200)},
       "ties_image_first": {"ndcg@10": float(M2[:, 3].mean()), "paired_new_minus_image": list(paired_ci(M2[:, 3] - M2[:, 1])),
                            "draw_power_50": draw_power(M2)},
       "wording_effect": {c: list(paired_ci(P[:, i] - O[:, i])) for i, c in enumerate(ALL)}}
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
print("DONE", flush=True)
