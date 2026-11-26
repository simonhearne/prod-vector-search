"""How much of hybrid's result is its settings? Same two arms (BM25, text vectors) on prod_naive, four fusions: RRF at
10 and 100 candidates per arm, and Milvus's WeightedRanker (normalised scores, 100 per arm) at 0.3 / 0.5 / 0.7 dense
weight. Scored as exact matches in the top 5, like probe_bm25. Nothing is judged: a query counts only if every result
in every arm's and fusion's top 5 has a label, so all rows share one query set. Rankings cached in
data/work/fusion_rankings.json. RRF at 100 is B1a's hybrid; the others are run live here."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker
from pymilvus.client.types import LoadState

from bench._common import CANONICAL_N, _src, client, load_eval, qrels
from bench.b1a_eval_sets import RANKINGS
from vsearch import judge
from vsearch.config import DATA_WORK
from vsearch.results import read_result
from vsearch.stats import paired_ci

COLLECTION, CACHE, N = "prod_naive", DATA_WORK / "fusion_rankings.json", 5
SHOWN = ["26141", "85295", "67830", "74720", "21072"]
B1A = read_result("b1a")["payload"]
FUSIONS = {"rrf_10": (10, RRFRanker(60)), "rrf_100": (100, RRFRanker(60)),
           **{f"weighted_{w}": (100, WeightedRanker(w, round(1 - w, 1), norm_score=True)) for w in (0.3, 0.5, 0.7)}}


def run() -> dict[str, dict[str, list[str]]]:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    q, tv, _ = load_eval("eval5k", CANONICAL_N)
    c = client()
    was_loaded = c.get_load_state(COLLECTION)["state"] == LoadState.Loaded
    out = {f: {} for f in FUSIONS}
    try:
        c.load_collection(COLLECTION)
        for i, (qid, text) in enumerate(zip(q.query_id.astype(str), q["query"])):
            for f, (depth, ranker) in FUSIONS.items():
                reqs = [AnnSearchRequest(data=[tv[i].tolist()], anns_field="text_vec",
                                         param={"params": {"ef": B1A["ef"]}}, limit=depth),
                        AnnSearchRequest(data=[text], anns_field="text_sparse", param={}, limit=depth)]
                out[f][qid] = [h.id for h in c.hybrid_search(COLLECTION, reqs=reqs, ranker=ranker, limit=10)[0]]
    finally:
        if not was_loaded:
            c.release_collection(COLLECTION)
    CACHE.write_text(json.dumps(out))
    return out


R = run()
b1a = json.loads(RANKINGS.read_text())["eval5k"]
R["bm25"] = json.loads((DATA_WORK / "bm25_rankings.json").read_text())["eval5k"]
R["dense_text"] = b1a["dense_text"]
assert judge.fingerprint() == B1A["judgments"], "judgment pool is not B1a's"
F = judge.filled_qrels(qrels())
lab = {str(q): dict(zip(g.product_id, g.label)) for q, g in F.groupby("query_id")}
qids = sorted(q for q in R["bm25"] if all(p in lab.get(q, {}) for a in R for p in R[a][q][:N]))
X = {a: np.array([sum(lab[q][p] == "E" for p in R[a][q][:N]) for q in qids]) for a in R}
best = np.maximum(X["bm25"], X["dense_text"])
agree = float(np.mean([R["rrf_100"][q] == b1a["hybrid_rrf"][q] for q in R["rrf_100"]]))
text = pd.read_parquet(_src("queries.parquet")).set_index("query_id")["query"]
overlap = [len(set(R["bm25"][q]) & set(R["dense_text"][q])) for q in R["bm25"]]
out = {"collection": COLLECTION, "n_queries": len(R["bm25"]), "n_judged": len(qids), "judgments": judge.fingerprint(),
       "rrf_100_same_top10_as_b1a": agree,
       "exact_top5": {a: float(X[a].mean()) for a in R},
       "zero_exact_top5": {a: float((X[a] == 0).mean()) for a in R},
       "matches_best_arm": {f: float((X[f] >= best).mean()) for f in FUSIONS},
       "beats_best_arm": {f: float((X[f] > best).mean()) for f in FUSIONS},
       "paired_exact_minus_rrf_10": {f: list(paired_ci(X[f] - X["rrf_10"])) for f in FUSIONS if f != "rrf_10"},
       "paired_zero_minus_rrf_10": {f: list(paired_ci((X[f] == 0) * 1.0 - (X["rrf_10"] == 0))) for f in FUSIONS if f != "rrf_10"},
       "oracle_best_arm_exact_top5": float(best.mean()),
       "top10_overlap": {"mean": float(np.mean(overlap)), "zero_share": float(np.mean(np.array(overlap) == 0))},
       "shown": {text[q]: {a: "".join(lab[q].get(p, "?") for p in R[a][q][:N]) for a in R} for q in SHOWN}}
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
