"""B1a follow-up: where does BM25 alone fall short? B1a has text, image and text + BM25, never BM25 on its own. Runs
BM25 over the 1,000 eval queries on prod_naive (cached in data/work/bm25_rankings.json) and compares exact matches in
the top 5 with B1a's saved rankings. Nothing is judged: BM25's top 10s are only partly covered by the pool, so nDCG@10
is not reported, and a query counts only if every BM25 result in its top 5 already has a label. Fewer than five BM25
results still counts (an empty page is a miss). B1a's text vectors came from HNSW, which can miss a whole brand at 0.98
mean recall, so the lopsided counts use exact vector search too, and each example carries its recall@10. The examples are picked for contrast; `examples` shows their top 5s."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymilvus.client.types import LoadState

from bench._common import CANONICAL_N, _src, client, corpus_rows, exact_ids, load_corpus, load_eval, qrels
from bench.b1a_eval_sets import RANKINGS
from vsearch import judge
from vsearch.config import DATA_WORK
from vsearch.metrics import recall_at_k
from vsearch.results import read_result
from vsearch.stats import paired_ci

COLLECTION, CACHE, N = "prod_naive", DATA_WORK / "bm25_rankings.json", 5
EXAMPLES = ["91765", "85295", "74720", "26141", "21072"]
ARMS = ["bm25", "dense_text", "dense_image", "hybrid_rrf"]


def bm25_rankings() -> dict[str, list[str]]:
    if CACHE.exists():
        return json.loads(CACHE.read_text())["eval5k"]
    q, _, _ = load_eval("eval5k", CANONICAL_N)
    c = client()
    was_loaded = c.get_load_state(COLLECTION)["state"] == LoadState.Loaded
    try:
        c.load_collection(COLLECTION)
        out = {str(i): [h.id for h in c.search(COLLECTION, data=[t], anns_field="text_sparse",
                                               search_params={"params": {}}, limit=10)[0]]
               for i, t in zip(q.query_id, q["query"])}
    finally:
        if not was_loaded:
            c.release_collection(COLLECTION)
    CACHE.write_text(json.dumps({"eval5k": out}))
    return out


R = {**json.loads(RANKINGS.read_text())["eval5k"], "bm25": bm25_rankings()}
R = {a: {str(q): ids for q, ids in R[a].items()} for a in ARMS}
pool = judge.fingerprint()
assert pool == read_result("b1a")["payload"]["judgments"], "judgment pool is not B1a's"
E = qrels()
human = set(zip(E.query_id.astype(str), E.product_id))
F = judge.filled_qrels(E)
lab = {str(q): dict(zip(g.product_id, g.label)) for q, g in F.groupby("query_id")}

allq = sorted(R["bm25"])
qids = [q for q in allq if all(p in lab.get(q, {}) for a in ARMS for p in R[a][q][:N])]
X = {a: np.array([sum(lab[q][p] == "E" for p in R[a][q][:N]) if R[a][q] else 0 for q in qids]) for a in ARMS}
short = [q for q in allq if len(R["bm25"][q]) < N]


def rescue(miss: str, hit: str) -> dict:
    m, h = X[miss] == 0, X[hit] == 0
    return {"missed": int(m.sum()), "found_by_other": int((m & ~h).sum()), "reverse": int((h & ~m).sum())}


corpus, text_vec, _, _ = load_corpus()
q, qvec, _ = load_eval("eval5k", CANONICAL_N)
exact = dict(zip(q.query_id.astype(str), exact_ids("probe_bm25_text", text_vec, corpus_rows(COLLECTION, corpus), qvec, corpus)))
eq = [x for x in qids if all(p in lab[x] for p in exact[x][:N])]


def lopsided(dense: dict[str, list[str]], among: list[str]) -> dict:
    b, d = (np.array([sum(lab[x][p] == "E" for p in r[x][:N]) for x in among]) for r in (R["bm25"], dense))
    return {"n": len(among), "bm25_0_dense_3plus": int(((b == 0) & (d >= 3)).sum()),
            "dense_0_bm25_3plus": int(((d == 0) & (b >= 3)).sum())}


products = corpus.set_index("product_id").title
text = pd.read_parquet(_src("queries.parquet")).set_index("query_id")["query"]
out = {"collection": COLLECTION, "n_queries": len(allq), "n_judged": len(qids), "n_bm25_short": len(short),
       "n_bm25_short_included": len(set(short) & set(qids)), "top": N, "judgments": pool,
       "exact_top5": {a: float(X[a].mean()) for a in ARMS},
       "zero_exact_top5": {a: float((X[a] == 0).mean()) for a in ARMS},
       "paired_exact": {f"{a}-{b}": list(paired_ci(X[a] - X[b])) for a, b in
                        [("hybrid_rrf", "bm25"), ("hybrid_rrf", "dense_text"), ("dense_text", "bm25")]},
       "zero_rescue": {f"{m}_missed_{h}_found": rescue(m, h) for m, h in
                       [("bm25", "hybrid_rrf"), ("dense_text", "hybrid_rrf"), ("bm25", "dense_text"), ("dense_text", "bm25")]},
       "lopsided": {"hnsw": lopsided(R["dense_text"], eq), "exact": lopsided(exact, eq)},
       "examples": [{"query_id": q, "query": text[q], "dense_recall@10": recall_at_k(R["dense_text"][q], exact[q]),
                     "top5": {a: [{"id": p, "label": lab[q][p], "source": "esci" if (q, p) in human else "judge",
                                   "title": products[p]} for p in R[a][q][:N]]
                              for a in ["bm25", "dense_text", "hybrid_rrf"]}} for q in EXAMPLES]}
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1))
for e in out["examples"]:
    print(e["query"], {a: " ".join(r["label"] + r["source"][0] for r in rs) for a, rs in e["top5"].items()})
