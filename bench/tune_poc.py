"""PoC tuning, as a typical PoC does it: smallest HNSW ef giving mean recall@10 >= 0.95
on the curated query set against the 20k Electronics slice.  -> results/poc_config.json"""
import numpy as np

from bench._common import POC_SET, client, corpus_rows, ensure_collection, exact_ids, load_corpus, load_eval, parser
from vsearch.metrics import recall_at_k
from vsearch.results import write_result
from vsearch.search import run_queries, summarise_latency

EF_GRID = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
TARGET = 0.95


def main(argv=None):
    a = parser(__doc__).parse_args(argv)
    c = client()
    products, tv, _, _ = load_corpus()
    q, qv, _ = load_eval(POC_SET, a.sample)
    name = ensure_collection(c, "poc_20k")
    truth = exact_ids("poc_curated", tv, corpus_rows("poc_20k", products), qv, products)
    grid = []
    for ef in EF_GRID:
        r = run_queries(c, name, "text_vec", qv, search_params={"ef": ef})
        rec = float(np.mean([recall_at_k(i, t) for i, t in zip(r.ids, truth)]))
        grid.append({"ef": ef, "recall@10": rec, "latency": summarise_latency(r.latencies_ms)})
        print(f"ef={ef} recall={rec:.3f}", flush=True)
    chosen = next((g for g in grid if g["recall@10"] >= TARGET), grid[-1])
    write_result("poc_config", {"ef": chosen["ef"], "recall@10": chosen["recall@10"], "target": TARGET,
                                "n_queries": len(q), "grid": grid, "sample": a.sample is not None}, client=c)


if __name__ == "__main__":
    main()
