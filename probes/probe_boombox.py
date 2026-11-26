"""Showcase follow-up: "boombox" on poc_20k and prod_naive, for the "Same query, bigger catalogue" slide. The poc50
set has few Electronics queries with labels an audience can check by eye, so this one was picked by hand from all
4,984 Electronics queries (scored with exact search). Same settings as showcase's poc_vs_prod: dense_text, B1a's ef.
Judges nothing: it stops if a result has no label. Won't run while a benchmark is running, and puts the loaded
collections back as it found them.  -> probes/probe_boombox.json"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from pymilvus.client.types import LoadState

from bench._common import _src, client, corpus_rows, load_corpus, qrels
from bench.showcase import label_maps, open_existing, side
from vsearch import judge
from vsearch.collections import release_all
from vsearch.config import POC_CATEGORY
from vsearch.results import read_result
from vsearch.search import run_queries

QUERY_ID, OUT = "18462", Path(__file__).with_suffix(".json")


def main():
    if subprocess.run(["pgrep", "-f", r"-m bench\."], capture_output=True).returncode == 0:
        raise SystemExit("a benchmark is running; try again when it has finished")
    c = client()
    products, _, _, _ = load_corpus()
    info = products.set_index("product_id")
    q = pd.read_parquet(_src("queries.parquet")).set_index("query_id").loc[QUERY_ID]
    i = list(np.load(_src("q_ids.npy"))).index(QUERY_ID)
    qv = np.array(np.load(_src("q_text_vec.npy"), mmap_mode="r")[[i]])
    ef = read_result("b1a")["payload"]["ef"]
    qr = qrels()
    labels = label_maps(qr, judge.filled_qrels(qr), [QUERY_ID])[QUERY_ID]
    loaded = [n for n in c.list_collections() if c.get_load_state(n)["state"] == LoadState.Loaded]
    row = {"query_id": QUERY_ID, "query": q["query"], "top_category": q.top_category}
    try:
        for key, name in (("poc", "poc_20k"), ("prod", "prod_naive")):
            open_existing(c, name, len(corpus_rows(name, products)))
            ids = run_queries(c, name, "text_vec", qv, search_params={"ef": ef}).ids[0]
            row[key] = side(ids, labels, info)
    finally:
        release_all(c)
        for n in loaded:
            c.load_collection(n)
    if min(row["poc"]["judged@10"], row["prod"]["judged@10"]) < 1:
        raise SystemExit("a result has no label; judge it before using this example")
    OUT.write_text(json.dumps({"ef": ef, "collections": ["poc_20k", "prod_naive"], "mode": "dense_text",
                               "poc_category": POC_CATEGORY, "rows": [row]}, indent=1) + "\n")
    print({k: round(row[k]["ndcg@10"], 3) for k in ("poc", "prod")})


if __name__ == "__main__":
    main()
