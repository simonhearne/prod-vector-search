"""B2c: what facets cost. Same queries/filters as B2 on prod_tuned:
search_only vs search + server-side facet counts (category, brand) vs facets from top-100.
-> results/b2c.json"""
import time
from collections import Counter

import numpy as np
import pandas as pd

from bench._common import DATA_WORK, client, ensure_collection, ensure_prices, load_corpus, load_eval, parser
from vsearch.collections import release_all
from vsearch.config import K
from vsearch.load import level_start
from vsearch.results import read_result, write_result
from vsearch.search import summarise_latency

FACETS = [("category", 50), ("brand", 20)]


def _vec(v):
    return np.asarray(v, dtype=np.float32).tolist()


def main(argv=None):
    a = parser(__doc__).parse_args(argv)
    c = client()
    ef = read_result("poc_config")["payload"]["ef"]
    q, qv, _ = load_eval("eval5k")
    mix = pd.read_parquet(DATA_WORK / "filter_mix.parquet")
    if a.sample:
        mix = mix.groupby("bucket", group_keys=False).head(max(1, a.sample // 5))
    pos = {qid: i for i, qid in enumerate(q.query_id)}
    release_all(c)
    name = ensure_collection(c, "prod_tuned")
    products = load_corpus()[0]
    drift = ensure_prices(c, name, products)
    _, level_quiet = level_start(c, name, len(products))
    sp = {"params": {"ef": ef}}
    lat = {"search_only": [], "search_plus_facets": [], "facets_from_topk": []}
    acc = {f: [] for f, _ in FACETS}
    for w in range(20):
        c.search(name, data=[_vec(qv[pos[mix.query_id.iloc[w % len(mix)]]])], anns_field="text_vec",
                 search_params=sp, limit=K)
    for r in mix.itertuples():
        v, expr = [_vec(qv[pos[r.query_id]])], r.expr or ""
        t0 = time.perf_counter()
        c.search(name, data=v, anns_field="text_vec", filter=expr, search_params=sp, limit=K)
        lat["search_only"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        c.search(name, data=v, anns_field="text_vec", filter=expr, search_params=sp, limit=K)
        true = {}
        for f, lim in FACETS:
            rows = c.query(name, filter=expr, group_by_fields=[f], output_fields=[f, "count(*)"], limit=lim)
            true[f] = Counter({x[f]: x["count(*)"] for x in rows})
        lat["search_plus_facets"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        hits = c.search(name, data=v, anns_field="text_vec", filter=expr, search_params={"params": {"ef": max(ef, 100)}},
                        limit=100, output_fields=[f for f, _ in FACETS])[0]
        approx = {f: Counter(h["entity"][f] for h in hits) for f, _ in FACETS}
        lat["facets_from_topk"].append((time.perf_counter() - t0) * 1000)
        for f, _ in FACETS:
            top_true = {k for k, _ in true[f].most_common(5)}
            top_approx = {k for k, _ in approx[f].most_common(5)}
            acc[f].append(top_true == top_approx)
    write_result("b2c", {**{k: {"latency": summarise_latency(np.array(v))} for k, v in lat.items()},
                         "facet_accuracy_topk": {f: float(np.mean(v)) for f, v in acc.items()},
                         "n": len(mix), "ef": ef, "price_drift_restored": drift, "level_quiet": level_quiet, "sample": a.sample is not None}, client=c)


if __name__ == "__main__":
    main()
