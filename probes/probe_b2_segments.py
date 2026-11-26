"""2.2 follow-up: is the partition-key "tax" on unfiltered queries a segment-count cost?

Four text-only HNSW collections over the same 448k rows (same HNSW params as prod's text index):
  plain    no partition key, one flush + compaction (Milvus picks the segment count)
  plain16  no partition key, 16 chunks each flushed, autocompaction off, never force-compacted
  pk16     partition key on category, 16 partitions (the default), compacted
  pk4      partition key on category, 4 partitions, compacted
Unfiltered latency (B2's TAX_EFS / TAX_QUERIES / TAX_PASSES) in two rounds, the second in reverse
order, with the loaded segment count per collection. If latency tracks segments rather than the
partition key, plain16 ~ pk16 and pk4 sits between plain and pk16.   --keep leaves the collections;
--reuse skips building a collection that already holds every row.
Before each collection's timed passes: settle, then the same 90 s compaction-quiet window as
level_start (no force-compaction, which would undo plain16's layout); the compaction counters are
recorded around the timed passes, so a measurement that overlapped compaction is flagged."""
import argparse
import json
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from pymilvus import connections, utility

from bench._common import client, corpus_rows, exact_ids, load_corpus, load_eval
from bench.b2_filters import TAX_EFS, TAX_PASSES, TAX_QUERIES
from vsearch.collections import (CollectionSpec, _rows, build_index_params, build_schema, compact_and_wait,
                                 release_all, wait_index, wait_loaded)
from vsearch.config import MILVUS_URI
from vsearch.load import _quiet, compaction_tasks, metrics_text, settle
from vsearch.metrics import recall_at_k
from vsearch.search import run_queries, summarise_latency

warnings.filterwarnings("ignore", category=DeprecationWarning)
VARIANTS = {"plain": dict(pk=False, parts=None, chunks=1, compact=True),
            "plain16": dict(pk=False, parts=None, chunks=16, compact=False),
            "pk16": dict(pk=True, parts=16, chunks=1, compact=True),
            "pk4": dict(pk=True, parts=4, chunks=1, compact=True)}


def build(c, label, v, products, tv, has, reuse=False):
    name = f"probe_seg_{label}"
    if reuse and c.has_collection(name) and c.get_collection_stats(name)["row_count"] == len(products):
        print(f"reused {name}", flush=True)
        return name
    spec = CollectionSpec(name, rows="full", image=False, bm25=False, partition_key=v["pk"])
    if c.has_collection(name):
        c.drop_collection(name)
    kw = {"num_partitions": v["parts"]} if v["parts"] else {}
    c.create_collection(name, schema=build_schema(spec), index_params=build_index_params(spec), **kw)
    if not v["compact"]:
        c.alter_collection_properties(name, properties={"collection.autocompaction.enabled": False})
    prods = products.sort_values("row_id")
    rows = prods.row_id.to_numpy()
    bounds = np.linspace(0, len(prods), v["chunks"] + 1).astype(int)
    t0 = time.time()
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        for s in range(lo, hi, 1000):
            e = min(s + 1000, hi)
            c.insert(name, list(_rows(spec, prods.iloc[s:e], tv[rows[s:e]], None, has[rows[s:e]])))
        c.flush(name)
    if v["compact"]:
        compact_and_wait(c, name)
    wait_index(c, name, len(prods))
    print(f"built {name} in {time.time() - t0:.0f}s", flush=True)
    return name


def segments(name):
    segs = utility.get_query_segment_info(name)
    n = sorted(s.num_rows for s in segs)
    return {"segments": len(segs), "partitions_used": len({s.partitionID for s in segs}),
            "rows_min": n[0], "rows_median": n[len(n) // 2], "rows_max": n[-1],
            "states": dict(Counter(str(s.state) for s in segs))}


def quiet_start(rounds=4):
    """settle, then require a 90 s window with no compaction active, starting or finishing."""
    for _ in range(rounds):
        settle()
        if _quiet(metrics_text, 18, 5):
            return True
    return False


def measure(c, name, n_rows, qv, truth):
    release_all(c)
    c.load_collection(name)
    wait_loaded(c, name, n_rows)
    quiet = quiet_start()
    before = compaction_tasks(metrics_text())
    run_queries(c, name, "text_vec", qv, search_params={"ef": max(TAX_EFS)}, warmup=0)  # warm, discarded
    out = {"layout": segments(name)}
    for ef in TAX_EFS:
        lat = [summarise_latency(run_queries(c, name, "text_vec", qv, search_params={"ef": ef}).latencies_ms)
               for _ in range(TAX_PASSES)]
        r = run_queries(c, name, "text_vec", qv, search_params={"ef": ef})
        out[f"ef={ef}"] = {"passes": lat, "p50": float(np.median([x["p50"] for x in lat])),
                           "p99": float(np.median([x["p99"] for x in lat])),
                           "recall": float(np.mean([recall_at_k(i, t) for i, t in zip(r.ids, truth)]))}
    after = compaction_tasks(metrics_text())
    out["level_quiet"] = quiet
    out["compactions_during"] = {"finished": after["done"] - before["done"], "active_at_end": after["active"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    c = client()
    connections.connect(uri=MILVUS_URI)
    products, tv, _, has = load_corpus()
    docs = tv
    q, qv, _ = load_eval("eval5k", TAX_QUERIES)
    truth = exact_ids("tax_unf", docs, corpus_rows("prod_naive", products), qv, products)
    release_all(c)
    names = {label: build(c, label, v, products, tv, has, a.reuse) for label, v in VARIANTS.items()}
    out_path = Path(__file__).with_suffix(".json")
    out = {"rounds": []}
    for order in (list(names), list(names)[::-1]):
        rnd = {}
        for label in order:
            rnd[label] = measure(c, names[label], len(products), qv, truth)
            print(time.strftime("%T"), label, rnd[label]["layout"]["segments"], "segs",
                  "quiet", rnd[label]["level_quiet"], "compactions", rnd[label]["compactions_during"],
                  {ef: (rnd[label][f"ef={ef}"]["p50"], rnd[label][f"ef={ef}"]["p99"]) for ef in TAX_EFS}, flush=True)
        out["rounds"].append({"order": order, **rnd})
        out_path.write_text(json.dumps(out, indent=1))
    release_all(c)
    if not a.keep:
        for n in names.values():
            c.drop_collection(n)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
