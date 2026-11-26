"""B3a: memory vs relevance frontier. dims {1024,512,256} x {HNSW fp32, HNSW fp16, HNSW_SQ8+refine,
IVF_RABITQ+refine}, text_vec only, 450k. Memory = Milvus internal_cache_loaded_bytes{location="memory"}
after load (RAM the query node holds); the mmap'd part (location disk/mixed, e.g. the raw fp32 field
beside a quantised index) is kept as mem_bytes_mmap; the container RSS delta is kept as mem_bytes_docker
but is noisy (allocator retention from the previous build can make it negative).
-> results/b3a.json"""
import json
import re
import subprocess
import time
import urllib.request

import anthropic
import numpy as np

from bench._common import (CANONICAL_N, client, corpus_rows, ensure_parent, exact_ids, load_corpus, load_eval,
                           parser, qrels, quality, rankings)
from vsearch import judge
from vsearch.config import DATA_WORK, MILVUS_METRICS_URL
from vsearch.collections import build_collection, frontier_specs, release_all, wait_loaded
from vsearch.embed import truncate_mrl
from vsearch.metrics import recall_at_k, summarise_quality
from vsearch.results import write_result
from vsearch.search import run_queries, summarise_latency

CONTAINER = "vsearch-talk-milvus-1"
RANKINGS = DATA_WORK / "b3a_rankings.json"
_UNITS = {"B": 1, "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40,
          "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}


def parse_mem(s: str) -> float:
    m = re.match(r"\s*([\d.]+)\s*([A-Za-z]+)", s.split("/")[0])
    return float(m.group(1)) * _UNITS[m.group(2).upper()]


_LOCATION = re.compile(r'location="([^"]*)"')


def loaded_bytes(text: str) -> dict[str, float]:
    """internal_cache_loaded_bytes split by location label: "memory" is RAM, "disk" is mmap'd
    (the raw vector field sits there beside a quantised index), "mixed" is both. A series with no
    location label counts as memory."""
    out = {"memory": 0.0, "disk": 0.0, "mixed": 0.0}
    for line in text.splitlines():
        if line.startswith("internal_cache_loaded_bytes{") or line.startswith("internal_cache_loaded_bytes "):
            m = _LOCATION.search(line)
            loc = m.group(1) if m else "memory"
            out[loc] = out.get(loc, 0.0) + float(line.rsplit(" ", 1)[1])
    return {**out, "total": sum(out.values())}


def milvus_loaded_bytes() -> dict[str, float]:
    with urllib.request.urlopen(MILVUS_METRICS_URL, timeout=5) as r:
        return loaded_bytes(r.read().decode())


def container_mem() -> float:
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", CONTAINER],
                         capture_output=True, text=True, check=True).stdout
    return parse_mem(out)


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=30_000, help="judge spend guard (each pair is a paid API call)")
    a = ap.parse_args(argv)
    c = client()
    products, tv, _, has = load_corpus()
    q, qv, _ = load_eval("eval5k", a.sample or CANONICAL_N)
    ranks = {}
    qr = qrels()
    rows = corpus_rows("prod_naive", products)
    points = []
    for spec in frontier_specs():
        release_all(c)
        time.sleep(30)
        base = container_mem()
        if not c.has_collection(spec.name):
            build_collection(c, spec, products, tv, None, has)
        c.load_collection(spec.name)
        wait_loaded(c, spec.name, len(rows))
        time.sleep(30)
        mem_docker = container_mem() - base
        loaded = milvus_loaded_bytes()
        mem = loaded["memory"]
        qd = truncate_mrl(qv, spec.text_dim)
        truth = exact_ids(f"dim{spec.text_dim}", truncate_mrl(np.asarray(tv), spec.text_dim),
                          rows, qd, products)
        sp = {"nprobe": 64} if spec.text_index.type == "IVF_RABITQ" else {"ef": 64}
        r = run_queries(c, spec.name, "text_vec", qd, search_params=sp)
        ranks[spec.name] = rankings(q, r.ids)
        points.append({"name": spec.name, "dim": spec.text_dim, "dtype": spec.text_dtype,
                       "index": spec.text_index.type, "mem_bytes": mem, "bytes_per_vector": mem / len(rows),
                       "mem_bytes_mmap": loaded["disk"] + loaded["mixed"],
                       "mem_bytes_docker": mem_docker,
                       **summarise_quality(rankings(q, r.ids), qr),
                       "recall@10": float(np.mean([recall_at_k(i, t) for i, t in zip(r.ids, truth)])),
                       "latency": summarise_latency(r.latencies_ms), "search_params": sp})
        print(f"{spec.name}: {mem / 1e9:.2f} GB ndcg={points[-1]['ndcg@10']:.4f}", flush=True)
        if not a.keep:
            c.drop_collection(spec.name)
    ensure_parent(RANKINGS).write_text(json.dumps(ranks))
    fill = None
    if not a.no_fill:
        fill = judge.fill(anthropic.Anthropic(), ranks.values(), qr, q[["query_id", "query"]], products,
                          max_pairs=a.max_pairs)
    for pt in points:
        pt["filled"] = quality(ranks[pt["name"]], qr)["filled"]
        print(f"{pt['name']}: filled ndcg={pt['filled']['ndcg@10']:.4f} judged={pt['filled']['judged@10']:.3f}")
    write_result("b3a", {"points": points, "n_queries": len(q), "fill": fill, "judgments": judge.fingerprint(), "sample": a.sample is not None},
                 client=c)


if __name__ == "__main__":
    main()
