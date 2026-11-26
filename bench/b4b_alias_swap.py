"""B4b: blue/green re-index behind an alias, under 20 QPS. Build products_v2 (HNSW_SQ8 re-index),
alter_alias, keep serving.  -> results/b4b.json"""
import threading
import time
from dataclasses import replace

import numpy as np

from bench._common import client, ensure_collection, load_corpus, load_eval, parser
from vsearch.collections import HNSW_SQ8, SPECS, IndexSpec, build_collection, release_all
from vsearch.load import run_load, windowed
from vsearch.results import read_result, write_result

ALIAS, V2 = "products", "products_v2"
QPS, THREADS, AFTER_S = 20, 4, 60
NOTES = ("Re-index only: query vectors are unchanged. A real model change must also switch the query "
         "encoder at the same instant as the alias, or every query hits the wrong vector space.")


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    c = client()
    ef = read_result("poc_config")["payload"]["ef"]
    products, tv, iv, has = load_corpus()
    _, qv, _ = load_eval("eval5k")
    release_all(c)
    v1 = ensure_collection(c, "prod_tuned")
    try:
        c.create_alias(v1, ALIAS)
    except Exception:  # noqa: BLE001 - alias already exists (possibly pointing at v2)
        c.alter_alias(v1, ALIAS)
    vecs = [np.asarray(v, dtype=np.float32).tolist() for v in qv]
    sp = {"params": {"ef": ef}}
    stop, marks = threading.Event(), {}
    t0 = time.perf_counter()

    def search(i):
        c.search(ALIAS, data=[vecs[i % len(vecs)]], anns_field="text_vec", search_params=sp, limit=10)

    def blue_green():
        time.sleep(30)
        b0 = time.perf_counter()
        spec = replace(SPECS["prod_tuned"], name=V2, text_index=IndexSpec("HNSW_SQ", HNSW_SQ8))
        build_collection(c, spec, products, tv, iv, has)
        marks["build_seconds"] = time.perf_counter() - b0
        c.alter_alias(V2, ALIAS)
        marks["swap_t"] = time.perf_counter() - t0
        time.sleep(AFTER_S)
        stop.set()

    bg = threading.Thread(target=blue_green)
    bg.start()
    samples = run_load(search, QPS, duration_s=24 * 3600, threads=THREADS, stop=stop)
    bg.join()
    write_result("b4b", {"windows": windowed(samples), "swap_t": marks["swap_t"],
                         "errors": sum(not s.ok for s in samples), "n": len(samples),
                         "build_seconds": marks["build_seconds"], "qps": QPS, "ef": ef, "notes": NOTES,
                         "sample": a.sample is not None}, client=c)
    if not a.keep:
        c.alter_alias(v1, ALIAS)
        c.drop_collection(V2)


if __name__ == "__main__":
    main()
