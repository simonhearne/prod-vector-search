"""Embed the corpus and queries.

    uv run python -m data.embed --text      # products.parquet  -> text_vec.npy
    uv run python -m data.embed --images    # images/           -> image_vec.npy, has_image.npy
    uv run python -m data.embed --queries   # queries + curated -> q_text_vec.npy, q_siglip_vec.npy, q_ids.npy

Rows follow products.row_id / the q_ids order. Work is checkpointed in
data/work/embed_chunks/ so an interrupted run resumes. Throughput goes to
results/b3c_throughput.json (input to the re-embed bill, B3c).
"""
import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from vsearch.config import DATA_WORK, RESULTS
from vsearch.results import read_result, write_result

CHUNKS = DATA_WORK / "embed_chunks"
CURATED = Path(__file__).parent / "curated_queries.jsonl"


def fill_missing(n: int, idx: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Place vecs at rows idx; other rows get the unit placeholder e_0 (never a zero vector)."""
    out = np.zeros((n, vecs.shape[1]), dtype=np.float32)
    out[:, 0] = 1.0
    out[idx] = vecs
    return out


def chunked(name: str, items: Sequence, encode: Callable[[Sequence], np.ndarray],
            chunk: int = 20_000, workdir: Path = CHUNKS, stats: dict | None = None) -> np.ndarray:
    """Encode items in resumable chunks. If `stats` is given, it receives n_encoded and encode_s
    for chunks encoded in this call only, so throughput stays honest after a resume."""
    workdir.mkdir(parents=True, exist_ok=True)
    parts = []
    n_enc, enc_s = 0, 0.0
    for i in range(0, len(items), chunk):
        p = workdir / f"{name}_{i:08d}.npy"
        if not p.exists():
            t = time.perf_counter()
            arr = encode(items[i:i + chunk])
            enc_s += time.perf_counter() - t
            n_enc += len(arr)
            with open(p.with_suffix(".part"), "wb") as f:
                np.save(f, arr)
            p.with_suffix(".part").rename(p)
            print(f"{name}: {min(i + chunk, len(items)):,}/{len(items):,}", flush=True)
        parts.append(np.load(p))
    if stats is not None:
        stats.update(n_encoded=n_enc, encode_s=enc_s)
    return np.concatenate(parts)


def _save(arr: np.ndarray, name: str) -> None:
    dest = DATA_WORK / name
    with open(dest.with_suffix(".part"), "wb") as f:
        np.save(f, arr)
    dest.with_suffix(".part").rename(dest)


def _throughput(**kv) -> None:
    cur = read_result("b3c_throughput")["payload"] if (RESULTS / "b3c_throughput.json").exists() else {}
    write_result("b3c_throughput", {**cur, **kv})


def embed_text() -> None:
    from vsearch.embed import TextEmbedder
    prods = pd.read_parquet(DATA_WORK / "products.parquet", columns=["row_id", "embed_text"])
    te = TextEmbedder()
    st: dict = {}
    vecs = chunked("text", prods.embed_text.tolist(), te.encode_docs, stats=st)
    _save(vecs, "text_vec.npy")
    if st["n_encoded"]:  # all chunks cached → keep the throughput already recorded
        _throughput(device=te.device, text_items_per_s=st["n_encoded"] / st["encode_s"],
                    n_text=len(prods), n_text_timed=st["n_encoded"])


def embed_images() -> None:
    from data.images import image_path
    from vsearch.embed import ImageEmbedder
    prods = pd.read_parquet(DATA_WORK / "products.parquet", columns=["row_id", "product_id"])
    man = pd.read_parquet(DATA_WORK / "images_manifest.parquet").set_index("product_id")
    has = prods.product_id.map(man.ok).fillna(False).to_numpy(bool)
    idx = np.flatnonzero(has)
    paths = [image_path(p) for p in prods.product_id.to_numpy()[idx]]
    ie = ImageEmbedder()
    st: dict = {}
    vecs = chunked("image", paths, ie.encode_images, stats=st)
    _save(fill_missing(len(prods), idx, vecs), "image_vec.npy")
    _save(has, "has_image.npy")
    if st["n_encoded"]:  # all chunks cached → keep the throughput already recorded
        _throughput(device=ie.device, image_items_per_s=st["n_encoded"] / st["encode_s"],
                    n_image=int(len(idx)), n_image_timed=st["n_encoded"])


def query_table() -> pd.DataFrame:
    q = pd.read_parquet(DATA_WORK / "queries.parquet")
    q = q[q.eligible][["query_id", "query"]]
    if CURATED.exists():
        cur = pd.read_json(CURATED, lines=True, dtype={"query_id": str})
        q = pd.concat([q, pd.DataFrame({"query_id": "curated:" + cur.query_id,
                                        "query": cur.curated_query})], ignore_index=True)
    return q.reset_index(drop=True)


def embed_queries() -> None:
    from vsearch.embed import ImageEmbedder, TextEmbedder
    q = query_table()
    texts = q["query"].tolist()
    _save(TextEmbedder().encode_queries(texts), "q_text_vec.npy")
    _save(ImageEmbedder().encode_texts(texts), "q_siglip_vec.npy")
    _save(q.query_id.to_numpy(str), "q_ids.npy")
    print(f"queries embedded: {len(q):,}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--queries", action="store_true")
    a = ap.parse_args(argv)
    if a.text:
        embed_text()
    if a.images:
        embed_images()
    if a.queries:
        embed_queries()


if __name__ == "__main__":
    main()
