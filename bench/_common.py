"""Shared loading/plumbing for bench scripts. Reads data/work/, else data/dataset/ (published)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymilvus import MilvusClient

from vsearch.collections import (SPECS, CollectionSpec, build_collection, drifted_ids, poc_slice, restore_prices,
                                 wait_loaded)
from vsearch.config import DATA_DATASET, DATA_WORK, MILVUS_URI, SEED
from vsearch.exact import exact_topk
from vsearch.judge import filled_qrels
from vsearch.metrics import summarise_quality

CURATED = Path(__file__).resolve().parent.parent / "data" / "curated_queries.jsonl"
GT_CACHE = DATA_WORK / "gt_cache"
POC_SETS = Path(__file__).resolve().parent.parent / "data" / "poc_sets.json"
# PoC query sets (data/poc_sets.py): "poc50" ESCI originals, "curated" paraphrases as drafted,
# "head50" head-query proxy. POC_SET is the one the PoC config is tuned on.
POC_SET = "poc50"
ALL_POC_SETS = ["poc50", "curated", "head50"]
# Canonical full-run query set: load_eval("eval5k", CANONICAL_N) — the B1c fill subsample, so judgments are shared.
CANONICAL_N = 1000


def parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--sample", type=int, default=None, help="spike run on N queries (marks result sample=True)")
    return ap


def ensure_parent(path: Path) -> Path:
    """Create path's directory and return path: a fresh clone has no data/work/ (data.fetch fills
    data/dataset/), but benchmarks still cache rankings, filter mixes and state there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def client() -> MilvusClient:
    return MilvusClient(uri=MILVUS_URI, timeout=600)


def _src(name: str) -> Path:
    p = DATA_WORK / name
    return p if p.exists() else DATA_DATASET / name


def load_corpus() -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, np.ndarray]:
    """image_vec is None until image embedding has run; has_image then comes from the
    download manifest (same definition data.embed uses), so text-only benches can run."""
    products = pd.read_parquet(_src("products.parquet"))
    tv = np.load(_src("text_vec.npy"), mmap_mode="r")
    if _src("image_vec.npy").exists():
        return products, tv, np.load(_src("image_vec.npy"), mmap_mode="r"), np.load(_src("has_image.npy"))
    man = pd.read_parquet(_src("images_manifest.parquet")).set_index("product_id")
    has = products.product_id.map(man.ok).fillna(False).to_numpy(bool)
    return products, tv, None, has


def load_eval(set_name: str, sample: int | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Returns (queries[key, query_id, query], q_text_vec, q_siglip_vec), rows aligned.

    key is the vector key in q_ids.npy; query_id is the qrels key (curated rows map
    back to their source ESCI query)."""
    keys = np.load(_src("q_ids.npy"))
    pos = {k: i for i, k in enumerate(keys)}
    if set_name == "eval5k":
        q = pd.read_parquet(_src("queries.parquet"))
        q = q[q.in_eval5k].sort_values("query_id")[["query_id", "query"]].assign(key=lambda d: d.query_id)
    elif set_name in ("poc50", "head50"):
        ids = sorted(json.loads(POC_SETS.read_text())[set_name])
        text = pd.read_parquet(_src("queries.parquet")).set_index("query_id")["query"]
        q = pd.DataFrame({"key": ids, "query_id": ids, "query": [text[i] for i in ids]})
    elif set_name == "curated":
        cur = pd.read_json(CURATED, lines=True, dtype={"query_id": str}).sort_values("query_id")
        q = pd.DataFrame({"key": "curated:" + cur.query_id, "query_id": cur.query_id, "query": cur.curated_query})
    else:
        raise ValueError(set_name)
    q = q[q.key.isin(pos)]
    if sample is not None and sample < len(q):
        q = q.sample(n=sample, random_state=SEED).sort_values("query_id")
    q = q.reset_index(drop=True)
    idx = np.array([pos[k] for k in q.key], dtype=np.int64)
    return (q, np.asarray(np.load(_src("q_text_vec.npy"), mmap_mode="r")[idx]),
            np.asarray(np.load(_src("q_siglip_vec.npy"), mmap_mode="r")[idx]))


def with_query_meta(q: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Attach queries.parquet columns (e.g. top_category) to load_eval's frame by query_id,
    keeping its row order (aligned with the query vectors)."""
    meta = pd.read_parquet(_src("queries.parquet"), columns=["query_id", *cols]).set_index("query_id")
    return q.assign(**{c: q.query_id.map(meta[c]).to_numpy() for c in cols})


def qrels() -> pd.DataFrame:
    return pd.read_parquet(_src("qrels.parquet"))


def qrels_for(query_ids) -> dict[str, dict[str, str]]:
    qr = qrels()
    qr = qr[qr.query_id.isin(set(query_ids))]
    return {qid: dict(zip(g.product_id, g.label)) for qid, g in qr.groupby("query_id")}


def rankings(queries: pd.DataFrame, ids: list[list[str]]) -> dict[str, list[str]]:
    """Keyed by query_id (qrels key)."""
    return dict(zip(queries.query_id, ids))


def corpus_rows(spec_name: str, products: pd.DataFrame) -> np.ndarray:
    spec = SPECS.get(spec_name)
    if spec is not None and spec.rows == "poc":
        return poc_slice(products).row_id.to_numpy()
    return products.sort_values("row_id").row_id.to_numpy()


def ensure_collection(c: MilvusClient, spec: str | CollectionSpec) -> str:
    """Reuse an existing collection only if it holds every row of its spec; a build killed
    mid-insert leaves a short one, which silently caps recall. Otherwise (re)build it."""
    spec = SPECS[spec] if isinstance(spec, str) else spec
    products, tv, iv, has = load_corpus()
    if c.has_collection(spec.name):
        c.load_collection(spec.name)
        want = len(poc_slice(products)) if spec.rows == "poc" else len(products)
        got = c.query(spec.name, filter="", output_fields=["count(*)"])[0]["count(*)"]
        if got == want:
            wait_loaded(c, spec.name, want)
            return spec.name
        print(f"{spec.name}: {got:,} rows, expected {want:,} -> rebuilding", flush=True)
    build_collection(c, spec, products, tv, iv, has)
    return spec.name


def exact_ids(tag: str, docs: np.ndarray, rows: np.ndarray, qvecs: np.ndarray, products: pd.DataFrame,
              k: int = 10, mask: np.ndarray | None = None) -> list[list[str]]:
    """Exact top-k product_ids over docs[rows] (optionally masked), cached on disk by content hash."""
    h = hashlib.sha1(np.ascontiguousarray(qvecs).tobytes() + rows.tobytes()
                     + (mask.tobytes() if mask is not None else b"") + tag.encode()).hexdigest()[:16]
    p = GT_CACHE / f"{tag}_{h}.json"
    if p.exists():
        return json.loads(p.read_text())
    sub = np.asarray(docs[rows], dtype=np.float32)
    idx, _ = exact_topk(sub, np.asarray(qvecs, dtype=np.float32), k=k, mask=mask)
    pids = products.set_index("row_id").product_id.reindex(rows).to_numpy()
    out = [[pids[j] for j in r if j >= 0] for r in idx]
    GT_CACHE.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return out


def quality(rankings: dict[str, list[str]], qr: pd.DataFrame) -> dict:
    """Raw quality (ESCI labels only) plus "filled" (ESCI + judge). Charts and slides read "filled"."""
    return {**summarise_quality(rankings, qr), "filled": summarise_quality(rankings, filled_qrels(qr))}


def ensure_prices(c: MilvusClient, name: str, products: pd.DataFrame) -> int:
    """Write benchmarks change prices; filtered ground truth uses products.parquet. Restore only the drifted
    rows (a partial update rewrites the whole row, so restoring everything doubles the collection in memory
    and can trip Milvus's write throttling), then fail rather than measure if any still drift."""
    ids = drifted_ids(c, name, products)
    if ids:
        print(f"{name}: {len(ids):,} prices drifted from products.parquet -> restoring them", flush=True)
        restore_prices(c, name, products, ids=ids)
        if (left := drifted_ids(c, name, products)):
            raise RuntimeError(f"{name}: still {len(left):,} drifted prices after restore")
    return len(ids)
