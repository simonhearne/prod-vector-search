"""Collection definitions for every benchmark, and the one builder that creates them.

Vectors are passed aligned to products.row_id (the full corpus); a spec with
rows="poc" selects the 20k Electronics slice itself.
"""
import json
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pymilvus import DataType, Function, FunctionType, MilvusClient

from vsearch.config import POC_CATEGORY, POC_ROWS, SEED
from vsearch.embed import truncate_mrl
from vsearch.merch import merch_fields
from vsearch.profiles import fold

ANALYZER = {"tokenizer": "standard",
            "filter": ["lowercase", "asciifolding",
                       {"type": "stemmer", "language": "english"},
                       {"type": "stop", "stop_words": ["_english_"]}]}
HNSW = {"M": 16, "efConstruction": 200}
HNSW_SQ8 = {"M": 16, "efConstruction": 200, "sq_type": "SQ8", "refine": True, "refine_type": "FP16"}
RABITQ = {"nlist": 1024, "refine": True, "refine_type": "SQ8"}
VARCHAR_LIMITS = {"product_id": 64, "category": 128, "brand": 512, "title": 2048, "image_url": 1024}


@dataclass(frozen=True)
class IndexSpec:
    type: str
    params: dict = field(default_factory=dict, hash=False)


@dataclass(frozen=True)
class CollectionSpec:
    name: str
    rows: Literal["poc", "full"]
    text_dim: int = 1024
    text_dtype: Literal["fp32", "fp16"] = "fp32"
    text_index: IndexSpec = IndexSpec("HNSW", HNSW)
    image: bool = True
    bm25: bool = True
    scalar_indexes: bool = False
    partition_key: bool = False
    merch: bool = False  # synthetic margin / campaigns / hidden (vsearch.merch, B5)
    brand_sparse: bool = False  # one-hot brand, for a user's sparse brand vector (B6)


SPECS: dict[str, CollectionSpec] = {
    "poc_20k": CollectionSpec("poc_20k", rows="poc"),
    "prod_naive": CollectionSpec("prod_naive", rows="full"),
    "prod_tuned": CollectionSpec("prod_tuned", rows="full", scalar_indexes=True, partition_key=True),
    "prod_merch": CollectionSpec("prod_merch", rows="full", image=False, merch=True),
    "prod_merch_pk": CollectionSpec("prod_merch_pk", rows="full", image=False, merch=True, partition_key=True),
    "prod_perso": CollectionSpec("prod_perso", rows="full", image=False, bm25=False, brand_sparse=True),
}


def brand_ids(products: pd.DataFrame) -> dict[str, int]:
    """Folded brand -> sparse dimension. 0 is reserved for products with no real brand."""
    return {b: i for i, b in enumerate(sorted({fold(b) for b in products.brand.unique()} - {""}), start=1)}


def frontier_specs() -> list[CollectionSpec]:
    variants = [("hnsw_fp32", "fp32", IndexSpec("HNSW", HNSW)),
                ("hnsw_fp16", "fp16", IndexSpec("HNSW", HNSW)),
                ("hnsw_sq8", "fp32", IndexSpec("HNSW_SQ", HNSW_SQ8)),
                ("rabitq", "fp32", IndexSpec("IVF_RABITQ", RABITQ))]
    return [CollectionSpec(f"f_{dim}_{label}", rows="full", text_dim=dim, text_dtype=dtype,
                           text_index=idx, image=False, bm25=False)
            for dim in (1024, 512, 256) for label, dtype, idx in variants]


def poc_slice(products: pd.DataFrame, category: str = POC_CATEGORY, n: int = POC_ROWS,
              seed: int = SEED) -> pd.DataFrame:
    pool = products[products.category == category].sort_values("product_id")
    return pool.sample(n=min(n, len(pool)), random_state=seed).sort_values("row_id")


def build_schema(spec: CollectionSpec):
    s = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    s.add_field("product_id", DataType.VARCHAR, is_primary=True, max_length=VARCHAR_LIMITS["product_id"])
    s.add_field("category", DataType.VARCHAR, max_length=VARCHAR_LIMITS["category"],
                is_partition_key=spec.partition_key)
    for name in ["brand", "title", "image_url"]:
        s.add_field(name, DataType.VARCHAR, max_length=VARCHAR_LIMITS[name])
    s.add_field("price", DataType.FLOAT)
    s.add_field("average_rating", DataType.FLOAT)
    s.add_field("rating_number", DataType.INT64)
    s.add_field("has_image", DataType.BOOL)
    s.add_field("row_id", DataType.INT64)
    if spec.merch:
        s.add_field("margin", DataType.FLOAT)
        s.add_field("campaigns", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=4, max_length=32)
        s.add_field("hidden", DataType.BOOL)
    vtype = DataType.FLOAT16_VECTOR if spec.text_dtype == "fp16" else DataType.FLOAT_VECTOR
    s.add_field("text_vec", vtype, dim=spec.text_dim)
    if spec.image:
        s.add_field("image_vec", DataType.FLOAT_VECTOR, dim=1024)
    if spec.brand_sparse:
        s.add_field("brand_sparse", DataType.SPARSE_FLOAT_VECTOR)
    if spec.bm25:
        s.add_field("embed_text", DataType.VARCHAR, max_length=8192, enable_analyzer=True,
                    enable_match=True, analyzer_params=ANALYZER)
        s.add_field("text_sparse", DataType.SPARSE_FLOAT_VECTOR)
        s.add_function(Function(name="embed_text_bm25", function_type=FunctionType.BM25,
                                input_field_names=["embed_text"], output_field_names=["text_sparse"]))
    return s


def build_index_params(spec: CollectionSpec):
    idx = MilvusClient.prepare_index_params()
    idx.add_index(field_name="text_vec", index_name="text_vec", index_type=spec.text_index.type,
                  metric_type="COSINE", params=dict(spec.text_index.params))
    if spec.image:
        idx.add_index(field_name="image_vec", index_name="image_vec", index_type="HNSW",
                      metric_type="COSINE", params=dict(HNSW))
    if spec.bm25:
        idx.add_index(field_name="text_sparse", index_name="text_sparse",
                      index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    if spec.brand_sparse:
        idx.add_index(field_name="brand_sparse", index_name="brand_sparse",
                      index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    if spec.scalar_indexes:
        for f, t in [("category", "INVERTED"), ("brand", "INVERTED"),
                     ("price", "STL_SORT"), ("average_rating", "STL_SORT")]:
            idx.add_index(field_name=f, index_name=f, index_type=t)
    return idx


def wait_index(client, name: str, n_rows: int, timeout_s: float = 1800, poll_s: float = 2) -> None:
    """Block until every index on `name` covers all n_rows with nothing pending. Loading and searching
    earlier hits unindexed segments, so recall and memory are measured on the wrong thing."""
    t0 = time.monotonic()
    while True:
        infos = [client.describe_index(name, ix) for ix in client.list_indexes(name)]
        if all(i.get("state") == "Finished" and int(i.get("pending_index_rows", 0)) == 0
               and int(i.get("indexed_rows", 0)) >= n_rows for i in infos):
            return
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(f"{name}: indexes not finished after {timeout_s:.0f}s: {infos}")
        time.sleep(poll_s)


def wait_loaded(client, name: str, n_rows: int, timeout_s: float = 600, poll_s: float = 2) -> None:
    """Block until the query node serves sealed segments covering all n_rows. Right after load it
    can still answer from growing segments (no HNSW, so `ef` has no effect) for ~20 s."""
    t0 = time.monotonic()
    while True:
        segs = client.list_loaded_segments(name)
        if sum(s.num_rows for s in segs if getattr(s.state, "name", s.state) == "Sealed") >= n_rows:
            return
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(f"{name}: sealed loaded rows short of {n_rows} after {timeout_s:.0f}s")
        time.sleep(poll_s)


def compact_and_wait(client, name: str, timeout_s: float = 600) -> None:
    """Best-effort force compaction."""
    try:
        job = client.compact(name)
    except Exception as e:  # noqa: BLE001
        print(f"compaction skipped ({name}): {e}")
        return
    if not job or int(job) <= 0:
        return
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if "Completed" in str(client.get_compaction_state(job)):
            return
        time.sleep(2)
    print(f"WARNING: compaction of {name} still running after {timeout_s:.0f}s")


def _clip(v, max_bytes: int) -> str:
    """Truncate to max_bytes of UTF-8 (VARCHAR max_length counts bytes)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v).encode("utf-8")[:max_bytes].decode("utf-8", "ignore")


def _rows(spec, prods, tv, iv, has, merch=None, brands=None):
    for j, p in enumerate(prods.itertuples(index=False)):
        r = {"product_id": p.product_id, "category": _clip(p.category, VARCHAR_LIMITS["category"]),
             "brand": _clip(p.brand, VARCHAR_LIMITS["brand"]), "title": _clip(p.title, VARCHAR_LIMITS["title"]),
             "image_url": _clip(p.image_url, VARCHAR_LIMITS["image_url"]),
             "price": -1.0 if pd.isna(p.price) else float(p.price),
             "average_rating": -1.0 if pd.isna(p.average_rating) else float(p.average_rating),
             "rating_number": int(p.rating_number), "has_image": bool(has[j]), "row_id": int(p.row_id),
             "text_vec": tv[j]}
        if spec.image:
            r["image_vec"] = iv[j]
        if spec.bm25:
            r["embed_text"] = _clip(p.embed_text, 8192)
        if spec.brand_sparse:
            r["brand_sparse"] = {brands.get(fold(p.brand), 0): 1.0}
        if spec.merch:
            m = merch.iloc[j]
            r |= {"margin": float(m.margin), "campaigns": list(m.campaigns), "hidden": bool(m.hidden)}
        yield r


def build_collection(client, spec: CollectionSpec, products: pd.DataFrame, text_vec: np.ndarray,
                     image_vec: np.ndarray | None, has_image: np.ndarray, *, batch: int = 1000) -> None:
    prods = poc_slice(products) if spec.rows == "poc" else products.sort_values("row_id")
    rows = prods.row_id.to_numpy()
    merch = merch_fields(products).loc[prods.index] if spec.merch else None
    brands = brand_ids(products) if spec.brand_sparse else None
    if client.has_collection(spec.name):
        client.drop_collection(spec.name)
    client.create_collection(spec.name, schema=build_schema(spec), index_params=build_index_params(spec))
    t0 = time.time()
    for s in range(0, len(prods), batch):
        r = rows[s:s + batch]
        tv = truncate_mrl(text_vec[r], spec.text_dim)
        tv = tv.astype(np.float16) if spec.text_dtype == "fp16" else tv
        iv = image_vec[r] if spec.image else None
        client.insert(spec.name, list(_rows(spec, prods.iloc[s:s + batch], tv, iv, has_image[r],
                                             merch.iloc[s:s + batch] if spec.merch else None, brands)))
    client.flush(spec.name)
    compact_and_wait(client, spec.name)
    wait_index(client, spec.name, len(prods))
    client.load_collection(spec.name)
    wait_loaded(client, spec.name, len(prods))
    print(f"built {spec.name}: {len(prods):,} rows in {time.time() - t0:.0f}s", flush=True)


def release_all(client) -> None:
    for name in client.list_collections():
        client.release_collection(name)


def _stored_price(p) -> float:
    return -1.0 if pd.isna(p) else float(p)  # same encoding as _rows


def restore_prices(client, name: str, products: pd.DataFrame, ids: set[str] | None = None, batch: int = 500) -> int:
    """Partial-update prices back to products.parquet's (write benchmarks change them). Returns rows sent."""
    p = products if ids is None else products[products.product_id.isin(ids)]
    for s in range(0, len(p), batch):
        ch = p.iloc[s:s + batch]
        client.upsert(name, [{"product_id": pid, "price": _stored_price(v)} for pid, v in zip(ch.product_id, ch.price)],
                      partial_update=True)
    client.flush(name)
    return len(p)


def drifted_ids(client, name: str, products: pd.DataFrame, tol: float = 0.005, **read_kw) -> set[str]:
    """product_ids whose stored price differs from products.parquet (or that are missing)."""
    it = client.query_iterator(name, batch_size=10_000, filter="", output_fields=["product_id", "price"], **read_kw)
    got = {}
    while rows := it.next():
        got.update({r["product_id"]: r["price"] for r in rows})
    it.close()
    return {pid for pid, v in zip(products.product_id, products.price)
            if pid not in got or abs(got[pid] - _stored_price(v)) > tol}


def price_drift(client, name: str, products: pd.DataFrame, tol: float = 0.005) -> int:
    """Rows whose stored price differs from products.parquet. Filtered ground truth is computed from the
    parquet, so a drifted collection silently scores the wrong rows."""
    return len(drifted_ids(client, name, products, tol))
