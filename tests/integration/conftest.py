import numpy as np
import pandas as pd
import pytest
from pymilvus import MilvusClient

from vsearch.config import MILVUS_URI

SMALL = "t_small"
N = 2000


def synthetic_corpus(n=N, dim=1024, seed=27):
    rng = np.random.default_rng(seed)
    unit = lambda a: (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)
    prods = pd.DataFrame({
        "row_id": np.arange(n), "product_id": [f"P{i:05d}" for i in range(n)],
        "title": [f"product {i}" for i in range(n)], "brand": [f"brand{i % 7}" for i in range(n)],
        "category": ["Books" if i % 3 == 0 else "Electronics" for i in range(n)],
        "price": [np.nan if i % 11 == 0 else float(i % 100) for i in range(n)],
        "average_rating": [float(i % 5) for i in range(n)], "rating_number": np.arange(n) % 50,
        "image_url": ["u"] * n,
        "embed_text": [f"product {i} usb cable charger" if i % 2 else f"product {i} garden hose" for i in range(n)]})
    has = np.arange(n) % 2 == 0
    return prods, unit(rng.normal(size=(n, dim))), unit(rng.normal(size=(n, dim))), has


@pytest.fixture(scope="session")
def client():
    return MilvusClient(uri=MILVUS_URI)


@pytest.fixture(scope="session")
def small(client):
    from dataclasses import replace

    from vsearch.collections import SPECS, build_collection
    corpus = synthetic_corpus()
    if not client.has_collection(SMALL):
        build_collection(client, replace(SPECS["prod_tuned"], name=SMALL), *corpus)
    else:
        client.load_collection(SMALL)
    return SMALL, corpus
