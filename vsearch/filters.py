"""Structured filters that render to a Milvus expression AND an equivalent pandas mask,
plus the production-like filter mix used by B2."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vsearch.config import K, SEED

BUCKETS = [("unfiltered", 1.0, 1.0), ("10-50%", 0.10, 0.50), ("1-10%", 0.01, 0.10),
           ("0.1-1%", 0.001, 0.01), ("<0.1%", 0.0, 0.001)]


def escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


@dataclass(frozen=True)
class Filter:
    category: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    rating_min: float | None = None
    brand: str | None = None

    def to_expr(self) -> str | None:
        parts = []
        if self.category is not None:
            parts.append(f'category == "{escape(self.category)}"')
        if self.price_min is not None or self.price_max is not None:
            parts.append("price >= 0")
        if self.price_min is not None:
            parts.append(f"price >= {self.price_min:g}")
        if self.price_max is not None:
            parts.append(f"price <= {self.price_max:g}")
        if self.rating_min is not None:
            parts.append(f"average_rating >= {self.rating_min}")
        if self.brand is not None:
            parts.append(f'brand == "{escape(self.brand)}"')
        return " and ".join(parts) or None

    def mask(self, products: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(products), dtype=bool)
        if self.category is not None:
            m &= (products.category == self.category).to_numpy()
        price = products["price"].to_numpy(float) if "price" in products else None
        if self.price_min is not None or self.price_max is not None:
            m &= np.nan_to_num(price, nan=-1.0) >= 0
        if self.price_min is not None:
            m &= np.nan_to_num(price, nan=-1.0) >= float(f"{self.price_min:g}")
        if self.price_max is not None:
            m &= np.where(np.isnan(price), False, price <= float(f"{self.price_max:g}"))
        if self.rating_min is not None:
            m &= np.nan_to_num(products.average_rating.to_numpy(float), nan=-1.0) >= self.rating_min
        if self.brand is not None:
            m &= (products.brand == self.brand).to_numpy()
        return m


def bucket_of(selectivity: float, filtered: bool) -> str | None:
    if not filtered:
        return "unfiltered"
    for name, lo, hi in BUCKETS[1:]:
        if lo < selectivity <= hi:
            return name
    return None


def _ladder(q, exact: pd.DataFrame) -> list[Filter]:
    cat = q.top_category or None
    steps = [Filter()]
    if cat is None:
        return steps
    steps.append(Filter(category=cat))
    prices = exact.price.dropna()
    prices = prices[prices >= 0]
    if len(prices):
        med = float(np.median(prices))
        band = dict(price_min=round(med * 0.5, 2), price_max=round(med * 2.0, 2))
        steps.append(Filter(category=cat, **band))
        steps.append(Filter(category=cat, rating_min=4.0, **band))
        brands = exact.brand[exact.brand.astype(str).str.len() > 0]
        if len(brands):
            steps.append(Filter(category=cat, rating_min=4.0, brand=brands.value_counts().index[0], **band))
    return steps


def make_filter_mix(queries: pd.DataFrame, products: pd.DataFrame, qrels: pd.DataFrame,
                    per_bucket: int = 200, seed: int = SEED) -> pd.DataFrame:
    """One candidate per (query, bucket) from a loose->tight ladder; drops filters with < K matches."""
    n = len(products)
    pinfo = products.set_index("product_id")[["price", "brand"]]
    ex = qrels[qrels.label == "E"].join(pinfo, on="product_id", how="inner")
    by_q = dict(tuple(ex.groupby("query_id")))
    empty = pinfo.iloc[0:0]
    rows, mask_cache = [], {}
    for q in queries.sort_values("query_id").itertuples():
        seen = set()
        for f in _ladder(q, by_q.get(q.query_id, empty)):
            expr = f.to_expr()
            if expr not in mask_cache:
                mask_cache[expr] = int(f.mask(products).sum())
            nm = mask_cache[expr]
            b = bucket_of(nm / n, filtered=expr is not None)
            if b is None or b in seen or nm < K:
                continue
            seen.add(b)
            rows.append({"query_id": q.query_id, "bucket": b, "filter": f, "expr": expr or "",
                         "selectivity": nm / n, "n_matches": nm})
    mix = pd.DataFrame(rows, columns=["query_id", "bucket", "filter", "expr", "selectivity", "n_matches"])
    order = {b[0]: i for i, b in enumerate(BUCKETS)}
    parts = [g.sample(n=min(per_bucket, len(g)), random_state=seed) for _, g in mix.groupby("bucket")]
    if not parts:
        return mix
    out = pd.concat(parts)
    out = out.assign(_o=out.bucket.map(order)).sort_values(["_o", "query_id"], kind="stable")
    return out.drop(columns="_o").reset_index(drop=True)
