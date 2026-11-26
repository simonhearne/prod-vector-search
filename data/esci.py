"""Fetch ESCI + the AR23 ASIN index, and build the covered-ID table.

    uv run python -m data.esci

Writes data/work/covered_ids.parquet (product_id, category) and
results/data_overlap.json.
"""
import json
import shutil
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

from vsearch.config import DATA_RAW, DATA_WORK
from vsearch.results import write_result

ESCI_URL = ("https://media.githubusercontent.com/media/amazon-science/esci-data/main/"
            "shopping_queries_dataset/shopping_queries_dataset_{name}.parquet")
A2C_URL = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
           "resolve/main/asin2category.json")


def _download(url: str, dest: Path) -> Path:
    """Download via a .part rename; no-op if dest exists."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    part.rename(dest)
    return dest


def fetch_esci(dest: Path = DATA_RAW / "esci") -> tuple[Path, Path]:
    return (_download(ESCI_URL.format(name="products"), dest / "products.parquet"),
            _download(ESCI_URL.format(name="examples"), dest / "examples.parquet"))


def fetch_asin2category(dest: Path = DATA_RAW / "ar23") -> Path:
    return _download(A2C_URL, dest / "asin2category.json")


def covered_ids(products: pd.DataFrame, a2c: dict[str, str]) -> pd.DataFrame:
    """US ESCI products whose product_id is an AR23 parent ASIN."""
    us = products[products.product_locale == "us"]
    hit = us[us.product_id.isin(a2c.keys())].drop_duplicates("product_id")
    return pd.DataFrame({"product_id": hit.product_id.values,
                         "category": [a2c[p] for p in hit.product_id]})


def overlap_report(products: pd.DataFrame, examples: pd.DataFrame,
                   a2c: dict[str, str]) -> dict:
    keys = a2c.keys()
    p = products.assign(hit=products.product_id.isin(keys))
    by_locale = {loc: {"products": int(len(g)), "covered": int(g.hit.sum()),
                       "share": float(g.hit.mean())}
                 for loc, g in p.groupby("product_locale")}
    ex = examples[examples.product_locale == "us"].assign(hit=lambda d: d.product_id.isin(keys))
    exact = ex.hit & (ex.esci_label == "E")
    q = ex.assign(exact=exact).groupby("query_id").agg(n=("hit", "size"), hit=("hit", "sum"),
                                                        exact=("exact", "sum"))
    cats = Counter(a2c[x] for x in p[(p.product_locale == "us") & p.hit].product_id)
    return {
        "a2c_keys": len(a2c),
        "by_locale": by_locale,
        "us_judgments": int(len(ex)),
        "us_judgment_coverage_by_label": {k: float(v) for k, v in ex.groupby("esci_label").hit.mean().items()},
        "us_queries": int(len(q)),
        "us_queries_ge1_exact": int((q.exact >= 1).sum()),
        "us_queries_ge3_exact": int((q.exact >= 3).sum()),
        "us_queries_ge80pct_covered": int(((q.hit / q.n) >= 0.8).sum()),
        "us_covered_by_category": dict(cats.most_common()),
    }


def main() -> None:
    prod_path, ex_path = fetch_esci()
    a2c = json.load(open(fetch_asin2category()))
    products = pd.read_parquet(prod_path, columns=["product_id", "product_locale"])
    examples = pd.read_parquet(ex_path)
    cov = covered_ids(products, a2c)
    DATA_WORK.mkdir(parents=True, exist_ok=True)
    cov.to_parquet(DATA_WORK / "covered_ids.parquet", index=False)
    write_result("data_overlap", overlap_report(products, examples, a2c))
    print(f"covered_ids: {len(cov):,} rows")


if __name__ == "__main__":
    main()
