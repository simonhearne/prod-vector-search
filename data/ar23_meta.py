"""Stream-filter Amazon Reviews 2023 metadata down to the covered product IDs.

    uv run python -m data.ar23_meta [--categories "Books" "Toys and Games" ...]

~101 GB is streamed over HTTP (never stored); each category's matches are
written to data/work/ar23_meta/<Category>.parquet. Categories whose parquet
already exists are skipped, so the job is resumable per category.
"""
import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator

import pandas as pd
import requests

from vsearch.config import DATA_WORK

BASE = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
        "resolve/main/raw/meta_categories/meta_{}.jsonl")
OUT = DATA_WORK / "ar23_meta"
_PARENT = re.compile(rb'"parent_asin":\s*"([^"]+)"')
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def meta_url(category: str) -> str:
    return BASE.format(category.replace(" ", "_"))


def parse_price(raw) -> float | None:
    """'$12.99' -> 12.99; ranges take the low end; anything unparseable -> None."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    m = _NUMBER.search(str(raw))
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def _image_url(images) -> str | None:
    if not images:
        return None
    first = images[0] or {}
    return first.get("hi_res") or first.get("large") or None


def filter_meta_lines(lines: Iterable[bytes], ids: set[str]) -> Iterator[dict]:
    for line in lines:
        m = _PARENT.search(line)
        if not m or m.group(1).decode() not in ids:
            continue
        r = json.loads(line)
        yield {"parent_asin": r["parent_asin"],
               "main_category": r.get("main_category"),
               "price": parse_price(r.get("price")),
               "average_rating": r.get("average_rating"),
               "rating_number": r.get("rating_number"),
               "store": r.get("store"),
               "image_url": _image_url(r.get("images"))}


def fetch_category(category: str, ids: set[str]) -> pd.DataFrame:
    with requests.get(meta_url(category), stream=True, timeout=60) as resp:
        resp.raise_for_status()
        rows = list(filter_meta_lines(resp.iter_lines(chunk_size=1 << 20), ids))
    return pd.DataFrame(rows, columns=["parent_asin", "main_category", "price", "average_rating",
                                       "rating_number", "store", "image_url"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*")
    args = ap.parse_args(argv)
    cov = pd.read_parquet(DATA_WORK / "covered_ids.parquet")
    by_cat = cov.groupby("category").product_id.apply(set)
    cats = args.categories or by_cat.sort_values(key=lambda s: s.map(len)).index.tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    for cat in cats:
        dest = OUT / f"{cat.replace(' ', '_')}.parquet"
        if dest.exists():
            print(f"skip {cat}", flush=True)
            continue
        df = fetch_category(cat, by_cat[cat])
        part = dest.with_suffix(".parquet.part")
        df.to_parquet(part, index=False)
        part.rename(dest)
        print(f"{cat}: {len(df):,}/{len(by_cat[cat]):,}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
