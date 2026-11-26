"""The PoC query sets compared against eval5k in B1a.

    uv run python -m data.poc_sets      # -> data/poc_sets.json

* poc50  - the 50 ESCI queries the curated paraphrases were drawn from (stratified eval5k sample).
* head50 - a head-query proxy: ESCI has no frequency data, so "head" = eligible, not in ESCI's
           hard (small_version) subset, <= 3 words, no digits, and broad (every word appears in
           >= 500 product titles); stratified across categories.
* curated (not listed here) - the paraphrases in data/curated_queries.jsonl, as drafted.
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from vsearch.config import DATA_WORK, SEED

ROOT = Path(__file__).parent
CURATED = ROOT / "curated_queries.jsonl"
OUT = ROOT / "poc_sets.json"
_DIGIT = re.compile(r"\d")
_WORD = re.compile(r"\w+")
MIN_DF = 500


def is_head_like(query: str) -> bool:
    return len(query.split()) <= 3 and not _DIGIT.search(query)


def title_df(titles: pd.Series) -> Counter:
    """Document frequency of lowercase word tokens across product titles."""
    df = Counter()
    for t in titles:
        df.update(set(_WORD.findall(str(t).lower())))
    return df


def is_broad(query: str, df, min_df: int = MIN_DF) -> bool:
    toks = _WORD.findall(query.lower())
    return bool(toks) and all(df.get(t, 0) >= min_df for t in toks)


def stratified_pick(pool: pd.DataFrame, n: int = 50, seed: int = SEED) -> pd.DataFrame:
    """One per top-10 category, the rest proportional (largest remainder), seeded."""
    pool = pool.sort_values("query_id")
    counts = pool.top_category.value_counts()
    alloc = pd.Series(0, index=counts.index)
    alloc[counts.index[:10]] = 1
    share = counts / counts.sum() * (n - int(alloc.sum()))
    alloc += share.astype(int)
    remainder = (share - share.astype(int)).sort_values(ascending=False, kind="stable")
    alloc[remainder.index[: n - int(alloc.sum())]] += 1
    picks = [pool[pool.top_category == c].sample(n=min(int(k), int(counts[c])), random_state=seed)
             for c, k in alloc.items() if k > 0]
    return pd.concat(picks).reset_index(drop=True)


def pick_head(queries: pd.DataFrame, n: int = 50, seed: int = SEED, df=None,
              min_df: int = MIN_DF) -> pd.DataFrame:
    pool = queries[queries.eligible & ~queries.hard.astype(bool) & queries["query"].map(is_head_like)]
    if df is not None:
        pool = pool[pool["query"].map(lambda x: is_broad(x, df, min_df))]
    return stratified_pick(pool, n, seed)


def write_poc_sets(queries: pd.DataFrame, curated: Path = CURATED, out: Path = OUT, n: int = 50,
                   df=None) -> dict:
    cur = pd.read_json(curated, lines=True, dtype={"query_id": str})
    sets = {"poc50": cur.query_id.tolist(), "head50": pick_head(queries, n, df=df).query_id.tolist()}
    out.write_text(json.dumps(sets, indent=1))
    return sets


def main() -> None:
    titles = pd.read_parquet(DATA_WORK / "products.parquet", columns=["title"]).title
    sets = write_poc_sets(pd.read_parquet(DATA_WORK / "queries.parquet"), df=title_df(titles))
    print({k: len(v) for k, v in sets.items()})


if __name__ == "__main__":
    main()
