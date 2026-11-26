"""Curated "PoC" query set: descriptive paraphrases of sampled ESCI eval queries.

    uv run python -m data.paraphrase [--regenerate]

Each paraphrase keeps its source query's information need, so it inherits the
source's ESCI judgments. Rows are written to data/curated_queries.jsonl with
approved=false. The benchmarks use every row as drafted (none of the 50 was reviewed);
setting approved=true protects a row you have edited by hand from --regenerate.
"""
import argparse
import json
from pathlib import Path

import anthropic
import pandas as pd
from pydantic import BaseModel

from vsearch.config import DATA_WORK, SEED

MODEL = "claude-opus-5-5"
PATH = Path(__file__).parent / "curated_queries.jsonl"
PROMPT = """A shopper typed this query into an e-commerce search box:

<query>{query}</query>

Rewrite it as the query an engineer would write into a proof-of-concept test set for a semantic \
search engine: a short, descriptive natural-language phrase with the same intent, written as the \
search text itself (for example "airpods pro case" -> "protective case for my wireless earbuds"). \
Do not describe the shopper ("A shopper is looking for..."), do not explain what the product is, \
and do not resolve ambiguity by guessing: if the query is just a name or brand, keep it that open. \
Do not add requirements the shopper did not express. Leave out SKUs and model numbers unless \
they are essential to the intent. Keep it under 15 words."""


class Paraphrase(BaseModel):
    curated_query: str


def pick_sources(queries: pd.DataFrame, n: int = 50, seed: int = SEED) -> pd.DataFrame:
    """Stratified across top_category from the eval5k sample."""
    from data.poc_sets import stratified_pick
    return stratified_pick(queries[queries.in_eval5k], n, seed)


def paraphrase(client, query: str) -> str:
    resp = client.messages.parse(model=MODEL, max_tokens=1024, output_format=Paraphrase,
                                 messages=[{"role": "user", "content": PROMPT.format(query=query)}])
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"no paraphrase for {query!r}: {resp.stop_reason}")
    return resp.parsed_output.curated_query.strip()


def run(client, queries: pd.DataFrame, path: Path = PATH, regenerate: bool = False) -> pd.DataFrame:
    existing = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[str(r["query_id"])] = r
    rows = []
    for s in pick_sources(queries).itertuples():
        old = existing.get(s.query_id)
        if old and (old.get("approved") or not regenerate):
            rows.append(old)
            continue
        rows.append({"query_id": s.query_id, "esci_query": s.query,
                     "curated_query": paraphrase(client, s.query), "approved": False})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true", help="redo unapproved rows")
    a = ap.parse_args(argv)
    out = run(anthropic.Anthropic(), pd.read_parquet(DATA_WORK / "queries.parquet"), regenerate=a.regenerate)
    print(f"{len(out)} curated queries; approved {int(out.approved.sum())}")


if __name__ == "__main__":
    main()
