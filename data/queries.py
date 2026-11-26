"""ESCI US queries + judgments restricted to the corpus.

    uv run python -m data.queries

Writes data/work/queries.parquet and data/work/qrels.parquet. query_id is a string.
"""
import pandas as pd

from vsearch.config import DATA_RAW, DATA_WORK, EVAL_SAMPLE, SEED

MIN_EXACT = 3


def build_qrels(examples: pd.DataFrame, product_ids: set[str]) -> pd.DataFrame:
    us = examples[(examples.product_locale == "us") & examples.product_id.isin(product_ids)]
    return pd.DataFrame({"query_id": us.query_id.astype(str).values,
                         "product_id": us.product_id.values,
                         "label": us.esci_label.values}).drop_duplicates(["query_id", "product_id"])


def build_queries(examples: pd.DataFrame, qrels: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    text = (examples[examples.product_locale == "us"]
            .assign(query_id=lambda d: d.query_id.astype(str))
            .drop_duplicates("query_id"))
    text = text.assign(hard=text.small_version.astype(int) == 1)[["query_id", "query", "hard"]]
    exact = qrels[qrels.label == "E"].merge(products[["product_id", "category"]], on="product_id")
    agg = exact.groupby("query_id").agg(
        n_exact=("product_id", "size"),
        top_category=("category", lambda s: s.value_counts().index[0]))
    q = text[text.query_id.isin(set(qrels.query_id))].merge(agg, on="query_id", how="left")
    q["n_exact"] = q.n_exact.fillna(0).astype("int64")
    q["top_category"] = q.top_category.fillna("")
    q["eligible"] = q.n_exact >= MIN_EXACT
    q["in_eval5k"] = q.query_id.isin(set(sample_eval(q)))
    return q.reset_index(drop=True)


def sample_eval(queries: pd.DataFrame, n: int = EVAL_SAMPLE, seed: int = SEED) -> pd.Index:
    elig = queries[queries.eligible].sort_values("query_id")
    return pd.Index(elig.sample(n=min(n, len(elig)), random_state=seed).query_id)


def main() -> None:
    examples = pd.read_parquet(DATA_RAW / "esci" / "examples.parquet")
    products = pd.read_parquet(DATA_WORK / "products.parquet", columns=["product_id", "category"])
    qrels = build_qrels(examples, set(products.product_id))
    queries = build_queries(examples, qrels, products)
    qrels.to_parquet(DATA_WORK / "qrels.parquet", index=False)
    queries.to_parquet(DATA_WORK / "queries.parquet", index=False)
    print(f"qrels {len(qrels):,} | queries {len(queries):,} | eligible {queries.eligible.sum():,} "
          f"| eval5k {queries.in_eval5k.sum():,}")


if __name__ == "__main__":
    main()
