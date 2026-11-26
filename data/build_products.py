"""Join ESCI US products with AR23 metadata into data/work/products.parquet.

    uv run python -m data.build_products

row_id is a dense 0..N-1 index in sorted product_id order; every vector array
in the project is aligned to it.
"""
import numpy as np
import pandas as pd

from vsearch.config import DATA_RAW, DATA_WORK

TEXT_COLS = ["title", "brand", "color", "bullets", "description"]
MAX_EMBED_CHARS = 2000


def build_products(esci: pd.DataFrame, covered: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    us = esci[esci.product_locale == "us"].drop_duplicates("product_id")
    us = us.rename(columns={"product_title": "title", "product_brand": "brand",
                            "product_color": "color", "product_bullet_point": "bullets",
                            "product_description": "description"})
    df = covered.merge(us[["product_id", *TEXT_COLS]], on="product_id", how="inner")
    for c in TEXT_COLS:
        df[c] = df[c].fillna("").astype(str)
    df = df[df.title.str.strip() != ""]
    df = df.merge(meta.drop_duplicates("parent_asin"), left_on="product_id",
                  right_on="parent_asin", how="left").drop(columns="parent_asin")
    for c in ["price", "average_rating"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["rating_number"] = pd.to_numeric(df.rating_number, errors="coerce").fillna(0).astype("int64")
    for c in ["main_category", "store"]:
        df[c] = df[c].fillna("").astype(str)
    df["image_url"] = df.image_url.where(df.image_url.notna(), None)
    df["embed_text"] = (df.title + "\nBrand: " + df.brand + "\n" + df.bullets).str.slice(0, MAX_EMBED_CHARS)
    df = df.sort_values("product_id").reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df), dtype="int64"))
    return df


def main() -> None:
    esci = pd.read_parquet(DATA_RAW / "esci" / "products.parquet")
    covered = pd.read_parquet(DATA_WORK / "covered_ids.parquet")
    meta = pd.concat([pd.read_parquet(p) for p in sorted((DATA_WORK / "ar23_meta").glob("*.parquet"))],
                     ignore_index=True)
    df = build_products(esci, covered, meta)
    df.to_parquet(DATA_WORK / "products.parquet", index=False)
    print(f"products: {len(df):,} | price null {df.price.isna().mean():.1%} | "
          f"image null {df.image_url.isna().mean():.1%} | meta joined {df.main_category.ne('').mean():.1%}")


if __name__ == "__main__":
    main()
