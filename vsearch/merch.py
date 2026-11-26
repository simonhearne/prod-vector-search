"""Seeded, synthetic merchandising fields for B5: the catalogue has no real margins, campaigns or hide flags.
Values depend only on the set of products and the seed, never on row order."""
import numpy as np
import pandas as pd

from vsearch.config import SEED

CAMPAIGN = "bf26"
CAMPAIGN_SHARE, CAMPAIGN_MAX = 0.05, 0.07
HIDDEN_SHARE = 0.05


def _campaign_brands(brand: pd.Series, rng: np.random.Generator) -> set[str]:
    """Whole brands, in seeded random order, until CAMPAIGN_SHARE of products are covered."""
    size = brand[brand != ""].value_counts()
    chosen, covered = set(), 0
    for b in rng.permutation(np.array(sorted(size.index), dtype=object)):
        if covered >= CAMPAIGN_SHARE * len(brand):
            break
        if covered + size[b] <= CAMPAIGN_MAX * len(brand):
            chosen.add(b)
            covered += size[b]
    return chosen


def merch_fields(products: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """margin (0.02-0.70, centred per category), campaigns ([CAMPAIGN] on whole brands) and hidden, aligned to
    `products`' index and order."""
    p = products.sort_values("product_id")
    rng = np.random.default_rng(seed)
    cats = sorted(p.category.unique())
    centre = p.category.map(dict(zip(cats, rng.uniform(0.10, 0.50, len(cats))))).to_numpy(float)
    margin = np.clip(centre + rng.normal(0, 0.10, len(p)), 0.02, 0.70)
    brand = p.brand.fillna("").astype(str)
    on = brand.isin(_campaign_brands(brand, rng))
    out = pd.DataFrame({"margin": margin, "campaigns": [[CAMPAIGN] if x else [] for x in on],
                        "hidden": rng.random(len(p)) < HIDDEN_SHARE}, index=p.index)
    return out.loc[products.index]
