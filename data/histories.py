"""Held-out purchases from Amazon Reviews 2023 (McAuley-Lab, 5-core, last-out), for B6.

    uv run python -I -c "import sys; sys.path.insert(0, '.'); from data.histories import main; main()"

Downloads the pinned CSVs to data/work/ar23_seq/ (3.2 GB) and writes data/work/b6_users.parquet: one row per user,
their earlier purchases, the last one held out, and an ESCI query nobody else has for which it is Exact.
Reviews stand in for purchases. User IDs are pseudonymous: everything here stays in data/work (git-ignored)."""
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from bench._common import _src
from vsearch.config import DATA_WORK, SEED

REV = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
BASE = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/{REV}/benchmark/5core/last_out"
CATEGORIES = ["Arts_Crafts_and_Sewing", "Automotive", "Baby_Products", "Beauty_and_Personal_Care",
              "Cell_Phones_and_Accessories", "Grocery_and_Gourmet_Food", "Health_and_Household",
              "Industrial_and_Scientific", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden",
              "Pet_Supplies", "Sports_and_Outdoors", "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games"]
SPLITS = ("train", "valid", "test")
SEQ, USERS = DATA_WORK / "ar23_seq", DATA_WORK / "b6_users.parquet"
HERE = Path(__file__).parent
N_TUNE, N_TEST = 1_000, 5_000


def protected_query_ids() -> set[str]:
    """Queries any talk number is scored on: eval5k (the monitor's golden set, B5 and the showcase sit inside it),
    the PoC sets and the curated rewrites."""
    q = pd.read_parquet(_src("queries.parquet"), columns=["query_id", "in_eval5k"])
    sets = json.loads((HERE / "poc_sets.json").read_text()).values()
    cur = pd.read_json(HERE / "curated_queries.jsonl", lines=True, dtype={"query_id": str})
    return set(q[q.in_eval5k].query_id) | {i for s in sets if isinstance(s, list) for i in s} | set(cur.query_id)


def build_interactions(frames: Iterable[tuple[str, pd.DataFrame]], product_ids: set[str]) -> pd.DataFrame:
    """Covered rows only, the first purchase per (user, product)."""
    parts = [f.loc[f.parent_asin.isin(product_ids), ["user_id", "parent_asin", "timestamp"]].assign(cat=cat)
             for cat, f in frames]
    x = pd.concat(parts, ignore_index=True).rename(columns={"parent_asin": "product_id"})
    x = x.sort_values(["user_id", "timestamp", "product_id", "cat"])
    return x.drop_duplicates(["user_id", "product_id"]).reset_index(drop=True)


def held_out(x: pd.DataFrame, min_history: int = 3) -> pd.DataFrame:
    """Each user's last purchase (item), and what they bought strictly before it (history)."""
    last = x.groupby("user_id", sort=True).tail(1).set_index("user_id")
    before = x[x.timestamp < x.user_id.map(last.timestamp)]
    hist = before.groupby("user_id", sort=True).product_id.agg(list)
    hist = hist[hist.map(len) >= min_history]
    last = last.loc[hist.index]
    return pd.DataFrame({"user_id": hist.index, "item": last.product_id.to_numpy(), "cat": last["cat"].to_numpy(),
                         "history": hist.to_numpy()})


def assign_queries(h: pd.DataFrame, qrels: pd.DataFrame, vector_ids: set[str], protected: set[str],
                   seed: int = SEED) -> pd.DataFrame:
    """One query per user, used by nobody else: rankings are keyed by query, and paired CIs need independent ones."""
    e = qrels[(qrels.label == "E") & qrels.query_id.isin(vector_ids) & ~qrels.query_id.isin(protected)]
    options = e.sort_values("query_id").groupby("product_id").query_id.agg(list).to_dict()
    h = h[h.item.isin(options)].sort_values("user_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    used, keep, qids = set(), [], []
    for i in rng.permutation(len(h)):
        free = [q for q in options[h.item[i]] if q not in used]
        if free:
            used.add(q := free[rng.integers(len(free))])
            keep.append(i)
            qids.append(q)
    return h.iloc[keep].reset_index(drop=True).assign(query_id=qids)


def _draw(rng, pool: np.ndarray, i: int, item: str, histories: list[set[str]], tries: int = 50) -> int:
    """A random member of pool other than i who hasn't bought i's held-out item, or -1."""
    for _ in range(tries):
        j = int(pool[rng.integers(len(pool))])
        if j != i and item not in histories[j]:
            return j
    return -1


def assign_controls(h: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """stranger: any other user. peer: another user whose held-out item is in the same category. Row positions."""
    h = h.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    histories = [set(x) for x in h.history]
    everyone = np.arange(len(h))
    by_cat = {c: np.flatnonzero((h["cat"] == c).to_numpy()) for c in h["cat"].unique()}
    stranger = [_draw(rng, everyone, i, it, histories) for i, it in enumerate(h.item)]
    peer = [_draw(rng, by_cat[c], i, it, histories) for i, (it, c) in enumerate(zip(h.item, h["cat"]))]
    return h.assign(stranger=stranger, peer=peer)


def _download() -> None:
    SEQ.mkdir(parents=True, exist_ok=True)
    for name in (f"{c}.{s}.csv" for c in CATEGORIES for s in SPLITS):
        dest = SEQ / name
        if dest.exists():
            continue
        part = dest.with_suffix(".csv.part")
        with requests.get(f"{BASE}/{name}", stream=True, timeout=60) as r, open(part, "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        part.rename(dest)
        print(name, flush=True)


def _frames() -> Iterable[tuple[str, pd.DataFrame]]:
    for c in CATEGORIES:
        yield c, pd.concat((pd.read_csv(SEQ / f"{c}.{s}.csv", usecols=["user_id", "parent_asin", "timestamp"])
                            for s in SPLITS), ignore_index=True)


def build_users() -> pd.DataFrame:
    ids = set(pd.read_parquet(_src("products.parquet"), columns=["product_id"]).product_id)
    h = held_out(build_interactions(_frames(), ids))
    u = assign_queries(h, pd.read_parquet(_src("qrels.parquet")),
                       set(np.load(_src("q_ids.npy")).tolist()), protected_query_ids())
    print(f"users with >=3 earlier items {len(h):,} | with their own query {len(u):,}", flush=True)
    if len(u) < N_TUNE + N_TEST:
        raise ValueError(f"{len(u)} users, need {N_TUNE + N_TEST}")
    tune = assign_controls(u.iloc[:N_TUNE]).assign(split="tune")
    test = assign_controls(u.iloc[N_TUNE:N_TUNE + N_TEST]).assign(split="test")
    return pd.concat([tune, test], ignore_index=True)


def load_users() -> pd.DataFrame:
    """stranger and peer are positions within the user's own split. Parquet returns arrays; history becomes lists."""
    u = pd.read_parquet(USERS)
    return u.assign(history=u.history.map(list))


def main() -> None:
    _download()
    u = build_users()
    u.to_parquet(USERS, index=False)
    print(u.groupby("split").agg(users=("user_id", "nunique"), queries=("query_id", "nunique"),
                                 no_stranger=("stranger", lambda s: int((s < 0).sum())),
                                 no_peer=("peer", lambda s: int((s < 0).sum()))))
    print("protected queries used:", len(set(u.query_id) & protected_query_ids()),
          "| queries shared:", len(u) - u.query_id.nunique())


if __name__ == "__main__":
    main()
