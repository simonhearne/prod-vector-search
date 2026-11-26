"""B6: does a user's profile help the right user? README row R11. Real review histories, last item held out (data.histories).
Every profile arm runs three times: with the user's own profile, a stranger's, and a same-category peer's. An arm counts
as personalisation only if the own profile beats no profile, popularity and the stranger. Weights are tuned on other users.
Already-bought items are dropped from every list. Judgments go to B6's own pool, so the talk's pool doesn't move.
The cache (data/work/b6_rankings.json) holds top 10s and held-out ranks by query id, never a user id.

    uv run python -m bench.b6_perso [--sample 50] [--tune-only] [--no-fill] [--score-only] [--n-guard 500]   -> results/b6.json"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bench._common import _src, client, ensure_collection, ensure_parent, load_corpus, parser, qrels, quality
from data.histories import load_users
from vsearch import judge
from vsearch.collections import brand_ids, release_all
from vsearch.config import DATA_WORK, K
from vsearch.load import level_start
from vsearch.metrics import ndcg_at_k
from vsearch.profiles import (blend, brand_filter, brand_variants, brand_weights, fold, moved, price_filter, rerank, rr,
                              user_vec)
from vsearch.results import _git, write_result
from vsearch.rules import boost, combined
from vsearch.search import run_brand_hybrid, run_queries
from vsearch.stats import paired_ci

COLL, EF, DEPTH = "prod_perso", 512, 200
BOOSTS, MIX = [1.02, 1.05, 1.1, 1.2, 1.5], [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
GRID = {"popularity": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0], "brand_boost": BOOSTS, "brand_sparse": [0.05, 0.1, 0.2, 0.35, 0.5],
        "price_band": BOOSTS, "user_blend": MIX, "user_rerank": MIX}
PROFILE_ARMS = ["brand_boost", "brand_sparse", "price_band", "user_blend", "user_rerank", "brand_price"]
PROFILES = ["own", "stranger", "peer"]
BROAD_MIN_BRANDS = 4
LIMITS = ["Reviews stand in for purchases, and the users did not issue these queries.",
          "Each query is chosen so the held-out item is Exact, so absolute MRR means little: only differences between runs count.",
          "Only covered products count, so histories are thin.",
          "rating_number is all-time, so popularity has seen the future.",
          "price_band is absolute price, whatever the product type.",
          "Engine boosts rescore up to 200 candidates per segment; application-side arms rerank the base 200.",
          "brand_sparse is weighted on normalised scores, and fetches the brand whatever the query.",
          "Arms are compared with their own controls, not with each other.",
          "With a stranger's profile the brand arms barely act (see moved@10): their brands rarely match the candidates, so for "
          "those arms own-minus-stranger is close to own-minus-none, and the peer is the nearer control."]


def rankings_path(sample: bool) -> Path:
    return DATA_WORK / ("b6_rankings.sample.json" if sample else "b6_rankings.json")


def pool_path(sample: bool) -> Path:
    return DATA_WORK / ("judgments_b6.sample.parquet" if sample else "judgments_b6.parquet")


def entry(ids: list[str], bought: set[str], item: str) -> dict:
    """The page and the held-out item's rank, once already-bought items are gone."""
    kept = [p for p in ids if p not in bought]
    return {"ids": kept[:K], "rank": kept.index(item) + 1 if item in kept else None}


def merge_base(base: list[list[str]], who: list[int], got: list[list[str]]) -> list[list[str]]:
    """The arm's lists for the users with a signal; the base list, untouched, for everyone else."""
    out = list(base)
    for i, ids in zip(who, got, strict=True):
        out[i] = ids
    return out


def profile_rows(u: pd.DataFrame, profile: str) -> list[int | None]:
    """Whose profile each row uses (positions in the split); None when there is no such user."""
    rows = range(len(u)) if profile == "own" else u[profile]
    return [int(j) if j >= 0 else None for j in rows]


def fill_keys() -> list[str]:
    return [f"test|{a}|own" for a in ("none", "popularity", *PROFILE_ARMS)]


class Runner:
    """Runs arms over one split. `n` limits the users searched; profiles cover the whole split, for the controls."""

    def __init__(self, c, u: pd.DataFrame, n: int, products: pd.DataFrame, tv: np.ndarray, qv: np.ndarray):
        self.c, self.u, self.qv, self.tv = c, u.iloc[:n].reset_index(drop=True), qv[:n], tv
        self.row_of = dict(zip(products.product_id, products.row_id))
        pop = np.log1p(products.rating_number.to_numpy(float))
        self.pop = dict(zip(products.product_id, pop / pop.max()))
        brand_of = dict(zip(products.product_id, products.brand))
        price_of = dict(zip(products.product_id, products.price))
        variants, dims = brand_variants(products.brand), brand_ids(products)
        self.prof = []
        for h in u.history:
            w = brand_weights(h, brand_of)
            self.prof.append({"brand": brand_filter(w, variants), "price": price_filter(h, price_of),
                              "sparse": {dims[b]: x for b, x in w.items()} or None,
                              "vec": user_vec([self.row_of[p] for p in h], tv)})
        self.bought = [set(h) for h in self.u.history]
        r = run_queries(c, COLL, "text_vec", self.qv, k=DEPTH, search_params={"ef": EF})
        self.base, self.scores = r.ids, r.scores

    def cached(self, lists: list[list[str]]) -> dict:
        e = [entry(i, b, it) for i, b, it in zip(lists, self.bought, self.u["item"])]
        return {"ids": {q: x["ids"] for q, x in zip(self.u.query_id, e)},
                "rank": {q: x["rank"] for q, x in zip(self.u.query_id, e)}}

    def _docs(self, i: int) -> np.ndarray:
        return self.tv[[self.row_of[p] for p in self.base[i]]]

    def _ranker(self, arm: str, p: dict, w) -> dict | None:
        wb, wp = w if arm == "brand_price" else (w, w)
        fs = []
        if arm != "price_band" and p["brand"]:
            fs.append(boost(p["brand"], wb, name="brand"))
        if arm != "brand_boost" and p["price"]:
            fs.append(boost(p["price"], wp, name="price"))
        if not fs:
            return None
        return {"ranker": fs[0] if len(fs) == 1 else combined(fs, "Multiply")}

    def run(self, arm: str, w, profile: str = "own") -> tuple[dict, float]:
        """The cached form of one run, and the share of users with a signal."""
        if arm == "none":
            return self.cached(self.base), 1.0
        n = len(self.u)
        if arm == "popularity":
            lists = [rerank(i, s, np.array([self.pop[p] for p in i]), w) for i, s in zip(self.base, self.scores)]
            return self.cached(lists), 1.0
        ps = [None if j is None else self.prof[j] for j in profile_rows(self.u, profile)]
        if arm == "user_rerank":
            who = [i for i in range(n) if ps[i]]
            got = [rerank(self.base[i], self.scores[i], self._docs(i) @ ps[i]["vec"], w) for i in who]
        elif arm == "user_blend":
            who = [i for i in range(n) if ps[i]]
            qv = np.stack([blend(self.qv[i], ps[i]["vec"], w) for i in who])
            got = run_queries(self.c, COLL, "text_vec", qv, k=DEPTH, search_params={"ef": EF}).ids
        elif arm == "brand_sparse":
            who = [i for i in range(n) if ps[i] and ps[i]["sparse"]]
            got = run_brand_hybrid(self.c, COLL, self.qv[who], [ps[i]["sparse"] for i in who], k=DEPTH,
                                   dense_weight=1 - w, dense_params={"ef": EF}).ids
        else:
            extra = [self._ranker(arm, p, w) if p else None for p in ps]
            who = [i for i in range(n) if extra[i]]
            got = run_queries(self.c, COLL, "text_vec", self.qv[who], k=DEPTH, search_params={"ef": EF},
                              extra=[extra[i] for i in who]).ids
        return self.cached(merge_base(self.base, who, got)), len(who) / n


def _ranks(cache: dict, key: str, qids: list[str]) -> np.ndarray:
    return np.array([rr(cache["runs"][key]["rank"][q]) for q in qids])


def tune_table(cache: dict) -> dict:
    qids = cache["query_ids"]["tune"]
    mrr = lambda key: float(_ranks(cache, key, qids).mean())
    return {"none": mrr("tune|none|0"), **{a: {str(w): mrr(f"tune|{a}|{w}") for w in g} for a, g in GRID.items()}}


def pick_weights(cache: dict) -> dict[str, float]:
    """Best tune-split MRR among non-zero weights (smallest on a tie): an arm that hurts is reported, not switched off."""
    t = tune_table(cache)
    return {a: max(g, key=lambda w: (t[a][str(w)], -w)) for a, g in GRID.items()}


def at_grid_edge(cache: dict) -> dict[str, bool]:
    """The best weight is the largest tried, so the arm may be under-tuned."""
    return {a: w == GRID[a][-1] for a, w in pick_weights(cache).items()}


def check_cache(cache: dict | None, qids: dict[str, list[str]]) -> None:
    """Refuse rankings from another run: other users, a shorter run, or test runs made at other weights."""
    if not cache or not set(fill_keys()) <= set(cache["runs"]):
        raise ValueError("no test runs in the cache; run without --score-only")
    if cache["query_ids"] != qids:
        raise ValueError("cached query_ids differ from the requested ones; rerun without --score-only")
    if cache.get("weights") != pick_weights(cache):
        raise ValueError("the cached test runs used other weights than the tuning runs pick; rerun without --score-only")


def score(cache: dict, users: pd.DataFrame, qr: pd.DataFrame, guard_ids: list[str], brand_of: dict[str, str]) -> dict:
    want = {s: users[users.split == s].query_id.tolist()[:len(cache["query_ids"][s])] for s in ("tune", "test")}
    if want != {s: list(cache["query_ids"][s]) for s in want}:
        raise ValueError("cached query_ids differ from the user set; rerun without --score-only")
    qids, runs = want["test"], cache["runs"]
    u = users[users.split == "test"].iloc[:len(qids)].reset_index(drop=True)
    hit = lambda key: np.array([(r := runs[key]["rank"][q]) is not None and r <= K for q in qids], float)
    vals = lambda key: {"rr": _ranks(cache, key, qids), "hit@10": hit(key)}
    none, pop = vals("test|none|own"), vals("test|popularity|own")
    none_ids = [runs["test|none|own"]["ids"][q] for q in qids]
    labels = judge.filled_qrels(qr[qr.query_id.isin(set(guard_ids))])
    lab = {q: dict(zip(g.product_id, g.label)) for q, g in labels.groupby("query_id")}
    nd = lambda key: np.array([ndcg_at_k(runs[key]["ids"][q], lab.get(q, {})) for q in guard_ids])
    exact = qr[(qr.label == "E") & qr.query_id.isin(set(qids))]
    n_brands = exact.assign(b=[fold(brand_of.get(p)) for p in exact.product_id]).query("b != ''").groupby("query_id").b.nunique()
    wide = u.query_id.map(n_brands).fillna(0).to_numpy() >= BROAD_MIN_BRANDS
    length = u.history.map(len).to_numpy()
    splits = {"query": {"specific": ~wide, "broad": wide},
              "history": {"3-4": length <= 4, "5-9": (length >= 5) & (length <= 9), "10+": length >= 10}}

    def cell(key: str) -> dict:
        v = vals(key)
        return {"mrr": float(v["rr"].mean()), "hit@10": float(v["hit@10"].mean()),
                "moved@10": float(np.mean([moved(a, runs[key]["ids"][q]) for a, q in zip(none_ids, qids)]))}

    def guard(key: str) -> dict:
        f = quality({q: runs[key]["ids"][q] for q in guard_ids}, qr[qr.query_id.isin(set(guard_ids))])["filled"]
        return {"n": len(guard_ids), **{k: f[k] for k in ("ndcg@10", "ndcg@10_ci95", "judged@10")},
                "own_minus_none": list(paired_ci(nd(key) - nd("test|none|own")))}

    def arm(a: str) -> dict:
        v = {p: vals(f"test|{a}|{p}") for p in PROFILES}
        diff = {f"own_minus_{name}": {m: list(paired_ci(v["own"][m] - rival[m])) for m in ("rr", "hit@10")}
                for name, rival in (("none", none), ("popularity", pop), ("stranger", v["stranger"]), ("peer", v["peer"]))}
        beats = {n: bool((c := diff[f"own_minus_{n}"]["rr"])[0] - c[1] > 0) for n in ("none", "popularity", "stranger")}
        d = v["own"]["rr"] - v["stranger"]["rr"]
        sp = {k: {name: {"n": int(m.sum()), "own_minus_stranger": list(paired_ci(d[m])) if m.sum() > 1 else None,
                         "own_minus_none": list(paired_ci((v["own"]["rr"] - none["rr"])[m])) if m.sum() > 1 else None}
                  for name, m in masks.items()} for k, masks in splits.items()}
        return {**{p: {**cell(f"test|{a}|{p}"), "eligible": cache["eligible"][a][p]} for p in PROFILES}, **diff,
                "beats": beats, "counts": all(beats.values()), "guard": guard(f"test|{a}|own"), "splits": sp}

    in_hist = [bool(b := fold(brand_of.get(it))) and b in {fold(brand_of.get(p)) for p in h} for it, h in zip(u["item"], u.history)]
    ranks = [runs["test|none|own"]["rank"][q] for q in qids]
    return {"n_tune": len(want["tune"]), "n_test": len(qids), "n_guard": len(guard_ids), "limits": LIMITS, "measured": cache.get("measured", {}),
            "ceiling": {"brand_in_history": float(np.mean(in_hist)),
                        "base_top10": float(np.mean([r is not None and r <= K for r in ranks])),
                        "base_top200": float(np.mean([r is not None for r in ranks]))},
            "tune": tune_table(cache), "weights": pick_weights(cache), "weight_at_grid_edge": at_grid_edge(cache),
            "none": {**cell("test|none|own"), "guard": guard("test|none|own")},
            "popularity": {**cell("test|popularity|own"), "guard": guard("test|popularity|own"),
                           "minus_none": {m: list(paired_ci(pop[m] - none[m])) for m in ("rr", "hit@10")}},
            "arms": {a: arm(a) for a in PROFILE_ARMS}}


def _vectors(qids: list[str]) -> np.ndarray:
    pos = {k: i for i, k in enumerate(np.load(_src("q_ids.npy")))}
    return np.asarray(np.load(_src("q_text_vec.npy"), mmap_mode="r")[[pos[q] for q in qids]])


def _measure(cache: dict, path: Path, c, users: pd.DataFrame, n: dict[str, int], products, tv, tune_only: bool) -> None:
    release_all(c)
    ensure_collection(c, COLL)
    quiet = level_start(c, COLL, len(products))[1]
    segments = sum(getattr(s.state, "name", s.state) == "Sealed" for s in c.list_loaded_segments(COLL))
    measured = {"segments": segments, "level_quiet": quiet, "milvus_version": c.get_server_version(), "git_sha": _git("rev-parse", "HEAD"),
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    docs = np.asarray(tv, dtype=np.float32)

    def runner(split: str) -> Runner:
        u = users[users.split == split].reset_index(drop=True)
        return Runner(c, u, n[split], products, docs, _vectors(u.query_id.tolist()))

    def go(r: Runner, key: str, arm: str, w, profile: str = "own") -> None:
        cache["runs"][key], share = r.run(arm, w, profile)
        if key.startswith("test|") and arm in PROFILE_ARMS:
            cache["eligible"].setdefault(arm, {})[profile] = share
        mrr = np.mean([rr(x) for x in cache["runs"][key]["rank"].values()])
        print(f"{key}: mrr {mrr:.4f} eligible {share:.2f}", flush=True)

    tune_keys = {"tune|none|0"} | {f"tune|{a}|{w}" for a, g in GRID.items() for w in g}
    if not tune_keys <= set(cache["runs"]):
        r = runner("tune")
        cache.setdefault("measured", {})["tune"] = measured
        go(r, "tune|none|0", "none", 0)
        for a, g in GRID.items():
            for w in g:
                go(r, f"tune|{a}|{w}", a, w)
        path.write_text(json.dumps(cache))
    weights = pick_weights(cache)
    print("weights", weights, flush=True)
    if tune_only:
        return
    r = runner("test")
    cache.setdefault("measured", {})["test"] = measured
    cache["weights"] = weights
    go(r, "test|none|own", "none", 0)
    go(r, "test|popularity|own", "popularity", weights["popularity"])
    for a in PROFILE_ARMS:
        w = (weights["brand_boost"], weights["price_band"]) if a == "brand_price" else weights[a]
        for p in PROFILES:
            go(r, f"test|{a}|{p}", a, w, p)
    path.write_text(json.dumps(cache))
    release_all(c)


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--tune-only", action="store_true")
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--score-only", action="store_true", help="no Milvus: score (and judge) the cached runs")
    ap.add_argument("--n-guard", type=int, default=500, help="test users judged for the nDCG guard (300 if refused)")
    ap.add_argument("--max-pairs", type=int, default=30_000, help="judge spend guard")
    a = ap.parse_args(argv)
    sample = a.sample is not None
    users = load_users()
    n = {s: min(a.sample or 10**9, int((users.split == s).sum())) for s in ("tune", "test")}
    qids = {s: users[users.split == s].query_id.tolist()[:n[s]] for s in n}
    products, tv, _, _ = load_corpus()
    path, c = ensure_parent(rankings_path(sample)), None
    cache = json.loads(path.read_text()) if path.exists() else None
    if cache is None or (not a.score_only and cache["query_ids"] != qids):
        cache = {"query_ids": qids, "runs": {}, "eligible": {}}
    if not a.score_only:
        c = client()
        _measure(cache, path, c, users, n, products, tv, a.tune_only)
        if a.tune_only:
            return
    check_cache(cache, qids)
    qr, brand_of = qrels(), dict(zip(products.product_id, products.brand))
    guard_ids, main_pool = qids["test"][:a.n_guard], judge.fingerprint()
    with judge.use_pool(pool_path(sample)):
        fill = None
        if not a.no_fill:
            import anthropic
            text = pd.read_parquet(_src("queries.parquet"), columns=["query_id", "query"])
            top = [{q: cache["runs"][k]["ids"][q] for q in guard_ids} for k in fill_keys()]
            fill = judge.fill(anthropic.Anthropic(), top, qr, text, products, max_pairs=a.max_pairs)
        payload = score(cache, users, qr, guard_ids, brand_of)
        write_result("b6", {"collection": COLL, "ef": EF, "depth": DEPTH, "n_arms": len(PROFILE_ARMS), **payload,
                            "fill": fill, "judged": bool(judge.fingerprint()["rows"]), "pool": judge.fingerprint(),
                            "main_pool": main_pool, "sample": sample}, client=c)


if __name__ == "__main__":
    main()
