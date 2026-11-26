"""Showcase follow-up: "xbox one x" with B5's synthetic Black Friday boost off, at x1.1 and at x1.5, for the "Tune
boost weights against nDCG" slide. Reads B5's cached rankings (plain layout, 200 results): no Milvus, judges nothing,
and stops if a result has no label. Picked by hand, at the extreme end; `spread` says how extreme.
-> probes/probe_campaign.json"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bench._common import _src, qrels
from bench.b5_merch import MAIN, RANKINGS, RULE_LIMIT
from bench.showcase import label_maps, side
from vsearch import judge
from vsearch.config import K
from vsearch.merch import CAMPAIGN, merch_fields
from vsearch.metrics import ndcg_at_k
from vsearch.results import read_result
from vsearch.stats import paired_ci

QUERY_ID, WEIGHTS, OUT = "114311", (1.1, 1.5), Path(__file__).with_suffix(".json")


def main():
    cols = ["product_id", "category", "brand", "title", "price"]
    products = pd.read_parquet(_src("products.parquet"), columns=cols)
    info = products.set_index("product_id")
    merch = merch_fields(products).set_axis(products.product_id.to_numpy())
    on = set(merch.index[merch.campaigns.map(bool)])
    b5 = read_result("b5")["payload"]
    cache = json.loads(RANKINGS.read_text())
    qids, runs = cache["query_ids"], cache["runs"]
    qr = qrels()
    filled = judge.filled_qrels(qr[qr.query_id.isin(set(qids))])
    flat = {q: dict(zip(g.product_id, g.label)) for q, g in filled.groupby("query_id")}
    labels = label_maps(qr, filled, [QUERY_ID])[QUERY_ID]
    text = pd.read_parquet(_src("queries.parquet"), columns=["query_id", "query"]).set_index("query_id")["query"]

    def page(key):
        return {q: runs[key]["ids"][q][:K] for q in qids}

    def ndcgs(key):
        return np.array([ndcg_at_k(ids, flat.get(q, {})) for q, ids in page(key).items()])

    off, m1 = ndcgs(f"base|{MAIN}"), b5["m1"]
    arms = [(None, f"base|{MAIN}")] + [(w, f"m1|{MAIN}|{w}|{RULE_LIMIT}") for w in WEIGHTS]
    out = []
    for w, key in arms:
        s, nd = side(page(key)[QUERY_ID], labels, info), ndcgs(key)
        if s["judged@10"] < 1:
            raise SystemExit("a result has no label; judge it before using this example")
        for card in s["results"]:
            card["on_sale"] = card["product_id"] in on
        row = next((r for r in m1["rows"] if r["collection"] == MAIN and r["weight"] == w and r["limit"] == RULE_LIMIT), None)
        mean = row["ndcg@10"] if row else b5["m2"]["base"]["ndcg@10"]
        if abs(mean - nd.mean()) > 1e-6:
            raise SystemExit(f"{key}: rescored to {nd.mean():.4f}, results/b5.json says {mean:.4f}; rescore B5 first")
        out.append({"weight": w, **s, "all": {"ndcg@10": mean,
                    "reach": row["reach"] if row else m1["base_reach"][MAIN],
                    "campaign_slots": row["campaign_slots"] if row else m1["base_campaign_slots"][MAIN],
                    "minus_off": None if w is None else list(paired_ci(nd - off))}})
    worst = ndcgs(arms[-1][1])
    brands = products.brand.fillna("")[products.product_id.isin(on)].nunique()
    OUT.write_text(json.dumps({
        "query_id": QUERY_ID, "query": text[QUERY_ID], "synthetic": True, "campaign": CAMPAIGN,
        "n_products": len(products), "n_on_sale": len(on), "n_brands": int(brands), "limit": RULE_LIMIT, "ef": b5["ef"],
        "n": len(qids), "n_eligible": m1["n_eligible"][MAIN], "arms": out,
        "spread": {"weight": WEIGHTS[-1], "below_0.1": float((worst < .1).mean()),
                   "below_0.1_of_off_above_0.9": float((worst[off > .9] < .1).mean()),
                   "n_off_above_0.9": int((off > .9).sum())}}, indent=1, ensure_ascii=False) + "\n")
    print([(a["weight"], round(a["ndcg@10"], 3), a["all"]) for a in out], brands)


if __name__ == "__main__":
    main()
