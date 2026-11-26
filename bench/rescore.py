"""Re-score every canonical judge-filled result at ONE judgment-pool snapshot, offline, from the rankings
the benchmarks cached (no Milvus, no new judging). Filled nDCG's ideal DCG uses every label a query has,
so a number scored before a later fill reads lower afterwards; rescoring makes all slides comparable.
Also recomputes B4c's marker stats from its stored events (censoring flags).

    uv run python -m bench.rescore            # b1a b1b b3a b4c showcase (b5 if run) -> results/<id>.json (measured meta kept)"""
import json

from scipy.stats import spearmanr

from bench import b5_merch
from bench._common import load_corpus, load_eval, qrels, quality
from bench.b1a_eval_sets import CONFIGS, power_block, wording_block
from bench.b4c_freshness import marker_stats
from bench.showcase import label_maps, rank_poc_vs_prod
from vsearch import judge
from vsearch.config import DATA_WORK, RESULTS
from vsearch.merch import merch_fields
from vsearch.metrics import judged_at_k, ndcg_at_k
from vsearch.results import read_result, write_result


def rescore_points(points: list[dict], ranks: dict[str, dict], qr, key: str) -> list[dict]:
    """Replace the quality fields of each point (raw + filled) from its cached rankings; keep measurements."""
    out = []
    for p in points:
        q = quality(ranks[str(p[key])], qr)
        out.append({**p, **q})
    return out


def rescore_b1a(payload: dict, ranks: dict, qr) -> dict:
    sets = {s: {cfg: {**payload["sets"][s][cfg], **quality(ranks[s][cfg], qr)} for cfg in CONFIGS} for s in ranks}
    nd = lambda s, k: sets[s][k]["filled"]["ndcg@10"]
    poc = payload["poc_sets"]
    return {**payload, "sets": sets, "rank": {s: sorted(CONFIGS, key=lambda k: -nd(s, k)) for s in sets},
            "spearman_vs_eval5k": {ps: float(spearmanr([nd(ps, k) for k in CONFIGS],
                                                      [nd("eval5k", k) for k in CONFIGS]).statistic) for ps in poc},
            "power": power_block(ranks["eval5k"], qr), "wording": wording_block(ranks["poc50"], ranks["curated"], qr)}


def rescore_b4c(payload: dict) -> dict:
    t = payload.get("timeout_s", 900)
    return {**payload, "variants": {k: {**v, **marker_stats(v["events"], t)} for k, v in payload["variants"].items()}}


def rescore_b5(payload: dict, cache: dict, qr, merch, query_ids: list[str]) -> dict:
    """Latency comes from the cache, so only the quality fields can change, provided it is this result's cache."""
    if cache["n"] != payload["n"]:
        raise ValueError(f"b5 cache has n={cache['n']}, the result n={payload['n']}: not the same run")
    return {**payload, **b5_merch.score(cache, qr, merch, query_ids)}


def rescore_showcase(payload: dict, qr) -> dict:
    """poc_vs_prod only, from its stored result cards: the latency split and narrow-filter runs are measurements."""
    block = payload["poc_vs_prod"]
    labels = label_maps(qr, judge.filled_qrels(qr), [r["query_id"] for r in block["rows"]])
    rows = []
    for r in block["rows"]:
        lab = labels[r["query_id"]]
        flat = {p: l for p, (l, _) in lab.items()}
        new = dict(r)
        for s in ("poc", "prod"):
            ids = [c["product_id"] for c in r[s]["results"]]
            cards = []
            for c in r[s]["results"]:
                label, source = lab.get(c["product_id"], (None, None))
                cards.append({**c, "label": label, "label_source": source})
            new[s] = {**r[s], "ndcg@10": ndcg_at_k(ids, flat), "judged@10": judged_at_k(ids, flat), "results": cards}
        new["ndcg_gap"] = new["prod"]["ndcg@10"] - new["poc"]["ndcg@10"]
        rows.append(new)
    cat = block["poc_category"]
    return {**payload, "poc_vs_prod": {**block, "rows": rows, "candidates": rank_poc_vs_prod(rows),
                                       "candidates_in_category": rank_poc_vs_prod([r for r in rows if r["top_category"] == cat])}}


def _write(bench_id: str, payload: dict, fp: dict | None) -> None:
    meta = read_result(bench_id)["meta"]
    measured = payload.get("measured") or {k: meta.get(k) for k in ("timestamp_utc", "git_sha", "milvus_version")}
    write_result(bench_id, {**payload, "measured": measured, **({"judgments": fp} if fp else {})})


def main() -> None:
    qr, fp = qrels(), judge.fingerprint()
    print("judgment pool", fp, flush=True)
    b1b = json.loads((DATA_WORK / "b1b_rankings.json").read_text())
    p = read_result("b1b")["payload"]
    _write("b1b", {**p, "points": rescore_points(p["points"], b1b["rankings"], qr, "ef")}, fp)
    p = read_result("b3a")["payload"]
    _write("b3a", {**p, "points": rescore_points(p["points"], json.loads((DATA_WORK / "b3a_rankings.json").read_text()),
                                                 qr, "name")}, fp)
    p = read_result("b1a")["payload"]
    _write("b1a", rescore_b1a(p, json.loads((DATA_WORK / "b1a_rankings.json").read_text()), qr), fp)
    _write("b4c", rescore_b4c(read_result("b4c")["payload"]), None)
    _write("showcase", rescore_showcase(read_result("showcase")["payload"], qr), fp)
    done = "b1b b3a b1a b4c showcase"
    if (RESULTS / "b5.json").exists() and b5_merch.RANKINGS.exists():
        products = load_corpus()[0]
        merch = merch_fields(products).set_axis(products.product_id.to_numpy())
        cache = json.loads(b5_merch.RANKINGS.read_text())
        p = read_result("b5")["payload"]
        qids = load_eval("eval5k", p["n"])[0].query_id.tolist()
        _write("b5", rescore_b5(p, cache, qr, merch, qids), fp)
        done += " b5"
    print(f"rescored {done}", flush=True)


if __name__ == "__main__":
    main()
