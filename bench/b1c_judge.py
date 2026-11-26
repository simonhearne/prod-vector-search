"""B1c: calibrate an LLM judge against ESCI labels, then fill judgment holes in B1a's top-10s.

    uv run python bench/b1c_judge.py calibrate   # 1000 labelled pairs (250/label) -> Haiku; 200 -> Opus
    uv run python bench/b1c_judge.py fill        # unjudged top-10 pairs, 1000-query eval5k subsample
    uv run python bench/b1c_judge.py report      # waits for batches, writes results/b1c.json

Batch ids live in data/work/b1c_state.json; judgments are cached, never re-bought."""
import argparse
import json
import time

import anthropic
import pandas as pd

from bench._common import (client as milvus_client, ensure_collection, ensure_parent, load_corpus, load_eval,
                           qrels, rankings)
from bench import _common
from bench.b1a_eval_sets import CONFIGS, run_config
from vsearch import judge
from vsearch.config import DATA_WORK, SEED
from vsearch.judge import JUDGE_MODEL, SPOTCHECK_MODEL, cohen_kappa
from vsearch.metrics import summarise_quality
from vsearch.results import read_result, write_result

STATE = DATA_WORK / "b1c_state.json"
RANKINGS = DATA_WORK / "b1c_rankings.json"
FILL_QUERIES = 1000


def _state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save_state(s: dict) -> None:
    ensure_parent(STATE).write_text(json.dumps(s, indent=2))


def _queries_text() -> pd.DataFrame:
    return pd.read_parquet(_common._src("queries.parquet"), columns=["query_id", "query"])


def _uncached(pairs: pd.DataFrame, model: str) -> pd.DataFrame:
    cache = judge.cached_judgments()
    done = set(zip(cache[cache.model == model].query_id, cache[cache.model == model].product_id))
    return pairs[[(q, p) not in done for q, p in zip(pairs.query_id, pairs.product_id)]]


def calibration_pairs() -> pd.DataFrame:
    qr = qrels()
    picks = pd.concat([qr[qr.label == l].sample(n=250, random_state=SEED) for l in judge.LABELS])
    return picks.merge(_queries_text(), on="query_id").reset_index(drop=True)


def calibrate(ac) -> None:
    pairs = calibration_pairs()
    products = load_corpus()[0]
    s = _state()
    for model, sub in [(JUDGE_MODEL, pairs), (SPOTCHECK_MODEL, pairs.sample(n=200, random_state=SEED))]:
        todo = _uncached(sub, model)
        if len(todo):
            s.setdefault("calibrate", {})[model] = judge.submit(ac, todo, products, model)
    _save_state(s)
    print(json.dumps(s.get("calibrate", {})))


def fill(ac, max_pairs: int = 30_000) -> None:
    c = milvus_client()
    ef = read_result("b1a")["payload"]["ef"]
    name = ensure_collection(c, "prod_naive")
    q, tv, sv = load_eval("eval5k", FILL_QUERIES)
    ranks = {cfg: rankings(q, run_config(c, name, cfg, q, tv, sv, ef).ids) for cfg in CONFIGS}
    ensure_parent(RANKINGS).write_text(json.dumps(ranks))
    s = _state()
    s["fill_result"] = judge.fill(ac, ranks.values(), qrels(), _queries_text(), load_corpus()[0],
                                  JUDGE_MODEL, max_pairs=max_pairs)
    s.pop("fill", None)  # pre-16 Nov async fill batch id; fill now blocks until its batch ends
    _save_state(s)


def _collect_all(ac) -> dict:
    s, counts = _state(), {}
    for model, bid in s.get("calibrate", {}).items():
        while ac.messages.batches.retrieve(bid).processing_status != "ended":
            time.sleep(60)
        df, cnt = judge.collect(ac, bid)
        judge.append_judgments(df.assign(model=model))
        counts[bid] = cnt
    return counts


def report(ac) -> None:
    counts = _collect_all(ac)
    cache = judge.cached_judgments()
    cal = calibration_pairs()
    h = cal.merge(cache[cache.model == JUDGE_MODEL], on=["query_id", "product_id"], suffixes=("", "_llm"))
    conf = pd.crosstab(h.label, h.label_llm).to_dict()
    o = h.merge(cache[cache.model == SPOTCHECK_MODEL], on=["query_id", "product_id"], suffixes=("", "_opus"))
    out = {"n_calibration": len(h),
           "kappa_4class": cohen_kappa(h.label, h.label_llm),
           "kappa_binary_E": cohen_kappa(h.label == "E", h.label_llm == "E"),
           "confusion_esci_by_llm": conf,
           "spotcheck_n": len(o),
           "spotcheck_kappa_haiku_vs_opus": cohen_kappa(o.label_llm, o.label_opus) if len(o) else None,
           "kappa_opus_vs_esci": cohen_kappa(o.label, o.label_opus) if len(o) else None,
           "batches": counts, "judgments": judge.fingerprint()}
    if RANKINGS.exists():
        ranks = json.loads(RANKINGS.read_text())
        before = qrels()
        after = judge.filled_qrels(before, JUDGE_MODEL)
        out["filled"] = {cfg: {"before": summarise_quality(r, before), "after": summarise_quality(r, after)}
                         for cfg, r in ranks.items()}
        out["rank_before"] = sorted(ranks, key=lambda k: -out["filled"][k]["before"]["ndcg@10"])
        out["rank_after"] = sorted(ranks, key=lambda k: -out["filled"][k]["after"]["ndcg@10"])
    write_result("b1c", out)
    print(json.dumps({k: out[k] for k in ("kappa_4class", "kappa_binary_E")}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["calibrate", "fill", "report"])
    ap.add_argument("--max-pairs", type=int, default=30_000, help="judge spend guard for fill (each pair is a paid API call)")
    a = ap.parse_args(argv)
    ac = anthropic.Anthropic()
    fill(ac, a.max_pairs) if a.cmd == "fill" else {"calibrate": calibrate, "report": report}[a.cmd](ac)


if __name__ == "__main__":
    main()
