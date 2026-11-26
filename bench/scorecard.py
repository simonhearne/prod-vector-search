"""§0 scorecard: one fixed PoC config, measured under PoC and under production conditions.
-> results/scorecard.json"""
from vsearch.results import read_result, write_result

METRICS = ["ndcg@10", "judged@10", "recall@10", "p99_ms"]


def _get(d: dict, m: str) -> float:
    return d["latency"]["p99"] if m == "p99_ms" else d[m]


def scorecard(poc: dict, prod: dict) -> dict:
    return {"rows": [{"metric": m, "poc": _get(poc, m), "prod": _get(prod, m),
                      "delta": _get(prod, m) - _get(poc, m)} for m in METRICS]}


def main():
    poc, prod = read_result("scorecard_poc"), read_result("scorecard_prod")
    write_result("scorecard", {**scorecard(poc["payload"], prod["payload"]),
                               "poc_conditions": poc["payload"]["queries"],
                               "prod_conditions": prod["payload"]["queries"],
                               "sample": poc["meta"]["sample"] or prod["meta"]["sample"]})


if __name__ == "__main__":
    main()
