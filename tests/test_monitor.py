from bench.b1b_recall_vs_ndcg import monitor_report


def g(n, judged=0.95):
    return {"ndcg@10": n, "filled": {"ndcg@10": n, "judged@10": judged}}


def test_monitor_report_needs_judging():
    r = monitor_report(0.97, g(0.30, judged=0.5), 0.36)
    assert r["recall_ok"] and r["ndcg_status"] == "needs_judging"


def test_monitor_report_drift_and_ok():
    assert monitor_report(0.97, g(0.33), 0.36)["ndcg_status"] == "drift"
    assert monitor_report(0.97, g(0.35), 0.36)["ndcg_status"] == "ok"
    r = monitor_report(0.90, g(0.35), None)
    assert not r["recall_ok"] and r["ndcg_status"] == "no_baseline"
    assert r["recall@10"] == 0.90 and r["ndcg@10"] == 0.35 and r["judged@10"] == 0.95


def test_monitor_baseline_pins_rankings_not_numbers():
    # review: a stored nDCG drifts as the judgment pool grows, and "latest run" ratchets; pin rankings instead
    from bench.b1b_recall_vs_ndcg import monitor_baseline
    b1b = {64: {"q": ["a"]}}
    prior = [{"ef": 192, "ndcg_status": "no_baseline", "golden_rankings": {"q": ["first"]}, "timestamp_utc": "1"},
             {"ef": 192, "ndcg_status": "drift", "golden_rankings": {"q": ["bad"]}, "timestamp_utc": "2"},
             {"ef": 192, "ndcg_status": "ok", "golden_rankings": {"q": ["later"]}, "timestamp_utc": "3"},
             {"ef": 192, "ndcg_status": "ok", "timestamp_utc": "0"}]          # pre-fix run: no rankings
    assert monitor_baseline(64, b1b, prior) == {"q": ["a"]}                 # swept ef: b1b's rankings
    assert monitor_baseline(192, b1b, prior) == {"q": ["first"]}            # else the earliest good run, pinned
    assert monitor_baseline(96, b1b, prior) is None


def test_monitor_report_needs_judging_relative_to_baseline():
    # judged 0.93 passes the absolute 0.9 floor but sits 0.07 below a fully judged baseline
    r = monitor_report(0.97, g(0.62, judged=0.93), 0.64, baseline_judged=1.0)
    assert r["ndcg_status"] == "needs_judging"
    assert monitor_report(0.97, g(0.63, judged=0.995), 0.64, baseline_judged=1.0)["ndcg_status"] == "ok"
