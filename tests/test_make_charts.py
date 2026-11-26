import json

import pytest

import bench.make_charts as mc
from vsearch import results


def _w(d, name, payload, sample=False):
    (d / f"{name}.json").write_text(json.dumps({"payload": {**payload, "sample": sample},
                                                "meta": {"sample": sample}}))


Q = {"ndcg@10": 0.5, "ndcg@10_ci95": [0.45, 0.55], "judged@10": 0.6, "n_queries": 10}
L = {"p50": 2.0, "p95": 3.0, "p99": 4.0, "mean": 2.2, "n": 10}


@pytest.fixture
def res(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS", tmp_path)
    monkeypatch.setattr(mc, "CHARTS", tmp_path / "charts")
    _w(tmp_path, "scorecard", {"rows": [{"metric": m, "poc": 1.0, "prod": 0.5, "delta": -0.5}
                                         for m in ["ndcg@10", "judged@10", "recall@10", "p99_ms"]]})
    _w(tmp_path, "b1a", {"sets": {s: {c: {**Q, "latency": L} for c in ["dense_text", "dense_image", "hybrid_rrf"]}
                                  for s in ["curated", "eval5k"]},
                         "power": {"n_queries": 1000, "draw_size": 50, "configs": ["dense_text", "dense_image", "hybrid_rrf"],
                                   "filled": {"draw_power": {"winner": 1, "winner_right": 0.58, "order_right": 0.43,
                                                             "winner_share": [0.02, 0.58, 0.40]}}}})
    _w(tmp_path, "b1b", {"points": [{"ef": ef, "recall@10": 0.9, **Q, "latency": L} for ef in [10, 64]]})
    _w(tmp_path, "b1c", {"kappa_4class": 0.5, "kappa_binary_E": 0.6,
                         "confusion_esci_by_llm": {"E": {"E": 5, "I": 1}, "I": {"E": 1, "I": 5}}})
    bucket = {"n": 5, "selectivity_median": 0.2, **{k: {"recall@10": 0.9, "latency": L, "ef": 10}
                                                    for k in ["poc_unfiltered_baseline", "naive", "tuned", "tuned_params"]}}
    bucket["naive"] = {**bucket["naive"], "short_rate": 0.0}
    bucket["rabitq"] = {f"nprobe={v}": {"recall@10": 0.7, "short_rate": r, "latency": L, "nprobe": v}
                        for v, r in [(8, 0.34), (16, 0.28), (32, 0.17), (64, 0.10)]}
    _w(tmp_path, "b2", {"buckets": {b: bucket for b in ["unfiltered", "10-50%", "<0.1%"]}})
    _w(tmp_path, "b2c", {k: {"latency": L} for k in ["search_only", "search_plus_facets", "facets_from_topk"]})
    _w(tmp_path, "b3a", {"points": [{"name": f"f_{d}_{i}", "dim": d, "index": i, "dtype": "fp32",
                                     "mem_bytes": 1e9 * d / 1024, **Q} for d in [1024, 256] for i in ["HNSW", "HNSW_SQ"]]})
    _w(tmp_path, "b3b", {"rows": [{"name": "f_1024_hnsw_fp32", "n_products": 10_000_000, "vectors": 2,
                                   "replicas": r, "gb": 80.0 * r} for r in (1, 3)]})
    win = [{"t0": t, "n": 5, "errors": 0, "p50": 2.0, "p99": 4.0} for t in range(0, 50, 5)]
    ph = {"baseline": [0, 10], "storm": [10, 40], "cooldown": [40, 50]}
    sph = {"baseline": [0, 10], "w0_partial": [10, 20], "w1_feed": [20, 25], "w2_partial": [25, 35], "tail_0_5m": [35, 50]}
    _w(tmp_path, "b4a", {"design": "stacking", "passes": [
        {"label": lab, "writes": wr, "phases": sph, "windows": win, "level_ok": True,
         "phase_summary": {k: {"p99": 40.0} for k in sph}}
        for lab, wr in [("control", []), ("single", ["partial"]), ("stacked", ["partial", "feed", "partial"]),
                        ("stacked_strong", ["partial", "feed", "partial"]), ("stacked", ["partial", "feed", "partial"]),
                        ("single", ["partial"]), ("control", [])]]})
    _w(tmp_path, "b4b", {"windows": win, "swap_t": 20.0, "errors": 0})
    _w(tmp_path, "b5", {"campaign_share": 0.05, "m1": {
        "segments": {"prod_merch": 3, "prod_merch_pk": 14}, "base_reach": {"prod_merch": 0.33, "prod_merch_pk": 0.33},
        "rows": [{"collection": c, "weight": w, "limit": l, "reach": min(1.0, 0.5 + w / 10)}
                 for c in ("prod_merch", "prod_merch_pk") for w in (1.05, 1.1, 1.2, 1.5, 2.0) for l in (10, 50, 200)]}})
    d = {"rr": [0.004, 0.002], "hit@10": [0.004, 0.003]}
    arm = {"own_minus_none": d, "own_minus_stranger": d, "guard": {"own_minus_none": [-0.01, 0.005]}, "counts": False}
    _w(tmp_path, "b6", {"n_test": 5000, "n_guard": 500, "arms": {a: arm for a in mc.B6_ARMS},
                        "popularity": {"minus_none": {"rr": [0.118, 0.007]}, "guard": {"own_minus_none": [-0.079, 0.016]}}})
    return tmp_path


def test_renders_every_chart(res):
    made = mc.render_all(allow_sample=False)
    names = {p.stem for p in (res / "charts").glob("*.png")}
    assert names == set(mc.CHART_FUNCS) and set(made) == names
    assert {p.stem for p in (res / "charts").glob("*.svg")} == names


def test_refuses_sample_results(res):
    _w(res, "b1b", {"points": []}, sample=True)
    made = mc.render_all(allow_sample=False)
    assert "b1b_recall_ndcg" not in made


def test_b1b_plots_filled_ndcg(res):
    _w(res, "b1b", {"points": [{"ef": ef, "recall@10": 0.9, **Q, "filled": {**Q, "ndcg@10": 0.7, "judged@10": 0.95},
                                "latency": L} for ef in [10, 64]]})
    fig = mc.b1b_recall_ndcg(False)
    ys = [list(line.get_ydata()) for line in fig.axes[0].get_lines()]
    assert [0.7, 0.7] in ys and [0.5, 0.5] not in ys


def test_render_only(res):
    assert mc.render_all(allow_sample=False, only=["b1b_recall_ndcg"]) == ["b1b_recall_ndcg"]


def test_b1a_power_bars_are_winner_shares(res):
    fig = mc.b1a_power(False)
    heights = [round(b.get_height(), 2) for b in fig.axes[0].patches]
    assert heights == [0.02, 0.58, 0.40]


def test_b2_short_pages_panel(res):
    fig = mc.b2_buckets(False)
    left = [round(b.get_height(), 2) for b in fig.axes[0].patches]
    assert sorted(set(left)) == [0.0, 0.1, 0.34]  # HNSW 0, RaBitQ nprobe 64 and 8


def test_b4a_phase_bars_use_level_passes_only(res):
    ps = lambda base, w, t: {"baseline": {"p99": base},
                             **({"w0_partial": {"p99": w}, "w1_feed": {"p99": w * 2}} if w else {}),
                             "tail_0_5m": {"p99": t}, "tail_5_10m": {"p99": t}, "tail_10_20m": {"p99": t}}
    passes = [{"label": "control", "level_ok": True, "phase_summary": ps(20, None, 30)},
              {"label": "single", "level_ok": False, "phase_summary": ps(600, 400, 40)},
              {"label": "stacked", "level_ok": True, "phase_summary": ps(25, 100, 300)},
              {"label": "stacked", "level_ok": True, "phase_summary": ps(35, 200, 100)},
              {"label": "control", "level_ok": True, "phase_summary": ps(40, None, 30)}]
    _w(res, "b4a", {"design": "stacking", "passes": passes})
    fig = mc.b4a_timeline(False)
    heights = sorted(round(b.get_height()) for b in fig.axes[0].patches)
    assert 600 not in heights and 400 not in heights          # the non-level single pass is left out
    assert 300 in heights and 150 not in heights              # writes = mean over replicates of each pass's worst write phase
    assert 30 in heights                                       # control tail


def test_b3a_frontier_plots_filled(res):
    _w(res, "b3a", {"points": [{"name": "f_1024_x", "dim": 1024, "index": "HNSW", "dtype": "fp32", "mem_bytes": 2e9,
                                **Q, "filled": {**Q, "ndcg@10": 0.8}}]})
    fig = mc.b3a_frontier(False)
    ys = [o.get_offsets()[0][1] for o in fig.axes[0].collections if len(o.get_offsets())]
    assert ys == [0.8]


def test_b4a_old_format_is_skipped(res):
    _w(res, "b4a", {"no_writes": {"windows": [], "phases": {}}})
    assert "b4a_timeline" not in mc.render_all(allow_sample=False)


def test_b5_reach_renders_alone_and_says_the_data_is_synthetic(res):
    assert mc.render_all(only=["b5_reach"]) == ["b5_reach"]
    assert (res / "charts" / "b5_reach.png").exists() and (res / "charts" / "b5_reach.svg").exists()
    fig = mc.b5_reach(False)
    text = " ".join(t.get_text() for t in fig.findobj(mc.plt.Text))
    assert "synthetic" in text and "3 segments" in text and "14 segments" in text


def test_b6_lift_names_every_arm_and_says_what_stands_in_for_purchases(res):
    fig = mc.b6_lift(False)
    text = " ".join(t.get_text() for t in fig.findobj(mc.matplotlib.text.Text))
    assert all(label in text for label in mc.B6_ARMS.values())
    assert "reviews stand in for purchases" in text and "5,000 users" in text
    assert "stranger's brands rarely match" in text
    assert "+0.118" in text and "-0.079" in text  # popularity is off scale, so it is written out
