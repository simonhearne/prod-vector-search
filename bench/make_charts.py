"""Render every slide chart from results/*.json -> results/charts/<name>.{svg,png} (1920x1080).

    uv run python bench/make_charts.py [--allow-sample]

Sample (spike) results are refused unless --allow-sample. Static, light theme, for slides.
Palette: the validated dataviz reference slots (blue, orange, aqua, yellow), fixed order; every
series is direct-labelled or legended because aqua/yellow sit below 3:1 contrast."""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from vsearch import results  # noqa: E402

CHARTS = results.RESULTS / "charts"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"]
LW = 2 * 72 / 100  # 2px at 100 dpi

plt.rcParams.update({
    "figure.figsize": (19.2, 10.8), "figure.dpi": 100, "savefig.dpi": 100,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 20, "axes.titlesize": 28, "axes.titleweight": "bold", "axes.labelsize": 22,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": LW, "lines.markersize": 9,
})


class Skip(Exception):
    pass


def _load(name: str, allow_sample: bool) -> dict:
    path = results.RESULTS / f"{name}.json"
    if not path.exists():
        raise Skip(f"{name}.json missing")
    r = json.loads(path.read_text())
    if r["meta"].get("sample") and not allow_sample:
        raise Skip(f"{name}.json is a sample run")
    return r["payload"]


def _table(ax, header, rows, title):
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(24); t.scale(1, 3.2)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor(GRID); cell.set_facecolor(SURFACE)
        cell.get_text().set_color(INK if r else INK2)
        if r == 0:
            cell.get_text().set_weight("bold")
    ax.set_title(title)


def _fmt(metric, v):
    return f"{v:.1f} ms" if metric == "p99_ms" else f"{v:.3f}"


def scorecard(a):
    p = _load("scorecard", a)
    fig, ax = plt.subplots()
    rows = [[r["metric"], _fmt(r["metric"], r["poc"]), _fmt(r["metric"], r["prod"]),
             ("+" if r["delta"] >= 0 else "") + _fmt(r["metric"], r["delta"])] for r in p["rows"]]
    _table(ax, ["", "PoC conditions", "Production conditions", "Δ"], rows, "Same system, two scorecards")
    return fig


def b1a_sets(a):
    p = _load("b1a", a)
    labels = {"poc50": "PoC: 50 ESCI queries", "curated": "PoC: 50 paraphrases", "head50": "PoC: 50 head queries",
              "eval5k": "5,000 real ESCI queries"}
    sets = [s for s in [*p.get("poc_sets", ["curated"]), "eval5k"] if s in p["sets"]]
    cfgs = list(p["sets"][sets[0]])
    fig, ax = plt.subplots()
    x, w = np.arange(len(cfgs)), 0.8 / len(sets)
    for j, s in enumerate(sets):
        vals = [p["sets"][s][c]["ndcg@10"] for c in cfgs]
        bars = ax.bar(x + (j - (len(sets) - 1) / 2) * w, vals, w * 0.95, color=S[j], label=labels.get(s, s))
        for b, c in zip(bars, cfgs):
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                        va="bottom", fontsize=15, color=INK2, xytext=(0, 4), textcoords="offset points")
    ax.set_xticks(x, cfgs); ax.set_ylabel("nDCG@10"); ax.legend(loc="upper right", fontsize=17)
    ax.set_title("Small PoC query sets vs real traffic")
    return fig


def b1a_power(a):
    p = _load("b1a", a)["power"]
    dp = p["filled"]["draw_power"]
    names = {"dense_text": "Text embedding", "dense_image": "Image embedding", "hybrid_rrf": "Hybrid (RRF)"}
    fig, ax = plt.subplots()
    x = np.arange(len(p["configs"]))
    bars = ax.bar(x, dp["winner_share"], 0.6, color=[S[0] if i == dp["winner"] else GRID for i in x])
    for i, b in enumerate(bars):
        tag = f"  ← wins on all {p['n_queries']:,}" if i == dp["winner"] else ""
        ax.annotate(f"{b.get_height():.0%}{tag}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                    va="bottom", xytext=(0, 6), textcoords="offset points", color=INK)
    ax.set_xticks(x, [names.get(c, c) for c in p["configs"]]); ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel(f"share of random {p['draw_size']}-query sets it wins")
    ax.set_title("Can your eval set tell your candidates apart?")
    return fig


def b1b_recall_ndcg(a):
    p = _load("b1b", a)
    ef = [q["ef"] for q in p["points"]]
    fig, ax = plt.subplots()
    filled = all("filled" in q for q in p["points"])
    nd = [(q["filled"] if filled else q)["ndcg@10"] for q in p["points"]]
    judged = min((q["filled"] if filled else q)["judged@10"] for q in p["points"])
    rec = [q["recall@10"] for q in p["points"]]
    pts = lambda ys: f"{100 * (ys[-1] - ys[0]):+.0f} pts"
    series = [(rec, f"ANN recall@10\n{pts(rec)}"),
              (nd, f"nDCG@10, {'judge-filled' if filled else 'ESCI only'}\n{pts(nd)} (judged@10 ≥ {judged:.2f})")]
    for i, (ys, label) in enumerate(series):
        ax.plot(ef, ys, color=S[i], marker="o")
        ax.annotate(label, (ef[-1], ys[-1]), xytext=(10, 0), textcoords="offset points", va="center", color=INK)
    ax.set_xscale("log", base=2); ax.set_xticks(ef, [str(e) for e in ef]); ax.set_ylim(0, 1.02)
    ax.set_xlabel("HNSW ef"); ax.set_ylabel("score (0–1)")
    fig.subplots_adjust(right=0.74)
    ax.set_title("Do you track relevance separately from ANN recall?")
    return fig


def b1c_kappa(a):
    p = _load("b1c", a)
    conf = p["confusion_esci_by_llm"]  # {llm_label: {esci_label: n}}
    labs = ["E", "S", "C", "I"]
    m = np.array([[conf.get(l, {}).get(e, 0) for l in labs] for e in labs], dtype=float)
    fig, ax = plt.subplots(figsize=(13, 10.8))
    ax.grid(False)
    from matplotlib.colors import ListedColormap
    ax.imshow(m, cmap=ListedColormap(SEQ))
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center",
                    color=SURFACE if m[i, j] > m.max() * 0.6 else INK, fontsize=24)
    ax.set_xticks(range(4), labs); ax.set_yticks(range(4), labs)
    ax.set_xlabel("LLM judge"); ax.set_ylabel("ESCI human label")
    ax.set_title(f"Judge vs humans: κ={p['kappa_4class']:.2f} (4-class), {p['kappa_binary_E']:.2f} (Exact vs rest)")
    return fig


def b2_buckets(a):
    p = _load("b2", a)["buckets"]
    order = [b for b in ["unfiltered", "10-50%", "1-10%", "0.1-1%", "<0.1%"] if b in p and p[b] and "rabitq" in p[b]]
    series = [(lambda b: p[b]["naive"], "HNSW (prod)"),
              (lambda b: p[b]["rabitq"]["nprobe=64"], "IVF_RABITQ, nprobe 64"),
              (lambda b: p[b]["rabitq"]["nprobe=8"], "IVF_RABITQ, nprobe 8")]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    x, w = np.arange(len(order)), 0.26
    for j, (get, label) in enumerate(series):
        off = (j - 1) * (w + 0.01)
        ax1.bar(x + off, [get(b)["short_rate"] for b in order], w, color=S[j], label=label)
        ax2.bar(x + off, [get(b)["recall@10"] for b in order], w, color=S[j])
    ax1.set_xticks(x, order); ax2.set_xticks(x, order)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_ylabel("queries returning < 10 results"); ax2.set_ylim(0, 1.05); ax2.set_ylabel("recall@10 vs exact filtered")
    ax1.set_xlabel("filter selectivity"); ax2.set_xlabel("filter selectivity")
    fig.legend(*ax1.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=18)
    fig.suptitle("Do narrow filters return full pages?", fontsize=28, fontweight="bold")
    fig.subplots_adjust(bottom=0.2)
    return fig


def b2c_facets(a):
    p = _load("b2c", a)
    keys = [("search_only", "Search only"), ("search_plus_facets", "Search + facet counts"),
            ("facets_from_topk", "Facets from top-100")]
    fig, ax = plt.subplots()
    vals = [p[k]["latency"]["p99"] for k, _ in keys]
    bars = ax.bar([l for _, l in keys], vals, 0.5, color=S[0])
    for b in bars:
        ax.annotate(f"{b.get_height():.1f} ms", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", xytext=(0, 4), textcoords="offset points", color=INK)
    ax.set_ylabel("p99 latency (ms)"); ax.set_title("What facets cost per request")
    return fig


def b3a_frontier(a):
    p = _load("b3a", a)["points"]
    dims = sorted({q["dim"] for q in p}, reverse=True)
    marks = {"HNSW": "o", "HNSW_SQ": "s", "IVF_RABITQ": "D"}
    nd = lambda q: (q["filled"] if "filled" in q else q)["ndcg@10"]
    fig, ax = plt.subplots()
    for q in p:
        color = S[dims.index(q["dim"])]
        fp16 = q.get("dtype") == "fp16"
        ax.scatter(q["mem_bytes"] / 1e9, nd(q), s=160, marker=marks.get(q["index"], "o"),
                   facecolor=SURFACE if fp16 else color, edgecolor=color, linewidth=LW * 1.5, zorder=3)
        lat = f" · {q['latency']['p50']:.0f} ms" if "latency" in q else ""
        ax.annotate(q["name"].removeprefix("f_") + lat, (q["mem_bytes"] / 1e9, nd(q)), xytext=(8, 6),
                    textcoords="offset points", fontsize=15, color=INK2)
    for d in dims:
        ax.scatter([], [], color=S[dims.index(d)], label=f"{d} dims")
    for k, m in marks.items():
        ax.scatter([], [], marker=m, color=INK2, label=k)
    ax.scatter([], [], facecolor=SURFACE, edgecolor=INK2, label="fp16 (hollow)")
    ax.legend(loc="lower right", fontsize=17)
    ax.set_xlabel("Milvus memory in RAM (GB, 450k text vectors) · labels: config · search p50")
    ax.set_ylabel("nDCG@10 (judge-filled)" if all("filled" in q for q in p) else "nDCG@10 (ESCI only)")
    ax.set_title("Did you choose dims and quantisation on relevance per GB?")
    return fig


def b3b_costs(a):
    p = _load("b3b", a)["rows"]
    rows = [[r["name"].removeprefix("f_"), f"{r['n_products']:,}", r["vectors"], r["replicas"], f"{r['gb']:.0f} GB"]
            for r in p if r["n_products"] == 10_000_000 and r["vectors"] == 2 and r["replicas"] in (1, 3)]
    fig, ax = plt.subplots()
    _table(ax, ["config", "products", "vectors", "replicas", "memory"], rows[:12], "At 10M products")
    return fig


def _timeline(ax, windows, color, label):
    t = [w["t0"] for w in windows if w["p99"] is not None]
    y = [w["p99"] for w in windows if w["p99"] is not None]
    ax.plot(t, y, color=color, label=label)
    if t:
        ax.annotate(label, (t[-1], y[-1]), xytext=(10, 0), textcoords="offset points", va="center", color=INK)


def b4a_timeline(a):
    """p99 by phase, level passes only, averaged over replicates. 'During writes' is each pass's worst
    write phase (a storm or the bulk feed)."""
    p = _load("b4a", a)
    if "passes" not in p:
        raise Skip("b4a.json predates the 16 Nov pass format")
    phases = [("baseline", "before"), ("writes", "during writes"), ("tail_0_5m", "0–5 min after"),
              ("tail_5_10m", "5–10 min after"), ("tail_10_20m", "10–20 min after")]
    series = [("control", "No writes"), ("single", "One storm"), ("stacked", "Storm + feed + storm")]

    def val(q, ph):
        s = q["phase_summary"]
        if ph == "writes":
            w = [v["p99"] for k, v in s.items() if k.startswith("w") and v["p99"] is not None]
            return max(w) if w else None
        return s.get(ph, {}).get("p99")

    fig, ax = plt.subplots()
    shown, dropped = [], []
    for lab, name in series:
        reps = [q for q in p["passes"] if q["label"] == lab]
        lvl = [q for q in reps if q.get("level_ok", True)]
        if reps and not lvl:
            dropped.append(name)
        if lvl:
            shown.append((name, lvl))
    x, w = np.arange(len(phases)), 0.8 / max(1, len(shown))
    for j, (name, lvl) in enumerate(shown):
        vals = [np.mean([v for q in lvl if (v := val(q, ph)) is not None]) if any(val(q, ph) is not None for q in lvl)
                else 0.0 for ph, _ in phases]
        bars = ax.bar(x + (j - (len(shown) - 1) / 2) * w, vals, w * 0.95, color=S[[n for _, n in series].index(name)],
                      label=f"{name} (n={len(lvl)})")
        for b in bars:
            if b.get_height() > 0:
                ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                            va="bottom", xytext=(0, 4), textcoords="offset points", fontsize=16, color=INK2)
    ax.set_xticks(x, [lbl for _, lbl in phases]); ax.set_ylabel("search p99 (ms)")
    ax.legend(loc="upper right", fontsize=18)
    if dropped:
        ax.text(0.01, 0.98, f"Not shown (baseline not level): {', '.join(dropped)}", transform=ax.transAxes,
                va="top", color=INK2, fontsize=15)
    ax.set_title("Did you test back-to-back write bursts?")
    return fig


def b4b_swap(a):
    p = _load("b4b", a)
    fig, ax = plt.subplots()
    _timeline(ax, p["windows"], S[0], "search p99")
    ax.axvline(p["swap_t"], color=INK2, lw=LW, ls="--")
    ax.text(p["swap_t"], 0.98, f"  alias swap ({p['errors']} failed queries)", transform=ax.get_xaxis_transform(),
            va="top", color=INK2)
    ax.set_xlabel("seconds"); ax.set_ylabel("search p99 (ms, 5 s windows)"); ax.set_ylim(bottom=0)
    ax.margins(x=0.12); ax.set_title("Blue/green re-index behind an alias")
    return fig


def b5_reach(a):
    p = _load("b5", a)
    m = p["m1"]
    weights = [w for w in sorted({r["weight"] for r in m["rows"]}) if w != 2.0]  # 2.0 sits on top of 1.5
    fig, axes = plt.subplots(1, 2, sharey=True)
    for ax, (name, layout) in zip(axes, (("prod_merch", "plain layout"), ("prod_merch_pk", "partitioned by category"))):
        limits = sorted({r["limit"] for r in m["rows"] if r["collection"] == name})
        for w, color in zip(weights, SEQ[1:]):
            y = [next(r["reach"] for r in m["rows"] if (r["collection"], r["weight"], r["limit"]) == (name, w, lim))
                 for lim in limits]
            ax.plot(range(len(limits)), y, color=color, marker="o", label=f"boost ×{w:g}")
            ax.annotate(f"×{w:g}", (len(limits) - 1, y[-1]), xytext=(12, 0), textcoords="offset points",
                        va="center", fontsize=18, color=INK2)
        ax.axhline(m["base_reach"][name], color=INK2, ls="--", lw=LW)
        ax.annotate("no boost", (len(limits) - 1, m["base_reach"][name]), xytext=(0, 8), textcoords="offset points",
                    ha="right", fontsize=18, color=INK2)
        ax.set_xticks(range(len(limits)), [str(x) for x in limits])
        ax.set_xlabel("results requested (limit)"); ax.set_ylim(0, 1.05); ax.margins(x=0.12)
        ax.set_title(f"{m['segments'][name]} segments, {layout}", fontsize=22)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("pages with a campaign product in the top 10")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(weights), fontsize=18)
    fig.suptitle("Do your boosts reach the page?", fontsize=28, fontweight="bold")
    fig.text(0.5, 0.925, f"synthetic campaign on {p['campaign_share']:.0%} of the catalogue, same products and queries in both",
             ha="center", fontsize=18, color=INK2)
    fig.subplots_adjust(bottom=0.17, top=0.85)
    return fig


B6_ARMS = {"brand_boost": "brand boost", "brand_sparse": "brand sparse vector", "price_band": "price band",
           "user_blend": "user vector in the query", "user_rerank": "user vector as a rerank", "brand_price": "brand + price boost"}


def b6_lift(a):
    p = _load("b6", a)
    arms = list(B6_ARMS)
    y = np.arange(len(arms))[::-1]
    fig, (mrr, nd) = plt.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [3, 2]})

    def dots(ax, vals, dy, color, label=None):
        for i, (m, h) in zip(y, vals):
            ax.plot([m - h, m + h], [i + dy] * 2, color=color, lw=2 * LW, solid_capstyle="round")
        ax.plot([m for m, _ in vals], y + dy, ls="", marker="o", color=color, label=label, markersize=11,
                markeredgecolor=SURFACE, markeredgewidth=LW)

    dots(mrr, [p["arms"][k]["own_minus_none"]["rr"] for k in arms], 0.13, S[0], "against no profile")
    dots(mrr, [p["arms"][k]["own_minus_stranger"]["rr"] for k in arms], -0.13, S[1], "against a stranger's profile")
    dots(nd, [p["arms"][k]["guard"]["own_minus_none"] for k in arms], 0, S[0])
    for ax, title in ((mrr, "MRR of the held-out purchase"), (nd, f"nDCG@10, {p['n_guard']} judged users")):
        ax.axvline(0, color=INK2, lw=LW)
        ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
        ax.set_title(title, fontsize=22)
        ax.set_xlabel("difference, 95% CI")
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.005))
    mrr.set_yticks(y, [B6_ARMS[k] for k in arms])
    mrr.set_ylim(-0.6, len(arms) - 0.4)
    fig.legend(*mrr.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=2, fontsize=18)
    pop, popn = p["popularity"]["minus_none"]["rr"], p["popularity"]["guard"]["own_minus_none"]
    fig.suptitle("Does the right user's profile help?", fontsize=28, fontweight="bold")
    fig.text(0.5, 0.925, f"{p['n_test']:,} users, reviews stand in for purchases. Off scale: ranking by popularity, "
             f"{pop[0]:+.3f} MRR and {popn[0]:+.3f} nDCG (all-time ratings, so leaky)", ha="center", fontsize=18, color=INK2)
    fig.text(0.5, 0.015, "A stranger's brands rarely match the results, so for the brand arms that control barely acts",
             ha="center", fontsize=16, color=INK2)
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.17, top=0.84, wspace=0.08)
    return fig


CHART_FUNCS = {"scorecard": scorecard, "b1a_sets": b1a_sets, "b1a_power": b1a_power, "b1b_recall_ndcg": b1b_recall_ndcg,
               "b1c_kappa": b1c_kappa, "b2_buckets": b2_buckets, "b2c_facets": b2c_facets,
               "b3a_frontier": b3a_frontier, "b3b_costs": b3b_costs, "b4a_timeline": b4a_timeline,
               "b4b_swap": b4b_swap, "b5_reach": b5_reach, "b6_lift": b6_lift}


def render_all(allow_sample: bool = False, only: list[str] | None = None) -> list[str]:
    CHARTS.mkdir(parents=True, exist_ok=True)
    made = []
    for name, fn in CHART_FUNCS.items():
        if only and name not in only:
            continue
        try:
            fig = fn(allow_sample)
        except Skip as e:
            print(f"skip {name}: {e}")
            continue
        for ext in ("svg", "png"):
            fig.savefig(CHARTS / f"{name}.{ext}")
        plt.close(fig)
        made.append(name)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-sample", action="store_true")
    ap.add_argument("--only", nargs="+", choices=list(CHART_FUNCS), help="render just these charts")
    a = ap.parse_args(argv)
    print("rendered:", ", ".join(render_all(a.allow_sample, a.only)))


if __name__ == "__main__":
    main()
