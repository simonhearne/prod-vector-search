"""B4d report: prints the per-operation tables from results/b4d_single_ops.json and the raw samples.

Everything is cut from raw samples, shown per repeat, next to the same repeat's control. Two repeats on one
laptop is thin evidence: both are printed, including where they disagree.

  uv run python -m bench.b4d_report [--id b4d_single_ops]
"""
import argparse
import gzip
import json

import numpy as np

from bench.b4d_single_ops import RECOVERY_TOL, RECOVERY_WINDOW_S, breached, load_raw, recovery_s
from vsearch.load import phase_summary
from vsearch.config import ROOT
from vsearch.results import read_result

COUNTERS = ("seal_", "flush_", "autoflush_")


def best_attempt(passes: list[dict], slot: int) -> dict | None:
    at = [p for p in passes if p["slot"] == slot]
    level = [p for p in at if p.get("level_ok")]
    return (level or at or [None])[-1]


def matching_control(passes: list[dict], repeat: int) -> dict | None:
    slots = [p["slot"] for p in passes if p["label"] == "control" and p["repeat"] == repeat]
    return best_attempt(passes, slots[0]) if slots else None


def rowset_mismatches(passes: list[dict]) -> list[int]:
    """Repeats in which promo and promo_flush did not write the same rows."""
    sha: dict[int, set] = {}
    for p in passes:
        if p["label"] in ("promo", "promo_flush"):
            sha.setdefault(p["repeat"], set()).add(p["rows_sha"])
    return sorted(r for r, s in sha.items() if len(s) > 1)


def cut(samples, phases) -> dict[str, dict]:
    return phase_summary(samples, phases)


def control_p99(samples, phases) -> float:
    """One reference per repeat: the control's p99 over everything after its run-in."""
    lat = [s.latency_ms for s in samples if s.ok and s.t >= phases["runin"][1]]
    return float(np.percentile(lat, 99))


def events(rows: list[dict], t0: float) -> list[dict]:
    """Changes between consecutive 5 s metric rows: counters that moved and loaded-segment gauges."""
    out, prev = [], None
    for r in rows:
        if "error" in r:
            continue
        if prev is not None:
            for k in sorted(set(r) & set(prev)):
                counted = k.startswith(COUNTERS) or (k.startswith("comp_") and k.endswith("_done")) \
                    or k == "idx_JobStateFinished" or k.startswith(("qn_L0_", "qn_L1_", "qn_L2_"))
                if counted and r[k] != prev[k]:
                    out.append({"t": r["t"] - t0, "key": k, "delta": r[k] - prev[k]})
        prev = r
    return out


def spikes(samples, t0: float, until: float, ref_p99: float, window: float = RECOVERY_WINDOW_S,
           tol: float = RECOVERY_TOL) -> list[dict]:
    """Windows after t0 whose p99 exceeds tol x ref, or in which no search succeeded."""
    out = []
    for i in range(int((until - t0) // window)):
        lo = t0 + i * window
        w = [s for s in samples if lo <= s.t < lo + window]
        ok = [s.latency_ms for s in w if s.ok]
        p99 = float(np.percentile(ok, 99)) if ok else None
        if p99 is None or p99 > tol * ref_p99:
            out.append({"t": i * window, "p99": p99, "n": len(w), "errors": len(w) - len(ok)})
    return out


def recovery_text(v: float | None) -> str:
    return "not reached" if v is None else "never left" if v == 0 else f"{v:.0f} s"


def read_jsonl_gz(path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f]


def _ms(v):
    return "   -" if v is None else f"{v:4.0f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--id", default="b4d_single_ops")
    ap.add_argument("--events", action="store_true", help="also print segment/compaction events after the write")
    a = ap.parse_args(argv)
    payload = read_result(a.id)["payload"]
    passes = payload["passes"]
    if (bad := rowset_mismatches(passes)):
        print(f"WARNING promo and promo_flush wrote different rows in repeats {bad}")
    for f in payload.get("failures", []):
        print(f"FAILED pass {f['slot']} {f['label']} attempt {f['attempt']}: {f['error']}")
    for op in payload["ops"]:
        for repeat in sorted({p["repeat"] for p in passes}):
            slots = sorted({p["slot"] for p in passes if p["label"] == op["label"] and p["repeat"] == repeat})
            if not slots:
                continue
            p = best_attempt(passes, slots[0])
            samples = load_raw(ROOT / p["raw"])
            c = matching_control(passes, repeat)
            cs = load_raw(ROOT / c["raw"]) if c else None
            ref = control_p99(cs, c["phases"]) if c else None
            cstat = cut(cs, c["phases"]) if c else {}
            t0 = p["timeline"]["runoff_start"]
            until = p["phases"][list(p["phases"])[-1]][1]
            ws = p["write_stats"]
            rate = f"{ws['rows_per_s']:.0f} rows/s in {ws['seconds']:.1f} s" if ws["rows_per_s"] else "no writes"
            print(f"\n== {op['label']} repeat {repeat + 1}  level={p['level_ok']} {p['level_reasons'] or ''} "
                  f"attempts={sum(q['slot'] == p['slot'] for q in passes)}")
            print(f"   write: {ws['rows']} rows, {rate}, {ws['errors']} write errors, truncated={p['write_truncated']}")
            for ph, v in cut(samples, p["phases"]).items():
                cv = cstat.get(ph, {})
                print(f"   {ph:12s} p50 {_ms(v['p50'])}  p99 {_ms(v['p99'])}  n {v['n']:6d}  err {v['errors']:3d}"
                      + (f"   | control p50 {_ms(cv['p50'])}  p99 {_ms(cv['p99'])}" if cv else ""))
            if ref:
                rec = recovery_s(samples, t0, ref)
                print(f"   control p99 {ref:.0f} ms, limit {RECOVERY_TOL * ref:.0f} ms; "
                      f"p99 back within it for 3 min: {recovery_text(rec)}")
                if op["label"] == "control":
                    print(f"   control itself breaches its own limit: {breached(samples, t0, ref, until - t0)}")
                for w in spikes(samples, t0, until, ref):
                    print(f"     slow window at +{w['t'] / 60:.1f} min p99 {_ms(w['p99'])} errors {w['errors']}")
            if a.events:
                for e in events(read_jsonl_gz(ROOT / p["metrics"]), t0):
                    print(f"     {e['t'] / 60:+6.1f} min  {e['key']} {e['delta']:+.0f}")


if __name__ == "__main__":
    main()
