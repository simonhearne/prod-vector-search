"""4.4 follow-up: does a flush or a forced compaction shorten the ignore_growing blackout?

Four B4c ignore_growing passes (each from a level start, which itself compacts = "before"):
none, flush right after the storm, compact mid-storm (non-blocking), compact after the storm
(waited). Per pass: B4c's marker stats, plus how long markers written before the hook took to
appear after it fired."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench._common import client, ensure_collection, ensure_prices, load_corpus
from bench.b4a_write_storm import ReadOpts
from bench.b4c_freshness import MARKER_TIMEOUT_S, run_variant
from vsearch.collections import compact_and_wait, release_all
from vsearch.results import read_result

STORM_S = 300
PASSES = {
    "none": {},
    "flush_after": {"flush": (STORM_S, lambda c, n: c.flush(n))},
    "compact_mid": {"compact": (STORM_S / 2, lambda c, n: c.compact(n))},
    "compact_after": {"compact": (STORM_S, lambda c, n: compact_and_wait(c, n))},
}


def after_hook(ev, t_hook):
    d = [e["t_visible"] - t_hook if e["t_visible"] is not None else None for e in ev if e["t_write"] < t_hook]
    seen = [x for x in d if x is not None]
    return {"n_before_hook": len(d), "still_hidden": len(d) - len(seen),
            "visible_after_hook_p50": float(np.percentile(seen, 50)) if seen else None,
            "visible_after_hook_max": float(max(seen)) if seen else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", nargs="+", choices=list(PASSES), default=list(PASSES),
                    help="run only these passes; results merge into the existing JSON")
    args = ap.parse_args()
    c = client()
    ef = read_result("poc_config")["payload"]["ef"]
    products, tv, iv, has = load_corpus()
    release_all(c)
    name = ensure_collection(c, "prod_tuned")
    ensure_prices(c, name, products)
    out_path = Path(__file__).with_suffix(".json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for k, (label, hooks) in enumerate(PASSES.items()):
        if label not in args.passes:
            continue
        r = run_variant(c, name, 10 + k, ReadOpts("Bounded", ignore_growing=True), products, tv, iv, has,
                        STORM_S, MARKER_TIMEOUT_S, ef, hooks=hooks)
        for h in hooks:
            r[f"{h}_effect"] = after_hook(r["events"], r["hooks"][h])
            r[f"{h}_took_s"] = r["hooks"][h + "_done"] - r["hooks"][h]
        out[label] = r
        print(time.strftime("%T"), label, {x: r[x] for x in ("n", "timed_out", "visible_p50", "visible_p99",
              "missing_share", "missing_p50")}, {h: r[f"{h}_effect"] for h in hooks}, flush=True)
        out_path.write_text(json.dumps(out, indent=1))
    ensure_prices(c, name, products)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
