"""Replicate B2's unfiltered tax (item 2.2) in reverse order (tuned first, then naive). The full B2 run
measured naive first and tuned straight after a 448k-row price restore; ~2x p50 vs the 3 Nov probe's 1.3x."""
import json
from pathlib import Path

from bench._common import client, load_corpus
from bench.b2_filters import unfiltered_tax

c = client()
products, docs, _, _ = load_corpus()
out = unfiltered_tax(c, products, docs, order=("tuned", "naive"))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
print("DONE", flush=True)
