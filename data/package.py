"""Assemble the published dataset in data/dataset/ (and optionally upload it).

    uv run python -m data.package [--publish]

Contents mirror data/work/ names so bench/_common.py reads either location:
products.parquet, queries.parquet, qrels.parquet, curated_queries.parquet (every paraphrase, as
drafted), judgments.parquet, text_vec.npy, image_vec.npy, has_image.npy, q_ids.npy, q_text_vec.npy,
q_siglip_vec.npy, SHA256SUMS and a README dataset card. Image URLs only, no image bytes.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd

from vsearch.config import DATA_DATASET, DATA_WORK
from vsearch.judge import fingerprint

REPO_ID = "simonhearne/vector-search-ecom"
CURATED = Path(__file__).parent / "curated_queries.jsonl"
COPY = ["products.parquet", "queries.parquet", "qrels.parquet", "judgments.parquet", "text_vec.npy",
        "image_vec.npy", "has_image.npy", "q_ids.npy", "q_text_vec.npy", "q_siglip_vec.npy"]

CARD = """---
license: other
license_name: mixed-see-card
task_categories: [text-retrieval]
language: [en]
pretty_name: Vector search in production (ESCI x Amazon Reviews 2023)
---
# vector-search-ecom

Data behind the talk *Lessons from deploying vector search in production* (London Search & AI
User Group, 26 Nov 2026). Code: https://github.com/simonhearne/prod-vector-search

* ~448k US products: Amazon Shopping Queries (ESCI) product text joined to Amazon Reviews 2023
  metadata (price, rating, category, **image URL** - no image bytes are redistributed).
* ESCI US queries and E/S/C/I judgments restricted to these products; `in_eval5k` marks the
  5,000-query evaluation sample (seed 27).
* `text_vec.npy` - Qwen/Qwen3-Embedding-0.6B (1024-d, unit norm), rows in `row_id` order.
* `image_vec.npy` / `has_image.npy` - SigLIP ViT-L-16-SigLIP-256 image vectors; rows without an
  image hold the unit placeholder e_0 and `has_image = False`.
* `q_*.npy` - query vectors (Qwen3 with the shopping-query instruction; SigLIP text tower).
* `judgments.parquet` - LLM judgments (Claude) used to fill unjudged results, with `model` and
  `prompt` columns. Only rows with `prompt == "{prompt}"` count; older rows are kept for the record.
  Every filled nDCG in the talk was scored at this pool: {rows} rows, fingerprint `{sha}`
  (`vsearch.judge.fingerprint()`).

## Licences and attribution

* Shopping Queries Dataset (ESCI): Reddy et al., 2022, Apache-2.0 (amazon-science/esci-data).
* Amazon Reviews 2023: Hou et al., 2024, McAuley Lab, UC San Diego. Redistributed fields are
  limited to item metadata and image URLs; check the source dataset's terms before reuse.

Licence decision (17 Nov 2026): the Amazon Reviews 2023 release states no licence, only
attribution. Its item metadata and image URLs are redistributed here unchanged, with attribution,
for research reproducibility; no review text and no image bytes. Contact the maintainer for removal.
"""


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package(dest: Path = DATA_DATASET) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        src = DATA_WORK / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dest / name)
    cur = pd.read_json(CURATED, lines=True, dtype={"query_id": str})
    cur.to_parquet(dest / "curated_queries.parquet", index=False)
    (dest / "README.md").write_text(CARD.format(**fingerprint()))
    files = sorted(p for p in dest.iterdir() if p.name != "SHA256SUMS" and not p.name.endswith(".part"))
    (dest / "SHA256SUMS").write_text("".join(f"{_sha256(p)}  {p.name}\n" for p in files))
    return files


def publish(dest: Path = DATA_DATASET) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=str(dest))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args(argv)
    files = package()
    print(f"packaged {len(files)} files in {DATA_DATASET}")
    if a.publish:
        publish()


if __name__ == "__main__":
    main()
