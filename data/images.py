"""Download product images (resumable), resized to <=384px longest side.

    uv run python -m data.images

Writes data/work/images/<last2>/<product_id>.jpg and data/work/images_manifest.parquet.
"""
import io
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from vsearch.config import DATA_WORK

IMAGES = DATA_WORK / "images"
MAX_SIDE = 384


def image_path(product_id: str, root: Path = IMAGES) -> Path:
    return root / product_id[-2:] / f"{product_id}.jpg"


_session = requests.Session()


def requests_fetch(url: str) -> bytes:
    r = _session.get(url, timeout=30)
    r.raise_for_status()
    return r.content


def _one(pid: str, url, fetch: Callable[[str], bytes], root: Path) -> dict:
    dest = image_path(pid, root)
    if dest.exists():
        return {"product_id": pid, "ok": True, "error": None}
    if not isinstance(url, str) or not url:
        return {"product_id": pid, "ok": False, "error": "no_url"}
    try:
        img = Image.open(io.BytesIO(fetch(url))).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".jpg.part")
        img.save(part, "JPEG", quality=90)
        part.rename(dest)
        return {"product_id": pid, "ok": True, "error": None}
    except Exception as e:  # noqa: BLE001 - record and continue
        return {"product_id": pid, "ok": False, "error": f"{type(e).__name__}: {e}"[:200]}


def download_all(products: pd.DataFrame, fetch: Callable[[str], bytes] = requests_fetch,
                 threads: int = 32, root: Path = IMAGES) -> pd.DataFrame:
    jobs = list(zip(products.product_id, products.image_url))
    with ThreadPoolExecutor(threads) as ex:
        rows = list(tqdm(ex.map(lambda j: _one(j[0], j[1], fetch, root), jobs),
                         total=len(jobs), disable=len(jobs) < 1000, mininterval=30))
    return pd.DataFrame(rows, columns=["product_id", "ok", "error"]).astype({"ok": bool})


def main() -> None:
    products = pd.read_parquet(DATA_WORK / "products.parquet", columns=["product_id", "image_url"])
    m = download_all(products)
    m.to_parquet(DATA_WORK / "images_manifest.parquet", index=False)
    print(f"images ok {m.ok.sum():,}/{len(m):,}; errors: {m.error.value_counts().head(5).to_dict()}")


if __name__ == "__main__":
    main()
