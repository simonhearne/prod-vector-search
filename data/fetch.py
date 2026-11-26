"""Download the published dataset (standard library only).

    python data/fetch.py            # -> data/dataset/

Every file listed in the remote SHA256SUMS is fetched via a .part rename and
verified; files already present and verified are skipped.
"""
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "dataset"
BASE_URL = "https://huggingface.co/datasets/simonhearne/vector-search-ecom/resolve/main"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sums(base_url: str) -> dict[str, str]:
    with urllib.request.urlopen(f"{base_url}/SHA256SUMS") as r:
        lines = r.read().decode().splitlines()
    return {name.strip(): sha for sha, name in (l.split(None, 1) for l in lines if l.strip())}


def fetch(dest: Path = DEST, verify: bool = True, base_url: str = BASE_URL) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for name, sha in _sums(base_url).items():
        path = dest / name
        if path.exists() and (not verify or _sha256(path) == sha):
            out.append(path)
            continue
        part = path.with_name(path.name + ".part")
        with urllib.request.urlopen(f"{base_url}/{name}") as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        if verify and _sha256(part) != sha:
            part.unlink()
            raise ValueError(f"sha256 mismatch for {name}")
        part.rename(path)
        out.append(path)
        print(f"fetched {name}", flush=True)
    return out


if __name__ == "__main__":
    sys.exit(print(f"{len(fetch())} files in {DEST}"))
