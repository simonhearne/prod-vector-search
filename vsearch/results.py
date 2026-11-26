"""The only way results reach results/*.json: payload + run metadata."""
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from vsearch.config import RESULTS, ROOT


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def _mem_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


def _cpu() -> str:
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if out.stdout.strip():
            return out.stdout.strip()
    return platform.processor() or platform.machine()


def _relpaths(obj):
    """Store paths under the repo as repo-relative, so result files don't carry the machine's home path."""
    root = f"{ROOT}{os.sep}"
    if isinstance(obj, str):
        return obj[len(root):] if obj.startswith(root) else obj
    if isinstance(obj, Path):
        return _relpaths(str(obj))
    if isinstance(obj, dict):
        return {k: _relpaths(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_relpaths(v) for v in obj]
    return obj


def write_result(bench_id: str, payload: dict, *, client=None) -> Path:
    payload = _relpaths(payload)
    meta = {"git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "milvus_version": client.get_server_version() if client is not None else None,
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu": _cpu(), "mem_gb": round(_mem_gb(), 1),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sample": bool(payload.get("sample", False))}
    path = RESULTS / f"{bench_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"payload": payload, "meta": meta}, indent=2, default=float))
    return path


def read_result(bench_id: str) -> dict:
    return json.loads((RESULTS / f"{bench_id}.json").read_text())
