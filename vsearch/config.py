"""Project-wide paths and constants. Values pinned by the design spec."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_WORK = ROOT / "data" / "work"
DATA_DATASET = ROOT / "data" / "dataset"
RESULTS = ROOT / "results"

SEED = 27
K = 10
GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
POC_CATEGORY = "Electronics"
POC_ROWS = 20_000
EVAL_SAMPLE = 5_000
QUERY_PROMPT = "Instruct: Given a shopping query, retrieve relevant product listings\nQuery: "



def _dotenv(path: Path) -> dict[str, str]:
    """Minimal .env reader: KEY=VALUE lines, # comments ignored."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


_ENV = {**_dotenv(ROOT / ".env"), **os.environ}
MILVUS_PORT = _ENV.get("MILVUS_PORT", "19540")
MILVUS_METRICS_PORT = _ENV.get("MILVUS_METRICS_PORT", "9095")
MILVUS_URI = _ENV.get("MILVUS_URI", f"http://127.0.0.1:{MILVUS_PORT}")
MILVUS_METRICS_URL = _ENV.get("MILVUS_METRICS_URL", f"http://127.0.0.1:{MILVUS_METRICS_PORT}/metrics")


def export_dotenv(path: Path = ROOT / ".env", keys=("ANTHROPIC_API_KEY", "HF_TOKEN")) -> None:
    """Copy selected secrets from .env into os.environ (never overriding what is already set)."""
    env = _dotenv(path)
    for k in keys:
        if k in env and k not in os.environ:
            os.environ[k] = env[k]


export_dotenv()
