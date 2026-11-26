"""LLM relevance judge (ESCI labels) via the Message Batches API.

Judgments are cached in data/work/judgments.parquet keyed on (query_id, product_id, model)
so re-running never re-buys a label. Errored/expired results stay unlabelled.
"""
import json
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from vsearch.config import DATA_DATASET, DATA_WORK

JUDGE_MODEL = "claude-haiku-5-5"
SPOTCHECK_MODEL = "claude-opus-5-5"
CACHE = DATA_WORK / "judgments.parquet"
PUBLISHED = DATA_DATASET / "judgments.parquet"  # read-only fallback for a fresh clone (data.fetch)
BATCH_LOG = DATA_WORK / "judge_batches.jsonl"
LABELS = ["E", "S", "C", "I"]
# Bump when the request (prompt, schema, product text) changes: only current-version judgments count.
# "nosystem-5699a45" tags the 3 Nov fills that ran without SYSTEM_PROMPT.
PROMPT_VERSION = "esci-v1"

SYSTEM_PROMPT = """You judge product search relevance using the Amazon ESCI labels.

E (Exact): the product is relevant for the query and satisfies all the query specifications.
S (Substitute): the product is somewhat relevant; it fails to fulfil some aspects of the query \
but can be used as a functional substitute.
C (Complement): the product does not fulfil the query but could be used in combination with an \
exact item.
I (Irrelevant): the product is irrelevant to the query or fails a central aspect of it.

Return only the single label that best fits."""

SCHEMA = {"type": "object",
          "properties": {"label": {"type": "string", "enum": LABELS}},
          "required": ["label"], "additionalProperties": False}


def _product_text(p: dict) -> str:
    return (f"Title: {p.get('title', '')}\nBrand: {p.get('brand', '')}\n"
            f"Details: {str(p.get('bullets', ''))[:1500]}")


SEP = "__"  # Batch API custom_id must match ^[a-zA-Z0-9_-]{1,64}$; ESCI ids are alphanumeric


def custom_id(query_id: str, product_id: str) -> str:
    return f"{query_id}{SEP}{product_id}"


def build_request(custom_id: str, query: str, product: dict, model: str = JUDGE_MODEL) -> Request:
    return Request(custom_id=custom_id, params=MessageCreateParamsNonStreaming(
        model=model, max_tokens=2048,  # Haiku/Opus 5.5 think by default; thinking counts toward max_tokens
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user",
                   "content": f"<query>{query}</query>\n<product>\n{_product_text(product)}\n</product>"}]))


def submit(client, pairs: pd.DataFrame, products: pd.DataFrame, model: str = JUDGE_MODEL) -> str:
    """pairs: query_id, query, product_id. Returns the batch id (also logged)."""
    prod = products.set_index("product_id")[["title", "brand", "bullets"]].to_dict("index")
    reqs = [build_request(custom_id(r.query_id, r.product_id), r.query, prod[r.product_id], model)
            for r in pairs.itertuples()]
    batch = client.messages.batches.create(requests=reqs)
    BATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(BATCH_LOG, "a") as f:
        f.write(json.dumps({"batch_id": batch.id, "model": model, "n": len(reqs)}) + "\n")
    return batch.id


def collect(client, batch_id: str) -> tuple[pd.DataFrame, dict]:
    counts = Counter({"succeeded": 0, "errored": 0, "expired": 0, "canceled": 0, "truncated": 0})
    rows = []
    for res in client.messages.batches.results(batch_id):
        if res.result.type != "succeeded":
            counts[res.result.type] += 1
            continue
        msg = res.result.message
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if msg.stop_reason == "max_tokens" or text is None:
            counts["truncated"] += 1  # left uncached, so a rerun of calibrate/fill resubmits it
            continue
        counts["succeeded"] += 1
        qid, pid = res.custom_id.split(SEP, 1)
        rows.append({"query_id": qid, "product_id": pid, "label": json.loads(text)["label"]})
    return pd.DataFrame(rows, columns=["query_id", "product_id", "label"]), dict(counts)


COLUMNS = ["query_id", "product_id", "label", "model", "prompt"]


@contextmanager
def use_pool(path: Path):
    """Read and write judgments in `path` alone (no published fallback), so a benchmark can keep its own pool
    and leave the main pool's fingerprint unmoved."""
    global CACHE, PUBLISHED
    before = CACHE, PUBLISHED
    CACHE = PUBLISHED = path
    try:
        yield
    finally:
        CACHE, PUBLISHED = before


def _all_judgments() -> pd.DataFrame:
    src = CACHE if CACHE.exists() else PUBLISHED
    df = pd.read_parquet(src) if src.exists() else pd.DataFrame(columns=COLUMNS)
    return df if "prompt" in df else df.assign(prompt=None)


def cached_judgments() -> pd.DataFrame:
    """Judgments made with the current request (PROMPT_VERSION). Older rows stay on disk, unused."""
    df = _all_judgments()
    return df[df.prompt == PROMPT_VERSION].reset_index(drop=True)


def append_judgments(df: pd.DataFrame) -> None:
    df = df.assign(prompt=PROMPT_VERSION) if "prompt" not in df else df
    out = pd.concat([_all_judgments(), df], ignore_index=True).drop_duplicates(
        ["query_id", "product_id", "model", "prompt"], keep="last")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    a, b = list(a), list(b)
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def filled_qrels(qrels: pd.DataFrame, model: str = JUDGE_MODEL) -> pd.DataFrame:
    """ESCI labels plus this model's judgments; ESCI wins on a duplicate (query_id, product_id)."""
    cache = cached_judgments()
    llm = cache[cache.model == model][["query_id", "product_id", "label"]]
    return pd.concat([qrels[["query_id", "product_id", "label"]], llm], ignore_index=True).drop_duplicates(
        ["query_id", "product_id"], keep="first").reset_index(drop=True)


def unjudged_pairs(rankings: Iterable[dict[str, list[str]]], qrels: pd.DataFrame) -> pd.DataFrame:
    judged = set(zip(qrels.query_id, qrels.product_id))
    pairs = sorted({(q, p) for r in rankings for q, ids in r.items() for p in ids if (q, p) not in judged})
    return pd.DataFrame(pairs, columns=["query_id", "product_id"])


def fill(client, rankings: Iterable[dict[str, list[str]]], qrels: pd.DataFrame, queries: pd.DataFrame,
         products: pd.DataFrame, model: str = JUDGE_MODEL, max_pairs: int = 30_000, poll_s: float = 60) -> dict:
    """Judge every unjudged, uncached pair in the rankings (against the ESCI query text for that
    query_id), wait for the batch, cache the labels. Refuses before submitting if over max_pairs."""
    pairs = unjudged_pairs(rankings, qrels)
    cache = cached_judgments()
    done = set(zip(cache[cache.model == model].query_id, cache[cache.model == model].product_id))
    todo = pairs[[(q, p) not in done for q, p in zip(pairs.query_id, pairs.product_id)]]
    out = {"pairs": len(pairs), "submitted": len(todo), "batch": None, "counts": {}}
    if len(todo) > max_pairs:
        raise ValueError(f"{len(todo)} uncached pairs > max_pairs={max_pairs}")
    print(f"fill: {len(pairs):,} pairs, {len(todo):,} submitted", flush=True)
    if not len(todo):
        return out
    todo = todo.merge(queries[["query_id", "query"]], on="query_id")
    out["batch"] = bid = submit(client, todo, products, model)
    while client.messages.batches.retrieve(bid).processing_status != "ended":
        time.sleep(poll_s)
    got, out["counts"] = collect(client, bid)
    append_judgments(got.assign(model=model))
    return out


def fingerprint() -> dict:
    """Identity of the current judgment pool. Filled nDCG depends on it (IDCG uses every label a query
    has), so every filled number records the pool it was scored at."""
    import hashlib
    c = cached_judgments()
    rows = sorted(zip(c.query_id.astype(str), c.product_id.astype(str), c.label.astype(str), c.model.astype(str)))
    return {"rows": len(rows), "sha": hashlib.sha1(repr(rows).encode()).hexdigest()[:12], "prompt": PROMPT_VERSION}
