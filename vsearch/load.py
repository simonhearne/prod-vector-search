"""Open-loop load generation and windowed latency, plus a Prometheus text scraper."""
import re
import threading
import time
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from vsearch.collections import compact_and_wait, wait_index
from vsearch.config import MILVUS_METRICS_URL


@dataclass
class Sample:
    t: float
    latency_ms: float
    ok: bool
    error: str | None


def run_load(fn: Callable[[int], None], qps: float, duration_s: float, threads: int = 4,
             stop: threading.Event | None = None) -> list[Sample]:
    """Query i is due at start + i/qps regardless of how slow earlier queries were. Latency is measured
    from the due time, so time spent queued behind a saturated pool counts (no coordinated omission),
    and Sample.t is the due time, so a sample lands in the phase in which it was offered."""
    out: list[Sample] = []
    lock = threading.Lock()
    start = time.perf_counter()

    def call(i: int) -> None:
        due = start + i / qps
        try:
            fn(i)
            s = Sample(due - start, (time.perf_counter() - due) * 1000, True, None)
        except Exception as e:  # noqa: BLE001 - errors are data here
            s = Sample(due - start, (time.perf_counter() - due) * 1000, False, f"{type(e).__name__}: {e}"[:200])
        with lock:
            out.append(s)

    with ThreadPoolExecutor(threads) as ex:
        i = 0
        while True:
            due = start + i / qps
            if i / qps >= duration_s or (stop is not None and stop.is_set()):
                break
            delay = due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            ex.submit(call, i)
            i += 1
    return sorted(out, key=lambda s: s.t)


def windowed(samples: list[Sample], window_s: float = 5.0) -> list[dict]:
    buckets: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        buckets[int(s.t // window_s)].append(s)
    rows = []
    for w in sorted(buckets):
        lat = np.array([s.latency_ms for s in buckets[w] if s.ok])
        rows.append({"t0": w * window_s, "n": len(buckets[w]),
                     "errors": sum(not s.ok for s in buckets[w]),
                     "p50": float(np.percentile(lat, 50)) if len(lat) else None,
                     "p99": float(np.percentile(lat, 99)) if len(lat) else None})
    return rows


def scrape_metrics_text(text: str, pattern: str) -> dict[str, float]:
    rx = re.compile(pattern)
    out: dict[str, float] = defaultdict(float)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if rx.fullmatch(name):
            try:
                out[name] += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return dict(out)


def scrape_metrics(url: str = MILVUS_METRICS_URL,
                   pattern: str = r"milvus_.*(segment|compaction).*") -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=5) as r:
        return scrape_metrics_text(r.read().decode(), pattern)


_LABEL = re.compile(r'(\w+)="([^"]*)"')


def labelled_sums(text: str, metric: str, by: list[str], where: dict[str, str] | None = None) -> dict[tuple, float]:
    """One metric's series, summed by the named labels, keeping only series matching `where`."""
    out: dict[tuple, float] = defaultdict(float)
    for line in text.splitlines():
        if not line.startswith(metric + "{"):
            continue
        labels = dict(_LABEL.findall(line))
        if all(labels.get(k) == v for k, v in (where or {}).items()):
            out[tuple(labels.get(k, "") for k in by)] += float(line.rsplit(" ", 1)[1])
    return dict(out)


_STATUS = re.compile(r'status="([^"]*)"')


def compaction_tasks(text: str) -> dict[str, float]:
    """milvus_datacoord_compaction_task_num split by status: active = pending + executing (3.0.3 books
    executing as +n on the scheduler and -n on the worker, so the sum nets out), done is cumulative."""
    out = {"active": 0.0, "done": 0.0}
    for line in text.splitlines():
        if not line.startswith("milvus_datacoord_compaction_task_num{"):
            continue
        m = _STATUS.search(line)
        status = m.group(1) if m else ""
        key = "done" if status == "done" else "active" if status in ("pending", "executing") else None
        if key:
            out[key] += float(line.rsplit(" ", 1)[1])
    return out


_QUERY_TYPE = re.compile(r'query_type="([^"]*)"')
_QN_CORE = {"milvus_querynode_sq_core_latency_count": "count", "milvus_querynode_sq_core_latency_sum": "ms"}


def querynode_requests(text: str) -> dict[str, float]:
    """Cumulative query-node requests and core time (ms) per query_type, e.g. qn_search_count,
    qn_upsert_query_ms. "upsert_query" is the read a partial update does before it writes."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        name = line.split("{", 1)[0]
        kind = _QN_CORE.get(name)
        if kind is None:
            continue
        m = _QUERY_TYPE.search(line)
        key = f"qn_{m.group(1) if m else 'unknown'}_{kind}"
        out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def metrics_text(url: str = MILVUS_METRICS_URL) -> str:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode()


def settle(fetch: Callable[[], str] = metrics_text, idle_polls: int = 3, poll_s: float = 5,
           timeout_s: float = 1800) -> int:
    """Block until no compaction is pending or executing for idle_polls consecutive polls, so a
    pass doesn't inherit the previous pass's backlog. Returns the number of polls taken."""
    t0, polls, idle = time.monotonic(), 0, 0
    while True:
        polls += 1
        idle = idle + 1 if compaction_tasks(fetch())["active"] <= 0 else 0
        if idle >= idle_polls:
            return polls
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(f"compaction still active after {timeout_s:.0f}s")
        time.sleep(poll_s)


def phase_summary(samples: list[Sample], phases: dict[str, list[float]]) -> dict[str, dict]:
    """p50/p99 per phase over the raw samples (not a median of window p99s)."""
    out = {}
    for name, (lo, hi) in phases.items():
        ph = [s for s in samples if lo <= s.t < hi]
        lat = np.array([s.latency_ms for s in ph if s.ok])
        out[name] = {"n": len(ph), "errors": sum(not s.ok for s in ph),
                     "p50": float(np.percentile(lat, 50)) if len(lat) else None,
                     "p99": float(np.percentile(lat, 99)) if len(lat) else None}
    return out


def _quiet(fetch: Callable[[], str], polls: int, poll_s: float) -> bool:
    """True if no compaction is active and none finishes for `polls` consecutive polls."""
    done0 = compaction_tasks(fetch())["done"]
    for _ in range(polls):
        time.sleep(poll_s)
        t = compaction_tasks(fetch())
        if t["active"] > 0 or t["done"] != done0:
            return False
    return True


def level_start(client, name: str, n_rows: int, fetch: Callable[[], str] = metrics_text,
                settle_timeout_s: float = 1800, quiet_polls: int = 18, poll_s: float = 5,
                max_rounds: int = 3) -> tuple[float, bool]:
    """Bring a collection to a level start before a measured pass: compact, wait for the re-index the
    compaction triggers, wait for compaction to go idle, then require a quiet window (default 90 s) with no
    compaction starting or finishing. Background compaction can begin after a short idle gap, so repeat
    up to max_rounds; a pass that still isn't level is caught by its baseline check. Raises TimeoutError
    rather than measure a collection that never settles. Returns (seconds, quiet): quiet is False when
    no round reached a quiet window, and callers record it with the measurement."""
    t0 = time.monotonic()
    quiet = False
    for _ in range(max_rounds):
        compact_and_wait(client, name)
        wait_index(client, name, n_rows)
        settle(fetch, timeout_s=settle_timeout_s, poll_s=poll_s)
        if (quiet := _quiet(fetch, quiet_polls, poll_s)):
            break
    if not quiet:
        print(f"WARNING level_start({name}): no compaction-quiet window after {max_rounds} rounds", flush=True)
    return time.monotonic() - t0, quiet
