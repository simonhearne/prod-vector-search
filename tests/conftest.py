import urllib.request

import pytest

from vsearch.config import MILVUS_METRICS_URL


def _milvus_up() -> bool:
    try:
        url = MILVUS_METRICS_URL.rsplit("/", 1)[0] + "/healthz"
        return urllib.request.urlopen(url, timeout=2).status == 200
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _milvus_up():
        return
    skip = pytest.mark.skip(reason="Milvus not reachable")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
