import pytest
from pymilvus import MilvusClient

from vsearch.config import MILVUS_URI


@pytest.mark.integration
def test_server_version():
    assert MilvusClient(uri=MILVUS_URI).get_server_version().lstrip("v").startswith("3.0")
