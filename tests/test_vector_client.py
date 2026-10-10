"""get_qdrant() must not cache a client whose collection check failed."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ragforge.app.retrieval import vector

SETTINGS = SimpleNamespace(qdrant_host="localhost", qdrant_port=6333, qdrant_collection="c", vector_size=384)


@pytest.fixture(autouse=True)
def reset_cached_client(monkeypatch):
    monkeypatch.setattr(vector, "_client", None)


def test_failed_collection_check_is_not_cached():
    first, second = MagicMock(name="first"), MagicMock(name="second")
    with (
        patch.object(vector, "QdrantClient", side_effect=[first, second]),
        patch.object(vector, "_ensure_collection", side_effect=[ConnectionError("down"), None]),
    ):
        with pytest.raises(ConnectionError):
            vector.get_qdrant(SETTINGS)
        first.close.assert_called_once()

        assert vector.get_qdrant(SETTINGS) is second


def test_successful_client_is_reused():
    client = MagicMock()
    with (
        patch.object(vector, "QdrantClient", return_value=client) as factory,
        patch.object(vector, "_ensure_collection"),
    ):
        assert vector.get_qdrant(SETTINGS) is client
        assert vector.get_qdrant(SETTINGS) is client
    factory.assert_called_once()
