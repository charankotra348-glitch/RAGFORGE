"""Opt-in integration test for a running RAGForge stack.

Set RAGFORGE_RUN_INTEGRATION=1 to run this test. It requires Qdrant and
Ollama to be running, the configured Ollama model to be pulled, and the
example document to have been indexed. Ordinary pytest runs do not make
network/model calls during test collection.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RAGFORGE_RUN_INTEGRATION") != "1",
    reason="Set RAGFORGE_RUN_INTEGRATION=1 to run the live RAG integration test",
)


def test_answer_query_returns_answer_and_typed_sources() -> None:
    from ragforge.app.generation.service import answer_query
    from ragforge.core.config import get_settings

    response = answer_query(
        get_settings(),
        "What happens to the security deposit when the tenant vacates?",
    )

    assert response.answer.strip()
    assert response.query == "What happens to the security deposit when the tenant vacates?"
    for source in response.sources:
        assert source.text.strip()
        assert isinstance(source.metadata, dict)
