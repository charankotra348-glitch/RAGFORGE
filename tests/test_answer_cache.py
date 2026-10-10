"""Answer cache and corpus version."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ragforge.app.cache import answers
from ragforge.app.main import app
from ragforge.app.retrieval.types import SearchResult
from ragforge.core.config import get_settings

from fakes import FakeRedis


def fake_result(query: str):
    return SimpleNamespace(
        query=query,
        answer="The monthly rent is $750 [Source 1].",
        sources=[SearchResult(text="Monthly rent of $750.", metadata={"page_number": 1, "section_number": "3"})],
        abstained=False,
        abstention_reason=None,
        timings_ms={"retrieval": 10.0, "generation": 500.0},
    )


@pytest.fixture()
def client(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(app.state, "redis", redis, raising=False)
    monkeypatch.setattr(app.state, "corpus_version", None, raising=False)
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 0)
    monkeypatch.setattr(get_settings(), "answer_cache_ttl_seconds", 600)
    return TestClient(app), redis


def ask(test_client, headers, query="What is the rent?", top_k=5):
    with patch("ragforge.app.main.answer_query", side_effect=lambda settings, query, top_k: fake_result(query)) as answer:
        response = test_client.post("/query", json={"query": query, "top_k": top_k}, headers=headers)
    return response, answer.call_count


def test_repeat_question_is_served_from_cache(client, api_headers, caplog) -> None:
    test_client, redis = client
    caplog.set_level(logging.INFO, logger="ragforge.access")
    first, calls_first = ask(test_client, api_headers)
    second, calls_second = ask(test_client, api_headers, query="  what IS   the rent? ")
    assert (calls_first, calls_second) == (1, 0)
    assert first.json()["cached"] is False and second.json()["cached"] is True
    assert second.json()["answer"] == first.json()["answer"]
    assert second.json()["sources"] == first.json()["sources"]
    assert second.json()["query"] == "what IS   the rent?"  # the caller's own question, not the cached one
    assert [r.fields["cache"] for r in caplog.records if r.name == "ragforge.access"] == ["miss", "hit"]
    assert list(redis.ttl.values()) == [600]


def test_cache_key_never_contains_the_question(client, api_headers) -> None:
    test_client, redis = client
    ask(test_client, api_headers, query="Is my SSN 123-45-6789 needed?")
    (key,) = [k for k in redis.data if k.startswith(answers.ANSWER_KEY_PREFIX)]
    assert "SSN" not in key and "6789" not in key
    assert "6789" not in redis.data[key]  # stored payload has no question either


def test_different_top_k_is_a_different_entry(client, api_headers) -> None:
    test_client, _ = client
    _, first = ask(test_client, api_headers, top_k=5)
    _, second = ask(test_client, api_headers, top_k=3)
    assert (first, second) == (1, 1)


def test_corpus_change_invalidates_cache_and_rebuilds_search_index(client, api_headers) -> None:
    test_client, redis = client
    ask(test_client, api_headers)
    answers.bump_corpus_version(redis)  # what the reindex script and uploads do
    with patch("ragforge.app.main.clear_hybrid_searcher_cache") as rebuild:
        response, calls = ask(test_client, api_headers)
    assert calls == 1 and response.json()["cached"] is False
    rebuild.assert_called_once()


def test_config_change_invalidates_cache(client, api_headers, monkeypatch) -> None:
    test_client, _ = client
    ask(test_client, api_headers)
    monkeypatch.setattr(get_settings(), "ollama_model", "another-model")
    _, calls = ask(test_client, api_headers)
    assert calls == 1


def test_redis_down_bypasses_cache(monkeypatch, api_headers, caplog) -> None:
    monkeypatch.setattr(app.state, "redis", FakeRedis(fail=True), raising=False)
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 0)
    caplog.set_level(logging.INFO, logger="ragforge.access")
    test_client = TestClient(app)
    calls = [ask(test_client, api_headers)[1] for _ in range(2)]
    assert calls == [1, 1]
    assert [r.fields["cache"] for r in caplog.records if r.name == "ragforge.access"] == ["bypass", "bypass"]


def test_cache_disabled_with_zero_ttl(client, api_headers, monkeypatch) -> None:
    test_client, redis = client
    monkeypatch.setattr(get_settings(), "answer_cache_ttl_seconds", 0)
    calls = [ask(test_client, api_headers)[1] for _ in range(2)]
    assert calls == [1, 1] and not [k for k in redis.data if k.startswith(answers.ANSWER_KEY_PREFIX)]


def test_errors_are_not_cached(client, api_headers) -> None:
    test_client, redis = client
    with patch("ragforge.app.main.answer_query", side_effect=ConnectionError("ollama down")):
        assert test_client.post("/query", json={"query": "q"}, headers=api_headers).status_code == 503
    assert not [k for k in redis.data if k.startswith(answers.ANSWER_KEY_PREFIX)]


def test_corpus_version_helpers() -> None:
    redis = FakeRedis()
    assert answers.get_corpus_version(redis) == "0"
    assert answers.bump_corpus_version(redis) == "1"
    assert answers.get_corpus_version(redis) == "1"
    assert answers.get_corpus_version(None) is None
    assert answers.bump_corpus_version(FakeRedis(fail=True)) is None


def test_reindex_bumps_corpus_version_and_returns_summary(monkeypatch) -> None:
    from langchain_core.documents import Document

    from ragforge.vector import reindex as reindex_module

    redis = FakeRedis()
    chunks = [
        Document(page_content="a", metadata={"document_id": "doc1", "source": "/x/lease.pdf", "chunk_index": 0, "page_number": 1, "page_end_number": 2}),
        Document(page_content="b", metadata={"document_id": "doc1", "source": "/x/lease.pdf", "chunk_index": 1, "page_number": 3}),
    ]
    monkeypatch.setattr(reindex_module, "ingest_file", lambda path: ["page"])
    monkeypatch.setattr(reindex_module, "chunk_documents", lambda docs, chunk_size, chunk_overlap: chunks)
    monkeypatch.setattr(reindex_module, "validate_chunks", lambda chunks, chunk_size: None)
    monkeypatch.setattr(reindex_module, "generate_embeddings", lambda chunks, settings: [[0.0] * 384] * len(chunks))
    monkeypatch.setattr(reindex_module, "get_qdrant", lambda settings: object())
    monkeypatch.setattr(reindex_module, "replace_document_chunks", lambda client, settings, chunks, embeddings: len(chunks))
    monkeypatch.setattr(reindex_module, "get_collection_count", lambda client, settings: 2)
    monkeypatch.setattr(reindex_module, "count_document_chunks", lambda client, settings, document_id: 2)
    monkeypatch.setattr("ragforge.app.cache.redis.get_redis", lambda settings: redis)

    summary = reindex_module.reindex("lease.pdf", get_settings())
    assert (summary.document_id, summary.chunks, summary.pages, summary.corpus_version) == ("doc1", 2, 3, "1")
    assert answers.get_corpus_version(redis) == "1"
