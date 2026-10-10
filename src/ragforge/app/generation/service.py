from __future__ import annotations

import logging
from dataclasses import dataclass

from ragforge.app.generation.grounding import (
    SAFE_ABSTENTION,
    is_abstention_answer,
    verify_answer_grounded,
)
from ragforge.app.generation.llm import generate_answer
from ragforge.app.retrieval.search import HybridSearcher
from ragforge.app.retrieval.types import SearchResult
from ragforge.app.retrieval.vector import get_qdrant
from ragforge.app.retrieval.vector_store import get_indexed_chunks
from ragforge.core.config import Settings
from ragforge.ingestion.embedder import get_embedding_model

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: list[SearchResult]


_searcher: HybridSearcher | None = None
_searcher_collection: str | None = None


def get_hybrid_searcher(settings: Settings) -> HybridSearcher:
    """
    Initialize the hybrid retriever from the indexed Qdrant corpus.

    Building BM25 from Qdrant payloads ensures lexical and dense retrieval
    search the same indexed chunks, rather than a hard-coded local PDF.
    """

    global _searcher, _searcher_collection

    if _searcher is not None and _searcher_collection == settings.qdrant_collection:
        return _searcher

    client = get_qdrant(settings)
    chunks = get_indexed_chunks(client, settings)

    if not chunks:
        raise RuntimeError(
            f"Qdrant collection {settings.qdrant_collection!r} is empty. "
            "Index at least one document before querying RAGForge."
        )

    logger.info("Loaded %d indexed chunks for BM25", len(chunks))

    _searcher = HybridSearcher(
        client=client,
        settings=settings,
        embedding_model=get_embedding_model(settings),
        chunks=chunks,
        vector_top_k=30,
        bm25_top_k=30,
    )
    _searcher_collection = settings.qdrant_collection

    return _searcher


def clear_hybrid_searcher_cache() -> None:
    """Invalidate the in-process BM25 index after the Qdrant corpus changes."""
    global _searcher, _searcher_collection
    _searcher = None
    _searcher_collection = None


def answer_query(
    settings: Settings,
    query: str,
    top_k: int = 5,
) -> RAGResponse:
    """Retrieve relevant chunks and generate a grounded answer."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    logger.info("Starting hybrid RAG query: %s", query)

    searcher = get_hybrid_searcher(settings)

    results = searcher.search(
        query,
        top_k=top_k,
    )

    logger.info(
        "Hybrid retrieval returned %d chunks",
        len(results),
    )

    answer = generate_answer(settings=settings, query=query, results=results)

    if not is_abstention_answer(answer):
        grounded = verify_answer_grounded(
            settings=settings,
            query=query,
            answer=answer,
            results=results,
        )

        if not grounded:
            logger.warning(
                "Answer failed grounding verification; returning safe abstention"
            )
            answer = SAFE_ABSTENTION

    logger.info("RAG query completed")
    return RAGResponse(query=query, answer=answer, sources=results)