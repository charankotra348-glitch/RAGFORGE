from __future__ import annotations

import logging
from dataclasses import dataclass

from ragforge.app.generation.llm import generate_answer
from ragforge.app.retrieval.search import SearchResult, search_vectors
from ragforge.app.retrieval.vector import get_qdrant
from ragforge.core.config import Settings
from ragforge.ingestion.embedder import get_embedding_model

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: list[SearchResult]


def answer_query(
    settings: Settings,
    query: str,
    top_k: int = 5,
) -> RAGResponse:
    """
    Run the vector-only RAG pipeline.

    Query
      ↓
    Query embedding
      ↓
    Vector retrieval
      ↓
    Prompt construction
      ↓
    LLM generation
      ↓
    Grounded answer
    """

    if not query.strip():
        raise ValueError("Query cannot be empty")

    logger.info("Starting RAG query: %s", query)

    # ------------------------------------------------------------------
    # Qdrant
    # ------------------------------------------------------------------

    client = get_qdrant(settings)

    # ------------------------------------------------------------------
    # Embedding model
    # ------------------------------------------------------------------

    embedding_model = get_embedding_model(settings)

    # ------------------------------------------------------------------
    # Vector retrieval
    # ------------------------------------------------------------------

    results = search_vectors(
        client=client,
        settings=settings,
        embedding_model=embedding_model,
        query=query,
        top_k=top_k,
    )

    logger.info(
        "Retrieved %d chunks for query",
        len(results),
    )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    answer = generate_answer(
        settings=settings,
        query=query,
        results=results,
    )

    logger.info("RAG query completed")

    return RAGResponse(
        query=query,
        answer=answer,
        sources=results,
    )