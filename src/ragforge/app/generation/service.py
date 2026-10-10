from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

from ragforge.app.generation.grounding import (
    SAFE_ABSTENTION,
    check_answer_grounded,
    is_abstention_answer,
)
from ragforge.app.generation.llm import generate_answer_with_stats
from ragforge.app.retrieval.search import HybridSearcher
from ragforge.app.retrieval.types import SearchResult
from ragforge.app.retrieval.vector import get_qdrant
from ragforge.app.retrieval.vector_store import get_indexed_chunks
from ragforge.core.config import Settings
from ragforge.ingestion.embedder import get_embedding_model

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Final answer plus what happened on the way.

    abstention_reason: None (answered), "no_results" (retrieval returned
    nothing), "model_abstained" (the model said the answer is not in the
    sources) or "failed_verification" (the verifier rejected the model's
    answer, or the check was malformed or raised; see ``verification``).
    """

    query: str
    answer: str
    sources: list[SearchResult]
    abstained: bool = False
    abstention_reason: str | None = None
    candidate_answer: str | None = None
    verification: dict | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    generation_stats: dict | None = None


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
    *,
    verify: bool = True,
    model: str | None = None,
    system_prompt: str | None = None,
    num_ctx: int | None = None,
) -> RAGResponse:
    """Retrieve relevant chunks, generate an answer, and verify its grounding.

    The keyword-only options exist for evaluation. The API uses the defaults.
    Note that the verifier always uses settings.ollama_model, even when
    ``model`` overrides the generation model.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    logger.info("Starting hybrid RAG query: %s", query)

    timings: dict[str, float] = {}
    start = time.perf_counter()
    searcher = get_hybrid_searcher(settings)
    results = searcher.search(query, top_k=top_k)
    timings["retrieval"] = _ms_since(start)

    logger.info("Hybrid retrieval returned %d chunks", len(results))

    start = time.perf_counter()
    candidate, generation_stats = generate_answer_with_stats(
        settings=settings,
        query=query,
        results=results,
        model=model,
        system_prompt=system_prompt,
        num_ctx=num_ctx,
    )
    timings["generation"] = _ms_since(start)

    response = RAGResponse(
        query=query,
        answer=candidate,
        sources=results,
        candidate_answer=candidate,
        timings_ms=timings,
        generation_stats=generation_stats,
    )

    if not results:
        response.abstained = True
        response.abstention_reason = "no_results"
    elif is_abstention_answer(candidate):
        response.abstained = True
        response.abstention_reason = "model_abstained"
    elif verify:
        verdict = check_answer_grounded(
            settings=settings,
            query=query,
            answer=candidate,
            results=results,
        )
        response.verification = asdict(verdict)
        if verdict.latency_ms is not None:
            timings["verification"] = verdict.latency_ms
        if not verdict.supported:
            logger.warning(
                "Answer failed grounding verification (%s); returning safe abstention",
                verdict.status,
            )
            response.answer = SAFE_ABSTENTION
            response.abstained = True
            response.abstention_reason = "failed_verification"

    logger.info("RAG query completed")
    return response


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
