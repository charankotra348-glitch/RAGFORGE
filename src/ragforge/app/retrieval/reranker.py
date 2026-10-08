from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from ragforge.app.retrieval.types import SearchResult

logger = logging.getLogger(__name__)


_model: CrossEncoder | None = None

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker() -> CrossEncoder:
    global _model

    if _model is None:
        logger.info(
            "Loading reranker model: %s",
            MODEL_NAME,
        )

        _model = CrossEncoder(MODEL_NAME)

        logger.info(
            "Reranker model loaded"
        )

    return _model


def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Rerank retrieved candidates using a cross-encoder.

    The cross-encoder reads the query and each chunk together
    and produces a direct relevance score.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if not results:
        return []

    model = get_reranker()

    pairs = [
        (query, result.text)
        for result in results
    ]

    scores = model.predict(pairs)

    for result, score in zip(results, scores):
        result.rerank_score = float(score)

    reranked = sorted(
        results,
        key=lambda result: result.rerank_score
        if result.rerank_score is not None
        else float("-inf"),
        reverse=True,
    )

    logger.info(
        "Reranked %d candidates, returning top %d",
        len(results),
        min(top_k, len(reranked)),
    )

    return reranked[:top_k]