from __future__ import annotations

import logging

from ragforge.app.retrieval.types import SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _rrf_score(
    rank: int,
    k: int = 60,
) -> float:
    """
    Calculate Reciprocal Rank Fusion score.

    RRF:
        score = 1 / (k + rank)

    Rank 1 receives the highest contribution.
    """
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    vector_results: list[SearchResult],
    bm25_results: list[SearchResult],
    top_k: int = 5,
    k: int = 60,
) -> list[SearchResult]:
    """
    Merge vector and BM25 rankings using Reciprocal Rank Fusion.

    A document appearing in both rankings receives contributions
    from both retrievers.
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if k <= 0:
        raise ValueError("RRF k must be greater than 0")

    scores: dict[str, SearchResult] = {}

    def result_key(result: SearchResult) -> str:
        metadata = result.metadata
        document_id = metadata.get("document_id")
        chunk_index = metadata.get("chunk_index")
        if document_id is not None and chunk_index is not None:
            return f"{document_id}:{chunk_index}"
        return result.text

    # -----------------------------------------------------------------------
    # Vector results
    # -----------------------------------------------------------------------

    for rank, result in enumerate(
        vector_results,
        start=1,
    ):
        key = result_key(result)

        if key not in scores:
            scores[key] = SearchResult(
                text=result.text,
                metadata=dict(result.metadata),
                vector_score=result.vector_score,
                bm25_score=result.bm25_score,
                rerank_score=result.rerank_score,
            )
        elif result.vector_score is not None:
            scores[key].vector_score = result.vector_score

        scores[key].rrf_score += _rrf_score(
            rank,
            k,
        )

    # -----------------------------------------------------------------------
    # BM25 results
    # -----------------------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):
        key = result_key(result)

        if key not in scores:
            scores[key] = SearchResult(
                text=result.text,
                metadata=dict(result.metadata),
                vector_score=result.vector_score,
                bm25_score=result.bm25_score,
                rerank_score=result.rerank_score,
            )
        elif result.bm25_score is not None:
            scores[key].bm25_score = result.bm25_score

        scores[key].rrf_score += _rrf_score(
            rank,
            k,
        )

    # -----------------------------------------------------------------------
    # Final ranking
    # -----------------------------------------------------------------------

    merged = sorted(
        scores.values(),
        key=lambda result: result.rrf_score,
        reverse=True,
    )

    logger.info(
        "RRF merged %d vector + %d BM25 results "
        "into %d unique results",
        len(vector_results),
        len(bm25_results),
        len(merged),
    )

    return merged[:top_k]


