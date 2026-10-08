from __future__ import annotations

import logging

from fastembed import TextEmbedding
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from ragforge.app.retrieval.bm25 import BM25Retriever
from ragforge.app.retrieval.reranker import rerank
from ragforge.app.retrieval.types import SearchResult
from ragforge.app.retrieval.vector import search_vectors
from ragforge.core.config import Settings
from ragforge.ingestion.embedder import embed_chunks

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

    # -----------------------------------------------------------------------
    # Vector results
    # -----------------------------------------------------------------------

    for rank, result in enumerate(
        vector_results,
        start=1,
    ):
        key = result.text

        if key not in scores:
            scores[key] = SearchResult(
                text=result.text,
                metadata=dict(result.metadata),
                vector_score=result.vector_score,
            )

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
        key = result.text

        if key not in scores:
            scores[key] = SearchResult(
                text=result.text,
                metadata=dict(result.metadata),
                bm25_score=result.bm25_score,
            )
        else:
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


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


class HybridSearcher:
    """
    Hybrid retrieval using:

        1. Qdrant vector search
        2. BM25 keyword search
        3. Reciprocal Rank Fusion
        4. Cross-encoder reranking
    """

    def __init__(
        self,
        *,
        client: QdrantClient,
        settings: Settings,
        embedding_model: TextEmbedding,
        chunks: list[Document],
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
    ):
        self.client = client
        self.settings = settings
        self.embedding_model = embedding_model

        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k

        self.bm25_retriever = BM25Retriever(
            chunks=chunks,
        )

        logger.info(
            "HybridSearcher initialized "
            "(vector_top_k=%d, bm25_top_k=%d)",
            vector_top_k,
            bm25_top_k,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        rrf_k: int = 60,
        rerank_top_k: int = 10,
    ) -> list[SearchResult]:

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be greater than 0")

        # -------------------------------------------------------------------
        # 1. Create query embedding
        # -------------------------------------------------------------------

        query_document = Document(
            page_content=query,
        )

        query_embedding = embed_chunks(
            self.settings,
            [query_document],
        )[0]

        # -------------------------------------------------------------------
        # 2. Vector search
        # -------------------------------------------------------------------

        vector_results = search_vectors(
            client=self.client,
            settings=self.settings,
            query_vector=query_embedding,
            top_k=self.vector_top_k,
        )

        # -------------------------------------------------------------------
        # 3. BM25 search
        # -------------------------------------------------------------------

        bm25_results_raw = self.bm25_retriever.search(
            query=query,
            top_k=self.bm25_top_k,
        )

        # Convert BM25Result objects into SearchResult objects.
        bm25_results = [
            SearchResult(
                text=result.text,
                metadata=dict(result.metadata),
                bm25_score=result.score,
            )
            for result in bm25_results_raw
        ]

        logger.info(
            "Hybrid search: query=%r vector=%d bm25=%d",
            query,
            len(vector_results),
            len(bm25_results),
        )

        # -------------------------------------------------------------------
        # 4. Reciprocal Rank Fusion
        # -------------------------------------------------------------------

        merged = reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=rerank_top_k,
            k=rrf_k,
        )

        # -------------------------------------------------------------------
        # 5. Cross-encoder reranking
        # -------------------------------------------------------------------

        return rerank(
            query=query,
            results=merged,
            top_k=top_k,
        )