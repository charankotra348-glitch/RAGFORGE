from __future__ import annotations

import logging

from fastembed import TextEmbedding
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from ragforge.app.retrieval.bm25 import BM25Retriever
from ragforge.app.retrieval.reranker import rerank
from ragforge.app.retrieval.fusion import reciprocal_rank_fusion
from ragforge.app.retrieval.types import SearchResult
from ragforge.app.retrieval.vector import search_vectors
from ragforge.core.config import Settings
from ragforge.ingestion.embedder import embed_chunks

logger = logging.getLogger(__name__)


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
        rerank_top_k: int = 30,
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

        # BM25Retriever already returns SearchResult objects. Keep the
        # shared result contract instead of converting through a nonexistent
        # `.score` attribute.
        bm25_results = bm25_results_raw

        logger.info(
            "Hybrid search: vector=%d bm25=%d",
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