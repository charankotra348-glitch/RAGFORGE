from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ragforge.app.retrieval.types import SearchResult

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Keyword-based BM25 retriever.

    BM25 is useful for exact terms and phrases that
    semantic vector search may rank poorly.
    """

    def __init__(self, chunks: Sequence[Document]):
        if not chunks:
            raise ValueError("BM25 requires at least one chunk")

        self.chunks = list(chunks)

        tokenized_documents = [
            self._tokenize(chunk.page_content)
            for chunk in self.chunks
        ]

        self._bm25 = BM25Okapi(tokenized_documents)

        logger.info(
            "BM25 index created with %d chunks",
            len(self.chunks),
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_tokens = self._tokenize(query)

        scores = self._bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: list[SearchResult] = []

        for index in ranked_indices:
            chunk = self.chunks[index]

            results.append(
                SearchResult(
                    text=chunk.page_content,
                    metadata=dict(chunk.metadata),
                    bm25_score=float(scores[index]),
                )
            )

        logger.info(
            "BM25 search returned %d result(s) for query: %s",
            len(results),
            query,
        )

        return results