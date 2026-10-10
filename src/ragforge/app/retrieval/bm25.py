from __future__ import annotations

import logging
import re
from typing import Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ragforge.app.retrieval.types import SearchResult

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Common English function words that carry no retrieval signal. Negations
# ("no", "not", "without") and legal modals ("shall", "may") are deliberately
# kept: they change what a clause means.
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "do",
        "does", "for", "from", "has", "have", "how", "if", "in", "into", "is",
        "it", "its", "of", "on", "or", "our", "that", "the", "their", "them",
        "then", "there", "these", "they", "this", "those", "to", "under",
        "was", "were", "what", "when", "where", "which", "who", "whom",
        "why", "will", "with",
    }
)


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
        """Lowercase, split on non-alphanumerics, drop stopwords and 1-letter words.

        "Landlord's rent?" -> ["landlord", "rent"]. Digits are kept, so "$750"
        and "60" remain searchable.
        """
        tokens = _TOKEN_PATTERN.findall(text.lower())
        return [
            token
            for token in tokens
            if token not in STOPWORDS and (len(token) > 1 or token.isdigit())
        ]

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
        if not query_tokens:
            logger.info("BM25 query has no searchable terms after tokenization")
            return []

        scores = self._bm25.get_scores(query_tokens)

        ranked_indices = [
            index
            for index in sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )
            if scores[index] > 0
        ][:top_k]

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
            "BM25 search returned %d result(s)",
            len(results),
        )

        return results