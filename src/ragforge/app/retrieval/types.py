from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    text: str
    metadata: dict[str, Any]
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None