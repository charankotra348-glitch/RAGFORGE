"""Retrieval metrics for chunk-level evaluation.

A chunk counts as *relevant* when its text contains at least one of the
question's evidence snippets. Matching ignores case, whitespace and
punctuation, so it survives PDF line breaks and re-chunking.

Definitions (all use the ranked list of retrieved chunk texts):

* Hit@K       1 if any relevant chunk is in the top K, else 0.
* Coverage@K  fraction of the question's evidence snippets found in the
              union of the top K chunks. For multi-fact questions this is
              the real "did we retrieve everything needed" recall.
* MRR         1 / rank of the first relevant chunk (0 if none retrieved).
* nDCG@K      binary-relevance nDCG; the ideal ranking places
              min(K, relevant chunks in corpus) relevant chunks first.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence


def normalize(text: str) -> str:
    """Lowercase and keep only letters and digits."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def relevance_flags(texts: Sequence[str], evidence: Sequence[str]) -> list[bool]:
    """For each ranked chunk text, is it relevant to this question?"""
    needles = [normalize(snippet) for snippet in evidence if normalize(snippet)]
    flags: list[bool] = []
    for text in texts:
        haystack = normalize(text)
        flags.append(any(needle in haystack for needle in needles))
    return flags


def hit_at_k(flags: Sequence[bool], k: int) -> float:
    _check_k(k)
    return 1.0 if any(flags[:k]) else 0.0


def mrr(flags: Sequence[bool]) -> float:
    for rank, is_relevant in enumerate(flags, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(flags: Sequence[bool], k: int, total_relevant: int) -> float:
    _check_k(k)
    if total_relevant <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(position + 2)
        for position, is_relevant in enumerate(flags[:k])
        if is_relevant
    )
    ideal_count = min(k, total_relevant)
    idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_count))
    return dcg / idcg


def missing_evidence(
    texts: Sequence[str], evidence: Sequence[str], k: int
) -> list[str]:
    """Evidence snippets NOT found in any of the top-K chunk texts."""
    _check_k(k)
    haystacks = [normalize(text) for text in texts[:k]]
    return [
        snippet
        for snippet in evidence
        if not any(normalize(snippet) in haystack for haystack in haystacks)
    ]


def coverage_at_k(texts: Sequence[str], evidence: Sequence[str], k: int) -> float:
    if not evidence:
        raise ValueError("coverage requires at least one evidence snippet")
    missing = missing_evidence(texts, evidence, k)
    return (len(evidence) - len(missing)) / len(evidence)


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than 0")
