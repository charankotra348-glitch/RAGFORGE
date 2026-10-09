from ragforge.app.retrieval.fusion import reciprocal_rank_fusion
from ragforge.app.retrieval.types import SearchResult


def test_rrf_merges_scores_for_same_chunk_from_both_retrievers() -> None:
    vector = [
        SearchResult(
            text="rent is due on the first day",
            metadata={"document_id": "lease-a", "chunk_index": 0},
            vector_score=0.9,
        )
    ]
    bm25 = [
        SearchResult(
            text="rent is due on the first day",
            metadata={"document_id": "lease-a", "chunk_index": 0},
            bm25_score=4.2,
        )
    ]

    results = reciprocal_rank_fusion(vector, bm25, top_k=5)

    assert len(results) == 1
    assert results[0].vector_score == 0.9
    assert results[0].bm25_score == 4.2
    assert results[0].rrf_score > 0


def test_rrf_does_not_merge_identical_text_from_different_documents() -> None:
    vector = [
        SearchResult("same text", {"document_id": "doc-a", "chunk_index": 0}, vector_score=0.8),
    ]
    bm25 = [
        SearchResult("same text", {"document_id": "doc-b", "chunk_index": 0}, bm25_score=3.0),
    ]

    results = reciprocal_rank_fusion(vector, bm25, top_k=5)

    assert len(results) == 2
