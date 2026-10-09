from langchain_core.documents import Document

from ragforge.app.retrieval.bm25 import BM25Retriever


def _retriever() -> BM25Retriever:
    texts = [
        "The monthly rent is $750 per month, due on the first day.",
        "Tenant pays 70% of utilities including water and electricity.",
        "Landlord may enter with 24 hours notice for inspection.",
        "Pets are not allowed without written consent.",
    ]
    chunks = [
        Document(page_content=text, metadata={"document_id": "d", "chunk_index": index})
        for index, text in enumerate(texts)
    ]
    return BM25Retriever(chunks)


def test_tokenizer_strips_punctuation_and_stopwords() -> None:
    assert BM25Retriever._tokenize("What is the rent?") == ["rent"]
    assert BM25Retriever._tokenize("Landlord’s consent.") == ["landlord", "consent"]


def test_tokenizer_keeps_numbers_and_negations() -> None:
    tokens = BM25Retriever._tokenize("No pets without 60 days notice, rent $750")
    assert {"no", "without", "60", "750", "pets"} <= set(tokens)


def test_punctuation_no_longer_blocks_a_match() -> None:
    results = _retriever().search("What is the rent?", top_k=3)
    assert results[0].text.startswith("The monthly rent")


def test_search_returns_only_positive_scores_without_filler() -> None:
    results = _retriever().search("zzzz qqqq", top_k=3)
    assert results == []

    partial = _retriever().search("pets", top_k=4)
    assert len(partial) == 1
    assert all(result.bm25_score and result.bm25_score > 0 for result in partial)


def test_query_made_only_of_stopwords_returns_no_results() -> None:
    assert _retriever().search("what is the", top_k=3) == []
