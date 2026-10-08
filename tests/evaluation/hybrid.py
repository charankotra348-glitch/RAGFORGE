from __future__ import annotations

from ragforge.app.generation.llm import generate_answer
from ragforge.app.retrieval.search import HybridSearcher
from ragforge.app.retrieval.vector import get_qdrant
from ragforge.core.config import Settings
from ragforge.ingestion.chunker import chunk_documents
from ragforge.ingestion.embedder import get_embedding_model
from ragforge.ingestion.loader import ingest_file


QUESTIONS = [
    "What is the monthly rent amount?",
    "When does the lease term start and end?",
    "What is the security deposit amount?",
    "What are all the conditions under which the landlord can enter the premises?",
    "What is the tenant responsible for paying regarding utilities?",
    "If the tenant pays rent on the 6th of the month, what is the total amount owed?",
    "Can the tenant sublet the apartment to a friend for two months while traveling?",
    "What is the landlord's phone number?",
    "What is the penalty for breaking the lease early?",
    "What must the tenant do before the lease expires to avoid automatic termination?",
]


def main() -> None:
    settings = Settings()

    # ---------------------------------------------------------------
    # Load and chunk the same PDF used for indexing.
    # ---------------------------------------------------------------

    documents = ingest_file("data/test.pdf")
    chunks = chunk_documents(documents)

    print(f"Loaded {len(chunks)} chunks for BM25")

    # ---------------------------------------------------------------
    # Initialize Qdrant + embedding model.
    # ---------------------------------------------------------------

    client = get_qdrant(settings)
    embedding_model = get_embedding_model(settings)

    # ---------------------------------------------------------------
    # Initialize hybrid retriever.
    # ---------------------------------------------------------------

    searcher = HybridSearcher(
        client=client,
        settings=settings,
        embedding_model=embedding_model,
        chunks=chunks,
        vector_top_k=10,
        bm25_top_k=10,
    )

    # ---------------------------------------------------------------
    # Run all evaluation questions.
    # ---------------------------------------------------------------

    for index, question in enumerate(QUESTIONS, start=1):

        print("\n" + "=" * 80)
        print(f"q{index:02d}: {question}")
        print("=" * 80)

        results = searcher.search(
            question,
            top_k=5,
        )

        answer = generate_answer(
            settings=settings,
            query=question,
            results=results,
        )

        print("\nANSWER:")
        print(answer)

        print("\nHYBRID SOURCES:")

        for source_index, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"[Source {source_index}] "
                f"rrf={result.rrf_score:.6f} "
                f"vector={result.vector_score} "
                f"bm25={result.bm25_score} "
                f"page={result.metadata.get('page')} "
                f"section={result.metadata.get('section_number')}"
            )

            print(result.text[:700])


if __name__ == "__main__":
    main()