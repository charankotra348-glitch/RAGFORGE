from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from ragforge.core.config import Settings
from ragforge.ingestion.chunker import (
    chunk_documents,
    validate_chunks,
)
from ragforge.ingestion.embedder import embed_chunks
from ragforge.ingestion.loader import ingest_file
from ragforge.app.retrieval.vector import get_collection_count, get_qdrant
from ragforge.app.retrieval.vector_store import (
    count_document_chunks,
    replace_document_chunks,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def generate_embeddings(
    documents,
    settings: Settings,
) -> list[list[float]]:
    """Generate and validate embeddings using the shared FastEmbed path."""
    if not documents:
        raise ValueError("Cannot generate embeddings for an empty document list.")

    logger.info("Generating embeddings for %d chunks...", len(documents))
    embeddings = embed_chunks(settings, documents)
    logger.info("Generated %d embeddings.", len(embeddings))
    return embeddings


# ---------------------------------------------------------------------------
# Re-indexing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndexSummary:
    document_id: str
    source: str
    chunks: int
    pages: int
    corpus_version: str | None


def reindex(
    file_path: str | Path,
    settings: Settings,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> IndexSummary:
    """
    Re-index one document in the shared Qdrant collection without deleting other documents.

    Pipeline:

        file
          ↓
        ingestion
          ↓
        cleaned chunks
          ↓
        embeddings
          ↓
        replace this document's vectors safely
          ↓
        verify count
    """

    file_path = Path(file_path)

    logger.info("=" * 80)
    logger.info("STARTING QDRANT RE-INDEX")
    logger.info("=" * 80)

    # -----------------------------------------------------------------------
    # 1. Ingest document
    # -----------------------------------------------------------------------

    logger.info(
        "Step 1/6: Loading document: %s",
        file_path,
    )

    documents = ingest_file(file_path)

    if not documents:
        raise RuntimeError(
            "Ingestion produced zero documents."
        )

    logger.info(
        "Loaded %d document(s).",
        len(documents),
    )

    # -----------------------------------------------------------------------
    # 2. Create cleaned chunks
    # -----------------------------------------------------------------------

    logger.info(
        "Step 2/6: Creating cleaned chunks...",
    )

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise RuntimeError(
            "Chunking produced zero chunks."
        )

    # Validate the exact chunks that will be indexed.
    validate_chunks(
        chunks,
        chunk_size=chunk_size,
    )

    logger.info(
        "Created %d cleaned chunk(s).",
        len(chunks),
    )

    # -----------------------------------------------------------------------
    # 3. Generate embeddings
    # -----------------------------------------------------------------------

    logger.info(
        "Step 3/6: Generating embeddings...",
    )

    embeddings = generate_embeddings(
        chunks,
        settings,
    )

    if not embeddings:
        raise RuntimeError(
            "Embedding generation produced zero vectors."
        )

    # Verify embedding dimensions.
    vector_dimension = len(embeddings[0])

    if vector_dimension != settings.vector_size:
        raise RuntimeError(
            "Unexpected embedding dimension: "
            f"{vector_dimension}. Expected {settings.vector_size}."
        )

    logger.info(
        "Embedding dimension: %d",
        vector_dimension,
    )

    # -----------------------------------------------------------------------
    # 4. Connect to Qdrant
    # -----------------------------------------------------------------------

    logger.info(
        "Step 4/6: Connecting to Qdrant...",
    )

    client = get_qdrant(settings)

    # -----------------------------------------------------------------------
    # 5. Replace this document's chunks without deleting other documents.
    # -----------------------------------------------------------------------

    logger.info(
        "Step 5/6: Safely replacing %d chunks for this document...",
        len(chunks),
    )

    upserted_count = replace_document_chunks(
        client=client,
        settings=settings,
        chunks=chunks,
        embeddings=embeddings,
    )
    if upserted_count != len(chunks):
        raise RuntimeError(
            f"Qdrant accepted {upserted_count} chunks; expected {len(chunks)}."
        )

    # Qdrant has changed; invalidate the in-process lexical index immediately.
    from ragforge.app.generation.service import clear_hybrid_searcher_cache

    clear_hybrid_searcher_cache()

    # Tell running API workers (other processes) that the corpus changed:
    # they rebuild their BM25 index and stop serving cached answers.
    from ragforge.app.cache.answers import bump_corpus_version
    from ragforge.app.cache.redis import get_redis

    new_version = bump_corpus_version(get_redis(settings))
    if new_version is None:
        logger.warning("Could not update the corpus version; restart the API to pick up this change.")
    else:
        logger.info("Corpus version is now %s", new_version)

    # -----------------------------------------------------------------------
    # 6. Verify
    # -----------------------------------------------------------------------

    logger.info(
        "Step 6/6: Verifying Qdrant index...",
    )

    collection_count = get_collection_count(
        client,
        settings,
    )

    logger.info(
        "Expected Qdrant points: %d",
        len(chunks),
    )

    logger.info(
        "Actual Qdrant points across all documents: %d",
        collection_count,
    )

    document_id = chunks[0].metadata["document_id"]
    document_count = count_document_chunks(
        client,
        settings,
        document_id,
    )
    logger.info("Verified chunks for this document: %d", document_count)
    if document_count != len(chunks):
        raise RuntimeError(
            "Qdrant verification failed for document "
            f"{document_id!r}: expected {len(chunks)} chunks, found {document_count}."
        )

    logger.info("=" * 80)
    logger.info("QDRANT RE-INDEX SUCCESSFUL")
    logger.info("=" * 80)

    logger.info(
        "Source: %s",
        file_path,
    )

    logger.info(
        "Clean chunks: %d",
        len(chunks),
    )

    logger.info(
        "Embeddings: %d",
        len(embeddings),
    )

    logger.info(
        "Qdrant points: %d",
        collection_count,
    )

    logger.info(
        "Collection: %s",
        settings.qdrant_collection,
    )

    logger.info("=" * 80)

    return IndexSummary(
        document_id=document_id,
        source=str(chunks[0].metadata.get("source", file_path)),
        chunks=len(chunks),
        pages=max(
            int(chunk.metadata.get("page_end_number") or chunk.metadata.get("page_number") or 1)
            for chunk in chunks
        ),
        corpus_version=new_version,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Command-line entry point.
    """

    parser = argparse.ArgumentParser(
        description="Re-index a document into Qdrant."
    )

    parser.add_argument(
        "file",
        type=str,
        help="Path to the document to index.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Chunk size used during indexing "
            f"(default: {DEFAULT_CHUNK_SIZE})."
        ),
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=(
            "Chunk overlap used during indexing "
            f"(default: {DEFAULT_CHUNK_OVERLAP})."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    settings = Settings()

    reindex(
        file_path=args.file,
        settings=settings,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()