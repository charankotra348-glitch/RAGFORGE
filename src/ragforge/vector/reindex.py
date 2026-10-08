from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ragforge.core.config import Settings
from ragforge.ingestion.chunker import (
    chunk_documents,
    validate_chunks,
)
from ragforge.ingestion.embedder import get_embedding_model
from ragforge.ingestion.loader import ingest_file
from ragforge.app.retrieval.vector import (
    get_collection_count,
    get_qdrant,
    recreate_collection,
    upsert_documents,
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
    """
    Generate embeddings for all documents using the configured
    FastEmbed model.
    """

    if not documents:
        raise ValueError(
            "Cannot generate embeddings for an empty document list."
        )

    model = get_embedding_model(settings)

    texts = [
        document.page_content
        for document in documents
    ]

    logger.info(
        "Generating embeddings for %d chunks...",
        len(texts),
    )

    embeddings = [
        list(map(float, embedding))
        for embedding in model.embed(texts)
    ]

    if len(embeddings) != len(documents):
        raise RuntimeError(
            "Embedding count does not match chunk count: "
            f"{len(embeddings)} != {len(documents)}"
        )

    logger.info(
        "Generated %d embeddings.",
        len(embeddings),
    )

    return embeddings


# ---------------------------------------------------------------------------
# Re-indexing
# ---------------------------------------------------------------------------

def reindex(
    file_path: str | Path,
    settings: Settings,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    """
    Rebuild the complete Qdrant index for a document.

    Pipeline:

        file
          ↓
        ingestion
          ↓
        cleaned chunks
          ↓
        embeddings
          ↓
        recreate Qdrant collection
          ↓
        upsert vectors
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

    if vector_dimension != 384:
        raise RuntimeError(
            "Unexpected embedding dimension: "
            f"{vector_dimension}. Expected 384."
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
    # 5. Recreate collection and index
    # -----------------------------------------------------------------------

    logger.info(
        "Step 5/6: Recreating Qdrant collection...",
    )

    recreate_collection(
        client,
        settings,
    )

    logger.info(
        "Indexing %d chunks into Qdrant...",
        len(chunks),
    )

    upsert_documents(
        client=client,
        settings=settings,
        documents=chunks,
        embeddings=embeddings,
    )

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
        "Actual Qdrant points: %d",
        collection_count,
    )

    if collection_count != len(chunks):
        raise RuntimeError(
            "Qdrant verification failed: "
            f"expected {len(chunks)} points, "
            f"found {collection_count}."
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