from __future__ import annotations

import logging

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from ragforge.app.retrieval.types import SearchResult
from ragforge.core.config import Settings
from ragforge.app.retrieval.vector_store import upsert_chunks

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None



# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_qdrant(settings: Settings) -> QdrantClient:
    """
    Create and return the shared Qdrant client.
    """

    global _client

    if _client is None:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        logger.info("Qdrant client created")

        # Cache the client only after the collection check succeeds. Caching it
        # first meant that if Qdrant was down at the first call, every later
        # call got the unchecked client back and the collection was never
        # created or validated.
        try:
            _ensure_collection(client, settings)
        except Exception:
            client.close()
            raise

        _client = client

    return _client


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def _ensure_collection(
    client: QdrantClient,
    settings: Settings,
) -> None:
    """
    Create the configured Qdrant collection if it does not exist.
    """

    collections = client.get_collections().collections
    existing_names = {collection.name for collection in collections}

    if settings.qdrant_collection not in existing_names:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.vector_size,
                distance=Distance.COSINE,
            ),
        )

        logger.info(
            "Created Qdrant collection: %s",
            settings.qdrant_collection,
        )

    else:
        collection_info = client.get_collection(settings.qdrant_collection)
        vector_config = collection_info.config.params.vectors
        configured_size = getattr(vector_config, "size", None)
        if configured_size != settings.vector_size:
            raise RuntimeError(
                f"Qdrant collection {settings.qdrant_collection!r} uses vector size "
                f"{configured_size}, but configuration requires {settings.vector_size}. "
                "Use a new collection or deliberately reindex; do not mix embedding dimensions."
            )
        logger.info("Qdrant collection already exists: %s", settings.qdrant_collection)


def recreate_collection(
    client: QdrantClient,
    settings: Settings,
    *,
    confirm_delete_all_documents: bool = False,
) -> None:
    """Destructively recreate the entire collection only after explicit confirmation."""

    if not confirm_delete_all_documents:
        raise ValueError(
            "Recreating the collection deletes every indexed document. "
            "Pass confirm_delete_all_documents=True only when a full rebuild is intended."
        )

    if client.collection_exists(settings.qdrant_collection):
        logger.info(
            "Deleting existing Qdrant collection: %s",
            settings.qdrant_collection,
        )

        client.delete_collection(
            collection_name=settings.qdrant_collection,
        )

    _ensure_collection(client, settings)

    logger.info(
        "Recreated Qdrant collection: %s",
        settings.qdrant_collection,
    )


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def upsert_documents(
    client: QdrantClient,
    settings: Settings,
    documents: list[Document],
    embeddings: list[list[float]],
) -> None:
    """Compatibility wrapper that stores chunks using stable deterministic IDs."""
    upsert_chunks(client, settings, documents, embeddings)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def get_collection_count(
    client: QdrantClient,
    settings: Settings,
) -> int:
    """
    Return the number of points currently stored in Qdrant.
    """

    info = client.get_collection(
        collection_name=settings.qdrant_collection,
    )

    return info.points_count or 0
# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def search_vectors(
    client: QdrantClient,
    settings: Settings,
    query_vector: list[float],
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Retrieve the most relevant chunks from Qdrant.
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if len(query_vector) != settings.vector_size:
        raise ValueError(
            f"Query vector dimension {len(query_vector)} does not match "
            f"configured vector_size {settings.vector_size}"
        )

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    results: list[SearchResult] = []

    for point in response.points:
        payload = point.payload or {}

        results.append(
            SearchResult(
                text=payload.get("text", ""),
                metadata=payload.get("metadata", {}),
                vector_score=point.score,
            )
        )

    return results