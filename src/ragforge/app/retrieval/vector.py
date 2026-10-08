from __future__ import annotations

import logging
from uuid import uuid4

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ragforge.app.retrieval.types import SearchResult
from ragforge.core.config import Settings

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None

# FastEmbed model currently produces 384-dimensional vectors.
VECTOR_SIZE = 384


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_qdrant(settings: Settings) -> QdrantClient:
    """
    Create and return the shared Qdrant client.
    """

    global _client

    if _client is None:
        _client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        logger.info("Qdrant client created")

        _ensure_collection(_client, settings)

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
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        logger.info(
            "Created Qdrant collection: %s",
            settings.qdrant_collection,
        )

    else:
        logger.info(
            "Qdrant collection already exists: %s",
            settings.qdrant_collection,
        )


def recreate_collection(
    client: QdrantClient,
    settings: Settings,
) -> None:
    """
    Delete the existing Qdrant collection and recreate it.

    This is used when chunking or embedding logic changes and
    the entire document index needs to be rebuilt.
    """

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
    """
    Insert document embeddings and metadata into Qdrant.
    """

    if len(documents) != len(embeddings):
        raise ValueError(
            f"Document count ({len(documents)}) does not match "
            f"embedding count ({len(embeddings)})."
        )

    if not documents:
        logger.warning("No documents to index.")
        return

    points: list[PointStruct] = []

    for document, embedding in zip(documents, embeddings):
        point_id = str(uuid4())

        payload = {
            "text": document.page_content,
            "metadata": document.metadata,
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        )

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )

    logger.info(
        "Upserted %d documents into Qdrant",
        len(points),
    )


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