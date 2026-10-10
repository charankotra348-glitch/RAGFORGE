from __future__ import annotations

import logging
from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
)

from ragforge.core.config import Settings

logger = logging.getLogger(__name__)


def _point_id(document_id: str, chunk_index: int) -> str:
    """Return a stable, valid UUID for a document chunk."""
    return str(uuid5(NAMESPACE_URL, f"ragforge:{document_id}:{chunk_index}"))


def _validate_chunks(chunks: Sequence[Document]) -> str:
    if not chunks:
        raise ValueError("Cannot replace a document with zero chunks")

    document_ids = {chunk.metadata.get("document_id") for chunk in chunks}
    if len(document_ids) != 1:
        raise ValueError("All chunks in one indexing operation must share document_id")

    document_id = next(iter(document_ids))
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("Every chunk must contain a non-empty document_id")

    indices: set[int] = set()
    for chunk in chunks:
        chunk_index = chunk.metadata.get("chunk_index")
        if not isinstance(chunk_index, int) or chunk_index < 0:
            raise ValueError("Every chunk must contain a non-negative integer chunk_index")
        if chunk_index in indices:
            raise ValueError(f"Duplicate chunk_index detected: {chunk_index}")
        indices.add(chunk_index)

    return document_id


def upsert_chunks(
    client: QdrantClient,
    settings: Settings,
    chunks: Sequence[Document],
    embeddings: Sequence[Sequence[float]],
) -> int:
    """Upsert chunks with stable IDs, making repeated indexing idempotent."""
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks must match number of embeddings")
    if not chunks:
        return 0

    points: list[PointStruct] = []
    for chunk, embedding in zip(chunks, embeddings):
        document_id = chunk.metadata.get("document_id")
        chunk_index = chunk.metadata.get("chunk_index")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("Every chunk must contain a non-empty document_id")
        if not isinstance(chunk_index, int) or chunk_index < 0:
            raise ValueError("Every chunk must contain a non-negative integer chunk_index")
        if len(embedding) != settings.vector_size:
            raise ValueError(
                f"Embedding dimension {len(embedding)} does not match "
                f"configured vector_size {settings.vector_size}"
            )

        points.append(
            PointStruct(
                id=_point_id(document_id, chunk_index),
                vector=list(map(float, embedding)),
                payload={"text": chunk.page_content, "metadata": dict(chunk.metadata)},
            )
        )

    client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)
    logger.info(
        "Upserted %d chunks into Qdrant collection '%s'",
        len(points),
        settings.qdrant_collection,
    )
    return len(points)


def replace_document_chunks(
    client: QdrantClient,
    settings: Settings,
    chunks: Sequence[Document],
    embeddings: Sequence[Sequence[float]],
) -> int:
    """Safely replace one document's indexed chunks without deleting other documents.

    New/current chunk IDs are upserted first. Only after that succeeds are
    obsolete point IDs for this document removed.
    """
    if not chunks:
        raise ValueError("Cannot replace a document with zero chunks")
    document_id = _validate_chunks(chunks)
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks must match number of embeddings")

    old_ids: list[str | int] = []
    offset = None
    document_filter = Filter(
        must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))]
    )
    while True:
        records, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=document_filter,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        old_ids.extend(record.id for record in records)
        if offset is None:
            break

    current_ids = {
        _point_id(document_id, int(chunk.metadata["chunk_index"]))
        for chunk in chunks
    }

    # Upsert first: if embedding/indexing fails, old points remain available.
    upserted = upsert_chunks(client, settings, chunks, embeddings)
    stale_ids = [point_id for point_id in old_ids if str(point_id) not in current_ids]
    if stale_ids:
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=PointIdsList(points=stale_ids),
            wait=True,
        )
        logger.info("Removed %d stale chunks for document %s", len(stale_ids), document_id)

    _delete_superseded_copies(client, settings, chunks, document_id)

    return upserted


def _delete_superseded_copies(
    client: QdrantClient,
    settings: Settings,
    chunks: Sequence[Document],
    document_id: str,
) -> None:
    """Delete chunks indexed from the same source under a different document_id.

    The same file indexed again (after an edit, or after the document_id
    scheme changed) must replace the old copy, not sit next to it. Runs only
    after the new chunks were upserted, so a failure never leaves the source
    with no chunks.
    """
    sources = {chunk.metadata.get("source") for chunk in chunks}
    if len(sources) != 1:
        return
    source = next(iter(sources))
    if not isinstance(source, str) or not source:
        return

    superseded = Filter(
        must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))],
        must_not=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))],
    )
    removed = client.count(
        collection_name=settings.qdrant_collection,
        count_filter=superseded,
        exact=True,
    ).count
    if removed:
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=superseded,
            wait=True,
        )
        logger.warning(
            "Removed %d chunks of an older copy of %s (different document_id)",
            removed,
            source,
        )


def delete_document(
    client: QdrantClient,
    settings: Settings,
    document_id: str,
) -> None:
    """Delete all chunks belonging to a document from Qdrant."""
    if not document_id:
        raise ValueError("document_id cannot be empty")

    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))]
        ),
        wait=True,
    )
    logger.info("Deleted chunks for document '%s'", document_id)


def count_document_chunks(
    client: QdrantClient,
    settings: Settings,
    document_id: str,
) -> int:
    """Count indexed chunks for one document without counting other documents."""
    if not document_id:
        raise ValueError("document_id cannot be empty")

    count = 0
    offset = None
    document_filter = Filter(
        must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))]
    )
    while True:
        records, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=document_filter,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        count += len(records)
        if offset is None:
            return count


def get_indexed_chunks(
    client: QdrantClient,
    settings: Settings,
) -> list[Document]:
    """Load the actual indexed chunk corpus for BM25, matching Qdrant exactly."""
    chunks: list[Document] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            text = payload.get("text")
            metadata = payload.get("metadata")
            if isinstance(text, str) and text.strip() and isinstance(metadata, dict):
                chunks.append(Document(page_content=text, metadata=dict(metadata)))
        if offset is None:
            break

    return chunks
