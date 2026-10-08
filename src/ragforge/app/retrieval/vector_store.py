from __future__ import annotations

import hashlib
import logging
from typing import Sequence

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)

from ragforge.core.config import Settings


logger = logging.getLogger(__name__)


def _point_id(
    document_id: str,
    chunk_index: int,
) -> str:
    """
    Generate a deterministic Qdrant point ID.

    The combination of document_id + chunk_index uniquely identifies
    a chunk within the indexed document.
    """

    return hashlib.sha256(
        f"{document_id}:{chunk_index}".encode("utf-8")
    ).hexdigest()[:32]


def upsert_chunks(
    client: QdrantClient,
    settings: Settings,
    chunks: Sequence[Document],
    embeddings: Sequence[Sequence[float]],
) -> int:
    """
    Store document chunks and their embeddings in Qdrant.
    """

    if not chunks:
        return 0

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks must match number of embeddings"
        )

    points: list[PointStruct] = []

    for chunk, embedding in zip(chunks, embeddings):
        chunk_index = chunk.metadata.get(
            "chunk_index"
        )

        document_id = chunk.metadata.get(
            "document_id"
        )

        if not isinstance(chunk_index, int):
            raise ValueError(
                "Every chunk must contain an integer chunk_index"
            )

        if (
            not isinstance(document_id, str)
            or not document_id
        ):
            raise ValueError(
                "Every chunk must contain a non-empty document_id"
            )

        payload = {
            "text": chunk.page_content,
            "metadata": dict(chunk.metadata),
        }

        points.append(
            PointStruct(
                id=_point_id(
                    document_id,
                    chunk_index,
                ),
                vector=list(embedding),
                payload=payload,
            )
        )

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )

    logger.info(
        "Upserted %d chunks into Qdrant collection '%s'",
        len(points),
        settings.qdrant_collection,
    )

    return len(points)


def delete_document(
    client: QdrantClient,
    settings: Settings,
    document_id: str,
) -> None:
    """
    Delete all chunks belonging to a document from Qdrant.
    """

    if not document_id:
        raise ValueError(
            "document_id cannot be empty"
        )

    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(
                        value=document_id
                    ),
                )
            ]
        ),
    )

    logger.info(
        "Deleted existing chunks for document '%s'",
        document_id,
    )