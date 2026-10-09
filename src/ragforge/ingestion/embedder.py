from __future__ import annotations

import logging
from typing import Sequence

from fastembed import TextEmbedding
from langchain_core.documents import Document

from ragforge.core.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_model: TextEmbedding | None = None


def get_embedding_model(
    settings: Settings,
) -> TextEmbedding:
    """
    Create the FastEmbed model once and reuse it.
    """

    global _model

    if _model is None:
        logger.info(
            "Loading embedding model: %s",
            settings.embedding_model,
        )

        _model = TextEmbedding(
            model_name=settings.embedding_model
        )

        logger.info(
            "Embedding model loaded"
        )

    return _model


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_chunks(
    settings: Settings,
    chunks: Sequence[Document],
) -> list[list[float]]:
    """
    Generate one embedding vector for every chunk.

    The order of the returned embeddings matches
    the order of the input chunks.
    """

    if not chunks:
        return []

    model = get_embedding_model(settings)

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    logger.info(
        "Embedding %d chunk(s)",
        len(texts),
    )

    embeddings = list(
        model.embed(texts)
    )

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: "
            f"{len(embeddings)} embeddings "
            f"for {len(chunks)} chunks"
        )

    result = [
        list(map(float, vector))
        for vector in embeddings
    ]

    dimensions = {len(vector) for vector in result}
    if dimensions != {settings.vector_size}:
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"got {sorted(dimensions)}, expected {settings.vector_size} "
            f"for model {settings.embedding_model!r}"
        )

    logger.info(
        "Finished embedding %d chunk(s)",
        len(result),
    )

    return result
