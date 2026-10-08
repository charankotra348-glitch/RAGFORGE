from __future__ import annotations

import logging

from ollama import Client

from ragforge.app.generation.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from ragforge.app.retrieval.search import SearchResult
from ragforge.core.config import Settings

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_llm_client(settings: Settings) -> Client:
    """
    Create and return the shared Ollama client.
    """

    global _client

    if _client is None:
        _client = Client(
            host="http://localhost:11434"
        )

        logger.info("Ollama client created")

    return _client


def generate_answer(
    settings: Settings,
    query: str,
    results: list[SearchResult],
    model: str = "llama3.2:3b",
    max_tokens: int = 1000,
) -> str:
    """
    Generate a grounded answer using a local Ollama model.
    """

    if not results:
        return (
            "The information is not available "
            "in the provided documents."
        )

    client = get_llm_client(settings)

    user_prompt = build_user_prompt(
        query,
        results,
    )

    logger.info(
        "Generating answer with Ollama model '%s' for query: %s",
        model,
        query,
    )

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": 0,
            "num_predict": max_tokens,
        },
    )

    answer = response["message"]["content"]

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty response"
        )

    logger.info(
        "Generated answer (%d chars)",
        len(answer),
    )

    return answer