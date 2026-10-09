from __future__ import annotations

import logging

from ollama import Client

from ragforge.app.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from ragforge.app.retrieval.types import SearchResult
from ragforge.core.config import Settings

logger = logging.getLogger(__name__)

_client: Client | None = None
_client_host: str | None = None


def get_llm_client(settings: Settings) -> Client:
    """Create and reuse an Ollama client for the configured local host."""

    global _client, _client_host

    if _client is None or _client_host != settings.ollama_host:
        _client = Client(host=settings.ollama_host)
        _client_host = settings.ollama_host
        logger.info("Ollama client created for %s", settings.ollama_host)

    return _client


def generate_answer(
    settings: Settings,
    query: str,
    results: list[SearchResult],
    model: str | None = None,
    max_tokens: int = 1000,
) -> str:
    """Generate a grounded answer using the configured local Ollama model."""

    if not results:
        return "The information is not available in the provided documents."
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    selected_model = model or settings.ollama_model
    client = get_llm_client(settings)
    user_prompt = build_user_prompt(query, results)

    logger.info(
        "Generating answer with Ollama model '%s' for query: %s",
        selected_model,
        query,
    )

    response = client.chat(
        model=selected_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0, "num_predict": max_tokens},
    )

    answer = response.message.content
    if not answer or not answer.strip():
        raise RuntimeError("Ollama returned an empty response")

    logger.info("Generated answer (%d chars)", len(answer))
    return answer
