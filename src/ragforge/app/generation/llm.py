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
    system_prompt: str | None = None,
    num_ctx: int | None = None,
) -> str:
    """Generate a grounded answer using the configured local Ollama model.

    ``system_prompt`` overrides the default SYSTEM_PROMPT. ``num_ctx`` sets
    Ollama's context window; when None, Ollama's own default is used.
    """

    answer, _ = generate_answer_with_stats(
        settings,
        query,
        results,
        model=model,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        num_ctx=num_ctx,
    )
    return answer


def generate_answer_with_stats(
    settings: Settings,
    query: str,
    results: list[SearchResult],
    model: str | None = None,
    max_tokens: int = 1000,
    system_prompt: str | None = None,
    num_ctx: int | None = None,
) -> tuple[str, dict[str, int | None]]:
    """Same as generate_answer, plus Ollama token counts for evaluation.

    Stats: prompt_eval_count (prompt tokens Ollama processed; can undercount
    when Ollama reuses a cached prompt prefix) and eval_count (output tokens).
    """

    if not results:
        return (
            "The information is not available in the provided documents.",
            {"prompt_eval_count": None, "eval_count": None},
        )
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")
    if num_ctx is not None and num_ctx <= 0:
        raise ValueError("num_ctx must be greater than 0")

    options: dict[str, int | float] = {"temperature": 0, "num_predict": max_tokens}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

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
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options=options,
    )

    answer = response.message.content
    if not answer or not answer.strip():
        raise RuntimeError("Ollama returned an empty response")

    stats = {
        "prompt_eval_count": getattr(response, "prompt_eval_count", None),
        "eval_count": getattr(response, "eval_count", None),
    }
    logger.info(
        "Generated answer (%d chars, prompt_tokens=%s, output_tokens=%s)",
        len(answer),
        stats["prompt_eval_count"],
        stats["eval_count"],
    )
    return answer, stats
