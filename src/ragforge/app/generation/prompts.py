from __future__ import annotations

from ragforge.app.retrieval.types import SearchResult


# Production system prompt. General rules only: nothing in it is written for a
# specific eval question. (The earlier prompt had a rule containing the answer
# to eval question q06; it was removed after the 2026-10-10 comparison, whose
# results are kept in data/eval/runs/current-3b.json and clean-3b.json.)
SYSTEM_PROMPT_CLEAN = """
You are RAGForge, an assistant that answers questions about documents.

Use only the numbered sources in the user message. Do not use outside knowledge.

- If the sources do not contain the answer, say: "The information is not
  available in the provided documents."
- If the sources answer only part of the question, answer that part and say
  what is missing.
- Include every relevant detail. If the question asks for all items or
  conditions, check every source before answering.
- If the question needs arithmetic, state the numbers from the sources and
  the result.
- Cite each claim as [Source N], using only source numbers that appear in
  the context.
- Be concise.
""".strip()

# Production default since 2026-10-10. Identical to SYSTEM_PROMPT_CLEAN except
# the two "missing information" lines. With SYSTEM_PROMPT_CLEAN, llama3.2:3b
# appended "The information is not available ... regarding X" to 19 of 28
# answerable questions. With this prompt: 6 of 28, fact recall 0.829 -> 0.858
# (within noise), generation time -34%, but 4 more uncited answers
# (data/eval/runs/clean-3b-verifier-off.json vs clean-v2-3b.json).
SYSTEM_PROMPT_CLEAN_V2 = """
You are RAGForge, an assistant that answers questions about documents.

Use only the numbered sources in the user message. Do not use outside knowledge.

- If the sources do not answer the question at all, reply only: "The
  information is not available in the provided documents."
- If the sources answer the question, give the answer. Do not add sentences
  about what the sources do not mention.
- Include every relevant detail. If the question asks for all items or
  conditions, check every source before answering.
- If the question needs arithmetic, state the numbers from the sources and
  the result.
- Cite each claim as [Source N], using only source numbers that appear in
  the context.
- Be concise.
""".strip()

SYSTEM_PROMPT = SYSTEM_PROMPT_CLEAN_V2

SYSTEM_PROMPTS = {
    "clean": SYSTEM_PROMPT_CLEAN,
    "clean-v2": SYSTEM_PROMPT_CLEAN_V2,
}


def build_context(results: list[SearchResult]) -> str:
    """
    Convert retrieved chunks into a structured context block.
    """

    context_parts = []

    for index, result in enumerate(results, start=1):
        metadata = result.metadata

        page = metadata.get("page_number", "unknown")
        section = metadata.get("section_number", "unknown")

        context_parts.append(
            f"""
[Source {index}]
Page: {page}
Section: {section}

{result.text}
""".strip()
        )

    return "\n\n".join(context_parts)


def build_user_prompt(
    query: str,
    results: list[SearchResult],
) -> str:
    """
    Build the user-facing RAG prompt.
    """

    context = build_context(results)

    return f"""
Answer the following question using only the provided context.

Question:
{query}

Context:
{context}

Citation instructions:
- Cite sources as [Source 1], [Source 2], etc.
- Only cite sources that actually support the claim.
- If the context does not contain the answer, say so explicitly.
""".strip()