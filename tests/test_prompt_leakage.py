"""Guard against eval-set text leaking into production prompts.

Twice already, text from the eval set ended up in a prompt (the q06 answer in
the generation prompt; the q35 clause and two refusal answers in the
grounding verifier). A rule written for one eval question makes that
question's score meaningless. These tests fail if it happens again.
"""

import json
import re
from pathlib import Path

import pytest

from ragforge.app.generation import grounding
from ragforge.app.generation.prompts import SYSTEM_PROMPTS

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = [
    q
    for path in sorted((ROOT / "data" / "eval").glob("*_eval.json"))
    for q in json.loads(path.read_text(encoding="utf-8"))["questions"]
]

PROMPT_TEXTS = {
    **{f"system prompt {name!r}": text for name, text in SYSTEM_PROMPTS.items()},
    "grounding verifier prompt": grounding.VERIFIER_SYSTEM_PROMPT,
    "grounding abstention prefixes": "\n".join(grounding._ABSTENTION_PREFIXES),
}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def eval_snippets() -> list[tuple[str, str]]:
    """(label, text) for every question, evidence phrase and expected answer."""
    snippets = []
    for q in QUESTIONS:
        snippets.append((f"{q['id']} question", q["question"]))
        snippets.extend((f"{q['id']} evidence", e) for e in q.get("evidence", []))
        if q["answerable"]:
            snippets.append((f"{q['id']} expected answer", q["expected"]))
    return snippets


@pytest.mark.parametrize("prompt_name", sorted(PROMPT_TEXTS))
def test_prompt_contains_no_eval_text(prompt_name: str) -> None:
    prompt = normalize(PROMPT_TEXTS[prompt_name])
    leaks = [
        f"{label}: {text!r}"
        for label, text in eval_snippets()
        if len(normalize(text)) >= 12 and normalize(text) in prompt
    ]
    assert not leaks, f"{prompt_name} contains eval-set text: {leaks}"


def test_guard_catches_the_old_q35_counterexample() -> None:
    old = "Such entry shall occur only during reasonable hours, defined as 9:00 A.M. to 8:00 P.M."
    q04 = next(q for q in QUESTIONS if q["id"] == "q04")
    assert any(normalize(e) in normalize(old) for e in q04["evidence"])
