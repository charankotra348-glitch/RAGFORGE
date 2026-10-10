"""Deterministic answer-level checks (no LLM judge).

Each answerable eval question carries ``answer_facts``: a list of fact
groups, where each group is a list of acceptable phrasings (any-of). A
generated answer gets credit for a group when any phrase in it appears in
the answer as a whole word or phrase. A question may also carry
``answer_must_not``: phrasings that state the opposite of the correct
answer (for example "does not have to give notice"). If one appears, the
answer is *contradicted* and scores zero fact recall.

Refusals are detected per sentence. A *refusal sentence* contains a
"missing information" pattern (``not available``, ``does not specify``,
``no information``...) AND refers to the material given to the model
(``context``, ``document``, ``source``, ``provided``). Requiring the
second part stops sentences such as "if the tenant does not provide
written notice" from counting as refusals.

Definitions:

* Fact recall        share of fact groups found (0 if contradicted).
* Complete           fact recall == 1.0.
* Leads with refusal the first sentence that is not a preamble ending in
                     ":" is a refusal sentence.
* Correct refusal    unanswerable question AND the answer leads with a
                     refusal. An answer that states a made-up fact first and
                     hedges afterwards is not a correct refusal.
* Contradicted       answerable question whose answer contains an
                     ``answer_must_not`` phrase (a wrong answer, not a refusal).
* False refusal      answerable question, a refusal sentence, zero facts,
                     not contradicted.
* Hedged             answerable question, a refusal sentence, facts > 0:
                     the model answered and also claimed information was
                     missing. Sometimes legitimate (part of the question is
                     really not covered); often boilerplate that contradicts
                     the answer it just gave.
* Citation valid     every [Source N] in the answer is in 1..n_sources.
* Citation support   at least one cited, in-range chunk contains one of the
                     question's evidence phrases (same matcher as the
                     retrieval metrics).

Known limits (documented, not hidden):

* Yes/no polarity is only partly checked. Polarity groups catch missing
  "no"/"cannot", and ``answer_must_not`` catches wrong-polarity phrasings,
  but must-not lists are added after a failure is seen, so new wrong
  phrasings can still pass.
* Refusal patterns are a fixed list; new wordings are missed until added.
* Phrase matching is lexical. "seven days" and "a week" are different
  strings; acceptable variants must be listed in the dataset.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from ragforge.evaluation.metrics import relevance_flags

# "Missing information" patterns, matched on normalized text.
_MISSING_PATTERNS: tuple[str, ...] = (
    r"\bnot (?:explicitly |specifically |directly |clearly )?"
    r"(?:available|stated|mentioned|specified|provided|included|found|addressed|given|defined)\b",
    r"\b(?:does|do|did)(?: not|n't) (?:explicitly |specifically |directly |clearly )?"
    r"(?:specify|mention|contain|provide|state|include|say|address|define|cover|indicate)\b",
    r"\bno (?:information|mention|details?|data)\b",
    r"\bfound no\b",
    r"\b(?:cannot|can't|could not|couldn't) (?:be )?(?:find|found|determine|determined|answer)\b",
    r"\bunable to (?:find|determine|answer)\b",
)
# The sentence must also refer to what the model was given.
_CONTEXT_WORDS = r"\b(?:context|contexts|documents?|sources?|provided|excerpts?|passages?)\b"

_MISSING_RE = re.compile("|".join(_MISSING_PATTERNS))
_CONTEXT_RE = re.compile(_CONTEXT_WORDS)
# "The lease does not specify X" is a refusal on its own: the document type is
# the grammatical subject. ("the tenant does not provide notice" is not.)
_DOCUMENT_SUBJECT_RE = re.compile(
    r"\b(?:the )?(?:lease|agreement|contract|policy|text) (?:itself )?"
    r"(?:does not|doesn't|do not|did not) (?:explicitly |specifically |directly |clearly )?"
    r"(?:specify|mention|contain|provide|state|include|say|address|define|cover|indicate)\b"
)

_DASHES = "‐‑‒–—−-"
_APOSTROPHES = "‘’ʼ`"
_DQUOTES = "“”"


def normalize_answer_text(text: str) -> str:
    """Lowercase, unify quotes and dashes, drop commas, collapse whitespace.

    Commas are dropped so "$1,800" becomes "$1800" (and cannot falsely match
    "800") and "August 1, 2026" matches "August 1 2026". Dashes become
    spaces so "wi-fi" matches "wi fi" and "broom-clean" matches "broom clean".
    """
    text = text.lower()
    for ch in _APOSTROPHES:
        text = text.replace(ch, "'")
    for ch in _DQUOTES:
        text = text.replace(ch, '"')
    for ch in _DASHES:
        text = text.replace(ch, " ")
    text = text.replace(",", "")
    return re.sub(r"\s+", " ", text).strip()


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    normalized = normalize_answer_text(phrase)
    if not normalized:
        raise ValueError(f"empty phrase after normalization: {phrase!r}")
    body = r"\s+".join(re.escape(part) for part in normalized.split(" "))
    # Whole-word boundaries that also work when the phrase starts or ends
    # with a symbol such as "$" or "%" (where \b would fail).
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


def contains_phrase(text: str, phrase: str) -> bool:
    """Whole-word, case-insensitive phrase match on normalized text."""
    return _phrase_pattern(phrase).search(normalize_answer_text(text)) is not None


def matched_groups(text: str, fact_groups: Sequence[Sequence[str]]) -> list[bool]:
    """For each fact group, does the text contain any of its phrases?"""
    return [any(contains_phrase(text, phrase) for phrase in group) for group in fact_groups]


def fact_recall(text: str, fact_groups: Sequence[Sequence[str]]) -> float:
    if not fact_groups:
        raise ValueError("fact recall requires at least one fact group")
    flags = matched_groups(text, fact_groups)
    return sum(flags) / len(flags)


def split_sentences(text: str) -> list[str]:
    """Split on newlines and on . ! ? followed by a capital, digit or bracket.

    Requiring a capital after the period keeps "9:00 A.M. to 8:00 P.M." in
    one sentence.
    """
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[\(\"'*“])|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


_CITATION_TOKEN_RE = re.compile(r"\(?\[?\bsources?\s*\d+[^\])]*[\])]?", re.IGNORECASE)


def is_refusal_sentence(sentence: str) -> bool:
    # Citation tokens ("[Source 1]", "(Source 2, Section 3)") do not count as
    # referring to the context, otherwise "According to [Source 1], if the
    # tenant does not provide notice..." would look like a refusal.
    without_citations = _CITATION_TOKEN_RE.sub(" ", sentence)
    normalized = normalize_answer_text(without_citations)
    if _DOCUMENT_SUBJECT_RE.search(normalized):
        return True
    return bool(_MISSING_RE.search(normalized) and _CONTEXT_RE.search(normalized))


def refusal_sentences(text: str) -> list[str]:
    return [s for s in split_sentences(text) if is_refusal_sentence(s)]


def is_refusal(text: str) -> bool:
    """The answer contains at least one refusal sentence (refusal or hedge)."""
    return bool(refusal_sentences(text))


def leads_with_refusal(text: str) -> bool:
    """The first non-preamble sentence is a refusal sentence."""
    for sentence in split_sentences(text):
        if sentence.rstrip().endswith(":"):
            continue
        return is_refusal_sentence(sentence)
    return False


_CITATION_RE = re.compile(
    r"\bsources?\s*#?\s*(\d+(?:\s*(?:,|;|&|\band\b|\bor\b|-|–|\bto\b)\s*(?:sources?\s*)?\d+)*)",
    re.IGNORECASE,
)


def parse_citations(text: str) -> list[int]:
    """Source numbers cited in the answer, in first-seen order, deduplicated.

    Handles "[Source 3]", "[Source 1, 2]", "Sources 1, 2 and 5",
    "[Sources 2-4]" (range expanded), "(Source 1, Section 6)" and
    "[Source 1][Source 2]".
    """
    cited: list[int] = []
    for match in _CITATION_RE.finditer(text):
        span = match.group(1)
        for part in re.split(r"\s*(?:,|;|&|\band\b|\bor\b)\s*", span, flags=re.IGNORECASE):
            part = re.sub(r"(?i)sources?\s*", "", part).strip()
            if not part:
                continue
            range_match = re.fullmatch(r"(\d+)\s*(?:-|–|to)\s*(\d+)", part, flags=re.IGNORECASE)
            if range_match:
                low, high = int(range_match.group(1)), int(range_match.group(2))
                numbers = range(low, high + 1) if 0 < high - low <= 20 else (low, high)
            else:
                numbers = [int(n) for n in re.findall(r"\d+", part)]
            for number in numbers:
                if number not in cited:
                    cited.append(number)
    return cited


@dataclass
class AnswerScore:
    qid: str
    answerable: bool
    refused: bool
    leads_with_refusal: bool
    citations: list[int]
    citations_valid: bool
    out_of_range_citations: list[int] = field(default_factory=list)
    fact_recall: float | None = None
    complete: bool | None = None
    false_refusal: bool | None = None
    hedged: bool | None = None
    contradicted: bool | None = None
    contradiction: str | None = None
    missing_facts: list[list[str]] = field(default_factory=list)
    citation_supported: bool | None = None
    correct_refusal: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def citation_support(cited: Sequence[int], source_has_evidence: Sequence[bool]) -> bool | None:
    """Does any cited, in-range source contain evidence? None if nothing valid is cited."""
    valid = [n for n in cited if 1 <= n <= len(source_has_evidence)]
    if not valid:
        return None
    return any(source_has_evidence[n - 1] for n in valid)


def score_answer(
    question: dict,
    answer: str,
    source_texts: Sequence[str] | None = None,
    *,
    source_has_evidence: Sequence[bool] | None = None,
) -> AnswerScore:
    """Score one generated answer against its eval question.

    Pass either ``source_texts`` (chunk texts in the order shown to the model,
    so ``source_texts[0]`` is [Source 1]) or ``source_has_evidence`` (one flag
    per source, for re-scoring saved runs that did not store chunk text).
    """
    if source_has_evidence is None:
        if source_texts is None:
            raise ValueError("pass source_texts or source_has_evidence")
        evidence = question.get("evidence") or []
        source_has_evidence = relevance_flags(source_texts, evidence) if evidence else [False] * len(source_texts)

    n_sources = len(source_has_evidence)
    citations = parse_citations(answer)
    out_of_range = [n for n in citations if not 1 <= n <= n_sources]
    refused = is_refusal(answer)

    score = AnswerScore(
        qid=question["id"],
        answerable=bool(question["answerable"]),
        refused=refused,
        leads_with_refusal=leads_with_refusal(answer),
        citations=citations,
        citations_valid=not out_of_range,
        out_of_range_citations=out_of_range,
    )

    if not question["answerable"]:
        score.correct_refusal = score.leads_with_refusal
        return score

    fact_groups = question.get("answer_facts")
    if not fact_groups:
        raise ValueError(f"{question['id']}: answerable question has no answer_facts")

    flags = matched_groups(answer, fact_groups)
    contradiction = next(
        (phrase for phrase in question.get("answer_must_not", []) if contains_phrase(answer, phrase)),
        None,
    )
    score.contradicted = contradiction is not None
    score.contradiction = contradiction
    score.fact_recall = 0.0 if contradiction else sum(flags) / len(flags)
    score.complete = score.fact_recall == 1.0
    score.false_refusal = refused and score.fact_recall == 0.0 and not score.contradicted
    score.hedged = refused and score.fact_recall > 0.0
    score.missing_facts = [list(group) for group, hit in zip(fact_groups, flags) if not hit]

    if question.get("evidence"):
        score.citation_supported = citation_support(citations, source_has_evidence)

    return score
