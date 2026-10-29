import json
from pathlib import Path

import pytest

from ragforge.evaluation.answer_checks import (
    citation_support,
    contains_phrase,
    fact_recall,
    is_refusal,
    leads_with_refusal,
    matched_groups,
    parse_citations,
    score_answer,
    split_sentences,
)

ROOT = Path(__file__).resolve().parents[1]
DATASETS = sorted((ROOT / "data" / "eval").glob("*_eval.json"))


def load_questions() -> list[dict]:
    """Questions from every eval set (lease, remote work policy); ids are unique across sets."""
    return [q for path in DATASETS for q in json.loads(path.read_text(encoding="utf-8"))["questions"]]


def test_question_ids_are_unique_across_datasets() -> None:
    ids = [q["id"] for q in load_questions()]
    assert len(DATASETS) >= 2
    assert len(ids) == len(set(ids))


# --- phrase matching -------------------------------------------------------


def test_contains_phrase_is_whole_word() -> None:
    assert contains_phrase("The rent is $750 per month.", "750")
    assert not contains_phrase("The rent is $7500 per month.", "750")
    assert not contains_phrase("total of $1,800", "800")
    assert contains_phrase("pay 70% of bills", "70%")
    assert not contains_phrase("inspection", "inspect")


def test_contains_phrase_ignores_case_dashes_commas_and_whitespace() -> None:
    assert contains_phrase("Wi-Fi and SNOW\nremoval", "wifi") is False
    assert contains_phrase("Wi-Fi and SNOW\nremoval", "wi-fi")
    assert contains_phrase("Wi-Fi and SNOW\nremoval", "snow removal")
    assert contains_phrase("starts August 1 2026", "august 1, 2026")
    assert contains_phrase("in “broom–clean” condition", "broom clean")


def test_contains_phrase_handles_curly_apostrophes() -> None:
    assert contains_phrase("the tenant’s responsibility", "tenant's responsibility")


def test_fact_recall_counts_groups_not_phrases() -> None:
    groups = [["70%", "70 percent"], ["electricity"], ["snow removal"]]
    answer = "Tenant pays 70 percent of utilities including electricity [Source 1]."
    assert matched_groups(answer, groups) == [True, True, False]
    assert fact_recall(answer, groups) == pytest.approx(2 / 3)


def test_fact_recall_requires_groups() -> None:
    with pytest.raises(ValueError):
        fact_recall("anything", [])


# --- refusals --------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The information is not available in the provided documents.",
        "The lease does not specify quiet hours.",
        "The context doesn\u2019t mention a pet deposit.",
        "This cannot be determined from the sources.",
        "The pet deposit amount is not explicitly stated in the provided context.",
        "Based on the provided sources, I found no information about a washer.",
    ],
)
def test_is_refusal_detects_known_phrasings(text: str) -> None:
    assert is_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "The monthly rent is $750 [Source 1].",
        # Real false positive from the first run: the clause itself says "does not provide".
        "According to [Source 1], Section 18, if the tenant does not provide written notice, the lease terminates.",
        "Late fee of $50 (Source 1, Section 3).",
    ],
)
def test_is_refusal_ignores_normal_answers(text: str) -> None:
    assert not is_refusal(text)


def test_split_sentences_keeps_am_pm_together() -> None:
    text = "Entry is allowed 9:00 A.M. to 8:00 P.M. on weekdays. Notice is required."
    assert split_sentences(text) == [
        "Entry is allowed 9:00 A.M. to 8:00 P.M. on weekdays.",
        "Notice is required.",
    ]


def test_leads_with_refusal_skips_colon_preamble() -> None:
    text = "Based on the provided context, I can answer the following question:\n\nThe information is not available in the provided documents."
    assert leads_with_refusal(text)


def test_answer_then_hedge_does_not_lead_with_refusal() -> None:
    # Real q35 answer pattern: a made-up fact first, a hedge afterwards.
    text = "The quiet hours are 9:00 A.M. to 8:00 P.M. [Source 1]. The information is not available in the provided documents regarding quiet hours."
    assert is_refusal(text)
    assert not leads_with_refusal(text)


# --- citations -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Rent is $750 [Source 1].", [1]),
        ("See [Source 2][Source 4].", [2, 4]),
        ("Per [Source 1, 3].", [1, 3]),
        ("Sources 1, 2 and 5 say so.", [1, 2, 5]),
        ("(Source 3 and Source 1)", [3, 1]),
        ("[Sources 2-4]", [2, 3, 4]),
        ("[Source 2] and again [Source 2]", [2]),
        ("No citations here.", []),
        ("Additional resources 3 are listed.", []),
        ("Late fee of $50 (Source 1, Section 3)", [1]),
    ],
)
def test_parse_citations(text: str, expected: list[int]) -> None:
    assert parse_citations(text) == expected


# --- score_answer ----------------------------------------------------------


ANSWERABLE = {
    "id": "qx",
    "answerable": True,
    "evidence": ["late fee of $50"],
    "answer_facts": [["800"]],
}
UNANSWERABLE = {"id": "qy", "answerable": False, "evidence": []}
SOURCES = ["Monthly rent of $750.", "Landlord may impose a late fee of $50."]


def test_score_answer_complete_and_supported() -> None:
    score = score_answer(ANSWERABLE, "Total owed is $800 [Source 1][Source 2].", SOURCES)
    assert score.complete and score.fact_recall == 1.0
    assert score.citations_valid and score.citation_supported
    assert score.false_refusal is False


def test_score_answer_flags_out_of_range_citation() -> None:
    score = score_answer(ANSWERABLE, "Total owed is $800 [Source 7].", SOURCES)
    assert not score.citations_valid
    assert score.out_of_range_citations == [7]
    assert score.citation_supported is None


def test_score_answer_unsupported_citation() -> None:
    score = score_answer(ANSWERABLE, "Total owed is $800 [Source 1].", SOURCES)
    assert score.citation_supported is False


def test_score_answer_false_refusal_only_with_zero_facts() -> None:
    refused = score_answer(ANSWERABLE, "The information is not available in the provided documents.", SOURCES)
    assert refused.false_refusal is True and refused.fact_recall == 0.0
    hedged = score_answer(ANSWERABLE, "$750 + $50 = $800 [Source 1]. The context does not specify a due date.", SOURCES)
    assert hedged.false_refusal is False and hedged.hedged is True


def test_score_answer_contradiction_scores_zero_and_is_not_a_refusal() -> None:
    question = {
        "id": "qn",
        "answerable": True,
        "evidence": [],
        "answer_facts": [["yes", "must provide"], ["reasonable notice"]],
        "answer_must_not": ["does not have to give notice"],
    }
    answer = (
        "The landlord does not have to give notice before showings, although reasonable notice is "
        "required for repairs. This is not explicitly stated in the provided context."
    )
    score = score_answer(question, answer, SOURCES)
    assert score.contradicted and score.fact_recall == 0.0 and not score.complete
    assert score.false_refusal is False


def test_score_answer_unanswerable() -> None:
    good = score_answer(UNANSWERABLE, "The information is not available in the provided documents.", SOURCES)
    bad = score_answer(UNANSWERABLE, "The landlord's number is 555-0100.", SOURCES)
    late = score_answer(
        UNANSWERABLE,
        "Quiet hours are 9 A.M. to 8 P.M. [Source 1]. The information is not available in the provided documents.",
        SOURCES,
    )
    assert good.correct_refusal is True and good.fact_recall is None
    assert bad.correct_refusal is False
    assert late.refused is True and late.correct_refusal is False


def test_score_answer_accepts_evidence_flags_instead_of_text() -> None:
    score = score_answer(ANSWERABLE, "Total $800 [Source 2].", source_has_evidence=[False, True])
    assert score.citation_supported is True and score.complete


def test_citation_support() -> None:
    assert citation_support([2], [False, True]) is True
    assert citation_support([1], [False, True]) is False
    assert citation_support([5], [False, True]) is None
    assert citation_support([], [True]) is None


def test_score_answer_rejects_answerable_without_facts() -> None:
    with pytest.raises(ValueError):
        score_answer({"id": "qz", "answerable": True, "evidence": []}, "x", SOURCES)


# --- dataset consistency ---------------------------------------------------


def test_every_answerable_question_has_fact_groups_and_unanswerable_has_none() -> None:
    for question in load_questions():
        facts = question.get("answer_facts")
        if question["answerable"]:
            assert facts, question["id"]
            assert all(group and all(phrase.strip() for phrase in group) for group in facts), question["id"]
        else:
            assert not facts, question["id"]


@pytest.mark.parametrize(
    "question",
    [q for q in load_questions() if q["answerable"]],
    ids=lambda q: q["id"],
)
def test_expected_answer_satisfies_its_own_fact_groups(question: dict) -> None:
    flags = matched_groups(question["expected"], question["answer_facts"])
    missing = [group for group, hit in zip(question["answer_facts"], flags) if not hit]
    assert not missing, f"{question['id']}: expected text misses {missing}"


@pytest.mark.parametrize(
    "question",
    [q for q in load_questions() if q["answerable"]],
    ids=lambda q: q["id"],
)
def test_fact_groups_are_not_satisfied_by_echoing_the_question(question: dict) -> None:
    flags = matched_groups(question["question"], question["answer_facts"])
    echoed = [group for group, hit in zip(question["answer_facts"], flags) if hit]
    assert not echoed, f"{question['id']}: question text alone satisfies {echoed}"


@pytest.mark.parametrize(
    "question",
    [q for q in load_questions() if q["answerable"]],
    ids=lambda q: q["id"],
)
def test_bare_refusal_scores_zero_facts(question: dict) -> None:
    refusal = "The information is not available in the provided documents."
    assert fact_recall(refusal, question["answer_facts"]) == 0.0


@pytest.mark.parametrize(
    "question",
    [q for q in load_questions() if q.get("answer_must_not")],
    ids=lambda q: q["id"],
)
def test_expected_answer_is_not_contradicted(question: dict) -> None:
    hits = [p for p in question["answer_must_not"] if contains_phrase(question["expected"], p)]
    assert not hits, f"{question['id']}: expected text contains must-not phrase {hits}"
