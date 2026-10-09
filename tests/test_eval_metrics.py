import json
import math
from pathlib import Path

import pytest
from pypdf import PdfReader

from ragforge.evaluation.metrics import (
    coverage_at_k,
    hit_at_k,
    missing_evidence,
    mrr,
    ndcg_at_k,
    normalize,
    relevance_flags,
)

ROOT = Path(__file__).resolve().parents[1]


def test_normalize_ignores_case_whitespace_and_punctuation() -> None:
    assert normalize("Landlord's  PRIOR\nwritten consent.") == "landlordspriorwrittenconsent"


def test_relevance_flags_match_across_line_breaks() -> None:
    texts = ["no later than sixty (60)\ndays prior", "unrelated text"]
    assert relevance_flags(texts, ["No later than sixty (60) days prior"]) == [True, False]


def test_hit_at_k_respects_cutoff() -> None:
    flags = [False, False, True]
    assert hit_at_k(flags, 2) == 0.0
    assert hit_at_k(flags, 3) == 1.0


def test_mrr_uses_first_relevant_rank_and_zero_when_none() -> None:
    assert mrr([False, True, True]) == 0.5
    assert mrr([False, False]) == 0.0


def test_ndcg_perfect_and_degraded_ranking() -> None:
    assert ndcg_at_k([True, True, False], k=3, total_relevant=2) == pytest.approx(1.0)
    degraded = ndcg_at_k([False, True, False], k=3, total_relevant=1)
    assert degraded == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k([False, False], k=2, total_relevant=0) == 0.0


def test_coverage_counts_evidence_found_in_union_of_top_k() -> None:
    texts = ["entry needs 24 hours notice", "showing during the last 60 days", "pets"]
    evidence = ["24 hours", "last 60 days", "bona fide emergency"]
    assert coverage_at_k(texts, evidence, k=2) == pytest.approx(2 / 3)
    assert missing_evidence(texts, evidence, k=1) == ["last 60 days", "bona fide emergency"]


def test_invalid_k_and_empty_evidence_raise() -> None:
    with pytest.raises(ValueError):
        hit_at_k([True], 0)
    with pytest.raises(ValueError):
        coverage_at_k(["x"], [], 3)


def test_every_evidence_label_exists_in_the_source_pdf() -> None:
    """Catches label typos without needing Qdrant or any model."""
    dataset = json.loads((ROOT / "data" / "eval" / "lease_eval.json").read_text(encoding="utf-8"))
    pdf_text = normalize("\n".join(page.extract_text() for page in PdfReader(ROOT / dataset["document"]).pages))

    ids = [question["id"] for question in dataset["questions"]]
    assert len(ids) == len(set(ids)), "duplicate question ids"

    for question in dataset["questions"]:
        assert question["tags"], question["id"]
        assert question["answerable"] == bool(question["evidence"]), question["id"]
        for snippet in question["evidence"]:
            assert normalize(snippet) in pdf_text, f"{question['id']}: {snippet!r}"
