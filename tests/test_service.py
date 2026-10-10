"""answer_query(): each abstention path sets the structured fields."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ragforge.app.generation import service
from ragforge.app.generation.grounding import SAFE_ABSTENTION, GroundingVerdict
from ragforge.app.retrieval.types import SearchResult

SETTINGS = SimpleNamespace(ollama_model="llama3.2:3b", qdrant_collection="test")
RESULTS = [SearchResult(text="Monthly rent of $750.", metadata={"page_number": 1})]


class FakeSearcher:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=5):
        return self.results


def run(candidate: str, verdict: GroundingVerdict | None = None, results=RESULTS, **kwargs):
    stats = {"prompt_eval_count": 10, "eval_count": 5}
    with (
        patch.object(service, "get_hybrid_searcher", return_value=FakeSearcher(results)),
        patch.object(service, "generate_answer_with_stats", return_value=(candidate, stats)) as gen,
        patch.object(service, "check_answer_grounded", return_value=verdict) as check,
    ):
        response = service.answer_query(SETTINGS, "What is the rent?", **kwargs)
    return response, gen, check


def test_supported_answer_is_returned_with_verification_details() -> None:
    response, _, check = run("The rent is $750 [Source 1].", GroundingVerdict(True, "supported", "stated", 5.0))
    assert response.answer == "The rent is $750 [Source 1]."
    assert not response.abstained and response.abstention_reason is None
    assert response.verification["status"] == "supported"
    assert response.timings_ms["verification"] == 5.0
    check.assert_called_once()


def test_rejected_answer_becomes_safe_abstention_and_keeps_candidate() -> None:
    response, _, _ = run("The rent is $900 [Source 1].", GroundingVerdict(False, "unsupported", "not stated", 5.0))
    assert response.answer == SAFE_ABSTENTION
    assert response.candidate_answer == "The rent is $900 [Source 1]."
    assert response.abstained and response.abstention_reason == "failed_verification"


@pytest.mark.parametrize("status", ["malformed", "error"])
def test_verifier_failure_fails_closed_and_records_status(status: str) -> None:
    response, _, _ = run("The rent is $750 [Source 1].", GroundingVerdict(False, status))
    assert response.abstention_reason == "failed_verification"
    assert response.verification["status"] == status


def test_model_abstention_skips_verifier() -> None:
    response, _, check = run("The information is not available in the provided documents.")
    assert response.abstained and response.abstention_reason == "model_abstained"
    check.assert_not_called()


def test_no_results_is_its_own_reason() -> None:
    response, _, check = run("The information is not available in the provided documents.", results=[])
    assert response.abstention_reason == "no_results"
    check.assert_not_called()


def test_verify_false_skips_verifier_and_passes_generation_options() -> None:
    response, gen, check = run(
        "The rent is $750 [Source 1].", verify=False, model="m", system_prompt="p", num_ctx=4096
    )
    assert response.verification is None and not response.abstained
    check.assert_not_called()
    kwargs = gen.call_args.kwargs
    assert (kwargs["model"], kwargs["system_prompt"], kwargs["num_ctx"]) == ("m", "p", 4096)
