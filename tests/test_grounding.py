from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ragforge.app.generation.grounding import (
    is_abstention_answer,
    verify_answer_grounded,
)
from ragforge.app.retrieval.types import SearchResult


def make_settings():
    return SimpleNamespace(ollama_model="llama3.2:3b")


def make_results():
    return [
        SearchResult(
            text="Monthly rent is $1,500.",
            metadata={"page": 1, "section": "Rent"},
        )
    ]


def make_response(content):
    return SimpleNamespace(
        message=SimpleNamespace(content=content)
    )


def test_supported_answer_passes_verification():
    client = MagicMock()
    client.chat.return_value = make_response(
        '{"supported": true, "reason": "The rent amount is stated."}'
    )

    with patch(
        "ragforge.app.generation.llm.get_llm_client",
        return_value=client,
    ):
        result = verify_answer_grounded(
            make_settings(),
            "What is the monthly rent?",
            "The monthly rent is $1,500.",
            make_results(),
        )

    assert result is True


def test_unsupported_answer_fails_verification():
    client = MagicMock()
    client.chat.return_value = make_response(
        '{"supported": false, "reason": "The quiet hours are not stated."}'
    )

    with patch(
        "ragforge.app.generation.llm.get_llm_client",
        return_value=client,
    ):
        result = verify_answer_grounded(
            make_settings(),
            "What are the quiet hours?",
            "Quiet hours are 9 PM to 8 AM.",
            make_results(),
        )

    assert result is False


def test_malformed_verifier_response_fails_closed():
    client = MagicMock()
    client.chat.return_value = make_response("not valid JSON")

    with patch(
        "ragforge.app.generation.llm.get_llm_client",
        return_value=client,
    ):
        result = verify_answer_grounded(
            make_settings(),
            "What are the quiet hours?",
            "Quiet hours are 9 PM to 8 AM.",
            make_results(),
        )

    assert result is False


def test_verifier_exception_fails_closed():
    client = MagicMock()
    client.chat.side_effect = RuntimeError("Ollama unavailable")

    with patch(
        "ragforge.app.generation.llm.get_llm_client",
        return_value=client,
    ):
        result = verify_answer_grounded(
            make_settings(),
            "What are the quiet hours?",
            "Quiet hours are 9 PM to 8 AM.",
            make_results(),
        )

    assert result is False


def test_existing_abstention_does_not_need_verification():
    with patch(
        "ragforge.app.generation.llm.get_llm_client"
    ) as get_client:
        result = verify_answer_grounded(
            make_settings(),
            "What are the quiet hours?",
            "The context does not contain information about quiet hours.",
            make_results(),
        )

    assert result is True
    get_client.assert_not_called()


def test_abstention_detection():
    assert is_abstention_answer(
        "The context does not specify the pet deposit."
    )
    assert not is_abstention_answer(
        "The pet deposit is $500."
    )


# --- added with the structured verdict and the abstention-bypass fix --------

from ragforge.app.generation.grounding import check_answer_grounded  # noqa: E402


def test_abstention_followed_by_a_claim_is_still_verified():
    answer = "The context does not specify quiet hours. However, quiet hours are 9 PM to 8 AM [Source 1]."
    assert not is_abstention_answer(answer)


def test_generic_not_stated_sentence_is_an_abstention():
    assert is_abstention_answer("The parking fee is not explicitly stated in the provided context.")
    assert is_abstention_answer(
        "The information is not available in the provided documents. "
        "The parking fee is not mentioned in any of the sources."
    )


def test_verdict_reports_why_verification_failed():
    def verdict_for(content=None, side_effect=None):
        client = MagicMock()
        if side_effect:
            client.chat.side_effect = side_effect
        else:
            client.chat.return_value = make_response(content)
        with patch("ragforge.app.generation.llm.get_llm_client", return_value=client):
            return check_answer_grounded(
                make_settings(), "What is the monthly rent?", "The monthly rent is $1,500.", make_results()
            )

    assert verdict_for('{"supported": true, "reason": "stated"}').status == "supported"
    assert verdict_for('{"supported": false, "reason": "no"}').status == "unsupported"
    assert verdict_for("not valid JSON").status == "malformed"
    assert verdict_for('{"supported": "yes"}').status == "malformed"
    assert verdict_for("[]").status == "malformed"
    assert verdict_for(side_effect=RuntimeError("Ollama unavailable")).status == "error"
    assert not verdict_for("not valid JSON").supported
