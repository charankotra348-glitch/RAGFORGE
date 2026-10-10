from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from ragforge.app.main import app


def test_root_returns_api_status():
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert response.json()["message"]
    assert response.json()["version"] == app.version


def test_query_rejects_whitespace():
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "   ", "top_k": 5},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Query cannot be empty or whitespace."
    )


def test_query_rejects_invalid_top_k():
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "What is the lease term?", "top_k": 0},
        )

    assert response.status_code == 422


def test_query_returns_answer_and_sources():
    fake_source = SimpleNamespace(
        metadata={"page_number": 3, "section_number": 19},
        text="The landlord must return the deposit within 14 days.",
    )
    fake_result = SimpleNamespace(
        query="When must the deposit be returned?",
        answer="The deposit must be returned within 14 days. [Source 1]",
        sources=[fake_source],
    )

    with patch(
        "ragforge.app.main.answer_query",
        return_value=fake_result,
    ) as mocked_answer:
        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={
                    "query": "When must the deposit be returned?",
                    "top_k": 5,
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == fake_result.answer
    assert len(body["sources"]) == 1
    assert body["sources"][0]["citation"] == "[Source 1]"
    assert body["sources"][0]["page"] == "3"
    assert body["sources"][0]["section"] == "19"
    assert "14 days" in body["sources"][0]["text"]

    mocked_answer.assert_called_once_with(
        settings=mocked_answer.call_args.kwargs["settings"],
        query="When must the deposit be returned?",
        top_k=5,
    )
