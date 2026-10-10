from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from ragforge.app.main import app


def test_root_returns_api_status():
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert response.json()["message"]
    assert response.json()["version"] == app.version


def test_query_rejects_whitespace(api_headers):
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "   ", "top_k": 5},
            headers=api_headers,
        )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Query cannot be empty or whitespace."
    )


def test_query_rejects_invalid_top_k(api_headers):
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "What is the lease term?", "top_k": 0},
            headers=api_headers,
        )

    assert response.status_code == 422


def test_query_returns_answer_and_sources(api_headers):
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
                headers=api_headers,
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


def test_query_reports_structured_abstention(api_headers):
    fake_result = SimpleNamespace(
        query="What are the quiet hours?",
        answer="I cannot provide a reliable answer because the retrieved context does not sufficiently support it.",
        sources=[],
        abstained=True,
        abstention_reason="failed_verification",
    )

    # No context manager: this test does not start Postgres, Redis or Qdrant.
    with patch("ragforge.app.main.answer_query", return_value=fake_result):
        response = TestClient(app).post(
            "/query", json={"query": "What are the quiet hours?", "top_k": 5}, headers=api_headers
        )

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["abstention_reason"] == "failed_verification"


# --- degraded mode: the API must work and report clearly without its services ---

from qdrant_client.http.exceptions import ResponseHandlingException  # noqa: E402

from ragforge.app import main as main_module  # noqa: E402


def test_api_starts_and_reports_degraded_when_services_are_down():
    def down(settings):
        raise ConnectionError("connection refused to secret-host:6333 password=hunter2")

    class BrokenRedis:
        def ping(self):
            raise ConnectionError("redis down at secret-host")

        def close(self):
            pass

    with (
        patch.object(main_module, "get_database", side_effect=down),
        patch.object(main_module, "get_redis", return_value=BrokenRedis()),
        patch.object(main_module, "get_qdrant", side_effect=down),
    ):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"] == {"postgres": "down", "redis": "down", "qdrant": "down"}
    assert "secret-host" not in response.text and "hunter2" not in response.text


def test_query_returns_503_when_a_dependency_is_unavailable(api_headers):
    for error in (
        ConnectionError("Failed to connect to Ollama"),
        ResponseHandlingException(ConnectionError("Connection refused")),
    ):
        with patch.object(main_module, "answer_query", side_effect=error):
            response = TestClient(app).post(
                "/query", json={"query": "What is the rent?", "top_k": 5}, headers=api_headers
            )
        assert response.status_code == 503, error
        assert "Connection refused" not in response.text
