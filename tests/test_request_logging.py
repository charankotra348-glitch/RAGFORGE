"""Request IDs and structured access logs."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from ragforge.app.main import app
from ragforge.core.logging import JsonFormatter, TextFormatter, configure_logging, request_id_var

QUERY = "What is the monthly rent for 48 Maple Hollow Road?"
FAKE_RESULT = SimpleNamespace(
    query=QUERY,
    answer="The information is not available in the provided documents.",
    sources=[],
    abstained=True,
    abstention_reason="model_abstained",
    timings_ms={"retrieval": 12.5, "generation": 900.0},
)


def access_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "ragforge.access"]


def test_response_has_a_request_id_and_an_incoming_one_is_kept() -> None:
    client = TestClient(app)
    generated = client.get("/")
    assert len(generated.headers["X-Request-ID"]) == 32
    echoed = client.get("/", headers={"X-Request-ID": "trace-123.abc_DEF"})
    assert echoed.headers["X-Request-ID"] == "trace-123.abc_DEF"


def test_unsafe_incoming_request_id_is_replaced() -> None:
    response = TestClient(app).get("/", headers={"X-Request-ID": "bad id\nwith newline" + "x" * 80})
    assert response.headers["X-Request-ID"] != "bad id"
    assert len(response.headers["X-Request-ID"]) == 32


def test_query_access_log_has_fields_but_never_the_query_text(api_headers, caplog) -> None:
    caplog.set_level(logging.INFO, logger="ragforge.access")
    with patch("ragforge.app.main.answer_query", return_value=FAKE_RESULT):
        response = TestClient(app).post("/query", json={"query": QUERY, "top_k": 3}, headers=api_headers)
    assert response.status_code == 200
    (record,) = access_records(caplog)
    fields = record.fields
    assert fields["path"] == "/query" and fields["status"] == 200 and fields["method"] == "POST"
    assert fields["key_id"].startswith("key_") and fields["top_k"] == 3
    assert fields["query_chars"] == len(QUERY)
    assert fields["abstained"] is True and fields["abstention_reason"] == "model_abstained"
    assert fields["generation_ms"] == 900.0 and fields["latency_ms"] >= 0
    line = JsonFormatter().format(record)
    assert "Maple Hollow" not in line and QUERY not in line


def test_rejected_request_is_logged_as_warning_without_key_id(api_headers, caplog) -> None:
    caplog.set_level(logging.INFO, logger="ragforge.access")
    TestClient(app).post("/query", json={"query": "x"}, headers={"X-API-Key": "rf_wrong"})
    (record,) = access_records(caplog)
    assert record.levelno == logging.WARNING
    assert record.fields["status"] == 401 and record.fields["key_id"] is None


def test_json_formatter_includes_request_id_and_extra_fields() -> None:
    token = request_id_var.set("req-1")
    try:
        record = logging.LogRecord("ragforge.x", logging.INFO, __file__, 1, "hello %s", ("world",), None)
        record.fields = {"status": 200}
        entry = json.loads(JsonFormatter().format(record))
        text = TextFormatter().format(record)
    finally:
        request_id_var.reset(token)
    assert entry["msg"] == "hello world" and entry["request_id"] == "req-1" and entry["status"] == 200
    assert entry["ts"].endswith("Z") and entry["level"] == "INFO"
    assert text.startswith("[req-1] ") and "status=200" in text


def test_configure_logging_rejects_unknown_format() -> None:
    import pytest

    with pytest.raises(ValueError):
        configure_logging("xml")


def test_configure_logging_routes_uvicorn_and_warnings_through_one_handler() -> None:
    uvicorn_error = logging.getLogger("uvicorn.error")
    uvicorn_error.addHandler(logging.StreamHandler())
    uvicorn_error.propagate = False
    configure_logging("json")
    assert uvicorn_error.handlers == [] and uvicorn_error.propagate is True
    assert logging.getLogger("uvicorn.access").disabled is True
    (handler,) = logging.getLogger().handlers
    assert isinstance(handler.formatter, JsonFormatter)
