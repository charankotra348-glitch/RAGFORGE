"""API-key authentication."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ragforge.app import security
from ragforge.app.main import app
from ragforge.core.config import get_settings

from conftest import TEST_API_KEY

BODY = {"query": "What is the rent?", "top_k": 5}
FAKE_RESULT = SimpleNamespace(query="What is the rent?", answer="$750 [Source 1].", sources=[], abstained=False, abstention_reason=None)


def post(headers=None):
    with patch("ragforge.app.main.answer_query", return_value=FAKE_RESULT) as answer:
        response = TestClient(app).post("/query", json=BODY, headers=headers or {})
    return response, answer


def test_valid_key_is_accepted(api_headers) -> None:
    response, answer = post(api_headers)
    assert response.status_code == 200
    answer.assert_called_once()


def test_missing_key_is_rejected_before_any_work(api_headers) -> None:
    response, answer = post()
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "ApiKey"
    answer.assert_not_called()


def test_wrong_key_is_rejected(api_headers) -> None:
    response, answer = post({"X-API-Key": "rf_wrong"})
    assert response.status_code == 401
    answer.assert_not_called()


def test_no_configured_keys_fails_closed(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "api_key_hashes", "")
    response, answer = post({"X-API-Key": TEST_API_KEY})
    assert response.status_code == 503
    answer.assert_not_called()


def test_malformed_hash_config_fails_closed(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "api_key_hashes", "not-a-hash")
    response, _ = post({"X-API-Key": TEST_API_KEY})
    assert response.status_code == 503
    assert "not-a-hash" not in response.text


def test_auth_can_be_disabled_for_local_development(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "auth_enabled", False)
    response, _ = post()
    assert response.status_code == 200


def test_health_and_root_stay_open(api_headers) -> None:
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code in (200, 503)  # never 401


def test_several_keys_and_key_id() -> None:
    key_a, hash_a = security.generate_key()
    key_b, hash_b = security.generate_key()
    hashes = security.configured_hashes(f"{hash_a}, {hash_b.upper()}")
    assert key_a.startswith("rf_") and key_a != key_b
    assert security.match_key(key_b, hashes) == f"key_{hash_b[:12]}"
    assert security.match_key("rf_other", hashes) is None


def test_key_id_never_contains_the_key() -> None:
    key, key_hash = security.generate_key()
    assert key not in security.key_id_for_hash(key_hash)


def test_new_key_command(capsys) -> None:
    assert security._main(["x", "new-key"]) == 0
    out = capsys.readouterr().out
    key = out.split("client; it is not stored anywhere): ")[1].split()[0]
    key_hash = out.split("API_KEY_HASHES in .env):")[1].split()[0]
    assert security.hash_key(key) == key_hash
    assert security._main(["x"]) == 2


def test_configured_hashes_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        security.configured_hashes("abc")
