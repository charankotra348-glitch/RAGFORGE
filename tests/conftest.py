"""Shared fixtures."""

import pytest

from ragforge.app.security import hash_key
from ragforge.core.config import get_settings

TEST_API_KEY = "rf_test-key-for-unit-tests-only"


@pytest.fixture()
def api_headers(monkeypatch) -> dict[str, str]:
    """Configure one valid API key and return headers that use it."""
    settings = get_settings()
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "api_key_hashes", hash_key(TEST_API_KEY))
    return {"X-API-Key": TEST_API_KEY}
