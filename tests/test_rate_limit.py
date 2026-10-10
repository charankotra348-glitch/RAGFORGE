"""Per-key rate limiting."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ragforge.app.main import app
from ragforge.app.rate_limit import check_rate_limit
from ragforge.core.config import get_settings

from fakes import FakeRedis

FAKE_RESULT = SimpleNamespace(query="q", answer="a", sources=[], abstained=False, abstention_reason=None)


def test_counts_per_key_and_resets_each_window() -> None:
    redis = FakeRedis()
    t = 60 * 16_667 + 20.0  # 20 s into a window
    results = [check_rate_limit(redis, "key_a", 3, now=t) for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]
    assert [r.remaining for r in results] == [2, 1, 0, 0]
    assert results[0].reset_seconds == 40
    assert check_rate_limit(redis, "key_b", 3, now=t).allowed  # separate key, separate count
    assert check_rate_limit(redis, "key_a", 3, now=t + 60).allowed  # next window
    assert all(seconds > 60 for seconds in redis.ttl.values())  # counters expire


def test_disabled_or_redis_down_means_not_limited() -> None:
    assert check_rate_limit(FakeRedis(), "key_a", 0) is None
    assert check_rate_limit(None, "key_a", 5) is None
    assert check_rate_limit(FakeRedis(fail=True), "key_a", 5) is None


@pytest.fixture()
def client_with(monkeypatch):
    """A TestClient whose app uses the given Redis; app.state is restored afterwards."""

    def make(redis) -> TestClient:
        monkeypatch.setattr(app.state, "redis", redis, raising=False)
        return TestClient(app)

    return make


def test_api_returns_429_with_retry_after(api_headers, monkeypatch, client_with) -> None:
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 2)
    client = client_with(FakeRedis())
    with patch("ragforge.app.main.answer_query", return_value=FAKE_RESULT) as answer:
        codes = [client.post("/query", json={"query": f"q{i}"}, headers=api_headers) for i in range(3)]
    assert [r.status_code for r in codes] == [200, 200, 429]
    assert codes[0].headers["X-RateLimit-Limit"] == "2"
    assert codes[1].headers["X-RateLimit-Remaining"] == "0"
    assert int(codes[2].headers["Retry-After"]) >= 1
    assert answer.call_count == 2  # the limited request did no work


def test_unauthenticated_requests_are_rejected_before_counting(api_headers, monkeypatch, client_with) -> None:
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 2)
    redis = FakeRedis()
    client = client_with(redis)
    response = client.post("/query", json={"query": "q"}, headers={"X-API-Key": "rf_wrong"})
    assert response.status_code == 401
    assert redis.data == {}


def test_api_serves_requests_when_redis_is_down(api_headers, monkeypatch, client_with) -> None:
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 1)
    client = client_with(FakeRedis(fail=True))
    with patch("ragforge.app.main.answer_query", return_value=FAKE_RESULT):
        codes = [client.post("/query", json={"query": "q"}, headers=api_headers).status_code for _ in range(3)]
    assert codes == [200, 200, 200]


def test_app_redis_client_fails_fast_when_redis_is_down(monkeypatch) -> None:
    import time

    from ragforge.app.cache import redis as redis_module

    monkeypatch.setattr(redis_module, "_client", None)
    client = redis_module.get_redis(SimpleNamespace(redis_host="127.0.0.1", redis_port=1, redis_db=0))
    start = time.perf_counter()
    assert check_rate_limit(client, "key_a", 5) is None
    assert time.perf_counter() - start < 1.5
