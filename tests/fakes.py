"""Small in-memory stand-ins for external services used by unit tests."""


class FakeRedis:
    """Implements the subset of redis.Redis that RAGForge uses."""

    def __init__(self, fail: bool = False):
        self.data: dict[str, str] = {}
        self.ttl: dict[str, int] = {}
        self.fail = fail

    def _check(self) -> None:
        if self.fail:
            raise ConnectionError("redis down")

    def incr(self, key: str) -> int:
        self._check()
        self.data[key] = str(int(self.data.get(key, "0")) + 1)
        return int(self.data[key])

    def expire(self, key: str, seconds: int) -> bool:
        self._check()
        self.ttl[key] = seconds
        return True

    def get(self, key: str):
        self._check()
        return self.data.get(key)

    def set(self, key: str, value, ex: int | None = None) -> bool:
        self._check()
        self.data[key] = str(value) if not isinstance(value, str) else value
        if ex is not None:
            self.ttl[key] = ex
        return True

    def ping(self) -> bool:
        self._check()
        return True

    def close(self) -> None:
        pass

    def pipeline(self):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis: FakeRedis):
        self.redis = redis
        self.calls = []

    def incr(self, key):
        self.calls.append(("incr", key))
        return self

    def expire(self, key, seconds):
        self.calls.append(("expire", key, seconds))
        return self

    def execute(self):
        return [getattr(self.redis, name)(*args) for name, *args in self.calls]
