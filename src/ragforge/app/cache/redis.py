import logging

import redis
from redis import Redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from ragforge.core.config import Settings

logger = logging.getLogger(__name__)

_client: Redis | None = None


def get_redis(settings: Settings) -> Redis:
    """
    Create (once) and return the Redis client.
    """

    global _client

    if _client is None:
        # Short timeouts and no retries: Redis only backs rate limiting and
        # caching, which fail open. With the library defaults, every request
        # waited several seconds for retries whenever Redis was down.
        _client = Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            retry=Retry(NoBackoff(), 0),
        )

        logger.info("Redis client created")

    return _client