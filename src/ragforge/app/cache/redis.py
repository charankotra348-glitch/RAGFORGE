import logging

import redis
from redis import Redis

from ragforge.core.config import Settings

logger = logging.getLogger(__name__)

_client: Redis | None = None


def get_redis(settings: Settings) -> Redis:
    """
    Create (once) and return the Redis client.
    """

    global _client

    if _client is None:
        _client = Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )

        logger.info("Redis client created")

    return _client