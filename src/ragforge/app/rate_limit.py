"""Per-API-key rate limiting with a Redis fixed window.

Each key may make RATE_LIMIT_PER_MINUTE requests per calendar minute. The
counter is one Redis key per (API key, minute), created with INCR and expired
shortly after the minute ends, so counts are shared by every API worker.

Fixed windows allow a burst of up to 2x the limit across a minute boundary;
that is accepted here for simplicity. If Redis is unreachable the limiter
fails open (the request is served and a warning is logged): availability is
preferred over strict limiting for this service. Both trade-offs are
documented in the README.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, Response

from ragforge.app.security import require_api_key
from ragforge.core.config import get_settings

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    reset_seconds: int


def check_rate_limit(
    redis_client,
    key_id: str,
    limit: int,
    *,
    now: float | None = None,
    window_seconds: int = WINDOW_SECONDS,
) -> RateLimitResult | None:
    """Count one request for ``key_id``. None means "not limited" (disabled or Redis down)."""
    if limit <= 0 or redis_client is None:
        return None
    now = time.time() if now is None else now
    window = int(now // window_seconds)
    counter = f"ratelimit:{key_id}:{window}"
    try:
        pipe = redis_client.pipeline()
        pipe.incr(counter)
        pipe.expire(counter, window_seconds + 5)
        count, _ = pipe.execute()
    except Exception:
        logger.warning("Rate limiter unavailable; serving request without a limit", exc_info=True)
        return None
    reset = max(1, int(window_seconds - (now % window_seconds)))
    return RateLimitResult(
        allowed=int(count) <= limit,
        limit=limit,
        remaining=max(limit - int(count), 0),
        reset_seconds=reset,
    )


def enforce_rate_limit(
    request: Request,
    response: Response,
    key_id: str = Depends(require_api_key),
) -> str:
    """FastAPI dependency: authenticate, then count the request. Returns the key_id."""
    result = check_rate_limit(
        getattr(request.app.state, "redis", None),
        key_id,
        get_settings().rate_limit_per_minute,
    )
    if result is None:
        return key_id

    headers = {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
        "X-RateLimit-Reset": str(result.reset_seconds),
    }
    if not result.allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Retry after the time in the Retry-After header.",
            headers={**headers, "Retry-After": str(result.reset_seconds)},
        )
    response.headers.update(headers)
    return key_id
