"""Request ID and access-log middleware.

Every response carries ``X-Request-ID``. A client may send its own (letters,
digits, ``.``, ``_``, ``-``; at most 64 characters), for example to trace a
call through several services; anything else is replaced with a new ID.

Each request produces one structured log line (logger ``ragforge.access``)
with method, path, status, latency, the caller's key_id, and any fields the
endpoint added to ``request.state.log_fields`` (for /query: top_k, query
length, abstention reason, cache hit). The query text itself is never
logged, because user questions can contain personal data.
"""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ragforge.core.logging import request_id_var

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

access_logger = logging.getLogger("ragforge.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        request.state.request_id = request_id
        request.state.log_fields = {}
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            fields = {
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "key_id": getattr(request.state, "key_id", None),
                **getattr(request.state, "log_fields", {}),
            }
            level = logging.ERROR if status >= 500 else logging.WARNING if status >= 400 else logging.INFO
            access_logger.log(level, "request", extra={"fields": fields})
            request_id_var.reset(token)
