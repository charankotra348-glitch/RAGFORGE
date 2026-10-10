"""Errors that mean "a backing service is unreachable", mapped to HTTP 503 on every endpoint."""

from __future__ import annotations

import logging

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ResponseHandlingException

logger = logging.getLogger(__name__)

DEPENDENCY_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    httpx.TransportError,
    ResponseHandlingException,
)

UNAVAILABLE_DETAIL = "A required service (vector store or language model) is unavailable. Try again shortly."


async def dependency_unavailable(request: Request, exc: Exception) -> JSONResponse:
    logger.warning("Dependency unavailable: %s", type(exc).__name__)
    return JSONResponse(status_code=503, content={"detail": UNAVAILABLE_DETAIL})
