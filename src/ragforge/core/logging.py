"""Logging setup: one JSON object per line, tagged with the current request ID.

``configure_logging("json")`` makes every log record (RAGForge's and its
libraries') a single JSON line such as:

    {"ts": "2026-10-10T17:52:01.120Z", "level": "INFO", "logger": "ragforge.access",
     "msg": "request", "request_id": "3f9c…", "method": "POST", "path": "/query",
     "status": 200, "latency_ms": 6890.4, "key_id": "key_1a2b3c4d5e6f", ...}

Structured fields are passed as ``logger.info("msg", extra={"fields": {...}})``.
The request ID comes from a context variable set by the request middleware,
so log lines written deep inside retrieval or generation carry it too.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "fields"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            entry["request_id"] = request_id
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            entry.update(fields)
        # Plain `extra={...}` keys from libraries.
        entry.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED and k not in entry})
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        request_id = request_id_var.get()
        fields = getattr(record, "fields", None)
        extras = " ".join(f"{k}={v}" for k, v in fields.items()) if isinstance(fields, dict) else ""
        prefix = f"[{request_id}] " if request_id else ""
        return f"{prefix}{line}{' | ' + extras if extras else ''}"


def configure_logging(fmt: str = "json", level: str = "INFO") -> None:
    """Replace the root handler with one that writes ``fmt`` ("json" or "text") to stdout."""
    if fmt not in ("json", "text"):
        raise ValueError("LOG_FORMAT must be 'json' or 'text'")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Uvicorn's own access log duplicates the structured request log; its
    # other messages (startup, shutdown, errors) go through the same handler.
    logging.getLogger("uvicorn.access").disabled = True
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    # Python warnings (for example from client libraries) become log records too.
    logging.captureWarnings(True)
    # HTTP client libraries log every Qdrant/Ollama call at INFO; keep them at WARNING.
    for name in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Kept for compatibility; configure_logging() sets the format for all loggers."""
    return logging.getLogger(name)
