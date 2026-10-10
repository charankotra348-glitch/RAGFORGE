"""Answer cache and corpus version, both in Redis.

Corpus version
    A counter (``ragforge:corpus_version``) incremented whenever indexed
    content changes: the reindex script, document upload and document
    delete. The API compares it with the version its in-memory BM25 index
    was built from and rebuilds when it changed, so re-indexing no longer
    needs an API restart. Cached answers are keyed by the version too, so
    they expire with the content they came from.

Answer cache
    Key: corpus version + a fingerprint of everything that shapes the answer
    (model, system prompt, verifier setting, collection, embedding model) +
    top_k + a SHA-256 of the normalized question (lowercased, whitespace
    collapsed). The question itself is never stored in the key. The cached
    value is the response without the question; it can still contain text
    from the answer, and it expires after ANSWER_CACHE_TTL_SECONDS.

Every Redis failure is treated as a cache miss: the request is answered
normally, just without caching.
"""

from __future__ import annotations

import hashlib
import json
import logging

logger = logging.getLogger(__name__)

CORPUS_VERSION_KEY = "ragforge:corpus_version"
ANSWER_KEY_PREFIX = "ragforge:answer"


def get_corpus_version(redis_client) -> str | None:
    """Current corpus version ("0" if never bumped), or None if Redis is unavailable."""
    if redis_client is None:
        return None
    try:
        value = redis_client.get(CORPUS_VERSION_KEY)
    except Exception:
        logger.warning("Redis unavailable: corpus version unknown, answer cache bypassed", exc_info=True)
        return None
    return str(value) if value is not None else "0"


def bump_corpus_version(redis_client) -> str | None:
    """Mark the indexed content as changed. Returns the new version, or None if Redis is down."""
    if redis_client is None:
        return None
    try:
        return str(redis_client.incr(CORPUS_VERSION_KEY))
    except Exception:
        logger.warning(
            "Could not bump the corpus version in Redis; running API workers will not "
            "notice this change until they restart, and may serve cached answers until they expire",
            exc_info=True,
        )
        return None


def normalize_question(query: str) -> str:
    return " ".join(query.lower().split())


def answer_cache_key(corpus_version: str, fingerprint: str, top_k: int, query: str) -> str:
    question_hash = hashlib.sha256(normalize_question(query).encode("utf-8")).hexdigest()
    return f"{ANSWER_KEY_PREFIX}:{corpus_version}:{fingerprint}:{top_k}:{question_hash}"


def config_fingerprint(*parts: object) -> str:
    joined = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def get_cached_answer(redis_client, key: str) -> dict | None:
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        logger.warning("Answer cache read failed; treating as a miss", exc_info=True)
        return None


def set_cached_answer(redis_client, key: str, payload: dict, ttl_seconds: int) -> bool:
    if redis_client is None or ttl_seconds <= 0:
        return False
    try:
        redis_client.set(key, json.dumps(payload, ensure_ascii=False), ex=ttl_seconds)
        return True
    except Exception:
        logger.warning("Answer cache write failed", exc_info=True)
        return False
