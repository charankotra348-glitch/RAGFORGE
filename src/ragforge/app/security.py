"""API-key authentication.

Clients send ``X-API-Key: <key>``. The server stores only SHA-256 hashes of
valid keys (``API_KEY_HASHES`` in the environment, comma-separated), so a
leaked config file or log does not leak a usable key. Each key is identified
in logs and rate limits by ``key_id``: the first 12 hex characters of its
hash, never the key itself.

Create a key:

    uv run python -m ragforge.app.security new-key

It prints the key (give it to the client, once) and the hash (put it in
API_KEY_HASHES).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sys

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from ragforge.core.config import get_settings

API_KEY_HEADER = "X-API-Key"
KEY_PREFIX = "rf_"

_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False, description="API key issued by the operator.")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def key_id_for_hash(key_hash: str) -> str:
    return f"key_{key_hash[:12]}"


def generate_key() -> tuple[str, str]:
    """A new random key and its hash."""
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    return key, hash_key(key)


def configured_hashes(raw: str) -> list[str]:
    hashes = [part.strip().lower() for part in raw.split(",") if part.strip()]
    for value in hashes:
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("API_KEY_HASHES must be comma-separated SHA-256 hex digests")
    return hashes


def match_key(key: str, hashes: list[str]) -> str | None:
    """The key_id of a valid key, else None. Compares every hash in constant time."""
    candidate = hash_key(key)
    matched = None
    for stored in hashes:
        if hmac.compare_digest(candidate, stored):
            matched = stored
    return key_id_for_hash(matched) if matched else None


def require_api_key(request: Request, api_key: str | None = Security(_api_key_header)) -> str:
    """FastAPI dependency: returns the caller's key_id or raises 401/503.

    With AUTH_ENABLED=false (local development only) every request is
    accepted as ``key_id = "anonymous"``.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        request.state.key_id = "anonymous"
        return "anonymous"

    try:
        hashes = configured_hashes(settings.api_key_hashes)
    except ValueError:
        raise HTTPException(status_code=503, detail="API authentication is misconfigured.") from None
    if not hashes:
        # Fail closed: no keys configured means nobody gets in.
        raise HTTPException(status_code=503, detail="API authentication is not configured.")

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail=f"Missing API key. Send it in the {API_KEY_HEADER} header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_id = match_key(api_key, hashes)
    if key_id is None:
        raise HTTPException(status_code=401, detail="Invalid API key.", headers={"WWW-Authenticate": "ApiKey"})

    request.state.key_id = key_id
    return key_id


def _main(argv: list[str]) -> int:
    if argv[1:] != ["new-key"]:
        print("usage: python -m ragforge.app.security new-key", file=sys.stderr)
        return 2
    key, key_hash = generate_key()
    print(f"API key (give to the client; it is not stored anywhere): {key}")
    print(f"Hash (add to API_KEY_HASHES in .env):                    {key_hash}")
    print(f"Key id (appears in logs):                                {key_id_for_hash(key_hash)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
