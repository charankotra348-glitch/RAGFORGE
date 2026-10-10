from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "RAGForge"
    environment: str = "development"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "ragforge"
    postgres_password: str = "ragforge"
    postgres_db: str = "ragforge"

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "ragforge_chunks"

    # Locked local model configuration
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    vector_size: int = 384

    # Grounding verifier: a second LLM call that checks each answer against the
    # retrieved context and replaces unsupported answers with a safe abstention.
    # Off by default: with llama3.2:3b it rejected 28 of 32 checked answers on
    # the lease eval (data/eval/runs/clean-3b-verifier-on.json), blocking 21
    # correct or partly correct answers to catch 1 hallucination.
    grounding_verifier_enabled: bool = False

    # API authentication (see ragforge/app/security.py). Comma-separated
    # SHA-256 hashes of valid API keys; create one with
    # `uv run python -m ragforge.app.security new-key`. With no hashes set,
    # protected endpoints return 503 (fail closed). AUTH_ENABLED=false turns
    # authentication off for local development only.
    auth_enabled: bool = True
    api_key_hashes: str = ""

    # Requests per API key per minute (Redis fixed window; 0 disables).
    rate_limit_per_minute: int = 30

    # Document uploads (POST /documents): where files are kept, and the size limit.
    upload_dir: str = "data/uploads"
    max_upload_mb: int = 10

    # Seconds a /query answer stays in the Redis cache (0 disables caching).
    answer_cache_ttl_seconds: int = 3600

    # Logging: "json" (one object per line, for log collectors) or "text".
    log_format: str = "json"
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
