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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
