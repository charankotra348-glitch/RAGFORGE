from ragforge.core.config import Settings


def test_locked_local_model_configuration_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.ollama_host == "http://localhost:11434"
    assert settings.ollama_model == "llama3.2:3b"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.vector_size == 384
    assert settings.qdrant_collection == "ragforge_chunks"
