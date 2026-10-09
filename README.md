# RAGForge

RAGForge is a Qdrant-based hybrid retrieval-augmented generation (RAG) project. The current retrieval design combines dense vector search, BM25 lexical search, Reciprocal Rank Fusion (RRF), cross-encoder reranking, and grounded answer generation with a local Ollama model.

## Locked local model configuration

- **Generation:** Ollama, default model `llama3.2:3b`
- **Embeddings:** FastEmbed, `BAAI/bge-small-en-v1.5`
- **Embedding dimensions:** 384
- **Vector store:** Qdrant, collection `ragforge_chunks`
- **Lexical retrieval:** BM25 built from the actual chunks stored in Qdrant

Keep the embedding model and vector dimensions consistent with the Qdrant collection. If the embedding model changes, intentionally re-index the affected documents; do not mix vectors from different embedding models in one collection.

## Local development

1. Create your local environment file:

   ```bash
   cp .env.example .env
   ```

2. Start the supporting services:

   ```bash
   docker compose up -d postgres redis qdrant
   ```

3. Install the locked Python dependencies:

   ```bash
   uv sync --frozen
   ```

4. Ensure Ollama is running locally and pull the configured model:

   ```bash
   ollama pull llama3.2:3b
   ```

5. Index the example document into Qdrant:

   ```bash
   uv run python -m ragforge.vector.reindex data/test.pdf
   ```

   Re-indexing replaces only the selected document's chunks; it does not recreate or delete the entire Qdrant collection. If the API is already running in a separate process while you re-index, restart the API afterward so its in-memory BM25 corpus is rebuilt. Cross-process cache invalidation will be addressed when document-management endpoints are implemented.

6. Run the API:

   ```bash
   uv run uvicorn ragforge.app.main:app --reload
   ```

   Current HTTP endpoints are `/` and `/health`. Document upload and query endpoints are planned work; use the evaluation scripts or `answer_query()` service function for the current RAG flow.

## Tests

Run the focused tests that don't require external services:

```bash
uv run pytest -q tests/test_config.py tests/test_retrieval_fusion.py
```

The live integration test in `test_rag.py` is skipped by default. Run it only when Qdrant and Ollama are available, the configured Ollama model is installed, and the example document has been indexed:

```bash
RAGFORGE_RUN_INTEGRATION=1 uv run pytest -q test_rag.py
```

## Security notes

- Never commit `.env`, API keys, passwords, or private documents.
- `.env.example` contains local development defaults, not production secrets.
- The current application is still under development and should not be treated as a fully hardened production service.
