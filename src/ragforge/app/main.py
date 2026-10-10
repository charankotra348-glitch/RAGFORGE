# app/main.py

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from sqlalchemy import text

from ragforge.core.config import get_settings
from ragforge.app.db.postgres import get_database
from ragforge.app.cache.redis import get_redis
from ragforge.app.retrieval.vector import get_qdrant
from ragforge.app.cache.answers import (
    answer_cache_key,
    bump_corpus_version,
    config_fingerprint,
    get_cached_answer,
    get_corpus_version,
    set_cached_answer,
)
from ragforge.app.generation.prompts import SYSTEM_PROMPT
from ragforge.app.generation.service import answer_query, clear_hybrid_searcher_cache
from ragforge.app.models.query import QueryRequest, QueryResponse, QuerySource
from ragforge.app.documents import router as documents_router
from ragforge.app.errors import DEPENDENCY_ERRORS, UNAVAILABLE_DETAIL, dependency_unavailable
from ragforge.app.middleware import RequestContextMiddleware
from ragforge.app.rate_limit import enforce_rate_limit
from ragforge.core.logging import configure_logging

settings = get_settings()

configure_logging(settings.log_format, settings.log_level)
logger = logging.getLogger(__name__)




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create service clients at startup.

    A dependency that is down does not stop the API from starting: /health
    reports it as down and /query returns 503 until it is reachable. Postgres
    and Redis clients connect lazily; Qdrant is contacted at startup and
    retried on the next query if that fails.
    """
    logger.info("Starting RAGForge API...")
    logger.info("Environment: %s", settings.environment)

    app.state.database = _start("PostgreSQL", get_database)
    app.state.redis = _start("Redis", get_redis)
    app.state.qdrant = _start("Qdrant", get_qdrant)
    app.state.corpus_version = None

    logger.info("RAGForge API ready")

    yield

    logger.info("Shutting down RAGForge API...")
    _stop("PostgreSQL", app.state.database, "dispose")
    _stop("Redis", app.state.redis, "close")
    _stop("Qdrant", app.state.qdrant, "close")
    logger.info("RAGForge shutdown complete")


def _start(name: str, factory):
    try:
        client = factory(settings)
    except Exception:
        logger.exception("%s unavailable at startup; continuing in degraded mode", name)
        return None
    logger.info("%s initialized", name)
    return client


def _stop(name: str, client, method: str) -> None:
    if client is None:
        return
    try:
        getattr(client, method)()
        logger.info("%s connection closed", name)
    except Exception:
        logger.exception("Error while closing %s", name)


app = FastAPI(
    title=settings.app_name,
    description=(
        "Production-grade Retrieval-Augmented Generation "
        "system with hybrid retrieval, reranking, "
        "evaluation, observability, and grounded generation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(RequestContextMiddleware)
app.include_router(documents_router)
for _error in DEPENDENCY_ERRORS:
    app.add_exception_handler(_error, dependency_unavailable)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} API is running", "version": app.version}


@app.get("/health")
def health_check(response: Response) -> dict:
    """Readiness check. Returns 503 when any dependency is down.

    Error details are logged, not returned: exception text can expose hosts,
    ports and credentials.
    """
    checks = {
        "postgres": _check("PostgreSQL", getattr(app.state, "database", None), _ping_postgres),
        "redis": _check("Redis", getattr(app.state, "redis", None), lambda client: client.ping()),
        "qdrant": _check("Qdrant", _current_qdrant(), lambda client: client.get_collections()),
    }
    healthy = all(value == "ok" for value in checks.values())
    if not healthy:
        response.status_code = 503
    return {"status": "healthy" if healthy else "degraded", "checks": checks}


def _ping_postgres(engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def _current_qdrant():
    """The startup client, or a new attempt if Qdrant was down at startup."""
    if getattr(app.state, "qdrant", None) is None:
        try:
            app.state.qdrant = get_qdrant(settings)
        except Exception:
            logger.warning("Qdrant still unavailable", exc_info=True)
            return None
    return app.state.qdrant


def _check(name: str, client, probe) -> str:
    if client is None:
        return "down"
    try:
        probe(client)
    except Exception:
        logger.warning("%s health check failed", name, exc_info=True)
        return "down"
    return "ok"


def _answer_fingerprint() -> str:
    """Everything besides the question and top_k that changes an answer."""
    return config_fingerprint(
        settings.ollama_model,
        SYSTEM_PROMPT,
        settings.grounding_verifier_enabled,
        settings.qdrant_collection,
        settings.embedding_model,
    )


def _refresh_searcher_if_corpus_changed(corpus_version: str | None) -> None:
    """Rebuild the in-memory BM25 index after another process changed the indexed documents."""
    if corpus_version is None:
        return
    seen = getattr(app.state, "corpus_version", None)
    if seen is not None and seen != corpus_version:
        logger.info("Corpus changed (version %s -> %s); rebuilding the search index", seen, corpus_version)
        clear_hybrid_searcher_cache()
    app.state.corpus_version = corpus_version


@app.post("/query", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    http_request: Request,
    key_id: str = Depends(enforce_rate_limit),
) -> QueryResponse:
    query = request.query.strip()
    log_fields = http_request.state.log_fields
    log_fields.update({"top_k": request.top_k, "query_chars": len(query)})

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty or whitespace.",
        )

    redis_client = getattr(app.state, "redis", None)
    corpus_version = get_corpus_version(redis_client)
    _refresh_searcher_if_corpus_changed(corpus_version)

    cache_key = None
    if corpus_version is not None and settings.answer_cache_ttl_seconds > 0:
        cache_key = answer_cache_key(corpus_version, _answer_fingerprint(), request.top_k, query)
        cached = get_cached_answer(redis_client, cache_key)
        if cached is not None:
            log_fields.update(
                {
                    "cache": "hit",
                    "abstained": cached.get("abstained"),
                    "abstention_reason": cached.get("abstention_reason"),
                    "sources": len(cached.get("sources", [])),
                }
            )
            return QueryResponse(query=query, cached=True, **cached)
    log_fields["cache"] = "miss" if cache_key else "bypass"

    try:
        result = answer_query(
            settings=settings,
            query=query,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DEPENDENCY_ERRORS as exc:
        logger.warning("RAG dependency unavailable: %r", exc)
        raise HTTPException(status_code=503, detail=UNAVAILABLE_DETAIL) from exc
    except RuntimeError as exc:
        logger.warning("RAG query unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="RAG service is not ready. Check document indexing and configuration.",
        ) from exc
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(
            status_code=502,
            detail="Retrieval or answer generation failed.",
        ) from exc

    sources = [
        QuerySource(
            citation=f"[Source {index}]",
            page=str(source.metadata.get("page_number", "unknown")),
            section=str(source.metadata.get("section_number", "unknown")),
            text=source.text,
        )
        for index, source in enumerate(result.sources, start=1)
    ]

    response = QueryResponse(
        query=result.query,
        answer=result.answer,
        sources=sources,
        abstained=getattr(result, "abstained", False),
        abstention_reason=getattr(result, "abstention_reason", None),
    )
    if cache_key:
        set_cached_answer(
            redis_client,
            cache_key,
            response.model_dump(exclude={"query", "cached"}),
            settings.answer_cache_ttl_seconds,
        )
    log_fields.update(
        {
            "abstained": response.abstained,
            "abstention_reason": response.abstention_reason,
            "sources": len(sources),
            **{f"{name}_ms": value for name, value in (getattr(result, "timings_ms", None) or {}).items()},
        }
    )
    return response