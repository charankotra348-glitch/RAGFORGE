# app/main.py

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Response
from qdrant_client.http.exceptions import ResponseHandlingException
from sqlalchemy import text

from ragforge.core.config import get_settings
from ragforge.app.db.postgres import get_database
from ragforge.app.cache.redis import get_redis
from ragforge.app.retrieval.vector import get_qdrant
from ragforge.app.generation.service import answer_query
from ragforge.app.models.query import QueryRequest, QueryResponse, QuerySource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


DEPENDENCY_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    httpx.TransportError,
    ResponseHandlingException,
)


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
        "postgres": _check("PostgreSQL", app.state.database, _ping_postgres),
        "redis": _check("Redis", app.state.redis, lambda client: client.ping()),
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


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty or whitespace.",
        )

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
        raise HTTPException(
            status_code=503,
            detail="A required service (vector store or language model) is unavailable. Try again shortly.",
        ) from exc
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

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        sources=sources,
        abstained=getattr(result, "abstained", False),
        abstention_reason=getattr(result, "abstention_reason", None),
    )