# app/main.py

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from ragforge.core.config import get_settings
from ragforge.app.db.postgres import get_database
from ragforge.app.cache.redis import get_redis
from ragforge.app.retrieval.vector import get_qdrant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAGForge API...")
    logger.info("Environment: %s", settings.environment)

    try:
        app.state.database = get_database(settings)
        logger.info("PostgreSQL initialized")
    except Exception:
        logger.exception("Failed to initialize PostgreSQL")
        raise

    try:
        app.state.redis = get_redis(settings)
        logger.info("Redis initialized")
    except Exception:
        logger.exception("Failed to initialize Redis")
        raise

    try:
        app.state.qdrant = get_qdrant(settings)
        logger.info("Qdrant initialized")
    except Exception:
        logger.exception("Failed to initialize Qdrant")
        raise

    logger.info("RAGForge API ready")

    yield

    logger.info("Shutting down RAGForge API...")

    try:
        app.state.database.dispose()
        logger.info("PostgreSQL connection closed")
    except Exception:
        logger.exception("Error while closing PostgreSQL")

    try:
        app.state.redis.close()
        logger.info("Redis connection closed")
    except Exception:
        logger.exception("Error while closing Redis")

    try:
        app.state.qdrant.close()
        logger.info("Qdrant connection closed")
    except Exception:
        logger.exception("Error while closing Qdrant")

    logger.info("RAGForge shutdown complete")


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
def health_check() -> dict:
    checks = {}

    try:
        with app.state.database.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"down: {e}"

    try:
        app.state.redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"down: {e}"

    try:
        app.state.qdrant.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"down: {e}"

    status = "healthy" if all(v == "ok" for v in checks.values()) else "degraded"

    return {"status": status, "checks": checks}