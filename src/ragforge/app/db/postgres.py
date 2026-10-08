import logging

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ragforge.core.config import Settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_database(settings: Settings) -> Engine:
    """
    Create (once) and return the SQLAlchemy engine for PostgreSQL.
    """

    global _engine, _SessionLocal

    if _engine is None:
        _engine = create_engine(
            settings.postgres_url,
            pool_pre_ping=True,
        )

        _SessionLocal = sessionmaker(
            bind=_engine,
            autoflush=False,
            autocommit=False,
        )

        logger.info("PostgreSQL engine created")

    return _engine


def get_session():
    """
    Yield a database session for FastAPI dependency injection.
    """

    if _SessionLocal is None:
        raise RuntimeError(
            "Database not initialized — call get_database() first"
        )

    session = _SessionLocal()

    try:
        yield session
    finally:
        session.close()