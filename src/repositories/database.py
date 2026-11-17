"""Async database engine, session factory and schema bootstrap.

The engine owns the connection pool; every request gets its own
:class:`~sqlalchemy.ext.asyncio.AsyncSession` via :func:`get_session`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy import String, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.repositories.schema import Base
from src.settings import settings
from src.utils.exceptions import DatabaseError
from src.utils.logger import get_logger

logger = get_logger(__name__)

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped database session."""

    async with SessionFactory() as session:
        yield session


async def init_db() -> None:
    """Enable pgvector and create all tables idempotently.

    Called once at application startup. ``CREATE EXTENSION IF NOT EXISTS`` and
    ``metadata.create_all`` both no-op when the objects already exist.
    """

    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
            # ``create_all`` never alters existing tables, so add newer columns here.
            await conn.execute(
                text(
                    "ALTER TABLE answer_sheets "
                    "ADD COLUMN IF NOT EXISTS student_name VARCHAR(200)"
                )
            )
    except SQLAlchemyError as exc:
        logger.exception("Database initialisation failed")
        raise DatabaseError(
            "Could not initialise the database. Verify PostgreSQL is running, the "
            "database exists and DATABASE_URL is correct.",
            detail={"error": str(exc)},
        ) from exc

    logger.info("Database ready", extra={"tables": len(Base.metadata.tables)})


async def dispose_engine() -> None:
    """Dispose the connection pool on application shutdown."""

    await engine.dispose()
    logger.info("Database engine disposed")


async def check_ready() -> dict[str, bool]:
    """Probe the database, the pgvector extension and the expected tables.

    Returns a mapping used by the readiness endpoint; never raises.
    """

    expected = tuple(Base.metadata.tables.keys())
    state = {"database": False, "pgvector": False, "tables": False}

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            state["database"] = True

            extension = await conn.scalar(
                text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
            )
            state["pgvector"] = bool(extension)

            stmt = text(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = ANY(:names)"
            ).bindparams(bindparam("names", type_=ARRAY(String)))
            found = await conn.scalar(stmt, {"names": list(expected)})
            state["tables"] = bool(found) and found >= len(expected)
    except SQLAlchemyError as exc:
        logger.warning("Readiness check failed", extra={"error": str(exc)})

    return state
