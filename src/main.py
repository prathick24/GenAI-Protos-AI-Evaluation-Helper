"""FastAPI application entrypoint.

Assembles the app: configures logging, initialises the database, registers
middleware, exception handlers and versioned routes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.middleware.correlation import CorrelationIdMiddleware
from src.middleware.exception_handlers import register_exception_handlers
from src.repositories.database import dispose_engine, init_db
from src.routes import answer_sheets, chat, default, exams, marks, papers, syllabus
from src.settings import API_V1_PREFIX, APP_NAME, APP_VERSION, settings
from src.utils.exceptions import DatabaseError
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and bootstrap the database for the app's lifetime."""

    configure_logging(settings.log_level)
    logger.info("Starting %s v%s", APP_NAME, APP_VERSION, extra={"env": settings.app_env})

    try:
        await init_db()
    except DatabaseError as exc:
        # The app still boots so /health works and /ready reports the fault.
        logger.warning("Database unavailable at startup", extra={"error": exc.message})

    try:
        yield
    finally:
        await dispose_engine()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""

    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(CorrelationIdMiddleware)
    register_exception_handlers(app)
    app.include_router(default.router, prefix=API_V1_PREFIX)
    app.include_router(syllabus.router, prefix=API_V1_PREFIX)
    app.include_router(papers.router, prefix=API_V1_PREFIX)
    app.include_router(chat.router, prefix=API_V1_PREFIX)
    app.include_router(chat.ask_router, prefix=API_V1_PREFIX)
    app.include_router(exams.router, prefix=API_V1_PREFIX)
    app.include_router(answer_sheets.router, prefix=API_V1_PREFIX)
    app.include_router(marks.router, prefix=API_V1_PREFIX)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host=settings.api_host, port=settings.api_port, reload=True)
