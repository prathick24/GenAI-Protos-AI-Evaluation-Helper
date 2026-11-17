"""System routes: root, liveness (health) and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from src.models import Health, Ready
from src.repositories.database import check_ready
from src.settings import APP_NAME, APP_VERSION

router = APIRouter(tags=["System"])


@router.get("/")
async def root() -> dict[str, str]:
    """Return the application name and version."""

    return {"name": APP_NAME, "version": APP_VERSION}


@router.get("/health", response_model=Health)
async def health() -> Health:
    """Liveness probe: the process is up and serving requests."""

    return Health(status="ok", version=APP_VERSION)


@router.get("/ready", response_model=Ready)
async def ready(response: Response) -> Ready:
    """Readiness probe: verify the database, pgvector and expected tables."""

    state = await check_ready()
    is_ready = all(state.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return Ready(status="ready" if is_ready else "not_ready", **state)
