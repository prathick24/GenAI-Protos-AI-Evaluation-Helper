"""System response models for the root, liveness and readiness endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class Health(BaseModel):
    """Liveness probe response."""

    status: str
    version: str


class Ready(BaseModel):
    """Readiness probe response describing the state of dependent subsystems."""

    status: str
    database: bool
    pgvector: bool
    tables: bool
