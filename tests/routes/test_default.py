"""Tests for the system routes (root, health, readiness)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.routes import default

client = TestClient(app)


def test_root_returns_app_name_and_version() -> None:
    response = client.get("/api/v1/")

    assert response.status_code == 200
    body = response.json()
    assert body["name"]
    assert body["version"]


def test_health_reports_ok() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_correlation_id_is_echoed() -> None:
    response = client.get("/api/v1/health", headers={"X-Correlation-ID": "abc123"})

    assert response.headers["X-Correlation-ID"] == "abc123"


def test_correlation_id_is_generated_when_missing() -> None:
    response = client.get("/api/v1/health")

    assert response.headers["X-Correlation-ID"]


def test_ready_returns_200_when_dependencies_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _ready() -> dict[str, bool]:
        return {"database": True, "pgvector": True, "tables": True}

    monkeypatch.setattr(default, "check_ready", _ready)

    response = client.get("/api/v1/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["database"] is True


def test_ready_returns_503_when_dependency_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _not_ready() -> dict[str, bool]:
        return {"database": False, "pgvector": False, "tables": False}

    monkeypatch.setattr(default, "check_ready", _not_ready)

    response = client.get("/api/v1/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["database"] is False
