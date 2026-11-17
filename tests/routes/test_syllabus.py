"""Tests for the syllabus routes (FR-001)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.dependencies import get_syllabus_service
from src.utils.exceptions import FileProcessingError, NotFoundError
from src.utils.exceptions import error_codes as code

client = TestClient(app)
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _syllabus_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "title": "Physics",
        "subject": None,
        "class_section": None,
        "file_path": "/storage/syllabus/abc.txt",
        "input_type": "typed",
        "ocr_status": "done",
        "status": "ready",
        "version": 1,
        "scope": ["1"],
        "chunk_count": 3,
        "created_at": _NOW.isoformat(),
        "updated_at": _NOW.isoformat(),
        "is_active": True,
    }
    payload.update(overrides)
    return payload


class _StubService:
    """Stub service returning preset values (or raising preset errors)."""

    def __init__(self, **results: object) -> None:
        self.results = results
        self.deleted: uuid.UUID | None = None

    @staticmethod
    def _resolve(value: object) -> object:
        if isinstance(value, Exception):
            raise value
        return value

    async def upload(self, **kwargs: object) -> object:
        return self._resolve(self.results["upload"])

    async def list(self) -> object:
        return self._resolve(self.results.get("list", []))

    async def get(self, syllabus_id: uuid.UUID) -> object:
        return self._resolve(self.results["get"])

    async def delete(self, syllabus_id: uuid.UUID) -> None:
        self.deleted = syllabus_id
        self._resolve(self.results.get("delete"))

    async def topics(self, syllabus_id: uuid.UUID) -> object:
        return self._resolve(self.results["topics"])


@pytest.fixture(autouse=True)
def _override_service() -> Iterator[None]:
    """Ensure dependency overrides never leak between tests."""

    yield
    app.dependency_overrides.clear()


def _use(service: _StubService) -> None:
    app.dependency_overrides[get_syllabus_service] = lambda: service


def test_upload_syllabus_returns_201() -> None:
    _use(_StubService(upload=_syllabus_payload()))

    response = client.post(
        "/api/v1/syllabus/upload",
        files={"file": ("syllabus.txt", b"Unit 1\n1.1 Topic", "text/plain")},
        data={"title": "Physics", "scope": "1, 2"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Physics"
    assert body["status"] == "ready"


def test_upload_syllabus_returns_400_on_bad_file() -> None:
    _use(_StubService(upload=FileProcessingError("Unsupported file type '.docx'.")))

    response = client.post(
        "/api/v1/syllabus/upload",
        files={"file": ("syllabus.docx", b"data", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json()["error"] == code.FILE_PROCESSING_ERROR


def test_list_syllabus_returns_items() -> None:
    _use(_StubService(list=[_syllabus_payload()]))

    response = client.get("/api/v1/syllabus")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["chunk_count"] == 3


def test_get_syllabus_returns_item() -> None:
    payload = _syllabus_payload()
    _use(_StubService(get=payload))

    response = client.get(f"/api/v1/syllabus/{payload['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == payload["id"]


def test_get_syllabus_not_found_returns_404() -> None:
    _use(_StubService(get=NotFoundError("missing", error_code=code.SYLLABUS_NOT_FOUND)))

    response = client.get(f"/api/v1/syllabus/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"] == code.SYLLABUS_NOT_FOUND


def test_get_syllabus_rejects_invalid_uuid() -> None:
    _use(_StubService(get=_syllabus_payload()))

    response = client.get("/api/v1/syllabus/not-a-uuid")

    assert response.status_code == 422
    assert response.json()["error"] == code.VALIDATION_ERROR


def test_delete_syllabus_returns_204() -> None:
    _use(_StubService())

    response = client.delete(f"/api/v1/syllabus/{uuid.uuid4()}")

    assert response.status_code == 204
    assert response.content == b""


def test_get_topics_returns_tree() -> None:
    _use(_StubService(topics=[{"unit_no": "1", "unit_name": "Mechanics", "topics": ["Friction"]}]))

    response = client.get(f"/api/v1/syllabus/{uuid.uuid4()}/topics")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["unit_no"] == "1"
    assert body[0]["topics"] == ["Friction"]
