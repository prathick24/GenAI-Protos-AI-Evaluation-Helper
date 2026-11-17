"""Tests for the teacher mark-override route (FR-009)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.dependencies import get_grading_service
from src.utils.exceptions import ConflictError, NotFoundError, RequestValidationError
from src.utils.exceptions import error_codes as code

client = TestClient(app)


def _mark_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "answer_id": str(uuid.uuid4()),
        "awarded_marks": 4.0,
        "max_marks": 5.0,
        "rubric_scores": [],
        "reasoning": "Teacher override: Rechecked the working.",
        "is_overridden": True,
        "original_marks": 3.0,
        "graded_by": "teacher",
    }
    payload.update(overrides)
    return payload


class _StubGradingService:
    """Stub grading service returning a preset mark (or raising a preset error)."""

    def __init__(self, result: object) -> None:
        self.result = result

    async def override(self, mark_id: uuid.UUID, payload: object) -> object:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture(autouse=True)
def _override_services() -> Iterator[None]:
    """Ensure dependency overrides never leak between tests."""

    yield
    app.dependency_overrides.clear()


def _use_grading(service: _StubGradingService) -> None:
    app.dependency_overrides[get_grading_service] = lambda: service


def test_override_returns_the_amended_mark() -> None:
    payload = _mark_payload()
    _use_grading(_StubGradingService(payload))

    response = client.patch(
        f"/api/v1/marks/{payload['id']}/override",
        json={"awarded_marks": 4, "reason": "Rechecked the working."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["awarded_marks"] == 4.0
    assert body["original_marks"] == 3.0
    assert body["is_overridden"] is True
    assert body["graded_by"] == "teacher"


def test_override_rejects_a_negative_mark() -> None:
    _use_grading(_StubGradingService(_mark_payload()))

    response = client.patch(
        f"/api/v1/marks/{uuid.uuid4()}/override", json={"awarded_marks": -1}
    )

    assert response.status_code == 422
    assert response.json()["error"] == code.VALIDATION_ERROR


def test_override_requires_awarded_marks() -> None:
    _use_grading(_StubGradingService(_mark_payload()))

    response = client.patch(
        f"/api/v1/marks/{uuid.uuid4()}/override", json={"reason": "typo"}
    )

    assert response.status_code == 422
    assert response.json()["error"] == code.VALIDATION_ERROR


def test_override_over_the_maximum_returns_422() -> None:
    _use_grading(
        _StubGradingService(
            RequestValidationError("too high", error_code=code.VALIDATION_ERROR)
        )
    )

    response = client.patch(
        f"/api/v1/marks/{uuid.uuid4()}/override", json={"awarded_marks": 9}
    )

    assert response.status_code == 422
    assert response.json()["error"] == code.VALIDATION_ERROR


def test_override_not_found_returns_404() -> None:
    _use_grading(
        _StubGradingService(NotFoundError("missing", error_code=code.MARK_NOT_FOUND))
    )

    response = client.patch(
        f"/api/v1/marks/{uuid.uuid4()}/override", json={"awarded_marks": 2}
    )

    assert response.status_code == 404
    assert response.json()["error"] == code.MARK_NOT_FOUND


def test_override_approved_sheet_returns_409() -> None:
    _use_grading(
        _StubGradingService(ConflictError("approved", error_code=code.CONFLICT))
    )

    response = client.patch(
        f"/api/v1/marks/{uuid.uuid4()}/override", json={"awarded_marks": 2}
    )

    assert response.status_code == 409
    assert response.json()["error"] == code.CONFLICT
