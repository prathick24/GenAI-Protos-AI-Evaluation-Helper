"""Tests for the answer sheet segmentation and grading routes (FR-005, FR-006)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.dependencies import get_grading_service
from src.utils.exceptions import ConflictError, NotFoundError
from src.utils.exceptions import error_codes as code

client = TestClient(app)
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _answer_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "answer_sheet_id": str(uuid.uuid4()),
        "question_id": str(uuid.uuid4()),
        "order_index": 0,
        "answer_text": "Force equals mass times acceleration.",
        "is_answered": True,
        "ocr_confidence": None,
        "mark": None,
    }
    payload.update(overrides)
    return payload


def _mark_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "answer_id": str(uuid.uuid4()),
        "awarded_marks": 3.0,
        "max_marks": 5.0,
        "rubric_scores": [],
        "is_overridden": False,
    }
    payload.update(overrides)
    return payload


def _segmentation_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "answer_sheet_id": str(uuid.uuid4()),
        "exam_id": str(uuid.uuid4()),
        "sheet_status": "uploaded",
        "total_questions": 2,
        "answered": 1,
        "unanswered": 1,
        "answers": [_answer_payload()],
        "unmatched_regions": [{"text": "Preamble", "reason": "Could not be aligned"}],
    }
    payload.update(overrides)
    return payload


def _grading_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "answer_sheet_id": str(uuid.uuid4()),
        "exam_id": str(uuid.uuid4()),
        "sheet_status": "graded",
        "total_awarded_marks": 3.0,
        "total_max_marks": 5.0,
        "marks": [_mark_payload()],
    }
    payload.update(overrides)
    return payload


def _detail_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "exam_id": str(uuid.uuid4()),
        "student_id": None,
        "student_name": "Asha",
        "file_path": "/storage/answer_sheets/sheet.txt",
        "input_type": "typed",
        "ocr_status": "done",
        "sheet_status": "graded",
        "total_awarded_marks": 3.0,
        "ocr_confidence": None,
        "created_at": _NOW.isoformat(),
        "updated_at": _NOW.isoformat(),
        "is_active": True,
        "answers": [_answer_payload()],
        "feedback": None,
    }
    payload.update(overrides)
    return payload


class _StubGradingService:
    """Stub grading service returning preset values (or raising preset errors)."""

    def __init__(self, **results: object) -> None:
        self.results = results

    @staticmethod
    def _resolve(value: object) -> object:
        if isinstance(value, Exception):
            raise value
        return value

    async def segment(self, sheet_id: uuid.UUID) -> object:
        return self._resolve(self.results["segment"])

    async def grade(self, sheet_id: uuid.UUID) -> object:
        return self._resolve(self.results["grade"])

    async def get_detail(self, sheet_id: uuid.UUID) -> object:
        return self._resolve(self.results["get_detail"])

    async def approve(self, sheet_id: uuid.UUID) -> object:
        return self._resolve(self.results["approve"])

    async def rename(self, sheet_id: uuid.UUID, payload: object) -> object:
        return self._resolve(self.results["rename"])


@pytest.fixture(autouse=True)
def _override_services() -> Iterator[None]:
    """Ensure dependency overrides never leak between tests."""

    yield
    app.dependency_overrides.clear()


def _use_grading(service: _StubGradingService) -> None:
    app.dependency_overrides[get_grading_service] = lambda: service


def test_segment_returns_200() -> None:
    _use_grading(_StubGradingService(segment=_segmentation_payload()))

    response = client.post(f"/api/v1/answer-sheets/{uuid.uuid4()}/segment")

    assert response.status_code == 200
    body = response.json()
    assert body["total_questions"] == 2
    assert body["answers"][0]["is_answered"] is True
    assert body["unmatched_regions"][0]["text"] == "Preamble"


def test_segment_not_found_returns_404() -> None:
    _use_grading(
        _StubGradingService(
            segment=NotFoundError("missing", error_code=code.ANSWER_SHEET_NOT_FOUND)
        )
    )

    response = client.post(f"/api/v1/answer-sheets/{uuid.uuid4()}/segment")

    assert response.status_code == 404
    assert response.json()["error"] == code.ANSWER_SHEET_NOT_FOUND


def test_grade_returns_200() -> None:
    _use_grading(_StubGradingService(grade=_grading_payload()))

    response = client.post(f"/api/v1/answer-sheets/{uuid.uuid4()}/grade")

    assert response.status_code == 200
    body = response.json()
    assert body["sheet_status"] == "graded"
    assert body["total_awarded_marks"] == 3.0
    assert body["marks"][0]["awarded_marks"] == 3.0


def test_grade_before_segmentation_returns_409() -> None:
    _use_grading(
        _StubGradingService(grade=ConflictError("not segmented", error_code=code.CONFLICT))
    )

    response = client.post(f"/api/v1/answer-sheets/{uuid.uuid4()}/grade")

    assert response.status_code == 409
    assert response.json()["error"] == code.CONFLICT


def test_rename_returns_200() -> None:
    payload = _detail_payload(student_name="Ravi")
    _use_grading(_StubGradingService(rename=payload))

    response = client.patch(
        f"/api/v1/answer-sheets/{payload['id']}", json={"student_name": "Ravi"}
    )

    assert response.status_code == 200
    assert response.json()["student_name"] == "Ravi"


def test_rename_rejects_an_overlong_name() -> None:
    _use_grading(_StubGradingService(rename=_detail_payload()))
    long_name = "x" * 201

    response = client.patch(
        f"/api/v1/answer-sheets/{uuid.uuid4()}", json={"student_name": long_name}
    )

    assert response.status_code == 422


def test_get_detail_returns_200() -> None:
    payload = _detail_payload()
    _use_grading(_StubGradingService(get_detail=payload))

    response = client.get(f"/api/v1/answer-sheets/{payload['id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == payload["id"]
    assert body["answers"][0]["order_index"] == 0
    assert body["feedback"] is None


def test_approve_returns_200() -> None:
    payload = _detail_payload(sheet_status="approved")
    _use_grading(_StubGradingService(approve=payload))

    response = client.post(f"/api/v1/answer-sheets/{payload['id']}/approve")

    assert response.status_code == 200
    assert response.json()["sheet_status"] == "approved"


def test_approve_ungraded_returns_409() -> None:
    _use_grading(
        _StubGradingService(
            approve=ConflictError("not graded", error_code=code.CONFLICT)
        )
    )

    response = client.post(f"/api/v1/answer-sheets/{uuid.uuid4()}/approve")

    assert response.status_code == 409
    assert response.json()["error"] == code.CONFLICT


def test_approve_not_found_returns_404() -> None:
    _use_grading(
        _StubGradingService(
            approve=NotFoundError("missing", error_code=code.ANSWER_SHEET_NOT_FOUND)
        )
    )

    response = client.post(f"/api/v1/answer-sheets/{uuid.uuid4()}/approve")

    assert response.status_code == 404
    assert response.json()["error"] == code.ANSWER_SHEET_NOT_FOUND
