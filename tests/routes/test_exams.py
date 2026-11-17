"""Tests for the exam and answer sheet routes (FR-004)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.dependencies import get_answer_sheet_service, get_exam_service
from src.utils.exceptions import ConflictError, NotFoundError
from src.utils.exceptions import error_codes as code

client = TestClient(app)
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _exam_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "title": "Midterm exam",
        "subject": "Science",
        "class_section": "B",
        "exam_date": "2026-03-01",
        "total_marks": 50,
        "status": "draft",
        "question_paper_id": str(uuid.uuid4()),
        "created_at": _NOW.isoformat(),
        "updated_at": _NOW.isoformat(),
        "is_active": True,
    }
    payload.update(overrides)
    return payload


def _sheet_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "exam_id": str(uuid.uuid4()),
        "student_id": None,
        "file_path": "/storage/answer_sheets/sheet.txt",
        "input_type": "typed",
        "ocr_status": "done",
        "sheet_status": "uploaded",
        "total_awarded_marks": None,
        "ocr_confidence": None,
        "created_at": _NOW.isoformat(),
        "updated_at": _NOW.isoformat(),
        "is_active": True,
    }
    payload.update(overrides)
    return payload


class _StubExamService:
    """Stub exam service returning preset values (or raising preset errors)."""

    def __init__(self, **results: object) -> None:
        self.results = results

    @staticmethod
    def _resolve(value: object) -> object:
        if isinstance(value, Exception):
            raise value
        return value

    async def create(self, payload: object) -> object:
        return self._resolve(self.results["create"])

    async def list(self) -> object:
        return self._resolve(self.results.get("list", []))

    async def get(self, exam_id: uuid.UUID) -> object:
        return self._resolve(self.results["get"])

    async def marksheet(self, exam_id: uuid.UUID) -> object:
        return self._resolve(self.results["marksheet"])

    async def analytics(self, exam_id: uuid.UUID) -> object:
        return self._resolve(self.results["analytics"])


class _StubAnswerSheetService:
    """Stub answer sheet service returning preset values (or raising errors)."""

    def __init__(self, **results: object) -> None:
        self.results = results

    @staticmethod
    def _resolve(value: object) -> object:
        if isinstance(value, Exception):
            raise value
        return value

    async def upload(
        self,
        exam_id: uuid.UUID,
        *,
        files: list[object],
        student_id: uuid.UUID | None = None,
        student_name: str | None = None,
        input_type: object = None,
    ) -> object:
        return self._resolve(self.results["upload"])

    async def list(self, exam_id: uuid.UUID) -> object:
        return self._resolve(self.results.get("list", []))


@pytest.fixture(autouse=True)
def _override_services() -> Iterator[None]:
    """Ensure dependency overrides never leak between tests."""

    yield
    app.dependency_overrides.clear()


def _use_exam(service: _StubExamService) -> None:
    app.dependency_overrides[get_exam_service] = lambda: service


def _use_sheets(service: _StubAnswerSheetService) -> None:
    app.dependency_overrides[get_answer_sheet_service] = lambda: service


def test_create_exam_returns_201() -> None:
    _use_exam(_StubExamService(create=_exam_payload()))

    response = client.post(
        "/api/v1/exams",
        json={"title": "Midterm exam", "question_paper_id": str(uuid.uuid4())},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Midterm exam"
    assert body["status"] == "draft"
    assert body["total_marks"] == 50


def test_create_exam_rejects_missing_paper_id() -> None:
    _use_exam(_StubExamService(create=_exam_payload()))

    response = client.post("/api/v1/exams", json={"title": "Midterm exam"})

    assert response.status_code == 422
    assert response.json()["error"] == code.VALIDATION_ERROR


def test_create_exam_returns_404_for_cold_paper() -> None:
    _use_exam(
        _StubExamService(
            create=NotFoundError("missing", error_code=code.QUESTION_PAPER_NOT_FOUND)
        )
    )

    response = client.post(
        "/api/v1/exams",
        json={"title": "Midterm exam", "question_paper_id": str(uuid.uuid4())},
    )

    assert response.status_code == 404
    assert response.json()["error"] == code.QUESTION_PAPER_NOT_FOUND


def test_create_exam_returns_409_for_unapproved_paper() -> None:
    _use_exam(_StubExamService(create=ConflictError("not approved", error_code=code.CONFLICT)))

    response = client.post(
        "/api/v1/exams",
        json={"title": "Midterm exam", "question_paper_id": str(uuid.uuid4())},
    )

    assert response.status_code == 409
    assert response.json()["error"] == code.CONFLICT


def test_list_exams_returns_items() -> None:
    _use_exam(_StubExamService(list=[_exam_payload()]))

    response = client.get("/api/v1/exams")

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_get_exam_returns_item() -> None:
    payload = _exam_payload()
    _use_exam(_StubExamService(get=payload))

    response = client.get(f"/api/v1/exams/{payload['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == payload["id"]


def test_get_exam_not_found_returns_404() -> None:
    _use_exam(_StubExamService(get=NotFoundError("missing", error_code=code.EXAM_NOT_FOUND)))

    response = client.get(f"/api/v1/exams/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"] == code.EXAM_NOT_FOUND


def test_upload_answer_sheets_returns_207() -> None:
    _use_sheets(
        _StubAnswerSheetService(
            upload=[
                {"file": "sheet.txt", "status": "uploaded", "answer_sheet_id": str(uuid.uuid4())},
                {"file": "scan.png", "status": "error", "reason": "OCR failed"},
            ]
        )
    )

    response = client.post(
        f"/api/v1/exams/{uuid.uuid4()}/answer-sheets/upload",
        files=[
            ("files", ("sheet.txt", b"F = m a", "text/plain")),
            ("files", ("scan.png", b"\x89PNG", "image/png")),
        ],
        data={"input_type": "auto", "student_name": "Asha"},
    )

    assert response.status_code == 207
    body = response.json()
    assert [item["status"] for item in body] == ["uploaded", "error"]
    assert body[1]["reason"] == "OCR failed"


def test_upload_answer_sheets_returns_404_for_missing_exam() -> None:
    _use_sheets(
        _StubAnswerSheetService(upload=NotFoundError("missing", error_code=code.EXAM_NOT_FOUND))
    )

    response = client.post(
        f"/api/v1/exams/{uuid.uuid4()}/answer-sheets/upload",
        files=[("files", ("sheet.txt", b"F = m a", "text/plain"))],
    )

    assert response.status_code == 404
    assert response.json()["error"] == code.EXAM_NOT_FOUND


def test_list_answer_sheets_returns_items() -> None:
    _use_sheets(_StubAnswerSheetService(list=[_sheet_payload()]))

    response = client.get(f"/api/v1/exams/{uuid.uuid4()}/answer-sheets")

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_upload_rejects_invalid_input_type() -> None:
    _use_sheets(_StubAnswerSheetService(upload=[]))

    response = client.post(
        f"/api/v1/exams/{uuid.uuid4()}/answer-sheets/upload",
        files=[("files", ("sheet.txt", b"F = m a", "text/plain"))],
        data={"input_type": "banana"},
    )

    assert response.status_code == 422
    assert response.json()["error"] == code.VALIDATION_ERROR


def _marksheet_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "exam_id": str(uuid.uuid4()),
        "total_marks": 10,
        "rows": [
            {
                "answer_sheet_id": str(uuid.uuid4()),
                "student_id": None,
                "student_name": "Asha",
                "total_awarded_marks": 7.0,
                "per_question": {str(uuid.uuid4()): 5.0},
            }
        ],
    }
    payload.update(overrides)
    return payload


def _analytics_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "exam_id": str(uuid.uuid4()),
        "average": 7.5,
        "highest": 8.0,
        "lowest": 7.0,
        "distribution": {
            "0-20": 0,
            "20-40": 0,
            "40-60": 0,
            "60-80": 1,
            "80-100": 1,
        },
        "per_topic_strength": {"Algebra": 0.8},
        "hardest_questions": [str(uuid.uuid4())],
        "easiest_questions": [str(uuid.uuid4())],
    }
    payload.update(overrides)
    return payload


def test_get_marksheet_returns_rows() -> None:
    payload = _marksheet_payload()
    _use_exam(_StubExamService(marksheet=payload))

    response = client.get(f"/api/v1/exams/{payload['exam_id']}/marksheet")

    assert response.status_code == 200
    body = response.json()
    assert body["total_marks"] == 10
    assert body["rows"][0]["student_name"] == "Asha"
    assert body["rows"][0]["total_awarded_marks"] == 7.0


def test_get_marksheet_not_found_returns_404() -> None:
    _use_exam(
        _StubExamService(
            marksheet=NotFoundError("missing", error_code=code.EXAM_NOT_FOUND)
        )
    )

    response = client.get(f"/api/v1/exams/{uuid.uuid4()}/marksheet")

    assert response.status_code == 404
    assert response.json()["error"] == code.EXAM_NOT_FOUND


def test_get_analytics_returns_summary() -> None:
    payload = _analytics_payload()
    _use_exam(_StubExamService(analytics=payload))

    response = client.get(f"/api/v1/exams/{payload['exam_id']}/analytics")

    assert response.status_code == 200
    body = response.json()
    assert body["average"] == 7.5
    assert body["distribution"]["80-100"] == 1
    assert body["per_topic_strength"]["Algebra"] == 0.8


def test_get_analytics_not_found_returns_404() -> None:
    _use_exam(
        _StubExamService(
            analytics=NotFoundError("missing", error_code=code.EXAM_NOT_FOUND)
        )
    )

    response = client.get(f"/api/v1/exams/{uuid.uuid4()}/analytics")

    assert response.status_code == 404
    assert response.json()["error"] == code.EXAM_NOT_FOUND
