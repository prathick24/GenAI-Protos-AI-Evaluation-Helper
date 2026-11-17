"""Tests for the question paper routes (FR-002)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models.paper import QuestionPaperRead
from src.services.dependencies import get_paper_service
from src.utils.exceptions import ConflictError, NotFoundError
from src.utils.exceptions import error_codes as code

client = TestClient(app)
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _paper_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "syllabus_id": str(uuid.uuid4()),
        "title": "Midterm",
        "subject": "Science",
        "class_section": "B",
        "duration_minutes": 60,
        "total_marks": 10,
        "version": 1,
        "source": "generated",
        "status": "draft",
        "blueprint": {"total_marks": 10},
        "created_at": _NOW.isoformat(),
        "updated_at": _NOW.isoformat(),
        "is_active": True,
        "questions": [
            {
                "id": str(uuid.uuid4()),
                "order_index": 0,
                "question_text": "State Newton's second law.",
                "question_type": "short",
                "marks": 10.0,
                "options": None,
                "correct_option": None,
                "model_answer": "F = ma",
                "topic": "Mechanics",
                "difficulty": "medium",
                "validation_status": "pending",
                "source_chunk_id": str(uuid.uuid4()),
                "rubric_criteria": [],
            }
        ],
    }
    payload.update(overrides)
    return payload


def _generate_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "syllabus_id": str(uuid.uuid4()),
        "title": "Midterm",
        "scope": ["1"],
        "blueprint": {
            "total_marks": 10,
            "duration_minutes": 60,
            "question_types": [{"type": "short", "count": 2, "marks_each": 5}],
        },
    }
    body.update(overrides)
    return body


def _validation_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "paper_id": str(uuid.uuid4()),
        "passed": True,
        "paper_status": "validated",
        "checks": [
            {
                "name": "marks_total",
                "passed": True,
                "summary": "Total marks match the blueprint.",
                "details": {"expected": 10.0, "actual": 10.0},
            }
        ],
        "questions": [
            {
                "question_id": str(uuid.uuid4()),
                "order_index": 0,
                "status": "passed",
                "issues": [],
            }
        ],
    }
    payload.update(overrides)
    return payload


def _generate_response(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "paper": _paper_payload(),
        "validation_report": _validation_payload(),
    }
    payload.update(overrides)
    return payload


class _StubService:
    """Stub service returning preset values (or raising preset errors)."""

    def __init__(self, **results: object) -> None:
        self.results = results

    @staticmethod
    def _resolve(value: object) -> object:
        if isinstance(value, Exception):
            raise value
        return value

    async def generate(self, payload: object) -> object:
        return self._resolve(self.results["generate"])

    async def list(self) -> object:
        return self._resolve(self.results.get("list", []))

    async def get(self, paper_id: uuid.UUID) -> object:
        return self._resolve(self.results["get"])

    async def approve(self, paper_id: uuid.UUID) -> object:
        return self._resolve(self.results["approve"])


@pytest.fixture(autouse=True)
def _override_service() -> Iterator[None]:
    """Ensure dependency overrides never leak between tests."""

    yield
    app.dependency_overrides.clear()


def _use(service: _StubService) -> None:
    app.dependency_overrides[get_paper_service] = lambda: service


def test_generate_returns_201() -> None:
    _use(_StubService(generate=_generate_response()))

    response = client.post("/api/v1/question-papers/generate", json=_generate_body())

    assert response.status_code == 201
    body = response.json()
    assert body["paper"]["title"] == "Midterm"
    assert body["paper"]["status"] == "draft"
    assert len(body["paper"]["questions"]) == 1
    assert body["validation_report"]["passed"] is True
    assert body["validation_report"]["paper_status"] == "validated"


def test_generate_rejects_blueprint_mismatch() -> None:
    _use(_StubService(generate=_generate_response()))
    body = _generate_body()
    body["blueprint"] = {
        "total_marks": 10,
        "question_types": [{"type": "short", "count": 1, "marks_each": 5}],
    }

    response = client.post("/api/v1/question-papers/generate", json=body)

    assert response.status_code == 422
    assert response.json()["error"] == code.VALIDATION_ERROR


def test_generate_returns_404_when_syllabus_missing() -> None:
    _use(_StubService(generate=NotFoundError("missing", error_code=code.SYLLABUS_NOT_FOUND)))

    response = client.post("/api/v1/question-papers/generate", json=_generate_body())

    assert response.status_code == 404
    assert response.json()["error"] == code.SYLLABUS_NOT_FOUND


def test_list_returns_items() -> None:
    _use(_StubService(list=[_paper_payload()]))

    response = client.get("/api/v1/question-papers")

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_get_returns_item() -> None:
    payload = _paper_payload()
    _use(_StubService(get=payload))

    response = client.get(f"/api/v1/question-papers/{payload['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == payload["id"]


def test_get_not_found_returns_404() -> None:
    _use(_StubService(get=NotFoundError("missing", error_code=code.QUESTION_PAPER_NOT_FOUND)))

    response = client.get(f"/api/v1/question-papers/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"] == code.QUESTION_PAPER_NOT_FOUND


def test_get_rejects_invalid_uuid() -> None:
    _use(_StubService(get=_paper_payload()))

    response = client.get("/api/v1/question-papers/not-a-uuid")

    assert response.status_code == 422
    assert response.json()["error"] == code.VALIDATION_ERROR


def test_approve_returns_paper() -> None:
    _use(_StubService(approve=_paper_payload(status="approved")))

    response = client.post(f"/api/v1/question-papers/{uuid.uuid4()}/approve")

    assert response.status_code == 200
    assert response.json()["status"] == "approved"


def test_approve_conflict_returns_409() -> None:
    _use(_StubService(approve=ConflictError("not validated", error_code=code.CONFLICT)))

    response = client.post(f"/api/v1/question-papers/{uuid.uuid4()}/approve")

    assert response.status_code == 409
    assert response.json()["error"] == code.CONFLICT


def _pdf_text(content: bytes) -> str:
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open(stream=content, filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in document)
    finally:
        document.close()


def test_export_student_pdf_contains_questions_without_answers() -> None:
    payload = _paper_payload()
    _use(_StubService(get=QuestionPaperRead.model_validate(payload)))

    response = client.get(f"/api/v1/question-papers/{payload['id']}/export.pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert 'filename="Midterm_student.pdf"' in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF")
    text = _pdf_text(response.content)
    assert "Section A" in text
    assert "State Newton" in text
    assert "Answer Key and Rubric" not in text


def test_export_teacher_pdf_includes_answer_key() -> None:
    payload = _paper_payload()
    _use(_StubService(get=QuestionPaperRead.model_validate(payload)))

    response = client.get(
        f"/api/v1/question-papers/{payload['id']}/export.pdf",
        params={"include_answers": "true"},
    )

    assert response.status_code == 200
    assert 'filename="Midterm_teacher.pdf"' in response.headers["content-disposition"]
    text = _pdf_text(response.content)
    assert "Section A" in text
    assert "Answer Key and Rubric" in text
    assert "F = ma" in text
