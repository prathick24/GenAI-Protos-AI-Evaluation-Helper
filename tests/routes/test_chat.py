"""Tests for the question paper chat route (FR-010)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.dependencies import get_assistant_service, get_paper_chat_service
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
        "duration_minutes": 90,
        "total_marks": 5,
        "version": 1,
        "source": "generated",
        "status": "validated",
        "blueprint": {"total_marks": 5},
        "created_at": _NOW.isoformat(),
        "updated_at": _NOW.isoformat(),
        "is_active": True,
        "questions": [
            {
                "id": str(uuid.uuid4()),
                "order_index": 0,
                "question_text": "State Newton's second law.",
                "question_type": "short",
                "marks": 5.0,
                "options": None,
                "correct_option": None,
                "model_answer": "F = ma",
                "topic": "Mechanics",
                "difficulty": "medium",
                "validation_status": "passed",
                "source_chunk_id": str(uuid.uuid4()),
                "rubric_criteria": [],
            }
        ],
    }
    payload.update(overrides)
    return payload


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
                "details": {"expected": 5.0, "actual": 5.0},
            }
        ],
        "questions": [],
    }
    payload.update(overrides)
    return payload


def _chat_response(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "conversation_id": str(uuid.uuid4()),
        "reply": "Removed the first question.",
        "applied": ["Removed 1 question(s)"],
        "paper": _paper_payload(),
        "validation_report": _validation_payload(),
    }
    payload.update(overrides)
    return payload


class _StubService:
    """Stub service returning a preset result (or raising a preset error)."""

    def __init__(self, result: object) -> None:
        self.result = result

    async def chat(self, paper_id: uuid.UUID, payload: object) -> object:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture(autouse=True)
def _override_service() -> Iterator[None]:
    """Ensure dependency overrides never leak between tests."""

    yield
    app.dependency_overrides.clear()


def _use(result: object) -> None:
    app.dependency_overrides[get_paper_chat_service] = lambda: _StubService(result)


def test_chat_returns_updated_paper() -> None:
    _use(_chat_response())

    response = client.post(
        f"/api/v1/question-papers/{uuid.uuid4()}/chat", json={"message": "drop the first one"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Removed the first question."
    assert body["applied"] == ["Removed 1 question(s)"]
    assert body["paper"]["total_marks"] == 5
    assert body["validation_report"]["passed"] is True


def test_chat_rejects_an_empty_message() -> None:
    _use(_chat_response())

    response = client.post(f"/api/v1/question-papers/{uuid.uuid4()}/chat", json={"message": ""})

    assert response.status_code == 422
    assert response.json()["error"] == code.VALIDATION_ERROR


def test_chat_rejects_an_invalid_paper_id() -> None:
    _use(_chat_response())

    response = client.post("/api/v1/question-papers/not-a-uuid/chat", json={"message": "hi"})

    assert response.status_code == 422
    assert response.json()["error"] == code.VALIDATION_ERROR


def test_chat_returns_404_when_paper_missing() -> None:
    _use(NotFoundError("missing", error_code=code.QUESTION_PAPER_NOT_FOUND))

    response = client.post(f"/api/v1/question-papers/{uuid.uuid4()}/chat", json={"message": "hi"})

    assert response.status_code == 404
    assert response.json()["error"] == code.QUESTION_PAPER_NOT_FOUND


def test_chat_returns_409_for_an_approved_paper() -> None:
    _use(ConflictError("approved", error_code=code.CONFLICT))

    response = client.post(f"/api/v1/question-papers/{uuid.uuid4()}/chat", json={"message": "hi"})

    assert response.status_code == 409
    assert response.json()["error"] == code.CONFLICT


def _ask_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "question": "What is Newton's second law?",
        "answer": "Force equals mass times acceleration. [1]",
        "citations": [
            {
                "source_type": "syllabus",
                "source_id": str(uuid.uuid4()),
                "unit_no": "1",
                "topic_name": "Mechanics",
                "snippet": "Newton's second law: F = ma",
            }
        ],
        "conversation_id": str(uuid.uuid4()),
    }
    payload.update(overrides)
    return payload


class _StubAssistantService:
    """Stub assistant service returning a preset result (or raising a preset error)."""

    def __init__(self, result: object) -> None:
        self.result = result

    async def ask(self, payload: object) -> object:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _use_assistant(result: object) -> None:
    app.dependency_overrides[get_assistant_service] = lambda: _StubAssistantService(result)


def test_ask_returns_grounded_answer_with_citations() -> None:
    payload = _ask_payload()
    _use_assistant(payload)

    response = client.post(
        "/api/v1/chat/ask", json={"question": "What is Newton's second law?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == payload["answer"]
    assert body["citations"][0]["source_type"] == "syllabus"
    assert body["conversation_id"] == payload["conversation_id"]


def test_ask_rejects_an_empty_question() -> None:
    _use_assistant(_ask_payload())

    response = client.post("/api/v1/chat/ask", json={"question": ""})

    assert response.status_code == 422
    assert response.json()["error"] == code.VALIDATION_ERROR


def test_ask_returns_404_when_exam_missing() -> None:
    _use_assistant(NotFoundError("missing", error_code=code.EXAM_NOT_FOUND))

    response = client.post(
        "/api/v1/chat/ask",
        json={"question": "What is covered?", "exam_id": str(uuid.uuid4())},
    )

    assert response.status_code == 404
    assert response.json()["error"] == code.EXAM_NOT_FOUND


def test_ask_returns_404_when_conversation_missing() -> None:
    _use_assistant(NotFoundError("missing", error_code=code.NOT_FOUND))

    response = client.post(
        "/api/v1/chat/ask",
        json={"question": "What is covered?", "conversation_id": str(uuid.uuid4())},
    )

    assert response.status_code == 404
    assert response.json()["error"] == code.NOT_FOUND
