"""Tests for the grounded Q&A assistant service (FR-010)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.agents import assistant_agent as agent_module
from src.models.chat import ChatAskRequest
from src.repositories.schema import (
    Conversation,
    ConversationMessage,
    Exam,
    Question,
    QuestionPaper,
    Syllabus,
    SyllabusChunk,
)
from src.services import assistant_service
from src.services.assistant_service import AssistantService
from src.settings import MessageRole
from src.utils.exceptions import NotFoundError
from src.utils.exceptions import error_codes as code

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _FakeSession:
    """Minimal AsyncSession stand-in that stamps audit fields and records writes."""

    def __init__(self) -> None:
        self.added: list[object] = []
        self.commits = 0
        self.flushes = 0

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        self.flushes += 1
        for item in self.added:
            _apply_defaults(item)

    async def commit(self) -> None:
        self.commits += 1


def _apply_defaults(row: object) -> None:
    """Emulate the column defaults SQLAlchemy applies at INSERT time."""

    if getattr(row, "id", None) is None:
        row.id = uuid.uuid4()
    for attr, value in (("created_at", _NOW), ("updated_at", _NOW), ("is_active", True)):
        if hasattr(row, attr) and getattr(row, attr) is None:
            setattr(row, attr, value)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the deterministic offline answer."""

    monkeypatch.setattr(agent_module.llm, "is_available", lambda: False)


def _question(text: str, topic: str) -> Question:
    question = Question(
        id=uuid.uuid4(),
        question_paper_id=uuid.uuid4(),
        order_index=0,
        question_text=text,
        question_type="short",
        marks=Decimal("5"),
        topic=topic,
        created_at=_NOW,
        updated_at=_NOW,
        is_active=True,
    )
    question.rubric_criteria = []
    return question


def _chunk(text: str, topic: str) -> SyllabusChunk:
    return SyllabusChunk(
        id=uuid.uuid4(),
        syllabus_id=uuid.uuid4(),
        unit_no="1",
        unit_name="Mechanics",
        topic_name=topic,
        chunk_index=0,
        raw_text=text,
        created_at=_NOW,
        updated_at=_NOW,
        is_active=True,
    )


def _paper(questions: list[Question], syllabus_id: uuid.UUID) -> QuestionPaper:
    paper = QuestionPaper(
        id=uuid.uuid4(),
        syllabus_id=syllabus_id,
        title="Midterm",
        total_marks=5,
        created_at=_NOW,
        updated_at=_NOW,
        is_active=True,
    )
    paper.questions = questions
    return paper


def _exam(paper: QuestionPaper) -> Exam:
    exam = Exam(
        id=uuid.uuid4(),
        title="Midterm exam",
        total_marks=5,
        status="draft",
        question_paper_id=paper.id,
        created_at=_NOW,
        updated_at=_NOW,
        is_active=True,
    )
    exam.question_paper = paper
    return exam


def _patch_chunks(
    monkeypatch: pytest.MonkeyPatch, chunks: list[SyllabusChunk]
) -> None:
    async def _list_chunks(session: object, syllabus_id: object) -> list[SyllabusChunk]:
        return chunks

    monkeypatch.setattr(assistant_service.syllabus_repo, "list_chunks", _list_chunks)


async def test_ask_answers_from_the_exam_and_persists_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk("Newton's second law: force equals mass times acceleration.", "Mechanics")
    question = _question("State Newton's second law.", "Mechanics")
    paper = _paper([question], chunk.syllabus_id)
    exam = _exam(paper)
    _patch_chunks(monkeypatch, [chunk])

    async def _get_exam(session: object, exam_id: object) -> Exam:
        return exam

    monkeypatch.setattr(assistant_service.exam_repo, "get", _get_exam)

    session = _FakeSession()
    result = await AssistantService(session).ask(
        ChatAskRequest(question="What is Newton's second law?", exam_id=exam.id)
    )

    assert result.question == "What is Newton's second law?"
    assert "force equals mass times acceleration" in result.answer
    assert {citation.source_type for citation in result.citations} == {"syllabus", "question"}
    assert result.citations[0].source_id in {chunk.id, question.id}

    conversations = [item for item in session.added if isinstance(item, Conversation)]
    messages = [item for item in session.added if isinstance(item, ConversationMessage)]
    assert len(conversations) == 1
    assert conversations[0].exam_id == exam.id
    assert [message.role for message in messages] == [
        MessageRole.USER.value,
        MessageRole.ASSISTANT.value,
    ]
    assert messages[1].agent_name == "assistant"
    assert messages[1].tool_calls
    assert session.commits == 1


async def test_ask_without_an_exam_searches_all_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk("Photosynthesis converts light into chemical energy.", "Biology")
    syllabus = Syllabus(
        id=chunk.syllabus_id,
        title="Biology",
        created_at=_NOW,
        updated_at=_NOW,
        is_active=True,
    )
    question = _question("What is photosynthesis?", "Biology")
    paper = _paper([question], syllabus.id)

    async def _list_all(session: object) -> list[Syllabus]:
        return [syllabus]

    async def _list_papers(session: object) -> list[QuestionPaper]:
        return [paper]

    _patch_chunks(monkeypatch, [chunk])
    monkeypatch.setattr(assistant_service.syllabus_repo, "list_all", _list_all)
    monkeypatch.setattr(assistant_service.paper_repo, "list_papers", _list_papers)

    session = _FakeSession()
    result = await AssistantService(session).ask(
        ChatAskRequest(question="What is photosynthesis?")
    )

    assert "chemical energy" in result.answer
    conversations = [item for item in session.added if isinstance(item, Conversation)]
    assert conversations[0].exam_id is None


async def test_ask_returns_a_scope_reply_when_nothing_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk("Solve quadratic equations by factoring.", "Algebra")
    syllabus = Syllabus(
        id=chunk.syllabus_id,
        title="Maths",
        created_at=_NOW,
        updated_at=_NOW,
        is_active=True,
    )

    async def _list_all(session: object) -> list[Syllabus]:
        return [syllabus]

    async def _list_papers(session: object) -> list[QuestionPaper]:
        return []

    _patch_chunks(monkeypatch, [chunk])
    monkeypatch.setattr(assistant_service.syllabus_repo, "list_all", _list_all)
    monkeypatch.setattr(assistant_service.paper_repo, "list_papers", _list_papers)

    result = await AssistantService(_FakeSession()).ask(
        ChatAskRequest(question="What is photosynthesis?")
    )

    assert "could not find" in result.answer
    assert "Algebra" in result.answer
    assert result.citations == []
    assert result.conversation_id is not None


async def test_ask_raises_when_the_exam_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _get_exam(session: object, exam_id: object) -> None:
        return None

    monkeypatch.setattr(assistant_service.exam_repo, "get", _get_exam)

    with pytest.raises(NotFoundError) as excinfo:
        await AssistantService(_FakeSession()).ask(
            ChatAskRequest(question="What is covered?", exam_id=uuid.uuid4())
        )

    assert excinfo.value.error_code == code.EXAM_NOT_FOUND


async def test_ask_resumes_an_existing_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk("Newton's second law: force equals mass times acceleration.", "Mechanics")
    conversation = Conversation(
        id=uuid.uuid4(),
        title="Earlier",
        created_at=_NOW,
        updated_at=_NOW,
        is_active=True,
    )

    async def _get_conversation(session: object, conversation_id: object) -> Conversation:
        return conversation

    async def _list_all(session: object) -> list[Syllabus]:
        return []

    async def _list_papers(session: object) -> list[QuestionPaper]:
        return []

    _patch_chunks(monkeypatch, [chunk])
    monkeypatch.setattr(assistant_service.syllabus_repo, "list_all", _list_all)
    monkeypatch.setattr(assistant_service.paper_repo, "list_papers", _list_papers)
    monkeypatch.setattr(
        assistant_service.chat_repo, "get_conversation", _get_conversation
    )

    session = _FakeSession()
    result = await AssistantService(session).ask(
        ChatAskRequest(
            question="What is Newton's second law?", conversation_id=conversation.id
        )
    )

    assert result.conversation_id == conversation.id
    assert not [item for item in session.added if isinstance(item, Conversation)]
    assert len([item for item in session.added if isinstance(item, ConversationMessage)]) == 2


async def test_ask_raises_when_the_conversation_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _get_conversation(session: object, conversation_id: object) -> None:
        return None

    monkeypatch.setattr(
        assistant_service.chat_repo, "get_conversation", _get_conversation
    )

    with pytest.raises(NotFoundError) as excinfo:
        await AssistantService(_FakeSession()).ask(
            ChatAskRequest(question="What is covered?", conversation_id=uuid.uuid4())
        )

    assert excinfo.value.error_code == code.NOT_FOUND
