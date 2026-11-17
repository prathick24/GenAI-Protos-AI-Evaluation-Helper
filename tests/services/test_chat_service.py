"""Tests for the chat-driven paper change service (FR-010)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.agents.paper_generation_agent import GeneratedQuestion
from src.client import llm
from src.models.chat import ChangeOperation, ChangePlan, PaperChatRequest
from src.repositories.schema import ConversationMessage, Question, QuestionPaper, SyllabusChunk
from src.services import chat_service, paper_service
from src.services.chat_service import PaperChatService
from src.settings import (
    ChangeOp,
    Difficulty,
    PaperStatus,
    QuestionType,
    ValidationStatus,
)
from src.utils.exceptions import ConflictError, NotFoundError
from src.utils.exceptions import error_codes as code

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _FakeSession:
    """Minimal AsyncSession stand-in: records commits/flushes and assigns ids."""

    def __init__(self, paper: QuestionPaper | None = None) -> None:
        self.paper = paper
        self.commits = 0
        self.flushes = 0
        self.added: list[object] = []

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def add_all(self, objs: list[object]) -> None:
        self.added.extend(objs)

    async def flush(self) -> None:
        self.flushes += 1
        self._identify(self.added)
        if self.paper is not None:
            self._identify(self.paper.questions)

    async def commit(self) -> None:
        self.commits += 1

    @staticmethod
    def _identify(objs: list[object]) -> None:
        for obj in objs:
            if getattr(obj, "id", "missing") is None:
                obj.id = uuid.uuid4()
            for criterion in getattr(obj, "rubric_criteria", None) or []:
                if criterion.id is None:
                    criterion.id = uuid.uuid4()


def _chunk(**overrides: object) -> SyllabusChunk:
    defaults: dict[str, object] = {
        "id": uuid.uuid4(),
        "syllabus_id": uuid.uuid4(),
        "chunk_index": 0,
        "raw_text": "Force equals mass times acceleration.",
        "unit_no": "1",
        "unit_name": "Mechanics",
        "topic_name": "Newton's Laws",
    }
    defaults.update(overrides)
    return SyllabusChunk(**defaults)


def _question(paper_id: uuid.UUID, **overrides: object) -> Question:
    defaults: dict[str, object] = {
        "id": uuid.uuid4(),
        "question_paper_id": paper_id,
        "source_chunk_id": None,
        "order_index": 0,
        "question_text": "State Newton's second law of motion.",
        "question_type": QuestionType.SHORT.value,
        "marks": Decimal(5),
        "options": None,
        "correct_option": None,
        "model_answer": "F = ma",
        "topic": "Mechanics",
        "difficulty": Difficulty.MEDIUM.value,
        "validation_status": ValidationStatus.PENDING.value,
    }
    defaults.update(overrides)
    return Question(**defaults)


def _blueprint_dict(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "total_marks": 10,
        "duration_minutes": 60,
        "question_types": [{"type": "short", "count": 2, "marks_each": 5}],
        "difficulty_mix": None,
        "topic_weightage": None,
        "scope": ["1"],
    }
    payload.update(overrides)
    return payload


def _paper(chunk: SyllabusChunk, **overrides: object) -> QuestionPaper:
    paper_id = overrides.pop("id", uuid.uuid4())
    defaults: dict[str, object] = {
        "id": paper_id,
        "created_at": _NOW,
        "updated_at": _NOW,
        "version": 1,
        "is_active": True,
        "title": "Midterm",
        "subject": "Science",
        "syllabus_id": chunk.syllabus_id,
        "total_marks": 10,
        "duration_minutes": 60,
        "source": "generated",
        "status": PaperStatus.DRAFT.value,
        "blueprint": _blueprint_dict(),
        "questions": [
            _question(
                paper_id,
                order_index=0,
                source_chunk_id=chunk.id,
                question_text="State Newton's second law of motion.",
            ),
            _question(
                paper_id,
                order_index=1,
                source_chunk_id=chunk.id,
                question_text="Define the SI unit of force.",
            ),
        ],
    }
    defaults.update(overrides)
    return QuestionPaper(**defaults)


class _StubRouter:
    """Chat router stand-in returning a preset plan."""

    def __init__(self, plan: ChangePlan | None = None) -> None:
        self._plan = plan or ChangePlan(reply="ok")
        self.calls = 0

    async def plan(self, *, paper: QuestionPaper, message: str) -> ChangePlan:
        self.calls += 1
        return self._plan


class _StubRevisionAgent:
    """Generation agent stand-in that only supports rewrite requests."""

    def __init__(self, *, returns: int | None = None) -> None:
        self._returns = returns
        self.revise_calls = 0

    async def revise(self, request: object) -> list[GeneratedQuestion]:
        self.revise_calls += 1
        targets = request.targets
        if self._returns is not None:
            targets = targets[: self._returns]
        chunk = request.chunks[0]
        return [
            GeneratedQuestion(
                question_text=f"Reworded: {target.question_text}",
                question_type=target.question_type,
                marks=Decimal(target.marks),
                difficulty=target.difficulty,
                topic="Mechanics",
                options=None,
                correct_option=None,
                model_answer="F = ma",
                source_chunk_id=chunk.id,
            )
            for target in targets
        ]


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the deterministic offline stubs instead of a live LLM."""

    monkeypatch.setattr(llm, "is_available", lambda: False)


def _patch_paper(monkeypatch: pytest.MonkeyPatch, paper: QuestionPaper | None) -> None:
    async def _get(session: object, paper_id: uuid.UUID) -> QuestionPaper | None:
        return paper

    monkeypatch.setattr(paper_service.paper_repo, "get_paper", _get)
    monkeypatch.setattr(chat_service.paper_repo, "get_paper", _get)


def _patch_chunks(monkeypatch: pytest.MonkeyPatch, chunks: list[SyllabusChunk]) -> None:
    async def _list_chunks(session: object, syllabus_id: uuid.UUID) -> list[SyllabusChunk]:
        return chunks

    async def _get_syllabus(session: object, syllabus_id: uuid.UUID) -> object:
        from src.repositories.schema import Syllabus

        return Syllabus(
            id=syllabus_id,
            created_at=_NOW,
            updated_at=_NOW,
            version=1,
            is_active=True,
            title="Science",
            subject="Science",
            class_section="B",
        )

    monkeypatch.setattr(chat_service.syllabus_repo, "list_chunks", _list_chunks)
    monkeypatch.setattr(chat_service.syllabus_repo, "get", _get_syllabus)


def _request(message: str = "please change it", **overrides: object) -> PaperChatRequest:
    payload: dict[str, object] = {"message": message}
    payload.update(overrides)
    return PaperChatRequest(**payload)


async def test_chat_returns_offline_reply_without_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk()
    paper = _paper(chunk)
    _patch_paper(monkeypatch, paper)
    _patch_chunks(monkeypatch, [chunk])
    session = _FakeSession(paper)

    result = await PaperChatService(session).chat(paper.id, _request())

    assert result.applied == []
    assert result.reply
    assert len(result.paper.questions) == 2
    assert result.validation_report.paper_id == paper.id
    messages = [item for item in session.added if isinstance(item, ConversationMessage)]
    assert [message.role for message in messages] == ["user", "assistant"]


async def test_chat_drops_selected_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunk()
    paper = _paper(chunk)
    _patch_paper(monkeypatch, paper)
    _patch_chunks(monkeypatch, [chunk])
    session = _FakeSession(paper)
    router = _StubRouter(
        ChangePlan(
            reply="Removed one",
            operations=[ChangeOperation(op=ChangeOp.DROP, order_indices=[0])],
        )
    )

    result = await PaperChatService(session, router=router).chat(paper.id, _request())

    assert result.applied == ["Removed 1 question(s)"]
    assert len(result.paper.questions) == 1
    assert result.paper.questions[0].order_index == 0
    assert result.paper.total_marks == 5
    assert result.validation_report.passed is True
    assert result.paper.status == PaperStatus.VALIDATED.value


async def test_chat_reorders_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunk()
    paper = _paper(chunk)
    _patch_paper(monkeypatch, paper)
    _patch_chunks(monkeypatch, [chunk])
    session = _FakeSession(paper)
    router = _StubRouter(
        ChangePlan(
            reply="Reordered",
            operations=[ChangeOperation(op=ChangeOp.REORDER, order=[1, 0])],
        )
    )

    result = await PaperChatService(session, router=router).chat(paper.id, _request())

    assert result.applied == ["Reordered the questions"]
    assert [question.order_index for question in result.paper.questions] == [0, 1]
    assert result.paper.questions[0].question_text == "Define the SI unit of force."


async def test_chat_ignores_an_invalid_reorder(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunk()
    paper = _paper(chunk)
    _patch_paper(monkeypatch, paper)
    _patch_chunks(monkeypatch, [chunk])
    session = _FakeSession(paper)
    router = _StubRouter(
        ChangePlan(
            reply="Reordered",
            operations=[ChangeOperation(op=ChangeOp.REORDER, order=[0, 0])],
        )
    )

    result = await PaperChatService(session, router=router).chat(paper.id, _request())

    assert result.applied == []
    assert result.paper.questions[0].question_text == "State Newton's second law of motion."


async def test_chat_sets_the_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunk()
    paper = _paper(chunk)
    _patch_paper(monkeypatch, paper)
    _patch_chunks(monkeypatch, [chunk])
    session = _FakeSession(paper)
    router = _StubRouter(
        ChangePlan(
            reply="Set the duration",
            operations=[ChangeOperation(op=ChangeOp.SET_DURATION, duration_minutes=90)],
        )
    )

    result = await PaperChatService(session, router=router).chat(paper.id, _request())

    assert result.paper.duration_minutes == 90
    assert result.applied == ["Set the duration to 90 minutes"]


async def test_chat_rewrites_selected_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunk()
    paper = _paper(chunk)
    _patch_paper(monkeypatch, paper)
    _patch_chunks(monkeypatch, [chunk])
    session = _FakeSession(paper)
    agent = _StubRevisionAgent()
    router = _StubRouter(
        ChangePlan(
            reply="Reworded",
            operations=[
                ChangeOperation(
                    op=ChangeOp.REWRITE, order_indices=[0], instructions="make it harder"
                )
            ],
        )
    )

    result = await PaperChatService(session, router=router, generation_agent=agent).chat(
        paper.id, _request()
    )

    assert agent.revise_calls == 1
    assert result.applied == ["Rewrote 1 question(s)"]
    assert result.paper.questions[0].question_text.startswith("Reworded:")


async def test_chat_reshapes_the_blueprint(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunk()
    paper = _paper(chunk)
    _patch_paper(monkeypatch, paper)
    _patch_chunks(monkeypatch, [chunk])
    session = _FakeSession(paper)
    router = _StubRouter(
        ChangePlan(
            reply="Reshaped",
            operations=[
                ChangeOperation(
                    op=ChangeOp.ADJUST_BLUEPRINT,
                    question_types=[
                        {"type": "mcq", "count": 1, "marks_each": 2},
                        {"type": "short", "count": 1, "marks_each": 3},
                    ],
                )
            ],
        )
    )

    result = await PaperChatService(session, router=router).chat(paper.id, _request())

    assert result.applied == ["Reshaped the paper to 5 marks across 2 question(s)"]
    assert result.paper.total_marks == 5
    assert [question.question_type for question in result.paper.questions] == ["mcq", "short"]
    assert [question.marks for question in result.paper.questions] == [2.0, 3.0]


async def test_chat_keeps_matching_questions_when_reshaping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk()
    paper = _paper(chunk)
    _patch_paper(monkeypatch, paper)
    _patch_chunks(monkeypatch, [chunk])
    session = _FakeSession(paper)
    router = _StubRouter(
        ChangePlan(
            reply="Reshaped",
            operations=[
                ChangeOperation(
                    op=ChangeOp.ADJUST_BLUEPRINT,
                    question_types=[{"type": "short", "count": 1, "marks_each": 5}],
                )
            ],
        )
    )

    result = await PaperChatService(session, router=router).chat(paper.id, _request())

    assert result.paper.total_marks == 5
    assert len(result.paper.questions) == 1
    assert result.paper.questions[0].question_text == "State Newton's second law of motion."


async def test_chat_resumes_an_existing_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunk()
    paper = _paper(chunk)
    _patch_paper(monkeypatch, paper)
    _patch_chunks(monkeypatch, [chunk])
    session = _FakeSession(paper)
    conversation_id = uuid.uuid4()
    seen: dict[str, object] = {}

    async def _get_conversation(session_: object, requested: uuid.UUID) -> object:
        seen["requested"] = requested
        from src.repositories.schema import Conversation

        return Conversation(id=requested, title="Midterm", is_active=True)

    monkeypatch.setattr(chat_service.chat_repo, "get_conversation", _get_conversation)

    result = await PaperChatService(session).chat(
        paper.id, _request(conversation_id=conversation_id)
    )

    assert seen["requested"] == conversation_id
    assert result.conversation_id == conversation_id


async def test_chat_creates_a_new_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunk()
    paper = _paper(chunk)
    _patch_paper(monkeypatch, paper)
    _patch_chunks(monkeypatch, [chunk])
    session = _FakeSession(paper)

    result = await PaperChatService(session).chat(paper.id, _request())

    assert isinstance(result.conversation_id, uuid.UUID)


async def test_chat_rejects_an_unknown_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunk()
    paper = _paper(chunk)
    _patch_paper(monkeypatch, paper)

    async def _get_conversation(session_: object, requested: uuid.UUID) -> None:
        return None

    monkeypatch.setattr(chat_service.chat_repo, "get_conversation", _get_conversation)

    with pytest.raises(NotFoundError) as excinfo:
        await PaperChatService(_FakeSession(paper)).chat(
            paper.id, _request(conversation_id=uuid.uuid4())
        )

    assert excinfo.value.error_code == code.NOT_FOUND


async def test_chat_rejects_a_missing_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_paper(monkeypatch, None)

    with pytest.raises(NotFoundError) as excinfo:
        await PaperChatService(_FakeSession()).chat(uuid.uuid4(), _request())

    assert excinfo.value.error_code == code.QUESTION_PAPER_NOT_FOUND


async def test_chat_rejects_an_approved_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunk()
    paper = _paper(chunk, status=PaperStatus.APPROVED.value)
    _patch_paper(monkeypatch, paper)

    with pytest.raises(ConflictError):
        await PaperChatService(_FakeSession(paper)).chat(paper.id, _request())


async def test_chat_rejects_dropping_every_question(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunk()
    paper = _paper(chunk)
    _patch_paper(monkeypatch, paper)
    _patch_chunks(monkeypatch, [chunk])
    session = _FakeSession(paper)
    router = _StubRouter(
        ChangePlan(
            reply="Removed all",
            operations=[ChangeOperation(op=ChangeOp.DROP, order_indices=[0, 1])],
        )
    )

    with pytest.raises(ConflictError) as excinfo:
        await PaperChatService(session, router=router).chat(paper.id, _request())

    assert excinfo.value.error_code == code.CONFLICT


async def test_chat_skips_rewrite_when_the_generator_cannot_revise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk()
    paper = _paper(chunk)
    _patch_paper(monkeypatch, paper)
    _patch_chunks(monkeypatch, [chunk])
    session = _FakeSession(paper)
    agent = _StubRevisionAgent(returns=0)
    router = _StubRouter(
        ChangePlan(
            reply="Reworded",
            operations=[ChangeOperation(op=ChangeOp.REWRITE, order_indices=[0, 1])],
        )
    )

    result = await PaperChatService(session, router=router, generation_agent=agent).chat(
        paper.id, _request()
    )

    assert result.applied == []
    assert result.paper.questions[0].question_text == "State Newton's second law of motion."
