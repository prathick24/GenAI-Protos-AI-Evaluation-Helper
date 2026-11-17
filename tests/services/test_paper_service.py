"""Tests for the question paper generation and validation service (FR-002, FR-003)."""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.agents import paper_generation_agent as agent_module
from src.agents.paper_generation_agent import (
    GeneratedQuestion,
    GeneratedRubric,
    QuestionSpec,
    RevisionRequest,
)
from src.agents.paper_validation_agent import QuestionReview
from src.models.paper import (
    Blueprint,
    DifficultyMix,
    GeneratePaperRequest,
    PaperValidationReport,
    QuestionTypeSpec,
    QuestionValidationResult,
)
from src.repositories.schema import Question, QuestionPaper, Syllabus, SyllabusChunk
from src.services import paper_service
from src.services.paper_service import (
    PaperService,
    _allocate,
    _check_duplication,
    _check_grounding,
    _difficulty_sequence,
    _even_counts,
    _largest_remainder,
    _merge_sections,
    _ordered_units,
    _plan_questions,
    _sectionise,
    _similarity,
    _stored_scope,
    _words,
)
from src.settings import (
    PAPER_REPAIR_MAX_ATTEMPTS,
    Difficulty,
    PaperStatus,
    QuestionType,
    SyllabusStatus,
    ValidationStatus,
)
from src.utils.exceptions import ConflictError, NotFoundError, RequestValidationError
from src.utils.exceptions import error_codes as code

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _FakeSession:
    """Minimal AsyncSession stand-in that records commits, flushes and adds."""

    def __init__(self) -> None:
        self.commits = 0
        self.flushes = 0
        self.added: list[object] = []

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def add_all(self, objs: list[object]) -> None:
        self.added.extend(objs)

    async def flush(self) -> None:
        self.flushes += 1

    async def commit(self) -> None:
        self.commits += 1


def _syllabus(**overrides: object) -> Syllabus:
    defaults: dict[str, object] = {
        "id": uuid.uuid4(),
        "created_at": _NOW,
        "updated_at": _NOW,
        "version": 1,
        "is_active": True,
        "title": "Science",
        "subject": "Science",
        "class_section": "B",
        "status": SyllabusStatus.READY.value,
    }
    defaults.update(overrides)
    return Syllabus(**defaults)


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


def _paper(**overrides: object) -> QuestionPaper:
    defaults: dict[str, object] = {
        "id": uuid.uuid4(),
        "created_at": _NOW,
        "updated_at": _NOW,
        "version": 1,
        "is_active": True,
        "title": "Midterm",
        "total_marks": 10,
        "source": "generated",
        "status": PaperStatus.DRAFT.value,
    }
    defaults.update(overrides)
    return QuestionPaper(**defaults)


def _blueprint(
    *,
    total: int,
    types: list[tuple[str, int, int]],
    difficulty_mix: DifficultyMix | None = None,
    topic_weightage: dict[str, float] | None = None,
    duration: int = 60,
) -> Blueprint:
    return Blueprint(
        total_marks=total,
        duration_minutes=duration,
        question_types=[
            QuestionTypeSpec(type=QuestionType(kind), count=count, marks_each=marks)
            for kind, count, marks in types
        ],
        difficulty_mix=difficulty_mix,
        topic_weightage=topic_weightage,
    )


def _request(syllabus_id: uuid.UUID, **overrides: object) -> GeneratePaperRequest:
    defaults: dict[str, object] = {
        "syllabus_id": syllabus_id,
        "title": "Midterm",
        "scope": None,
        "blueprint": _blueprint(total=5, types=[("short", 1, 5)]),
    }
    defaults.update(overrides)
    return GeneratePaperRequest(**defaults)


def _patch_syllabus(
    monkeypatch: pytest.MonkeyPatch,
    *,
    syllabus: Syllabus | None,
    chunks: list[SyllabusChunk] | None = None,
) -> None:
    async def _get(session: object, syllabus_id: uuid.UUID) -> Syllabus | None:
        return syllabus

    async def _list_chunks(session: object, syllabus_id: uuid.UUID) -> list[SyllabusChunk]:
        return chunks or []

    monkeypatch.setattr(paper_service.syllabus_repo, "get", _get)
    monkeypatch.setattr(paper_service.syllabus_repo, "list_chunks", _list_chunks)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the deterministic offline stub instead of a live LLM."""

    monkeypatch.setattr(agent_module.llm, "is_available", lambda: False)


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the paper repository with in-memory stand-ins that assign ids."""

    async def _add_paper(session: _FakeSession, paper: QuestionPaper) -> QuestionPaper:
        paper.id = uuid.uuid4()
        paper.created_at = _NOW
        paper.updated_at = _NOW
        paper.version = 1
        paper.is_active = True
        session.add(paper)
        return paper

    async def _add_questions(session: _FakeSession, questions: list[object]) -> None:
        for question in questions:
            question.id = uuid.uuid4()
            for criterion in question.rubric_criteria:
                criterion.id = uuid.uuid4()
        session.add_all(questions)

    monkeypatch.setattr(paper_service.paper_repo, "add_paper", _add_paper)
    monkeypatch.setattr(paper_service.paper_repo, "add_questions", _add_questions)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
async def test_generate_validates_a_consistent_paper(
    repo: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    syllabus = _syllabus()
    chunks = [
        _chunk(syllabus_id=syllabus.id, chunk_index=0, unit_no="1", unit_name="Mechanics"),
        _chunk(syllabus_id=syllabus.id, chunk_index=1, unit_no="2", unit_name="Optics"),
    ]
    _patch_syllabus(monkeypatch, syllabus=syllabus, chunks=chunks)
    session = _FakeSession()

    result = await PaperService(session).generate(
        _request(syllabus.id, blueprint=_blueprint(total=5, types=[("mcq", 1, 2), ("short", 1, 3)]))
    )

    paper = result.paper
    assert paper.status == PaperStatus.VALIDATED.value
    assert paper.title == "Midterm"
    assert paper.subject == "Science"
    assert paper.total_marks == 5
    assert len(paper.questions) == 2
    assert {question.topic for question in paper.questions} == {"Mechanics", "Optics"}
    assert all(question.source_chunk_id is not None for question in paper.questions)
    assert all(
        question.validation_status == ValidationStatus.PASSED.value
        for question in paper.questions
    )
    assert result.validation_report.passed is True
    assert result.validation_report.paper_status == PaperStatus.VALIDATED.value
    assert session.commits == 1


async def test_generate_honours_scope(repo: None, monkeypatch: pytest.MonkeyPatch) -> None:
    syllabus = _syllabus()
    chunks = [
        _chunk(syllabus_id=syllabus.id, unit_no="1", unit_name="Mechanics"),
        _chunk(syllabus_id=syllabus.id, chunk_index=1, unit_no="2", unit_name="Optics"),
    ]
    _patch_syllabus(monkeypatch, syllabus=syllabus, chunks=chunks)

    result = await PaperService(_FakeSession()).generate(_request(syllabus.id, scope=["2"]))

    assert [question.topic for question in result.paper.questions] == ["Optics"]


async def test_generate_apportions_by_topic_weightage(
    repo: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    syllabus = _syllabus()
    chunks = [
        _chunk(syllabus_id=syllabus.id, unit_no="1", unit_name="Mechanics"),
        _chunk(syllabus_id=syllabus.id, chunk_index=1, unit_no="2", unit_name="Optics"),
    ]
    _patch_syllabus(monkeypatch, syllabus=syllabus, chunks=chunks)

    result = await PaperService(_FakeSession()).generate(
        _request(
            syllabus.id,
            blueprint=_blueprint(
                total=20,
                types=[("short", 4, 5)],
                topic_weightage={"1": 3.0, "2": 1.0},
            ),
        )
    )

    topics = [question.topic for question in result.paper.questions]
    assert topics.count("Mechanics") == 3
    assert topics.count("Optics") == 1


async def test_generate_raises_when_syllabus_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_syllabus(monkeypatch, syllabus=None)

    with pytest.raises(NotFoundError) as excinfo:
        await PaperService(_FakeSession()).generate(_request(uuid.uuid4()))

    assert excinfo.value.error_code == code.SYLLABUS_NOT_FOUND


async def test_generate_raises_when_syllabus_inactive(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_syllabus(monkeypatch, syllabus=_syllabus(is_active=False))

    with pytest.raises(NotFoundError):
        await PaperService(_FakeSession()).generate(_request(uuid.uuid4()))


async def test_generate_raises_when_syllabus_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_syllabus(monkeypatch, syllabus=_syllabus(status=SyllabusStatus.PROCESSING.value))

    with pytest.raises(ConflictError):
        await PaperService(_FakeSession()).generate(_request(uuid.uuid4()))


async def test_generate_raises_when_no_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_syllabus(monkeypatch, syllabus=_syllabus(), chunks=[])

    with pytest.raises(NotFoundError) as excinfo:
        await PaperService(_FakeSession()).generate(_request(uuid.uuid4()))

    assert excinfo.value.error_code == code.SYLLABUS_CHUNKS_NOT_FOUND


async def test_generate_raises_when_scope_matches_nothing(
    repo: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    syllabus = _syllabus()
    _patch_syllabus(monkeypatch, syllabus=syllabus, chunks=[_chunk(syllabus_id=syllabus.id)])

    with pytest.raises(RequestValidationError):
        await PaperService(_FakeSession()).generate(_request(syllabus.id, scope=["99"]))


# --------------------------------------------------------------------------- #
# Repair loop
# --------------------------------------------------------------------------- #
_DUPLICATE_TEXT = "Define force and state its SI unit."


def _generated(
    *,
    text: str,
    marks: int = 5,
    question_type: str = QuestionType.SHORT.value,
    difficulty: str = Difficulty.MEDIUM.value,
    chunk_id: uuid.UUID | None = None,
) -> GeneratedQuestion:
    return GeneratedQuestion(
        question_text=text,
        question_type=question_type,
        marks=Decimal(marks),
        difficulty=difficulty,
        topic="Mechanics",
        options=None,
        correct_option=None,
        model_answer="F = ma",
        source_chunk_id=chunk_id,
        rubrics=[
            GeneratedRubric(criterion_text="Explains the concept.", max_marks=Decimal(marks))
        ],
    )


class _StubGenerationAgent:
    """Generation agent stand-in: emits duplicate questions, then revises them.

    ``mode`` selects the revision behaviour: ``fix`` returns distinct wording so
    the paper validates, ``noop`` returns the same duplicate wording so the loop
    exhausts its attempts, and ``none`` returns nothing so the loop stops early.
    """

    def __init__(self, *, mode: str = "fix") -> None:
        self.mode = mode
        self.generate_calls = 0
        self.revise_calls = 0

    async def generate(self, request: object) -> list[GeneratedQuestion]:
        self.generate_calls += 1
        chunk_id = request.chunks[0].id if request.chunks else None
        return [
            _generated(
                text=_DUPLICATE_TEXT,
                marks=spec.marks,
                question_type=spec.question_type.value,
                difficulty=spec.difficulty.value,
                chunk_id=chunk_id,
            )
            for spec in request.specs
        ]

    async def revise(self, request: object) -> list[GeneratedQuestion]:
        self.revise_calls += 1
        if self.mode == "none":
            return []
        chunk_id = request.chunks[0].id if request.chunks else None
        return [
            _generated(
                text=(
                    _DUPLICATE_TEXT
                    if self.mode == "noop"
                    else f"Explain physics concept {target.order_index + 1} in detail."
                ),
                marks=target.marks,
                question_type=target.question_type,
                difficulty=target.difficulty,
                chunk_id=chunk_id,
            )
            for target in request.targets
        ]


class _CapturingRevisionAgent:
    """Revision stand-in that records the request and rewrites nothing."""

    def __init__(self) -> None:
        self.requests: list[RevisionRequest] = []

    async def revise(self, request: RevisionRequest) -> list[GeneratedQuestion]:
        self.requests.append(request)
        return []


def _duplicate_blueprint() -> Blueprint:
    return _blueprint(total=10, types=[("short", 2, 5)])


async def test_generate_repairs_flagged_questions(
    repo: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    syllabus = _syllabus()
    chunks = [_chunk(syllabus_id=syllabus.id, unit_no="1", unit_name="Mechanics")]
    _patch_syllabus(monkeypatch, syllabus=syllabus, chunks=chunks)
    agent = _StubGenerationAgent(mode="fix")

    result = await PaperService(_FakeSession(), agent=agent).generate(
        _request(syllabus.id, blueprint=_duplicate_blueprint())
    )

    assert agent.generate_calls == 1
    assert agent.revise_calls == 1
    assert result.validation_report.passed is True
    assert result.paper.status == PaperStatus.VALIDATED.value
    assert len({question.question_text for question in result.paper.questions}) == 2


async def test_generate_stops_repairing_when_generator_cannot_revise(
    repo: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    syllabus = _syllabus()
    chunks = [_chunk(syllabus_id=syllabus.id, unit_no="1", unit_name="Mechanics")]
    _patch_syllabus(monkeypatch, syllabus=syllabus, chunks=chunks)
    agent = _StubGenerationAgent(mode="none")

    result = await PaperService(_FakeSession(), agent=agent).generate(
        _request(syllabus.id, blueprint=_duplicate_blueprint())
    )

    assert agent.revise_calls == 1
    assert result.validation_report.passed is False
    assert result.paper.status == PaperStatus.DRAFT.value


async def test_generate_caps_repair_attempts(
    repo: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    syllabus = _syllabus()
    chunks = [_chunk(syllabus_id=syllabus.id, unit_no="1", unit_name="Mechanics")]
    _patch_syllabus(monkeypatch, syllabus=syllabus, chunks=chunks)
    agent = _StubGenerationAgent(mode="noop")

    result = await PaperService(_FakeSession(), agent=agent).generate(
        _request(syllabus.id, blueprint=_duplicate_blueprint())
    )

    assert agent.revise_calls == PAPER_REPAIR_MAX_ATTEMPTS
    assert result.validation_report.passed is False


async def test_repair_sends_the_rest_of_the_paper_to_the_generator() -> None:
    chunk = _chunk()
    paper = _paper(syllabus_id=chunk.syllabus_id)
    failed = _question(
        question_paper_id=paper.id,
        order_index=0,
        source_chunk_id=chunk.id,
        question_text="Which of the following is NOT a force?",
    )
    other = _question(
        question_paper_id=paper.id,
        order_index=1,
        source_chunk_id=chunk.id,
        question_text="Define momentum and state its SI unit.",
    )
    report = PaperValidationReport(
        paper_id=paper.id,
        passed=False,
        paper_status=PaperStatus.DRAFT.value,
        questions=[
            QuestionValidationResult(
                question_id=failed.id,
                order_index=0,
                status=ValidationStatus.FAILED.value,
                issues=["Near-duplicate of question 2"],
            ),
            QuestionValidationResult(
                question_id=other.id,
                order_index=1,
                status=ValidationStatus.PASSED.value,
            ),
        ],
    )
    agent = _CapturingRevisionAgent()

    changed = await PaperService(_FakeSession(), agent=agent)._repair(
        paper, [failed, other], report, [chunk]
    )

    assert changed is False
    assert len(agent.requests) == 1
    request = agent.requests[0]
    assert [question.order_index for question in request.paper_questions] == [0, 1]
    assert request.paper_questions[1].question_text == other.question_text
    assert request.targets[0].issues == ["Near-duplicate of question 2"]


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #
async def test_get_returns_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    paper = _paper()

    async def _get(session: object, paper_id: uuid.UUID) -> QuestionPaper:
        return paper

    monkeypatch.setattr(paper_service.paper_repo, "get_paper", _get)

    result = await PaperService(_FakeSession()).get(paper.id)

    assert result.id == paper.id


async def test_get_raises_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _get(session: object, paper_id: uuid.UUID) -> None:
        return None

    monkeypatch.setattr(paper_service.paper_repo, "get_paper", _get)

    with pytest.raises(NotFoundError) as excinfo:
        await PaperService(_FakeSession()).get(uuid.uuid4())

    assert excinfo.value.error_code == code.QUESTION_PAPER_NOT_FOUND


async def test_list_returns_active_papers(monkeypatch: pytest.MonkeyPatch) -> None:
    paper = _paper()

    async def _list(session: object) -> list[QuestionPaper]:
        return [paper]

    monkeypatch.setattr(paper_service.paper_repo, "list_papers", _list)

    result = await PaperService(_FakeSession()).list()

    assert len(result) == 1
    assert result[0].id == paper.id


# --------------------------------------------------------------------------- #
# Planning helpers
# --------------------------------------------------------------------------- #
def test_plan_questions_expands_counts() -> None:
    blueprint = _blueprint(total=10, types=[("mcq", 2, 2), ("short", 2, 3)])

    specs = _plan_questions(blueprint)

    assert [spec.question_type for spec in specs] == [
        QuestionType.MCQ,
        QuestionType.MCQ,
        QuestionType.SHORT,
        QuestionType.SHORT,
    ]
    assert [spec.marks for spec in specs] == [2, 2, 3, 3]
    assert all(spec.difficulty is Difficulty.MEDIUM for spec in specs)


def test_plan_questions_applies_difficulty_mix() -> None:
    blueprint = _blueprint(
        total=8,
        types=[("short", 4, 2)],
        difficulty_mix=DifficultyMix(easy=1, medium=1, hard=2),
    )

    specs = _plan_questions(blueprint)

    counts = {
        level: sum(1 for spec in specs if spec.difficulty is level)
        for level in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
    }
    assert counts == {Difficulty.EASY: 1, Difficulty.MEDIUM: 1, Difficulty.HARD: 2}


def test_difficulty_sequence_without_mix() -> None:
    assert _difficulty_sequence(2, None) == [Difficulty.MEDIUM, Difficulty.MEDIUM]


def test_ordered_units_groups_and_filters_scope() -> None:
    chunks = [
        _chunk(unit_no="1"),
        _chunk(unit_no="2"),
        _chunk(unit_no="1"),
    ]

    units = _ordered_units(chunks, None)

    assert [key for key, _ in units] == ["1", "2"]
    assert len(units[0][1]) == 2
    assert [key for key, _ in _ordered_units(chunks, ["2"])] == ["2"]


def test_ordered_units_drops_front_matter_when_units_exist() -> None:
    chunks = [
        _chunk(unit_no=None, topic_name=None),
        _chunk(unit_no="1"),
        _chunk(unit_no="2"),
    ]

    units = _ordered_units(chunks, None)

    assert [key for key, _ in units] == ["1", "2"]


def test_ordered_units_keeps_untagged_chunks_without_units() -> None:
    chunks = [
        _chunk(unit_no=None, topic_name="Intro"),
        _chunk(unit_no=None, topic_name="Body"),
    ]

    units = _ordered_units(chunks, None)

    assert [key for key, _ in units] == [""]
    assert len(units[0][1]) == 2


def test_allocate_interleaves_questions_across_units() -> None:
    specs = [QuestionSpec(QuestionType.SHORT, 5, Difficulty.EASY)] * 4
    units = [("1", [_chunk()]), ("2", [_chunk()])]

    allocation = _allocate(specs, units, None)

    assert len(allocation["1"]) == 2
    assert len(allocation["2"]) == 2


def test_allocate_respects_topic_weightage() -> None:
    specs = [QuestionSpec(QuestionType.SHORT, 5, Difficulty.EASY)] * 4
    units = [("1", [_chunk()]), ("2", [_chunk()])]

    allocation = _allocate(specs, units, {"1": 3.0, "2": 1.0})

    assert len(allocation["1"]) == 3
    assert len(allocation["2"]) == 1


def test_largest_remainder_splits_remainder_by_fraction() -> None:
    assert _largest_remainder(5, {"a": 1.0, "b": 1.0, "c": 1.0}) == {"a": 2, "b": 2, "c": 1}


def test_largest_remainder_handles_zero_total() -> None:
    assert _largest_remainder(0, {"a": 1.0}) == {"a": 0}


def test_sectionise_splits_when_over_budget() -> None:
    first = _chunk(raw_text="a" * 10)
    second = _chunk(raw_text="b" * 10)

    sections = _sectionise([first, second], budget=15)

    assert len(sections) == 2


def test_merge_sections_caps_section_count() -> None:
    sections = [[_chunk()] for _ in range(5)]

    merged = _merge_sections(sections, 2)

    assert len(merged) == 2
    assert sum(len(group) for group in merged) == 5


def test_even_counts_distributes_remainder() -> None:
    assert _even_counts(5, 2) == [3, 2]
    assert _even_counts(4, 2) == [2, 2]
    assert _even_counts(3, 0) == []


# --------------------------------------------------------------------------- #
# Validation (FR-003)
# --------------------------------------------------------------------------- #
class _StubValidationAgent:
    """Validation agent stand-in returning preset advisory reviews."""

    def __init__(self, reviews: list[QuestionReview] | None = None) -> None:
        self.reviews = reviews or []
        self.calls = 0
        self.questions: list[object] = []

    async def review(self, *, subject: object, questions: list[object]) -> list[QuestionReview]:
        self.calls += 1
        self.questions = questions
        return self.reviews


def _question(**overrides: object) -> Question:
    defaults: dict[str, object] = {
        "id": uuid.uuid4(),
        "question_paper_id": uuid.uuid4(),
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
        "difficulty_mix": {"easy": 1, "medium": 1, "hard": 0},
        "topic_weightage": None,
        "scope": None,
    }
    payload.update(overrides)
    return payload


def _patch_validation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    paper: QuestionPaper | None,
    chunks: list[SyllabusChunk] | None = None,
) -> None:
    async def _get_paper(session: object, paper_id: uuid.UUID) -> QuestionPaper | None:
        return paper

    async def _list_chunks(session: object, syllabus_id: uuid.UUID) -> list[SyllabusChunk]:
        return chunks or []

    monkeypatch.setattr(paper_service.paper_repo, "get_paper", _get_paper)
    monkeypatch.setattr(paper_service.syllabus_repo, "list_chunks", _list_chunks)


def _check(report: object, name: str) -> object:
    return next(check for check in report.checks if check.name == name)


def _two_unit_paper(syllabus_id: uuid.UUID) -> tuple[QuestionPaper, SyllabusChunk, SyllabusChunk]:
    """Build a paper whose two questions each cover one of two units."""

    paper = _paper(syllabus_id=syllabus_id, total_marks=10, blueprint=_blueprint_dict())
    first = _chunk(syllabus_id=syllabus_id, unit_no="1", unit_name="Mechanics")
    second = _chunk(syllabus_id=syllabus_id, chunk_index=1, unit_no="2", unit_name="Optics")
    paper.questions = [
        _question(
            question_paper_id=paper.id,
            order_index=0,
            source_chunk_id=first.id,
            difficulty=Difficulty.EASY.value,
            question_text="Define force and state its SI unit.",
        ),
        _question(
            question_paper_id=paper.id,
            order_index=1,
            source_chunk_id=second.id,
            question_text="Explain how a convex lens forms an image.",
        ),
    ]
    return paper, first, second


async def test_generate_persists_scope_for_validation(
    repo: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    syllabus = _syllabus()
    chunks = [
        _chunk(syllabus_id=syllabus.id, unit_no="1"),
        _chunk(syllabus_id=syllabus.id, chunk_index=1, unit_no="2"),
    ]
    _patch_syllabus(monkeypatch, syllabus=syllabus, chunks=chunks)
    session = _FakeSession()

    await PaperService(session).generate(_request(syllabus.id, scope=["2"]))

    assert session.added[0].blueprint["scope"] == ["2"]


async def test_validate_passes_for_a_consistent_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    syllabus_id = uuid.uuid4()
    paper, first, second = _two_unit_paper(syllabus_id)
    _patch_validation(monkeypatch, paper=paper, chunks=[first, second])
    agent = _StubValidationAgent()

    report = await PaperService(_FakeSession(), validation_agent=agent).validate(paper.id)

    assert report.passed is True
    assert report.paper_status == PaperStatus.VALIDATED.value
    assert all(item.status == ValidationStatus.PASSED.value for item in report.questions)
    assert agent.calls == 1


async def test_validate_flags_a_marks_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    syllabus_id = uuid.uuid4()
    paper, first, second = _two_unit_paper(syllabus_id)
    paper.questions = paper.questions[:1]
    _patch_validation(monkeypatch, paper=paper, chunks=[first, second])

    report = await PaperService(_FakeSession(), validation_agent=_StubValidationAgent()).validate(
        paper.id
    )

    assert report.passed is False
    assert report.paper_status == PaperStatus.DRAFT.value
    assert _check(report, "marks_total").passed is False


async def test_validate_flags_ungrounded_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    syllabus_id = uuid.uuid4()
    paper, first, second = _two_unit_paper(syllabus_id)
    paper.questions[0].source_chunk_id = uuid.uuid4()
    _patch_validation(monkeypatch, paper=paper, chunks=[first, second])

    report = await PaperService(_FakeSession(), validation_agent=_StubValidationAgent()).validate(
        paper.id
    )

    assert report.passed is False
    assert _check(report, "source_grounding").passed is False
    assert report.questions[0].status == ValidationStatus.FAILED.value
    assert report.questions[0].issues


async def test_validate_flags_uncovered_units(monkeypatch: pytest.MonkeyPatch) -> None:
    syllabus_id = uuid.uuid4()
    paper, first, second = _two_unit_paper(syllabus_id)
    for question in paper.questions:
        question.source_chunk_id = first.id
    _patch_validation(monkeypatch, paper=paper, chunks=[first, second])

    report = await PaperService(_FakeSession(), validation_agent=_StubValidationAgent()).validate(
        paper.id
    )

    assert report.passed is False
    coverage = _check(report, "syllabus_coverage")
    assert coverage.passed is False
    assert coverage.details["uncovered_units"] == ["2"]


async def test_validate_flags_near_duplicate_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    syllabus_id = uuid.uuid4()
    paper, first, second = _two_unit_paper(syllabus_id)
    paper.questions[1].question_text = paper.questions[0].question_text
    _patch_validation(monkeypatch, paper=paper, chunks=[first, second])

    report = await PaperService(_FakeSession(), validation_agent=_StubValidationAgent()).validate(
        paper.id
    )

    assert report.passed is False
    assert _check(report, "duplication").passed is False
    assert all(item.status == ValidationStatus.FAILED.value for item in report.questions)


async def test_validate_applies_the_advisory_review(monkeypatch: pytest.MonkeyPatch) -> None:
    syllabus_id = uuid.uuid4()
    paper, first, second = _two_unit_paper(syllabus_id)
    _patch_validation(monkeypatch, paper=paper, chunks=[first, second])
    agent = _StubValidationAgent(
        reviews=[QuestionReview(index=0, ambiguous=True, duplicate_of=None, notes="too vague")]
    )

    report = await PaperService(_FakeSession(), validation_agent=agent).validate(paper.id)

    assert report.passed is False
    assert report.questions[0].status == ValidationStatus.FAILED.value
    assert report.questions[1].status == ValidationStatus.PASSED.value


async def test_validate_sends_mcq_options_to_the_advisory_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    syllabus_id = uuid.uuid4()
    paper, first, second = _two_unit_paper(syllabus_id)
    paper.questions[0].question_type = QuestionType.MCQ.value
    paper.questions[0].options = ["Speed", "Accuracy", "Reliability", "Versatility"]
    _patch_validation(monkeypatch, paper=paper, chunks=[first, second])
    agent = _StubValidationAgent()

    await PaperService(_FakeSession(), validation_agent=agent).validate(paper.id)

    assert agent.questions[0].options == ["Speed", "Accuracy", "Reliability", "Versatility"]
    assert agent.questions[1].options is None


async def test_validate_raises_when_paper_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_validation(monkeypatch, paper=None)

    with pytest.raises(NotFoundError) as excinfo:
        await PaperService(_FakeSession()).validate(uuid.uuid4())

    assert excinfo.value.error_code == code.QUESTION_PAPER_NOT_FOUND


async def test_validate_raises_when_paper_approved(monkeypatch: pytest.MonkeyPatch) -> None:
    paper = _paper(status=PaperStatus.APPROVED.value, blueprint=_blueprint_dict())
    _patch_validation(monkeypatch, paper=paper)

    with pytest.raises(ConflictError):
        await PaperService(_FakeSession()).validate(paper.id)


def _patch_paper(monkeypatch: pytest.MonkeyPatch, *, paper: QuestionPaper | None) -> None:
    async def _get_paper(session: object, paper_id: uuid.UUID) -> QuestionPaper | None:
        return paper

    monkeypatch.setattr(paper_service.paper_repo, "get_paper", _get_paper)


async def test_approve_moves_a_validated_paper_to_approved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper = _paper(status=PaperStatus.VALIDATED.value)
    _patch_paper(monkeypatch, paper=paper)
    session = _FakeSession()

    result = await PaperService(session).approve(paper.id)

    assert result.status == PaperStatus.APPROVED.value
    assert paper.status == PaperStatus.APPROVED.value
    assert session.commits == 1


async def test_approve_raises_when_paper_not_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper = _paper(status=PaperStatus.DRAFT.value)
    _patch_paper(monkeypatch, paper=paper)

    with pytest.raises(ConflictError):
        await PaperService(_FakeSession()).approve(paper.id)


async def test_approve_raises_when_paper_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_paper(monkeypatch, paper=None)

    with pytest.raises(NotFoundError) as excinfo:
        await PaperService(_FakeSession()).approve(uuid.uuid4())

    assert excinfo.value.error_code == code.QUESTION_PAPER_NOT_FOUND


def test_similarity_measures_word_overlap() -> None:
    assert (
        _similarity(_words("Define force and state its unit"), _words("define FORCE & its unit"))
        < 1
    )
    assert _similarity(_words("Explain optics"), _words("Describe lenses")) == 0.0
    assert _similarity(set(), _words("anything")) == 0.0


def test_check_duplication_flags_identical_questions() -> None:
    first = _question(order_index=0, question_text="Define force and state its SI unit.")
    second = _question(order_index=1, question_text="Define force and state its SI unit.")
    issues: dict[uuid.UUID, list[str]] = defaultdict(list)

    check = _check_duplication([first, second], issues)

    assert check.passed is False
    assert issues[first.id] and issues[second.id]


def test_check_grounding_requires_resolvable_chunks() -> None:
    chunk = _chunk()
    question = _question(source_chunk_id=uuid.uuid4())
    issues: dict[uuid.UUID, list[str]] = defaultdict(list)

    check = _check_grounding([question], {chunk.id}, issues)

    assert check.passed is False
    assert issues[question.id]


def test_stored_scope_reads_the_blueprint() -> None:
    assert _stored_scope(_paper(blueprint={"scope": ["1", "2"]})) == ["1", "2"]
    assert _stored_scope(_paper(blueprint={"scope": None})) is None
    assert _stored_scope(_paper(blueprint=None)) is None

