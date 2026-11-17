"""Tests for the answer segmentation and grading service (FR-005, FR-006)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import inspect

from src.agents import grading_agent as grading_agent_module
from src.models.answer_sheet import AnswerSheetRename
from src.models.grading import MarkOverrideRequest
from src.repositories import grading_repository as repo
from src.repositories.schema import (
    AnswerSheet,
    Exam,
    Mark,
    Question,
    QuestionPaper,
    RubricCriterion,
)
from src.services.grading_service import (
    GradingService,
    _build_answer,
    _marker,
    _segment,
    _sheet_text,
)
from src.settings import GradedBy, SheetStatus
from src.utils.exceptions import (
    ConflictError,
    FileProcessingError,
    NotFoundError,
    RequestValidationError,
)
from src.utils.exceptions import error_codes as code

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_MODEL_ANSWER = "Force equals mass times acceleration."


class _FakeSession:
    """Minimal AsyncSession stand-in that stamps audit fields and records writes."""

    def __init__(self, *, sheet: AnswerSheet | None = None) -> None:
        self._sheet = sheet
        self.commits = 0
        self.flushes = 0
        self.added: list[object] = []
        self.marks: dict[uuid.UUID, Mark] = {}

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def add_all(self, items: object) -> None:
        self.added.extend(items)  # type: ignore[arg-type]

    async def flush(self) -> None:
        self.flushes += 1
        for item in self.added:
            _apply_defaults(item)
            if isinstance(item, Mark):
                self.marks[item.id] = item

    async def commit(self) -> None:
        self.commits += 1

    async def get(
        self, model: object, key: object, *, options: object = None
    ) -> object | None:
        if model is AnswerSheet:
            return self._sheet
        if model is Mark:
            return self.marks.get(key)  # type: ignore[arg-type]
        return None


def _apply_defaults(row: object) -> None:
    """Emulate the column defaults SQLAlchemy applies at INSERT time."""

    if getattr(row, "id", None) is None:
        row.id = uuid.uuid4()
    for attr, value in (("created_at", _NOW), ("updated_at", _NOW), ("is_active", True)):
        if hasattr(row, attr) and getattr(row, attr) is None:
            setattr(row, attr, value)


class _RecordingMarkSession:
    """Session stand-in that records the loader options used to fetch a mark."""

    def __init__(self, mark: Mark) -> None:
        self.mark = mark
        self.options: object = None

    async def get(
        self, model: object, key: object, *, options: object = None
    ) -> object | None:
        self.options = options
        return self.mark


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the deterministic offline grading heuristic."""

    monkeypatch.setattr(grading_agent_module.llm, "is_available", lambda: False)


def _question(
    order_index: int,
    marks: float,
    *,
    question_type: str = "short",
    model_answer: str = _MODEL_ANSWER,
    rubrics: tuple[tuple[str, float], ...] = (),
) -> Question:
    question = Question(
        id=uuid.uuid4(),
        question_paper_id=uuid.uuid4(),
        order_index=order_index,
        question_text=f"Question {order_index + 1}",
        question_type=question_type,
        marks=Decimal(str(marks)),
        model_answer=model_answer,
    )
    question.rubric_criteria = [
        RubricCriterion(
            id=uuid.uuid4(),
            question_id=question.id,
            criterion_text=text,
            max_marks=Decimal(str(value)),
            order_index=index,
        )
        for index, (text, value) in enumerate(rubrics)
    ]
    return question


def _exam(questions: list[Question]) -> Exam:
    paper = QuestionPaper(id=uuid.uuid4(), title="Paper", total_marks=len(questions))
    paper.questions = questions
    exam = Exam(
        id=uuid.uuid4(),
        title="Midterm exam",
        subject="Science",
        total_marks=len(questions),
        status="draft",
        question_paper_id=paper.id,
        created_at=_NOW,
        updated_at=_NOW,
        is_active=True,
    )
    exam.question_paper = paper
    return exam


def _sheet(exam: Exam, path: object, **overrides: object) -> AnswerSheet:
    sheet = AnswerSheet(
        id=uuid.uuid4(),
        exam_id=exam.id,
        file_path=str(path),
        input_type="typed",
        ocr_status="done",
        sheet_status=SheetStatus.UPLOADED.value,
        ocr_confidence={"provider": "stub", "mean": 0.9},
        created_at=_NOW,
        updated_at=_NOW,
        is_active=True,
    )
    sheet.exam = exam
    for key, value in overrides.items():
        setattr(sheet, key, value)
    return sheet


# --------------------------------------------------------------------------- #
# Pure segmentation helpers
# --------------------------------------------------------------------------- #
def test_segment_aligns_blocks_in_paper_order() -> None:
    questions = [_question(0, 1), _question(1, 1), _question(2, 1)]
    text = "Preamble\nQ1: alpha\nQ2: beta\nQ3: gamma"

    blocks, unmatched = _segment(text, questions)

    assert blocks[questions[0].id] == "alpha"
    assert blocks[questions[1].id] == "beta"
    assert blocks[questions[2].id] == "gamma"
    assert unmatched == ["Preamble"]


def test_segment_handles_section_local_numbering() -> None:
    questions = [
        _question(0, 1, question_type="mcq"),
        _question(1, 5, question_type="short"),
        _question(2, 1, question_type="mcq"),
        _question(3, 5, question_type="long"),
    ]
    text = (
        "Section - A\n1. alpha\n2. beta\n"
        "Section B\n1. gamma\n2. delta"
    )

    blocks, _ = _segment(text, questions)

    assert blocks[questions[0].id] == "alpha"
    assert blocks[questions[2].id] == "beta"
    assert blocks[questions[1].id] == "gamma"
    assert blocks[questions[3].id] == "delta"


def test_segment_uses_section_types_for_paper_wide_numbering() -> None:
    questions = [
        _question(0, 1, question_type="mcq"),
        _question(1, 5, question_type="short"),
        _question(2, 1, question_type="mcq"),
        _question(3, 5, question_type="long"),
    ]
    text = (
        "SECTION A: MULTIPLE CHOICE QUESTIONS\n"
        "Q1: alpha\nQ2: gamma\n"
        "SECTION B: DESCRIPTIVE QUESTIONS\n"
        "Q3: beta\nQ4: delta"
    )

    blocks, _ = _segment(text, questions)

    assert [blocks[question.id] for question in questions] == [
        "alpha",
        "beta",
        "gamma",
        "delta",
    ]


def test_segment_maps_continuous_section_numbering_to_grouped_display_order() -> None:
    questions = [
        _question(0, 1, question_type="mcq"),
        _question(1, 2, question_type="short"),
        _question(2, 1, question_type="mcq"),
        _question(3, 5, question_type="long"),
        _question(4, 1, question_type="mcq"),
        _question(5, 2, question_type="short"),
    ]
    text = (
        "Section A - Multiple Choice Questions\n"
        "1. alpha\n2. gamma\n3. epsilon\n"
        "Section B - Short Answer Questions\n"
        "4. beta\n5. zeta\n"
        "Section C\n"
        "6. delta"
    )

    blocks, _ = _segment(text, questions)

    assert [blocks[question.id] for question in questions] == [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
    ]


def test_segment_keeps_a_numbered_list_inside_its_answer() -> None:
    questions = [_question(0, 1), _question(1, 1)]
    text = "Q2: Here is the answer.\n1) first\n2) second"

    blocks, _ = _segment(text, questions)

    assert blocks[questions[1].id] == "Here is the answer.\n1) first\n2) second"
    assert questions[0].id not in blocks


def test_segment_leaves_an_unanswered_question_without_a_block() -> None:
    questions = [_question(0, 1), _question(1, 1)]

    blocks, _ = _segment("Q1: only this one", questions)

    assert questions[0].id in blocks
    assert questions[1].id not in blocks


def test_marker_recognises_prefixes_and_rejects_decimals() -> None:
    assert _marker("Q3: foo") == (3, "foo")
    assert _marker("Answer 12) text") == (12, "text")
    assert _marker("3) bar") == (3, "bar")
    assert _marker("3.5 is a number") is None


def test_sheet_text_prefers_the_ocr_sidecar(tmp_path) -> None:
    path = tmp_path / "sheet.txt"
    path.write_text("plain extraction", encoding="utf-8")
    (tmp_path / "sheet.txt.ocr.txt").write_text("ocr text", encoding="utf-8")
    sheet = AnswerSheet(exam_id=uuid.uuid4(), file_path=str(path))

    assert _sheet_text(sheet) == "ocr text"


def test_sheet_text_falls_back_to_the_original_file(tmp_path) -> None:
    path = tmp_path / "sheet.txt"
    path.write_text("plain extraction", encoding="utf-8")
    sheet = AnswerSheet(exam_id=uuid.uuid4(), file_path=str(path))

    assert _sheet_text(sheet) == "plain extraction"


def test_sheet_text_requires_a_file() -> None:
    with pytest.raises(FileProcessingError) as excinfo:
        _sheet_text(AnswerSheet(exam_id=uuid.uuid4()))

    assert excinfo.value.error_code == code.FILE_PROCESSING_ERROR


def test_build_answer_declares_an_empty_mark() -> None:
    sheet = AnswerSheet(id=uuid.uuid4(), exam_id=uuid.uuid4())

    answer = _build_answer(sheet, _question(0, 1), "alpha", None)

    assert inspect(answer).attrs.mark.loaded_value is None


# --------------------------------------------------------------------------- #
# segment (FR-005)
# --------------------------------------------------------------------------- #
async def test_segment_persists_aligned_answers(tmp_path) -> None:
    questions = [_question(0, 1), _question(1, 1), _question(2, 1)]
    exam = _exam(questions)
    path = tmp_path / "sheet.txt"
    path.write_text(
        "Preamble\nQ1: Force equals mass times acceleration.\nQ2: Momentum is mass times velocity.",
        encoding="utf-8",
    )
    sheet = _sheet(exam, path)
    session = _FakeSession(sheet=sheet)

    result = await GradingService(session).segment(sheet.id)

    assert result.total_questions == 3
    assert result.answered == 2
    assert result.unanswered == 1
    assert [answer.order_index for answer in result.answers] == [0, 1, 2]
    assert result.answers[0].answer_text == "Force equals mass times acceleration."
    assert result.answers[2].is_answered is False
    assert result.answers[2].answer_text is None
    assert len(result.unmatched_regions) == 1
    assert sheet.ocr_confidence["segmentation"]["answers"] == 3
    assert session.commits == 1


async def test_segment_requires_the_sheet(tmp_path) -> None:
    session = _FakeSession(sheet=None)

    with pytest.raises(NotFoundError) as excinfo:
        await GradingService(session).segment(uuid.uuid4())

    assert excinfo.value.error_code == code.ANSWER_SHEET_NOT_FOUND


async def test_segment_rejects_an_approved_sheet(tmp_path) -> None:
    questions = [_question(0, 1)]
    exam = _exam(questions)
    path = tmp_path / "sheet.txt"
    path.write_text("Q1: alpha", encoding="utf-8")
    sheet = _sheet(exam, path, sheet_status=SheetStatus.APPROVED.value)

    with pytest.raises(ConflictError):
        await GradingService(_FakeSession(sheet=sheet)).segment(sheet.id)


async def test_segment_requires_paper_questions(tmp_path) -> None:
    exam = _exam([])
    path = tmp_path / "sheet.txt"
    path.write_text("Q1: alpha", encoding="utf-8")
    sheet = _sheet(exam, path)

    with pytest.raises(NotFoundError) as excinfo:
        await GradingService(_FakeSession(sheet=sheet)).segment(sheet.id)

    assert excinfo.value.error_code == code.QUESTION_NOT_FOUND


# --------------------------------------------------------------------------- #
# grade (FR-006)
# --------------------------------------------------------------------------- #
async def test_grade_creates_one_mark_per_answer(tmp_path) -> None:
    questions = [
        _question(0, 5, rubrics=(("States the law", 5),)),
        _question(1, 5, rubrics=(("Names momentum", 5),)),
    ]
    exam = _exam(questions)
    path = tmp_path / "sheet.txt"
    path.write_text(
        "Q1: Force equals mass times acceleration.\nQ2: Momentum is mass times velocity.",
        encoding="utf-8",
    )
    sheet = _sheet(exam, path)
    session = _FakeSession(sheet=sheet)
    service = GradingService(session)

    await service.segment(sheet.id)
    result = await service.grade(sheet.id)

    assert result.sheet_status == SheetStatus.GRADED.value
    assert result.total_max_marks == 10.0
    assert result.total_awarded_marks == 7.0
    assert len(result.marks) == 2
    assert all(mark.graded_by == GradedBy.AGENT.value for mark in result.marks)
    assert result.marks[0].awarded_marks == 5.0
    assert result.marks[0].evidence == "Force equals mass times acceleration."
    assert result.marks[1].awarded_marks == 2.0
    assert sheet.total_awarded_marks == Decimal("7.00")


async def test_regrading_updates_marks_without_duplicating(tmp_path) -> None:
    questions = [_question(0, 5)]
    exam = _exam(questions)
    path = tmp_path / "sheet.txt"
    path.write_text("Q1: Force equals mass times acceleration.", encoding="utf-8")
    sheet = _sheet(exam, path)
    session = _FakeSession(sheet=sheet)
    service = GradingService(session)

    await service.segment(sheet.id)
    first = await service.grade(sheet.id)
    second = await service.grade(sheet.id)

    assert len(first.marks) == 1
    assert len(second.marks) == 1
    assert first.marks[0].id == second.marks[0].id
    assert len([item for item in session.added if isinstance(item, Mark)]) == 1


async def test_grade_requires_segmentation_first(tmp_path) -> None:
    exam = _exam([_question(0, 5)])
    path = tmp_path / "sheet.txt"
    path.write_text("Q1: alpha", encoding="utf-8")
    sheet = _sheet(exam, path)

    with pytest.raises(ConflictError) as excinfo:
        await GradingService(_FakeSession(sheet=sheet)).grade(sheet.id)

    assert "Segment" in excinfo.value.message


async def test_grade_rejects_an_approved_sheet(tmp_path) -> None:
    exam = _exam([_question(0, 5)])
    path = tmp_path / "sheet.txt"
    path.write_text("Q1: alpha", encoding="utf-8")
    sheet = _sheet(exam, path, sheet_status=SheetStatus.APPROVED.value)

    with pytest.raises(ConflictError):
        await GradingService(_FakeSession(sheet=sheet)).grade(sheet.id)


# --------------------------------------------------------------------------- #
# detail
# --------------------------------------------------------------------------- #
async def test_detail_returns_answers_sorted_by_question_order(tmp_path) -> None:
    questions = [_question(0, 5), _question(1, 5)]
    exam = _exam(questions)
    path = tmp_path / "sheet.txt"
    path.write_text("Q1: alpha\nQ2: beta", encoding="utf-8")
    sheet = _sheet(exam, path)
    session = _FakeSession(sheet=sheet)
    service = GradingService(session)

    await service.segment(sheet.id)
    sheet.answers = list(reversed(sheet.answers))
    detail = await service.get_detail(sheet.id)

    assert [answer.order_index for answer in detail.answers] == [0, 1]
    assert detail.id == sheet.id
    assert detail.feedback is None


# --------------------------------------------------------------------------- #
# feedback (FR-008)
# --------------------------------------------------------------------------- #
async def test_grade_writes_grounded_feedback(tmp_path) -> None:
    exam = _exam([_question(0, 4, rubrics=(("States the law", 4),))])
    path = tmp_path / "sheet.txt"
    path.write_text("Q1: Force equals mass times acceleration.", encoding="utf-8")
    sheet = _sheet(exam, path)
    service = GradingService(_FakeSession(sheet=sheet))

    await service.segment(sheet.id)
    await service.grade(sheet.id)

    assert sheet.feedback is not None
    assert "scored 4 out of 4" in sheet.feedback.overall_comments
    assert sheet.feedback.strengths
    assert sheet.feedback.improvements is not None


# --------------------------------------------------------------------------- #
# override (FR-009)
# --------------------------------------------------------------------------- #
async def test_get_mark_eager_loads_the_sheets_answers_and_marks() -> None:
    mark = Mark(id=uuid.uuid4(), answer_id=uuid.uuid4())
    session = _RecordingMarkSession(mark)

    assert await repo.get_mark(session, mark.id) is mark
    assert session.options


async def _graded_sheet(tmp_path):
    questions = [_question(0, 5), _question(1, 5)]
    exam = _exam(questions)
    path = tmp_path / "sheet.txt"
    path.write_text(
        "Q1: Force equals mass times acceleration.\nQ2: Momentum is mass times velocity.",
        encoding="utf-8",
    )
    sheet = _sheet(exam, path)
    session = _FakeSession(sheet=sheet)
    service = GradingService(session)
    await service.segment(sheet.id)
    graded = await service.grade(sheet.id)
    return service, session, sheet, graded


async def test_override_preserves_original_and_updates_total(tmp_path) -> None:
    service, session, sheet, graded = await _graded_sheet(tmp_path)
    mark_id = graded.marks[0].id

    result = await service.override(
        mark_id, MarkOverrideRequest(awarded_marks=1, reason="Wrong law")
    )

    assert result.awarded_marks == 1.0
    assert result.original_marks == 5.0
    assert result.is_overridden is True
    assert result.graded_by == GradedBy.TEACHER.value
    assert "Wrong law" in (result.reasoning or "")
    assert sheet.total_awarded_marks == Decimal("3.00")
    assert sheet.sheet_status == SheetStatus.REVIEWED.value
    assert sheet.feedback is not None
    assert "3 out of 10" in sheet.feedback.overall_comments
    assert session.commits == 3


async def test_second_override_keeps_the_first_original(tmp_path) -> None:
    service, _session, sheet, graded = await _graded_sheet(tmp_path)
    mark_id = graded.marks[0].id

    await service.override(mark_id, MarkOverrideRequest(awarded_marks=1))
    result = await service.override(mark_id, MarkOverrideRequest(awarded_marks=4))

    assert result.original_marks == 5.0
    assert result.awarded_marks == 4.0
    assert sheet.total_awarded_marks == Decimal("6.00")


async def test_override_rejects_marks_over_the_maximum(tmp_path) -> None:
    service, _session, _sheet, graded = await _graded_sheet(tmp_path)

    with pytest.raises(RequestValidationError) as excinfo:
        await service.override(
            graded.marks[0].id, MarkOverrideRequest(awarded_marks=6)
        )

    assert excinfo.value.error_code == code.VALIDATION_ERROR


async def test_override_raises_when_mark_missing(tmp_path) -> None:
    service, _session, _sheet, _graded = await _graded_sheet(tmp_path)

    with pytest.raises(NotFoundError) as excinfo:
        await service.override(uuid.uuid4(), MarkOverrideRequest(awarded_marks=1))

    assert excinfo.value.error_code == code.MARK_NOT_FOUND


async def test_override_rejects_an_approved_sheet(tmp_path) -> None:
    service, _session, sheet, graded = await _graded_sheet(tmp_path)
    sheet.sheet_status = SheetStatus.APPROVED.value

    with pytest.raises(ConflictError):
        await service.override(
            graded.marks[0].id, MarkOverrideRequest(awarded_marks=1)
        )


# --------------------------------------------------------------------------- #
# approve (FR-009)
# --------------------------------------------------------------------------- #
async def test_approve_finalises_a_graded_sheet(tmp_path) -> None:
    service, session, sheet, _graded = await _graded_sheet(tmp_path)

    result = await service.approve(sheet.id)

    assert result.sheet_status == SheetStatus.APPROVED.value
    assert sheet.sheet_status == SheetStatus.APPROVED.value
    assert session.commits == 3


async def test_approve_rejects_an_ungraded_sheet(tmp_path) -> None:
    exam = _exam([_question(0, 5)])
    path = tmp_path / "sheet.txt"
    path.write_text("Q1: alpha", encoding="utf-8")
    sheet = _sheet(exam, path)

    with pytest.raises(ConflictError):
        await GradingService(_FakeSession(sheet=sheet)).approve(sheet.id)


async def test_approve_rejects_an_already_approved_sheet(tmp_path) -> None:
    service, _session, sheet, _graded = await _graded_sheet(tmp_path)
    await service.approve(sheet.id)

    with pytest.raises(ConflictError):
        await service.approve(sheet.id)


async def test_rename_sets_a_trimmed_student_name(tmp_path) -> None:
    exam = _exam([_question(0, 5)])
    path = tmp_path / "sheet.txt"
    path.write_text("Q1: alpha", encoding="utf-8")
    sheet = _sheet(exam, path)
    session = _FakeSession(sheet=sheet)

    result = await GradingService(session).rename(
        sheet.id, AnswerSheetRename(student_name="  Asha  ")
    )

    assert result.student_name == "Asha"
    assert sheet.student_name == "Asha"
    assert session.commits == 1


async def test_rename_can_clear_a_student_name(tmp_path) -> None:
    exam = _exam([_question(0, 5)])
    path = tmp_path / "sheet.txt"
    path.write_text("Q1: alpha", encoding="utf-8")
    sheet = _sheet(exam, path, student_name="Asha")
    session = _FakeSession(sheet=sheet)

    result = await GradingService(session).rename(sheet.id, AnswerSheetRename())

    assert result.student_name is None
    assert sheet.student_name is None
