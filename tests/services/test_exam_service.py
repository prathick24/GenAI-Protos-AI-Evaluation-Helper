"""Tests for the exam service (FR-004)."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from src.models.exam import ExamCreate
from src.repositories.schema import (
    Answer,
    AnswerSheet,
    Exam,
    Mark,
    Question,
    QuestionPaper,
    Student,
)
from src.services import exam_service
from src.services.exam_service import ExamService
from src.settings import PaperStatus
from src.utils.exceptions import ConflictError, NotFoundError
from src.utils.exceptions import error_codes as code

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _FakeSession:
    """Minimal AsyncSession stand-in that stamps audit fields and records writes."""

    def __init__(
        self, *, paper: QuestionPaper | None = None, exam: Exam | None = None
    ) -> None:
        self._paper = paper
        self._exam = exam
        self.commits = 0
        self.flushes = 0
        self.added: list[object] = []

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        self.flushes += 1
        self._stamp(obj=None)

    async def commit(self) -> None:
        self.commits += 1

    def _stamp(self, obj: object | None) -> None:
        for item in self.added:
            _apply_defaults(item)

    async def get(self, model: object, key: object) -> object | None:
        if model is QuestionPaper:
            return self._paper
        if model is Exam:
            return self._exam
        return None


def _apply_defaults(row: object) -> None:
    """Emulate the column defaults SQLAlchemy applies at INSERT time."""

    if getattr(row, "id", None) is None:
        row.id = uuid.uuid4()
    for attr, value in (("created_at", _NOW), ("updated_at", _NOW), ("is_active", True)):
        if hasattr(row, attr) and getattr(row, attr) is None:
            setattr(row, attr, value)
    if hasattr(row, "status") and row.status is None:  # type: ignore[attr-defined]
        row.status = "draft"


def _paper(**overrides: object) -> QuestionPaper:
    defaults: dict[str, object] = {
        "id": uuid.uuid4(),
        "syllabus_id": uuid.uuid4(),
        "title": "Midterm",
        "subject": "Science",
        "class_section": "B",
        "total_marks": 50,
        "version": 1,
        "source": "generated",
        "status": PaperStatus.APPROVED.value,
        "created_at": _NOW,
        "updated_at": _NOW,
        "is_active": True,
    }
    defaults.update(overrides)
    return QuestionPaper(**defaults)


def _exam(**overrides: object) -> Exam:
    defaults: dict[str, object] = {
        "id": uuid.uuid4(),
        "title": "Midterm exam",
        "total_marks": 50,
        "status": "draft",
        "question_paper_id": uuid.uuid4(),
        "created_at": _NOW,
        "updated_at": _NOW,
        "is_active": True,
    }
    defaults.update(overrides)
    return Exam(**defaults)


async def test_create_links_an_approved_paper() -> None:
    paper = _paper()
    session = _FakeSession(paper=paper)
    payload = ExamCreate(
        title="Midterm exam", question_paper_id=paper.id, exam_date=date(2026, 3, 1)
    )

    result = await ExamService(session).create(payload)

    assert result.title == "Midterm exam"
    assert result.question_paper_id == paper.id
    assert result.total_marks == 50
    assert result.status == "draft"
    assert result.subject == "Science"
    assert result.class_section == "B"
    assert session.commits == 1


async def test_create_prefers_explicit_metadata() -> None:
    paper = _paper(subject="Science", class_section="B")
    session = _FakeSession(paper=paper)
    payload = ExamCreate(
        title="Midterm exam",
        question_paper_id=paper.id,
        subject="Physics",
        class_section="A",
    )

    result = await ExamService(session).create(payload)

    assert result.subject == "Physics"
    assert result.class_section == "A"


async def test_create_rejects_an_unapproved_paper() -> None:
    paper = _paper(status=PaperStatus.VALIDATED.value)
    session = _FakeSession(paper=paper)

    with pytest.raises(ConflictError):
        await ExamService(session).create(
            ExamCreate(title="Midterm exam", question_paper_id=paper.id)
        )


async def test_create_raises_when_paper_missing() -> None:
    session = _FakeSession(paper=None)

    with pytest.raises(NotFoundError) as excinfo:
        await ExamService(session).create(
            ExamCreate(title="Midterm exam", question_paper_id=uuid.uuid4())
        )

    assert excinfo.value.error_code == code.QUESTION_PAPER_NOT_FOUND


async def test_get_returns_exam() -> None:
    exam = _exam()
    session = _FakeSession(exam=exam)

    result = await ExamService(session).get(exam.id)

    assert result.id == exam.id


async def test_get_raises_when_missing() -> None:
    session = _FakeSession(exam=None)

    with pytest.raises(NotFoundError) as excinfo:
        await ExamService(session).get(uuid.uuid4())

    assert excinfo.value.error_code == code.EXAM_NOT_FOUND


async def test_list_returns_exams(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_exam(), _exam(title="Final exam")]

    async def _list_all(session: object) -> list[Exam]:
        return rows

    monkeypatch.setattr(exam_service.repo, "list_all", _list_all)

    result = await ExamService(_FakeSession()).list()

    assert [item.title for item in result] == ["Midterm exam", "Final exam"]


# --------------------------------------------------------------------------- #
# marksheet and analytics (FR-008)
# --------------------------------------------------------------------------- #
def _graded_question(order_index: int, marks: float, topic: str) -> Question:
    question = Question(
        id=uuid.uuid4(),
        question_paper_id=uuid.uuid4(),
        order_index=order_index,
        question_text=f"Question {order_index + 1}",
        question_type="short",
        marks=Decimal(str(marks)),
        topic=topic,
    )
    question.rubric_criteria = []
    question.answers = []
    return question


def _graded_sheet(
    exam: Exam, name: str | None, scores: list[tuple[Question, float]]
) -> AnswerSheet:
    sheet = AnswerSheet(
        id=uuid.uuid4(),
        exam_id=exam.id,
        file_path=None,
        input_type="typed",
        ocr_status="done",
        sheet_status="graded",
        total_awarded_marks=Decimal(str(sum(awarded for _, awarded in scores))),
        ocr_confidence=None,
        created_at=_NOW,
        updated_at=_NOW,
        is_active=True,
    )
    sheet.exam = exam
    sheet.student = (
        Student(
            id=uuid.uuid4(),
            name=name,
            roll_no=name,
            class_section="B",
            created_at=_NOW,
            updated_at=_NOW,
            is_active=True,
        )
        if name
        else None
    )
    answers: list[Answer] = []
    for question, awarded in scores:
        answer = Answer(
            id=uuid.uuid4(),
            answer_sheet_id=sheet.id,
            question_id=question.id,
            is_answered=True,
        )
        answer.question = question
        mark = Mark(
            id=uuid.uuid4(),
            answer_id=answer.id,
            awarded_marks=Decimal(str(awarded)),
            max_marks=question.marks,
        )
        mark.answer = answer
        answer.mark = mark
        answers.append(answer)
    sheet.answers = answers
    return sheet


def _exam_with_paper(questions: list[Question]) -> Exam:
    paper = QuestionPaper(id=uuid.uuid4(), title="Paper", total_marks=10)
    paper.questions = questions
    exam = _exam(total_marks=10, question_paper_id=paper.id)
    exam.question_paper = paper
    return exam


async def test_marksheet_prefers_the_uploaded_student_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = _graded_question(0, 5, "Algebra")
    exam = _exam_with_paper([question])
    sheet = _graded_sheet(exam, None, [(question, 4)])
    sheet.student_name = "Asha"

    async def _list_for_exam(session: object, exam_id: object) -> list[AnswerSheet]:
        return [sheet]

    monkeypatch.setattr(exam_service.sheet_repo, "list_for_exam", _list_for_exam)

    result = await ExamService(_FakeSession(exam=exam)).marksheet(exam.id)

    assert result.rows[0].student_name == "Asha"


async def test_marksheet_builds_one_row_per_sheet(monkeypatch: pytest.MonkeyPatch) -> None:
    q1 = _graded_question(0, 5, "Algebra")
    q2 = _graded_question(1, 5, "Geometry")
    exam = _exam_with_paper([q1, q2])
    sheets = [
        _graded_sheet(exam, "Asha", [(q1, 5), (q2, 2)]),
        _graded_sheet(exam, None, [(q1, 3), (q2, 5)]),
    ]

    async def _list_for_exam(session: object, exam_id: object) -> list[AnswerSheet]:
        return sheets

    monkeypatch.setattr(exam_service.sheet_repo, "list_for_exam", _list_for_exam)

    result = await ExamService(_FakeSession(exam=exam)).marksheet(exam.id)

    assert result.exam_id == exam.id
    assert result.total_marks == 10
    assert [row.student_name for row in result.rows][0] == "Asha"
    assert result.rows[0].total_awarded_marks == 7.0
    assert result.rows[0].per_question == {str(q1.id): 5.0, str(q2.id): 2.0}
    assert result.rows[1].student_id is None
    assert result.rows[1].student_name.startswith("Unassigned")


async def test_marksheet_skips_ungraded_sheets(monkeypatch: pytest.MonkeyPatch) -> None:
    question = _graded_question(0, 5, "Algebra")
    exam = _exam_with_paper([question])
    graded = _graded_sheet(exam, "Asha", [(question, 4)])
    uploaded = _graded_sheet(exam, "Ravi", [(question, 0)])
    uploaded.sheet_status = "uploaded"

    async def _list_for_exam(session: object, exam_id: object) -> list[AnswerSheet]:
        return [graded, uploaded]

    monkeypatch.setattr(exam_service.sheet_repo, "list_for_exam", _list_for_exam)

    result = await ExamService(_FakeSession(exam=exam)).marksheet(exam.id)

    assert [row.student_name for row in result.rows] == ["Asha"]


async def test_analytics_summarises_class_performance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q1 = _graded_question(0, 5, "Algebra")
    q2 = _graded_question(1, 5, "Geometry")
    exam = _exam_with_paper([q1, q2])
    sheets = [
        _graded_sheet(exam, "Asha", [(q1, 5), (q2, 2)]),
        _graded_sheet(exam, "Ravi", [(q1, 3), (q2, 5)]),
    ]

    async def _list_for_exam(session: object, exam_id: object) -> list[AnswerSheet]:
        return sheets

    monkeypatch.setattr(exam_service.sheet_repo, "list_for_exam", _list_for_exam)

    result = await ExamService(_FakeSession(exam=exam)).analytics(exam.id)

    assert result.average == 7.5
    assert result.highest == 8.0
    assert result.lowest == 7.0
    assert result.distribution["60-80"] == 1
    assert result.distribution["80-100"] == 1
    assert result.distribution["0-20"] == 0
    assert result.per_topic_strength == {"Algebra": 0.8, "Geometry": 0.7}
    assert result.hardest_questions == [q2.id, q1.id]
    assert result.easiest_questions == [q1.id, q2.id]


async def test_analytics_handles_an_exam_without_marks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exam = _exam_with_paper([_graded_question(0, 5, "Algebra")])

    async def _list_for_exam(session: object, exam_id: object) -> list[AnswerSheet]:
        return []

    monkeypatch.setattr(exam_service.sheet_repo, "list_for_exam", _list_for_exam)

    result = await ExamService(_FakeSession(exam=exam)).analytics(exam.id)

    assert result.average == 0.0
    assert result.highest == 0.0
    assert result.lowest == 0.0
    assert result.distribution == {
        "0-20": 0,
        "20-40": 0,
        "40-60": 0,
        "60-80": 0,
        "80-100": 0,
    }
    assert result.hardest_questions == []
    assert result.easiest_questions == []
