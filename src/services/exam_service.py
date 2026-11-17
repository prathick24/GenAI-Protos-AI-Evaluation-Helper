"""Business logic for exams (FR-004).

An exam is the event that administers an *approved* question paper; answer
sheets are uploaded against it. Creating an exam therefore requires the linked
paper to exist and to have been approved in FR-003.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.exam import ExamCreate, ExamRead
from src.models.results import ClassAnalytics, Marksheet, MarksheetRow
from src.repositories import answer_sheet_repository as sheet_repo
from src.repositories import exam_repository as repo
from src.repositories import paper_repository as paper_repo
from src.repositories.schema import AnswerSheet, Exam
from src.settings import PaperStatus, SheetStatus
from src.utils.exceptions import ConflictError, NotFoundError
from src.utils.exceptions import error_codes as code
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExamService:
    """Orchestrates exam creation and read operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, payload: ExamCreate) -> ExamRead:
        """Create an exam linked to an approved question paper."""

        paper = await paper_repo.get_paper(self._session, payload.question_paper_id)
        if paper is None or not paper.is_active:
            raise NotFoundError(
                f"Question paper {payload.question_paper_id} not found",
                error_code=code.QUESTION_PAPER_NOT_FOUND,
            )
        if paper.status != PaperStatus.APPROVED.value:
            raise ConflictError(
                "An exam requires an approved question paper.",
                error_code=code.CONFLICT,
            )

        exam = Exam(
            title=payload.title,
            subject=payload.subject or paper.subject,
            class_section=payload.class_section or paper.class_section,
            exam_date=payload.exam_date,
            total_marks=paper.total_marks,
            question_paper_id=paper.id,
        )
        await repo.add(self._session, exam)
        await self._session.commit()
        logger.info("Exam created", extra={"exam_id": str(exam.id), "paper_id": str(paper.id)})
        return ExamRead.from_model(exam)

    async def list(self) -> list[ExamRead]:
        """Return all active exams."""

        rows = await repo.list_all(self._session)
        return [ExamRead.from_model(row) for row in rows]

    async def get(self, exam_id: uuid.UUID) -> ExamRead:
        """Return a single exam or raise :class:`NotFoundError`."""

        return ExamRead.from_model(await self._require(exam_id))

    async def marksheet(self, exam_id: uuid.UUID) -> Marksheet:
        """Return the per-student, per-question marksheet for an exam (FR-008)."""

        exam = await self._require(exam_id)
        sheets = [
            sheet
            for sheet in await sheet_repo.list_for_exam(self._session, exam_id)
            if sheet.sheet_status
            in {
                SheetStatus.GRADED.value,
                SheetStatus.REVIEWED.value,
                SheetStatus.APPROVED.value,
            }
        ]
        rows = [
            MarksheetRow(
                answer_sheet_id=sheet.id,
                student_id=sheet.student_id,
                student_name=_student_name(sheet),
                total_awarded_marks=_sheet_total(sheet),
                per_question=_per_question(sheet),
            )
            for sheet in sheets
        ]
        total_marks = exam.total_marks or max(
            (int(sum(float(question.marks) for question in _paper_questions(sheet)))
             for sheet in sheets),
            default=0,
        )
        logger.info(
            "Marksheet produced", extra={"exam_id": str(exam.id), "rows": len(rows)}
        )
        return Marksheet(exam_id=exam.id, total_marks=total_marks, rows=rows)

    async def analytics(self, exam_id: uuid.UUID) -> ClassAnalytics:
        """Compute the class-level analytics for an exam from stored marks (FR-008)."""

        exam = await self._require(exam_id)
        sheets = await sheet_repo.list_for_exam(self._session, exam_id)

        totals: list[float] = []
        percentages: list[float] = []
        question_stats: dict[uuid.UUID, list[float]] = {}
        topic_stats: dict[str, list[float]] = {}
        for sheet in sheets:
            awarded, maximum = _sheet_scores(sheet)
            if maximum <= 0:
                continue
            totals.append(round(awarded, 2))
            percentages.append(awarded / maximum * 100)
            for answer in sheet.answers:
                mark = answer.mark
                if mark is None:
                    continue
                earned = float(mark.awarded_marks)
                possible = float(mark.max_marks)
                _accumulate(question_stats, answer.question_id, earned, possible)
                topic = (answer.question.topic or "").strip()
                if topic:
                    _accumulate(topic_stats, topic, earned, possible)

        ranked = sorted(
            question_stats.items(),
            key=lambda entry: (_ratio(entry[1]), str(entry[0])),
        )
        logger.info(
            "Class analytics produced",
            extra={"exam_id": str(exam.id), "graded_sheets": len(totals)},
        )
        return ClassAnalytics(
            exam_id=exam.id,
            average=_round(sum(totals) / len(totals)) if totals else 0.0,
            highest=_round(max(totals)) if totals else 0.0,
            lowest=_round(min(totals)) if totals else 0.0,
            distribution=_distribution(percentages),
            per_topic_strength={
                topic: round(_ratio(stats), 4) for topic, stats in topic_stats.items()
            },
            hardest_questions=[question_id for question_id, _ in ranked[:3]],
            easiest_questions=[question_id for question_id, _ in reversed(ranked[-3:])],
        )

    async def _require(self, exam_id: uuid.UUID) -> Exam:
        """Load an exam, raising :class:`NotFoundError` when absent or deleted."""

        row = await repo.get(self._session, exam_id)
        if row is None or not row.is_active:
            raise NotFoundError(
                f"Exam {exam_id} not found",
                error_code=code.EXAM_NOT_FOUND,
            )
        return row


_DISTRIBUTION_BUCKETS = (
    ("0-20", 0.0, 20.0),
    ("20-40", 20.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60-80", 60.0, 80.0),
    ("80-100", 80.0, 100.0),
)


def _student_name(sheet: AnswerSheet) -> str:
    """Return the student's name, or a stable label for an unassigned sheet."""

    if sheet.student_name:
        return sheet.student_name
    if sheet.student is not None:
        return sheet.student.name
    return f"Unassigned ({str(sheet.id)[:8]})"


def _paper_questions(sheet: AnswerSheet) -> list[object]:
    """Return the exam's paper questions in paper order."""

    paper = sheet.exam.question_paper if sheet.exam is not None else None
    if paper is None:
        return []
    return sorted(paper.questions, key=lambda question: question.order_index)


def _per_question(sheet: AnswerSheet) -> dict[str, float]:
    """Map each graded question's id to the marks awarded for it."""

    return {
        str(answer.question_id): float(answer.mark.awarded_marks)
        for answer in sheet.answers
        if answer.mark is not None
    }


def _sheet_scores(sheet: AnswerSheet) -> tuple[float, float]:
    """Return the sheet's summed awarded and maximum marks."""

    awarded = maximum = 0.0
    for answer in sheet.answers:
        mark = answer.mark
        if mark is None:
            continue
        awarded += float(mark.awarded_marks)
        maximum += float(mark.max_marks)
    return awarded, maximum


def _sheet_total(sheet: AnswerSheet) -> float:
    """Return the sheet total, falling back to the sum of its marks."""

    if sheet.total_awarded_marks is not None:
        return round(float(sheet.total_awarded_marks), 2)
    awarded, _ = _sheet_scores(sheet)
    return round(awarded, 2)


def _accumulate(
    stats: dict[object, list[float]], key: object, earned: float, possible: float
) -> None:
    """Add one mark to an ``[earned, possible]`` accumulator."""

    entry = stats.setdefault(key, [0.0, 0.0])
    entry[0] += earned
    entry[1] += possible


def _ratio(stats: list[float]) -> float:
    """Return the awarded/possible ratio of an accumulator."""

    return stats[0] / stats[1] if stats[1] > 0 else 0.0


def _distribution(percentages: list[float]) -> dict[str, int]:
    """Bucket the per-sheet percentages into fixed score bands."""

    counts = {label: 0 for label, _, _ in _DISTRIBUTION_BUCKETS}
    for percentage in percentages:
        for label, low, high in _DISTRIBUTION_BUCKETS:
            if low <= percentage < high or (high == 100.0 and percentage >= 100.0):
                counts[label] += 1
                break
    return counts


def _round(value: float) -> float:
    """Round an analytics value to two decimal places."""

    return round(value, 2)
