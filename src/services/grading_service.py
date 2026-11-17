"""Answer segmentation/alignment (FR-005) and rubric-based grading (FR-006).

Segmentation is deterministic: each answer sheet's extracted text is split on
the paper's question numbering and every block is aligned to a real
``QUESTIONS`` row. Questions the student did not answer are recorded with
``is_answered = false`` rather than dropped, and any text that cannot be aligned
is reported as an unmatched region for the teacher.

Grading marks each segmented answer against its question's rubric and model
answer, producing one ``MARKS`` row per answer with an evidence span copied
verbatim from the student's answer. Re-running either step is safe: answers are
replaced rather than duplicated, and an existing mark is updated in place.
"""

from __future__ import annotations

import re
import uuid
from decimal import Decimal
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.feedback_agent import (
    FeedbackAgent,
    FeedbackItem,
    FeedbackOutcome,
    FeedbackTarget,
)
from src.agents.grading_agent import (
    CriterionScore,
    GradingAgent,
    GradingOutcome,
    GradingTarget,
    RubricTarget,
)
from src.agents.verification_agent import (
    MarkVerificationAgent,
    VerificationOutcome,
    VerificationTarget,
)
from src.models.answer_sheet import AnswerSheetRead, AnswerSheetRename
from src.models.grading import (
    AnswerRead,
    AnswerSheetDetail,
    GradingResult,
    MarkOverrideRequest,
    MarkRead,
    SegmentationResult,
    UnmatchedRegion,
)
from src.repositories import answer_sheet_repository as sheet_repo
from src.repositories import grading_repository as repo
from src.repositories.observability_repository import agent_run
from src.repositories.schema import Answer, AnswerSheet, Feedback, Mark, Question
from src.settings import GradedBy, SheetStatus, VerifierStatus
from src.utils.exceptions import (
    ConflictError,
    FileProcessingError,
    NotFoundError,
    RequestValidationError,
)
from src.utils.exceptions import error_codes as code
from src.utils.logger import get_logger

logger = get_logger(__name__)

_GRADING_TASK = "sheet_grading"
_VERIFICATION_TASK = "mark_verification"
_FEEDBACK_TASK = "student_feedback"

_PREFIX = r"(?:q(?:uestion)?|ans(?:wer)?)"
# "Q3", "Question 3:", "Ans 2)" — the prefix makes the delimiter optional.
_PREFIXED_MARKER = re.compile(rf"^\s*{_PREFIX}\s*[.\-:]?\s*(\d{{1,3}})\b[).:\-]?\s*", re.I)
# "3)", "3:", "3-" or "3." — a bare ".", however, must not begin a decimal.
_BARE_MARKER = re.compile(r"^\s*(\d{1,3})\s*(?:[):\-]|\.(?!\d))\s*", re.I)


class GradingService:
    """Segments an answer sheet and grades its answers against the paper's rubric."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        agent: GradingAgent | None = None,
        verifier: MarkVerificationAgent | None = None,
        feedback_agent: FeedbackAgent | None = None,
    ) -> None:
        self._session = session
        self._agent = agent or GradingAgent()
        self._verifier = verifier or MarkVerificationAgent()
        self._feedback_agent = feedback_agent or FeedbackAgent()

    async def segment(self, sheet_id: uuid.UUID) -> SegmentationResult:
        """Split a sheet into per-question answers aligned to the paper (FR-005)."""

        sheet = await self._require_sheet(sheet_id)
        if sheet.sheet_status == SheetStatus.APPROVED.value:
            raise ConflictError(
                "An approved answer sheet can no longer be re-segmented.",
                error_code=code.CONFLICT,
            )

        questions = _paper_questions(sheet)
        if not questions:
            raise NotFoundError(
                "The exam's question paper has no questions to align answers against.",
                error_code=code.QUESTION_NOT_FOUND,
            )

        blocks, unmatched = _segment(_sheet_text(sheet), questions)
        confidence = _mean_confidence(sheet)

        # Replace any previous segmentation so re-running never duplicates answers
        # (and drops the marks that belonged to the answers being replaced).
        sheet.answers.clear()
        await self._session.flush()

        answers = [
            _build_answer(sheet, question, blocks.get(question.id), confidence)
            for question in questions
        ]
        await repo.add_answers(self._session, answers)
        sheet.answers.extend(answers)
        sheet.total_awarded_marks = None
        sheet.ocr_confidence = _with_segmentation(sheet.ocr_confidence, answers, unmatched)
        await self._session.commit()

        logger.info(
            "Answer sheet segmented",
            extra={
                "answer_sheet_id": str(sheet.id),
                "answers": len(answers),
                "unanswered": sum(1 for answer in answers if not answer.is_answered),
                "unmatched_regions": len(unmatched),
            },
        )
        return SegmentationResult(
            answer_sheet_id=sheet.id,
            exam_id=sheet.exam_id,
            sheet_status=sheet.sheet_status,
            total_questions=len(questions),
            answered=sum(1 for answer in answers if answer.is_answered),
            unanswered=sum(1 for answer in answers if not answer.is_answered),
            answers=[AnswerRead.from_model(answer) for answer in answers],
            unmatched_regions=[
                UnmatchedRegion(text=text, reason="Could not be aligned to a question")
                for text in unmatched
            ],
        )

    async def grade(self, sheet_id: uuid.UUID) -> GradingResult:
        """Mark every segmented answer against its question's rubric (FR-006).

        Each mark is then independently re-checked (FR-007); the verdict is
        recorded on the mark but never applied silently, so the teacher always
        reviews any disagreement.
        """

        sheet = await self._require_sheet(sheet_id)
        if sheet.sheet_status == SheetStatus.APPROVED.value:
            raise ConflictError(
                "An approved answer sheet can no longer be re-graded.",
                error_code=code.CONFLICT,
            )

        answers = sorted(sheet.answers, key=lambda item: item.question.order_index)
        if not answers:
            raise ConflictError(
                "Segment the answer sheet before grading it.",
                error_code=code.CONFLICT,
            )

        subject = sheet.exam.subject if sheet.exam is not None else None
        marks: list[Mark] = []
        async with agent_run(
            _GRADING_TASK,
            task=_GRADING_TASK,
            exam_id=sheet.exam_id,
            answer_sheet_id=sheet.id,
            input_summary={"answers": len(answers)},
        ) as run:
            for answer in answers:
                outcome = await self._agent.grade(subject=subject, target=_target(answer))
                mark = answer.mark
                if mark is None:
                    mark = Mark(answer_id=answer.id)
                    answer.mark = mark
                    await repo.add_mark(self._session, mark)
                _apply_mark(mark, outcome)
                marks.append(mark)
            flagged = await self._verify(subject, answers, marks, sheet)
            total = _round(sum(float(mark.awarded_marks) for mark in marks))
            feedback = await self._compose_feedback(sheet, answers, marks, subject)
            run["marks"] = len(marks)
            run["total_awarded_marks"] = total
            run["flagged"] = flagged
            run["feedback"] = bool(feedback.overall_comments)

        sheet.total_awarded_marks = Decimal(f"{total:.2f}")
        sheet.sheet_status = SheetStatus.GRADED.value
        await self._session.commit()

        logger.info(
            "Answer sheet graded",
            extra={
                "answer_sheet_id": str(sheet.id),
                "marks": len(marks),
                "total_awarded_marks": total,
                "flagged": flagged,
            },
        )
        return GradingResult(
            answer_sheet_id=sheet.id,
            exam_id=sheet.exam_id,
            sheet_status=sheet.sheet_status,
            total_awarded_marks=total,
            total_max_marks=_round(sum(float(mark.max_marks) for mark in marks)),
            marks=[MarkRead.from_model(mark) for mark in marks],
        )

    async def _verify(
        self,
        subject: str | None,
        answers: list[Answer],
        marks: list[Mark],
        sheet: AnswerSheet,
    ) -> int:
        """Independently re-check every mark; return how many were disputed (FR-007)."""

        flagged = 0
        async with agent_run(
            _VERIFICATION_TASK,
            task=_VERIFICATION_TASK,
            exam_id=sheet.exam_id,
            answer_sheet_id=sheet.id,
            input_summary={"marks": len(marks)},
        ) as run:
            for answer, mark in zip(answers, marks, strict=True):
                outcome = await self._verifier.verify(
                    subject=subject, target=_verification_target(answer, mark)
                )
                _apply_verification(mark, outcome)
                if outcome.status != VerifierStatus.AGREED.value:
                    flagged += 1
            run["flagged"] = flagged
        return flagged

    async def _compose_feedback(
        self,
        sheet: AnswerSheet,
        answers: list[Answer],
        marks: list[Mark],
        subject: str | None,
    ) -> FeedbackOutcome:
        """Write and persist the sheet's per-student feedback from the marks (FR-008)."""

        target = _feedback_target(sheet, answers, marks, subject)
        async with agent_run(
            _FEEDBACK_TASK,
            task=_FEEDBACK_TASK,
            exam_id=sheet.exam_id,
            answer_sheet_id=sheet.id,
            input_summary={"marks": len(marks)},
        ) as run:
            outcome = await self._feedback_agent.compose(target=target)
            feedback = sheet.feedback
            if feedback is None:
                feedback = Feedback(answer_sheet_id=sheet.id)
                sheet.feedback = feedback
                self._session.add(feedback)
            feedback.overall_comments = outcome.overall_comments
            feedback.strengths = outcome.strengths
            feedback.improvements = outcome.improvements
            run["overall_comments"] = outcome.overall_comments
        return outcome

    async def override(self, mark_id: uuid.UUID, payload: MarkOverrideRequest) -> MarkRead:
        """Apply a teacher's override to one AI mark (FR-009).

        The first override preserves the agent's value in ``original_marks`` so the
        audit trail survives later edits, and the sheet total is recomputed from
        its marks. An approved sheet is final and can no longer be changed.
        """

        mark = await repo.get_mark(self._session, mark_id)
        if mark is None or not mark.is_active:
            raise NotFoundError(
                f"Mark {mark_id} not found", error_code=code.MARK_NOT_FOUND
            )
        sheet = mark.answer.answer_sheet
        if sheet.sheet_status == SheetStatus.APPROVED.value:
            raise ConflictError(
                "An approved answer sheet's marks can no longer be changed.",
                error_code=code.CONFLICT,
            )
        if round(payload.awarded_marks, 2) > float(mark.max_marks):
            raise RequestValidationError(
                f"The awarded mark cannot exceed the question's maximum of "
                f"{float(mark.max_marks):g}.",
                error_code=code.VALIDATION_ERROR,
            )

        if not mark.is_overridden:
            mark.original_marks = mark.awarded_marks
        mark.awarded_marks = _decimal(payload.awarded_marks)
        mark.is_overridden = True
        mark.graded_by = GradedBy.TEACHER.value
        if payload.reason:
            mark.reasoning = _override_reason(mark.reasoning, payload.reason)

        sheet.total_awarded_marks = _decimal(_marks_total(sheet))
        if sheet.sheet_status == SheetStatus.GRADED.value:
            sheet.sheet_status = SheetStatus.REVIEWED.value

        answers: list[Answer] = []
        marks: list[Mark] = []
        for answer in sorted(sheet.answers, key=lambda item: item.question.order_index):
            if answer.mark is not None:
                answers.append(answer)
                marks.append(answer.mark)
        subject = sheet.exam.subject if sheet.exam is not None else None
        await self._compose_feedback(sheet, answers, marks, subject)

        await self._session.commit()

        logger.info(
            "Mark overridden by teacher",
            extra={
                "mark_id": str(mark.id),
                "answer_sheet_id": str(sheet.id),
                "awarded_marks": float(mark.awarded_marks),
            },
        )
        return MarkRead.from_model(mark)

    async def approve(self, sheet_id: uuid.UUID) -> AnswerSheetRead:
        """Approve a graded sheet, finalising its results (FR-009)."""

        sheet = await self._require_sheet(sheet_id)
        if sheet.sheet_status == SheetStatus.APPROVED.value:
            raise ConflictError(
                "The answer sheet is already approved.", error_code=code.CONFLICT
            )
        if sheet.sheet_status not in {
            SheetStatus.GRADED.value,
            SheetStatus.REVIEWED.value,
        }:
            raise ConflictError(
                "Only a graded answer sheet can be approved.",
                error_code=code.CONFLICT,
            )

        sheet.sheet_status = SheetStatus.APPROVED.value
        await self._session.commit()

        logger.info(
            "Answer sheet approved",
            extra={
                "answer_sheet_id": str(sheet.id),
                "total_awarded_marks": float(sheet.total_awarded_marks or 0),
            },
        )
        return AnswerSheetRead.from_model(sheet)

    async def get_detail(self, sheet_id: uuid.UUID) -> AnswerSheetDetail:
        """Return a sheet with its segmented answers, marks and feedback."""

        sheet = await self._require_sheet(sheet_id)
        return AnswerSheetDetail.from_model(sheet)

    async def rename(
        self, sheet_id: uuid.UUID, payload: AnswerSheetRename
    ) -> AnswerSheetRead:
        """Set or clear the student name shown for an uploaded sheet."""

        sheet = await self._require_sheet(sheet_id)
        sheet.student_name = (payload.student_name or "").strip() or None
        await self._session.commit()
        logger.info(
            "Answer sheet renamed",
            extra={
                "answer_sheet_id": str(sheet.id),
                "student_name": sheet.student_name,
            },
        )
        return AnswerSheetRead.from_model(sheet)

    async def _require_sheet(self, sheet_id: uuid.UUID) -> AnswerSheet:
        """Load an answer sheet, raising :class:`NotFoundError` when absent."""

        sheet = await sheet_repo.get(self._session, sheet_id)
        if sheet is None or not sheet.is_active:
            raise NotFoundError(
                f"Answer sheet {sheet_id} not found",
                error_code=code.ANSWER_SHEET_NOT_FOUND,
            )
        return sheet


def _paper_questions(sheet: AnswerSheet) -> list[Question]:
    """Return the exam's paper questions in paper order."""

    paper = sheet.exam.question_paper if sheet.exam is not None else None
    if paper is None:
        return []
    return sorted(paper.questions, key=lambda question: question.order_index)


def _sheet_text(sheet: AnswerSheet) -> str:
    """Read the text extracted (or OCR'd) at ingestion time for this sheet."""

    if not sheet.file_path:
        raise FileProcessingError(
            "The answer sheet has no stored file to segment.",
            error_code=code.FILE_PROCESSING_ERROR,
        )
    path = Path(sheet.file_path)
    for candidate in (path.with_name(f"{path.name}.ocr.txt"), path):
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                return text
    raise FileProcessingError(
        f"No extracted text is available for '{path.name}'.",
        error_code=code.FILE_PROCESSING_ERROR,
    )


def _segment(
    text: str, questions: list[Question]
) -> tuple[dict[uuid.UUID, str], list[str]]:
    """Split the sheet text into per-question blocks.

    Two numbering conventions occur in practice: paper-wide numbering (``Q1``..
    ``Q12``) and section-local numbering (``Section A: 1..10``, ``Section B:
    1..2``). Both are parsed and the one that aligns more answers is kept, so a
    sectioned sheet no longer collapses its last section into one question.

    Section headers win ties because they describe the question type, while the
    sequential parser only knows the paper's storage order. That matters for
    papers printed in grouped sections, where the display order (MCQs, then
    short, then long answers) differs from the order rows were stored in.
    """

    sequential, sequential_unmatched = _segment_sequential(text, questions)
    sectioned, sectioned_unmatched = _segment_sections(text, questions)
    if _answered_count(sectioned) >= _answered_count(sequential):
        return sectioned, sectioned_unmatched
    return sequential, sequential_unmatched


def _segment_sequential(
    text: str, questions: list[Question]
) -> tuple[dict[uuid.UUID, str], list[str]]:
    """Split the sheet text into per-question blocks, in paper order.

    A line only starts a new block when it carries a marker whose number matches
    a question on the paper *and* is greater than the block already open, so a
    numbered list inside an answer cannot hijack the alignment. Text before the
    first marker is returned as an unmatched region.
    """

    labels = {question.order_index + 1: question.id for question in questions}
    blocks: dict[uuid.UUID, list[str]] = {}
    unmatched: list[str] = []
    current: uuid.UUID | None = None
    current_label = 0
    buffer: list[str] = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if current is None:
            if content:
                unmatched.append(content)
        else:
            blocks[current] = list(buffer)
        buffer.clear()

    for line in text.splitlines():
        if _SECTION_HEADER.match(line):
            # A new section restarts numbering but the paper-wide labels still
            # apply (some sheets keep Q1..Q12 across sections).
            flush()
            current = None
            current_label = 0
            continue
        marker = _marker(line)
        if marker is not None and marker[0] in labels and marker[0] > current_label:
            flush()
            current = labels[marker[0]]
            current_label = marker[0]
            buffer.append(marker[1])
            continue
        buffer.append(line)
    flush()

    return {qid: "\n".join(lines).strip() for qid, lines in blocks.items()}, unmatched


# "Section A", "Section - A", "SECTION B: DESCRIPTIVE QUESTIONS".
_SECTION_HEADER = re.compile(r"^\s*(?:section|part)\s*[-:.]?\s*([a-z]|\d{1,2})\b", re.I)


_TYPE_ORDER = {"mcq": 0, "short": 1, "long": 2}
_MARKS_IN_HEADER = re.compile(r"(\d+(?:\.\d+)?)\s*marks?\b", re.I)


def _segment_sections(
    text: str, questions: list[Question]
) -> tuple[dict[uuid.UUID, str], list[str]]:
    """Split sectioned numbering into blocks by matching each section's type.

    Recognised headings (``Multiple Choice``, ``Short Answer``, ``Long Answer``,
    etc.) select the matching question group, including sections that number
    continuously across the paper (``Section B: 11..15``). An unlabelled
    ``Section A``/``Section B`` falls back to the group at that position (MCQs
    first, then written questions); later unlabelled sections are matched by
    their paper-wide numbers against the displayed question order.
    """

    lines = text.splitlines()
    header_at = [
        index for index, line in enumerate(lines) if _SECTION_HEADER.match(line)
    ]
    if not header_at:
        content = text.strip()
        return {}, [content] if content else []

    display = _display_order(questions)
    mcq = [question for question in display if question.question_type == "mcq"]
    short = [question for question in display if question.question_type == "short"]
    long_answers = [
        question for question in display if question.question_type == "long"
    ]
    written = [question for question in display if question.question_type != "mcq"]
    positional = [group for group in (mcq, written) if group]

    blocks: dict[uuid.UUID, str] = {}
    unmatched: list[str] = []
    preamble = "\n".join(lines[: header_at[0]]).strip()
    if preamble:
        unmatched.append(preamble)

    for section_index, start in enumerate(header_at):
        end = header_at[section_index + 1] if section_index + 1 < len(header_at) else len(lines)
        header = lines[start]
        body = lines[start + 1 : end]
        labels = _section_labels(
            header,
            body,
            section_index=section_index,
            positional=positional,
            mcq=mcq,
            short=short,
            long_answers=long_answers,
            written=written,
            display=display,
        )
        section_blocks, section_unmatched = _segment_blocks(body, labels)
        blocks.update(section_blocks)
        unmatched.extend(section_unmatched)

    return blocks, unmatched


def _display_order(questions: list[Question]) -> list[Question]:
    """Return questions in the grouped order used by the paper and PDF export."""

    return sorted(
        questions,
        key=lambda question: (
            float(question.marks),
            _TYPE_ORDER.get(question.question_type, len(_TYPE_ORDER)),
            question.question_type,
            question.order_index,
        ),
    )


def _section_labels(
    header: str,
    body: list[str],
    *,
    section_index: int,
    positional: list[list[Question]],
    mcq: list[Question],
    short: list[Question],
    long_answers: list[Question],
    written: list[Question],
    display: list[Question],
) -> dict[int, uuid.UUID]:
    """Map a section's question numbers to the paper's question ids."""

    group = _section_group(
        header,
        section_index=section_index,
        positional=positional,
        mcq=mcq,
        short=short,
        long_answers=long_answers,
        written=written,
    )
    if group is None:
        return {index + 1: question.id for index, question in enumerate(display)}
    if not group:
        return {}

    first = next(
        (marker[0] for line in body if (marker := _marker(line)) is not None), None
    )
    if first is None:
        return {}
    base = 1 if first <= len(group) else first
    return {base + index: question.id for index, question in enumerate(group)}


def _section_group(
    header: str,
    *,
    section_index: int,
    positional: list[list[Question]],
    mcq: list[Question],
    short: list[Question],
    long_answers: list[Question],
    written: list[Question],
) -> list[Question] | None:
    """Pick the question group a section heading refers to."""

    lowered = header.lower()
    marks_match = _MARKS_IN_HEADER.search(header)
    marks = float(marks_match.group(1)) if marks_match else None

    def matching(group: list[Question]) -> list[Question]:
        if marks is None:
            return group
        return [question for question in group if float(question.marks) == marks]

    if any(word in lowered for word in ("multiple choice", "mcq", "objective")):
        group = matching(mcq)
    elif any(word in lowered for word in ("long", "essay")):
        group = matching(long_answers) or matching(written)
    elif any(word in lowered for word in ("short", "very short")):
        group = matching(short) or matching(written)
    elif any(
        word in lowered
        for word in ("descriptive", "subjective", "written", "answer questions")
    ):
        group = matching(written)
    elif 0 <= section_index < len(positional):
        group = positional[section_index]
    else:
        return None

    if group:
        return group
    if 0 <= section_index < len(positional):
        return positional[section_index]
    return None


def _segment_blocks(
    lines: list[str], labels: dict[int, uuid.UUID]
) -> tuple[dict[uuid.UUID, str], list[str]]:
    """Split one section's lines into blocks using its marker labels."""

    blocks: dict[uuid.UUID, list[str]] = {}
    unmatched: list[str] = []
    current: uuid.UUID | None = None
    current_label = 0
    buffer: list[str] = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if current is None:
            if content:
                unmatched.append(content)
        else:
            blocks[current] = list(buffer)
        buffer.clear()

    for line in lines:
        marker = _marker(line)
        if marker is not None and marker[0] in labels and marker[0] > current_label:
            flush()
            current = labels[marker[0]]
            current_label = marker[0]
            buffer.append(marker[1])
            continue
        buffer.append(line)
    flush()

    return {qid: "\n".join(content).strip() for qid, content in blocks.items()}, unmatched


def _answered_count(blocks: dict[uuid.UUID, str]) -> int:
    """Return how many blocks carry a non-empty answer."""

    return sum(1 for content in blocks.values() if content.strip())


def _marker(line: str) -> tuple[int, str] | None:
    """Return a line's question ``(number, remainder)`` when it starts a marker."""

    for pattern in (_PREFIXED_MARKER, _BARE_MARKER):
        match = pattern.match(line)
        if match:
            return int(match.group(1)), line[match.end() :]
    return None


def _build_answer(
    sheet: AnswerSheet,
    question: Question,
    raw: str | None,
    confidence: Decimal | None,
) -> Answer:
    """Build one per-question answer row for ``question``."""

    answer_text = (raw or "").strip()
    answered = bool(answer_text)
    return Answer(
        answer_sheet_id=sheet.id,
        question_id=question.id,
        question=question,
        mark=None,
        answer_text=answer_text if answered else None,
        is_answered=answered,
        ocr_confidence=confidence,
    )


def _target(answer: Answer) -> GradingTarget:
    """Build the grading target for one answer and its question's rubric."""

    question = answer.question
    rubrics = sorted(question.rubric_criteria, key=lambda item: item.order_index)
    max_marks = float(question.marks)
    criteria = [
        RubricTarget(item.id, item.criterion_text, float(item.max_marks)) for item in rubrics
    ]
    return GradingTarget(
        question_text=question.question_text,
        question_type=question.question_type,
        model_answer=question.model_answer,
        max_marks=max_marks,
        answer_text=answer.answer_text,
        is_answered=answer.is_answered,
        rubrics=criteria
        or [
            RubricTarget(
                None,
                "Award marks for an answer consistent with the model answer.",
                max_marks,
            )
        ],
    )


def _feedback_target(
    sheet: AnswerSheet,
    answers: list[Answer],
    marks: list[Mark],
    subject: str | None,
) -> FeedbackTarget:
    """Build the feedback writer's view of a marked sheet from the awarded marks."""

    items = [
        FeedbackItem(
            question_text=answer.question.question_text,
            topic=answer.question.topic,
            max_marks=float(mark.max_marks),
            awarded_marks=float(mark.awarded_marks),
            is_answered=answer.is_answered,
        )
        for answer, mark in zip(answers, marks, strict=True)
    ]
    return FeedbackTarget(
        total_awarded_marks=_round(sum(item.awarded_marks for item in items)),
        total_max_marks=_round(sum(item.max_marks for item in items)),
        student_name=sheet.student.name if sheet.student is not None else None,
        subject=subject,
        items=items,
    )


def _apply_mark(mark: Mark, outcome: GradingOutcome) -> None:
    """Write a grading outcome onto a mark row, replacing any previous value."""

    mark.awarded_marks = _decimal(outcome.awarded_marks)
    mark.max_marks = _decimal(outcome.max_marks)
    mark.rubric_scores = [
        {
            "criterion_id": str(score.criterion_id) if score.criterion_id else None,
            "criterion_text": score.criterion_text,
            "max_marks": score.max_marks,
            "awarded_marks": score.awarded_marks,
        }
        for score in outcome.rubric_scores
    ]
    mark.evidence = outcome.evidence
    mark.reasoning = outcome.reasoning
    mark.confidence = _decimal(outcome.confidence, places=4)
    mark.graded_by = GradedBy.AGENT.value
    mark.is_overridden = False
    mark.original_marks = None


def _verification_target(answer: Answer, mark: Mark) -> VerificationTarget:
    """Build the verifier's view of one mark.

    The grader's reasoning is deliberately withheld so the check is genuinely
    independent; only the question, model answer, rubric, the student's answer,
    the cited evidence and the awarded mark are exposed.
    """

    question = answer.question
    max_marks = mark.max_marks if mark.max_marks is not None else question.marks
    return VerificationTarget(
        question_text=question.question_text,
        question_type=question.question_type,
        model_answer=question.model_answer,
        max_marks=float(max_marks),
        answer_text=answer.answer_text,
        is_answered=answer.is_answered,
        awarded_marks=float(mark.awarded_marks),
        evidence=mark.evidence,
        rubric_scores=_criterion_scores(mark.rubric_scores),
    )


def _criterion_scores(rubric_scores: object) -> list[CriterionScore]:
    """Rebuild the stored rubric scores as verifier-facing criterion scores."""

    if not isinstance(rubric_scores, list):
        return []
    scores: list[CriterionScore] = []
    for entry in rubric_scores:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("criterion_id")
        scores.append(
            CriterionScore(
                criterion_id=_uuid_or_none(raw_id),
                criterion_text=str(entry.get("criterion_text") or ""),
                max_marks=_number(entry.get("max_marks")),
                awarded_marks=_number(entry.get("awarded_marks")),
            )
        )
    return scores


def _apply_verification(mark: Mark, outcome: VerificationOutcome) -> None:
    """Record the verifier's verdict on a mark without altering the awarded mark."""

    mark.verifier_status = outcome.status
    mark.verifier_notes = outcome.notes


def _marks_total(sheet: AnswerSheet) -> float:
    """Sum the current marks on a sheet (used after a teacher override)."""

    return _round(
        sum(
            float(answer.mark.awarded_marks)
            for answer in sheet.answers
            if answer.mark is not None
        )
    )


def _override_reason(reasoning: str | None, reason: str) -> str:
    """Append the teacher's override reason to the mark's reasoning for audit."""

    note = f"Teacher override: {reason.strip()}"
    return f"{reasoning}\n\n{note}" if reasoning else note


def _uuid_or_none(value: object) -> uuid.UUID | None:
    """Coerce a stored criterion id back to a UUID, or ``None`` when unusable."""

    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str):
        try:
            return uuid.UUID(value)
        except ValueError:
            return None
    return None


def _number(value: object) -> float:
    """Coerce a stored JSON number to a float, defaulting to ``0.0``."""

    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _mean_confidence(sheet: AnswerSheet) -> Decimal | None:
    """Return the sheet's OCR confidence mean, when it has one."""

    payload = sheet.ocr_confidence
    value = payload.get("mean") if isinstance(payload, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return Decimal(str(round(float(value), 4)))


def _with_segmentation(
    confidence: dict | None, answers: list[Answer], unmatched: list[str]
) -> dict:
    """Merge a segmentation summary into the sheet's confidence payload."""

    payload = dict(confidence) if isinstance(confidence, dict) else {}
    payload["segmentation"] = {
        "answers": len(answers),
        "unanswered": sum(1 for answer in answers if not answer.is_answered),
        "unmatched_regions": len(unmatched),
    }
    return payload


def _decimal(value: float, *, places: int = 2) -> Decimal:
    """Convert a float mark to a fixed-precision :class:`Decimal`."""

    return Decimal(str(round(value, places)))


def _round(value: float) -> float:
    """Round a mark total to two decimal places."""

    return round(value, 2)


__all__ = ["GradingService"]
