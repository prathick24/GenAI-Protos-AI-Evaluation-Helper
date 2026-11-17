"""Pydantic response models for answer segmentation and grading (FR-005, FR-006)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.answer_sheet import AnswerSheetRead

if TYPE_CHECKING:
    from src.repositories.schema import Answer, Feedback, Mark


class RubricScoreRead(BaseModel):
    """One rubric criterion's awarded score within a mark."""

    criterion_id: UUID | None = None
    criterion_text: str
    max_marks: float
    awarded_marks: float

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RubricScoreRead:
        """Build the model from one ``rubric_scores`` JSON entry."""

        raw_id = payload.get("criterion_id")
        return cls(
            criterion_id=UUID(str(raw_id)) if raw_id else None,
            criterion_text=str(payload.get("criterion_text") or ""),
            max_marks=float(payload.get("max_marks") or 0.0),
            awarded_marks=float(payload.get("awarded_marks") or 0.0),
        )


class MarkRead(BaseModel):
    """The awarded mark for one answer, with its evidence and rubric breakdown."""

    id: UUID
    answer_id: UUID
    awarded_marks: float
    max_marks: float
    rubric_scores: list[RubricScoreRead] = Field(default_factory=list)
    evidence: str | None = None
    reasoning: str | None = None
    verifier_status: str | None = None
    verifier_notes: str | None = None
    confidence: float | None = None
    is_overridden: bool
    original_marks: float | None = None
    graded_by: str | None = None

    @classmethod
    def from_model(cls, mark: Mark) -> MarkRead:
        """Build the response model from a persisted :class:`Mark` row."""

        raw_scores = mark.rubric_scores if isinstance(mark.rubric_scores, list) else []
        return cls(
            id=mark.id,
            answer_id=mark.answer_id,
            awarded_marks=float(mark.awarded_marks),
            max_marks=float(mark.max_marks),
            rubric_scores=[
                RubricScoreRead.from_payload(item)
                for item in raw_scores
                if isinstance(item, dict)
            ],
            evidence=mark.evidence,
            reasoning=mark.reasoning,
            verifier_status=mark.verifier_status,
            verifier_notes=mark.verifier_notes,
            confidence=float(mark.confidence) if mark.confidence is not None else None,
            is_overridden=mark.is_overridden,
            original_marks=(
                float(mark.original_marks) if mark.original_marks is not None else None
            ),
            graded_by=mark.graded_by,
        )


class MarkOverrideRequest(BaseModel):
    """Request body for ``PATCH /marks/{markId}/override`` (FR-009)."""

    awarded_marks: float = Field(ge=0)
    reason: str | None = Field(default=None, max_length=2000)


class AnswerRead(BaseModel):
    """One per-question answer segmented from an answer sheet (FR-005)."""

    id: UUID
    answer_sheet_id: UUID
    question_id: UUID
    order_index: int
    answer_text: str | None = None
    is_answered: bool
    ocr_confidence: float | None = None
    mark: MarkRead | None = None

    @classmethod
    def from_model(cls, answer: Answer) -> AnswerRead:
        """Build the response model from a persisted :class:`Answer` row."""

        return cls(
            id=answer.id,
            answer_sheet_id=answer.answer_sheet_id,
            question_id=answer.question_id,
            order_index=answer.question.order_index,
            answer_text=answer.answer_text,
            is_answered=answer.is_answered,
            ocr_confidence=(
                float(answer.ocr_confidence) if answer.ocr_confidence is not None else None
            ),
            mark=MarkRead.from_model(answer.mark) if answer.mark is not None else None,
        )


class UnmatchedRegion(BaseModel):
    """Sheet text that could not be aligned to any question (FR-005)."""

    text: str
    reason: str


class SegmentationResult(BaseModel):
    """Response for ``POST /answer-sheets/{sheetId}/segment`` (FR-005)."""

    answer_sheet_id: UUID
    exam_id: UUID
    sheet_status: str
    total_questions: int
    answered: int
    unanswered: int
    answers: list[AnswerRead] = Field(default_factory=list)
    unmatched_regions: list[UnmatchedRegion] = Field(default_factory=list)


class GradingResult(BaseModel):
    """Response for ``POST /answer-sheets/{sheetId}/grade`` (FR-006)."""

    answer_sheet_id: UUID
    exam_id: UUID
    sheet_status: str
    total_awarded_marks: float
    total_max_marks: float
    marks: list[MarkRead] = Field(default_factory=list)


class FeedbackRead(BaseModel):
    """Per-student qualitative feedback (FR-008)."""

    overall_comments: str | None = None
    strengths: str | None = None
    improvements: str | None = None

    @classmethod
    def from_model(cls, feedback: Feedback) -> FeedbackRead:
        """Build the response model from a persisted :class:`Feedback` row."""

        return cls(
            overall_comments=feedback.overall_comments,
            strengths=feedback.strengths,
            improvements=feedback.improvements,
        )


class AnswerSheetDetail(AnswerSheetRead):
    """An answer sheet with its segmented answers, marks and feedback."""

    answers: list[AnswerRead] = Field(default_factory=list)
    feedback: FeedbackRead | None = None

    @classmethod
    def from_model(cls, sheet: Any) -> AnswerSheetDetail:
        """Build the response model from a persisted sheet and its children."""

        base = AnswerSheetRead.from_model(sheet)
        return cls(
            **base.model_dump(),
            answers=[
                AnswerRead.from_model(answer)
                for answer in sorted(
                    sheet.answers, key=lambda item: item.question.order_index
                )
            ],
            feedback=(
                FeedbackRead.from_model(sheet.feedback) if sheet.feedback is not None else None
            ),
        )
