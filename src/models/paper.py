"""Pydantic request/response models for question paper endpoints (FR-002)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from src.settings import QuestionType

if TYPE_CHECKING:
    from src.repositories.schema import Question, QuestionPaper, RubricCriterion


class QuestionTypeSpec(BaseModel):
    """How many questions of one format the paper needs, and their marks."""

    type: QuestionType
    count: int = Field(ge=1)
    marks_each: int = Field(ge=1)


class DifficultyMix(BaseModel):
    """Relative weights of easy / medium / hard questions in the paper."""

    easy: float = Field(default=0.0, ge=0)
    medium: float = Field(default=0.0, ge=0)
    hard: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _require_positive_total(self) -> DifficultyMix:
        if self.easy + self.medium + self.hard <= 0:
            raise ValueError("difficulty_mix must assign a positive weight to at least one level")
        return self


class Blueprint(BaseModel):
    """The teacher-defined shape of a question paper."""

    total_marks: int = Field(gt=0)
    duration_minutes: int | None = Field(default=None, gt=0)
    question_types: list[QuestionTypeSpec] = Field(min_length=1)
    difficulty_mix: DifficultyMix | None = None
    topic_weightage: dict[str, float] | None = None

    @model_validator(mode="after")
    def _marks_must_match_total(self) -> Blueprint:
        planned = sum(spec.count * spec.marks_each for spec in self.question_types)
        if planned != self.total_marks:
            raise ValueError(
                f"question_types produce {planned} marks but total_marks is {self.total_marks}"
            )
        return self


class GeneratePaperRequest(BaseModel):
    """Request body for ``POST /question-papers/generate``."""

    syllabus_id: UUID
    title: str = Field(min_length=1, max_length=300)
    scope: list[str] | None = Field(
        default=None, description="Unit numbers to restrict generation to; all units when omitted"
    )
    blueprint: Blueprint


class RubricCriterionRead(BaseModel):
    """One scoring criterion of a question's rubric."""

    id: UUID
    criterion_text: str
    max_marks: float
    order_index: int

    @classmethod
    def from_model(cls, criterion: RubricCriterion) -> RubricCriterionRead:
        """Build the response model from a persisted rubric criterion."""

        return cls(
            id=criterion.id,
            criterion_text=criterion.criterion_text,
            max_marks=float(criterion.max_marks),
            order_index=criterion.order_index,
        )


class QuestionRead(BaseModel):
    """A generated question with its rubric and syllabus grounding."""

    id: UUID
    order_index: int
    question_text: str
    question_type: str
    marks: float
    options: list[str] | None = None
    correct_option: str | None = None
    model_answer: str | None = None
    topic: str | None = None
    difficulty: str | None = None
    validation_status: str | None = None
    source_chunk_id: UUID | None = None
    rubric_criteria: list[RubricCriterionRead] = Field(default_factory=list)

    @classmethod
    def from_model(cls, question: Question) -> QuestionRead:
        """Build the response model from a persisted question."""

        return cls(
            id=question.id,
            order_index=question.order_index,
            question_text=question.question_text,
            question_type=question.question_type,
            marks=float(question.marks),
            options=question.options,
            correct_option=question.correct_option,
            model_answer=question.model_answer,
            topic=question.topic,
            difficulty=question.difficulty,
            validation_status=question.validation_status,
            source_chunk_id=question.source_chunk_id,
            rubric_criteria=[
                RubricCriterionRead.from_model(criterion)
                for criterion in sorted(
                    question.rubric_criteria, key=lambda item: item.order_index
                )
            ],
        )


class QuestionPaperRead(BaseModel):
    """A generated question paper and its questions."""

    id: UUID
    syllabus_id: UUID | None = None
    title: str
    subject: str | None = None
    class_section: str | None = None
    duration_minutes: int | None = None
    total_marks: int
    version: int
    source: str
    status: str
    blueprint: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime
    is_active: bool
    questions: list[QuestionRead] = Field(default_factory=list)

    @classmethod
    def from_model(
        cls, paper: QuestionPaper, questions: list[Question] | None = None
    ) -> QuestionPaperRead:
        """Build the response model from a persisted paper and its questions."""

        rows = paper.questions if questions is None else questions
        return cls(
            id=paper.id,
            syllabus_id=paper.syllabus_id,
            title=paper.title,
            subject=paper.subject,
            class_section=paper.class_section,
            duration_minutes=paper.duration_minutes,
            total_marks=paper.total_marks,
            version=paper.version,
            source=paper.source,
            status=paper.status,
            blueprint=paper.blueprint,
            created_at=paper.created_at,
            updated_at=paper.updated_at,
            is_active=paper.is_active,
            questions=[
                QuestionRead.from_model(question)
                for question in sorted(rows, key=lambda item: item.order_index)
            ],
        )


class ValidationCheck(BaseModel):
    """The outcome of one validation check run against a paper (FR-003)."""

    name: str
    passed: bool
    summary: str
    details: dict[str, Any] | None = None


class QuestionValidationResult(BaseModel):
    """Per-question validation outcome."""

    question_id: UUID
    order_index: int
    status: str
    issues: list[str] = Field(default_factory=list)


class PaperValidationReport(BaseModel):
    """Paper-level validation report produced by the validation checks (FR-003)."""

    paper_id: UUID
    passed: bool
    paper_status: str
    checks: list[ValidationCheck] = Field(default_factory=list)
    questions: list[QuestionValidationResult] = Field(default_factory=list)


class GeneratePaperResponse(BaseModel):
    """Response for ``POST /question-papers/generate``.

    Generation and validation run in one step, so the caller receives the
    persisted paper (``validated`` when every check passed, otherwise ``draft``)
    alongside the validation report.
    """

    paper: QuestionPaperRead
    validation_report: PaperValidationReport
