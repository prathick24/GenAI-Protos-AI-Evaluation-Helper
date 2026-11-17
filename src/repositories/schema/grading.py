"""Answer sheets, segmented answers, marks and feedback."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, ForeignKey, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.repositories.schema.base import AuditMixin, Base
from src.settings import InputType, OcrStatus, SheetStatus

if TYPE_CHECKING:
    from src.repositories.schema.exams import Exam
    from src.repositories.schema.papers import Question
    from src.repositories.schema.users import Student


class AnswerSheet(Base, AuditMixin):
    """One student's submitted answer sheet for an exam."""

    __tablename__ = "answer_sheets"

    exam_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("exams.id"), nullable=False
    )
    student_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("students.id"), nullable=True
    )
    student_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    input_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default=InputType.TYPED.value
    )
    ocr_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=OcrStatus.PENDING.value
    )
    sheet_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=SheetStatus.UPLOADED.value
    )
    total_awarded_marks: Mapped[Decimal | None] = mapped_column(Numeric(7, 2), nullable=True)
    ocr_confidence: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    exam: Mapped[Exam] = relationship(back_populates="answer_sheets", lazy="selectin")
    student: Mapped[Student | None] = relationship(lazy="selectin")
    answers: Mapped[list[Answer]] = relationship(
        back_populates="answer_sheet",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    feedback: Mapped[Feedback | None] = relationship(
        back_populates="answer_sheet",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<AnswerSheet id={self.id} status={self.sheet_status!r}>"


class Answer(Base, AuditMixin):
    """A per-question answer segmented from an answer sheet."""

    __tablename__ = "answers"

    answer_sheet_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("answer_sheets.id"), nullable=False
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("questions.id"), nullable=False
    )
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_answered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    ocr_confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)

    answer_sheet: Mapped[AnswerSheet] = relationship(back_populates="answers", lazy="selectin")
    question: Mapped[Question] = relationship(back_populates="answers", lazy="selectin")
    mark: Mapped[Mark | None] = relationship(
        back_populates="answer",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Answer id={self.id} question={self.question_id} answered={self.is_answered}>"


class Mark(Base, AuditMixin):
    """The awarded mark for one answer, with evidence and verification."""

    __tablename__ = "marks"

    answer_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("answers.id"), nullable=False, unique=True
    )
    awarded_marks: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False, default=0)
    max_marks: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False, default=0)
    rubric_scores: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    verifier_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    verifier_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    is_overridden: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    original_marks: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    graded_by: Mapped[str | None] = mapped_column(String(20), nullable=True)

    answer: Mapped[Answer] = relationship(back_populates="mark", lazy="selectin")

    def __repr__(self) -> str:
        return (
            f"<Mark id={self.id} awarded={self.awarded_marks}/{self.max_marks} "
            f"verifier={self.verifier_status!r}>"
        )


class Feedback(Base, AuditMixin):
    """Per-student qualitative feedback for an answer sheet."""

    __tablename__ = "feedback"

    answer_sheet_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("answer_sheets.id"), nullable=False
    )
    overall_comments: Mapped[str | None] = mapped_column(Text, nullable=True)
    strengths: Mapped[str | None] = mapped_column(Text, nullable=True)
    improvements: Mapped[str | None] = mapped_column(Text, nullable=True)

    answer_sheet: Mapped[AnswerSheet] = relationship(back_populates="feedback", lazy="selectin")

    def __repr__(self) -> str:
        return f"<Feedback id={self.id} sheet={self.answer_sheet_id}>"
