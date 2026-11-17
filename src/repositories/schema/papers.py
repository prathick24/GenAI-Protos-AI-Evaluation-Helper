"""Question papers, their questions and rubric criteria."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.repositories.schema.base import AuditMixin, Base
from src.settings import PaperStatus

if TYPE_CHECKING:
    from src.repositories.schema.grading import Answer
    from src.repositories.schema.syllabus import Syllabus, SyllabusChunk


class QuestionPaper(Base, AuditMixin):
    """A generated (or uploaded) question paper grounded in a syllabus."""

    __tablename__ = "question_papers"

    syllabus_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("syllabus.id"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    subject: Mapped[str | None] = mapped_column(String(150), nullable=True)
    class_section: Mapped[str | None] = mapped_column(String(100), nullable=True)
    duration_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_marks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="generated")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=PaperStatus.DRAFT.value)
    blueprint: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    syllabus: Mapped[Syllabus | None] = relationship(lazy="selectin")
    questions: Mapped[list[Question]] = relationship(
        back_populates="question_paper",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<QuestionPaper id={self.id} title={self.title!r} status={self.status!r}>"


class Question(Base, AuditMixin):
    """A single question, traceable to the syllabus chunk it was grounded in."""

    __tablename__ = "questions"
    __table_args__ = (
        UniqueConstraint("question_paper_id", "order_index", name="uq_questions_paper_order"),
    )

    question_paper_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("question_papers.id"), nullable=False
    )
    source_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("syllabus_chunks.id"), nullable=True
    )
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    question_type: Mapped[str] = mapped_column(String(50), nullable=False, default="short")
    marks: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False, default=0)
    options: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    correct_option: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    topic: Mapped[str | None] = mapped_column(String(300), nullable=True)
    difficulty: Mapped[str | None] = mapped_column(String(20), nullable=True)
    validation_status: Mapped[str | None] = mapped_column(String(30), nullable=True)

    question_paper: Mapped[QuestionPaper] = relationship(
        back_populates="questions", lazy="selectin"
    )
    source_chunk: Mapped[SyllabusChunk | None] = relationship(
        back_populates="questions", lazy="selectin"
    )
    rubric_criteria: Mapped[list[RubricCriterion]] = relationship(
        back_populates="question",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    answers: Mapped[list[Answer]] = relationship(back_populates="question")

    def __repr__(self) -> str:
        return f"<Question id={self.id} index={self.order_index} type={self.question_type!r}>"


class RubricCriterion(Base, AuditMixin):
    """One scoring criterion of a question's rubric."""

    __tablename__ = "rubric_criteria"
    __table_args__ = (
        UniqueConstraint("question_id", "order_index", name="uq_rubric_question_order"),
    )

    question_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("questions.id"), nullable=False
    )
    criterion_text: Mapped[str] = mapped_column(Text, nullable=False)
    max_marks: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False, default=0)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    question: Mapped[Question] = relationship(back_populates="rubric_criteria", lazy="selectin")

    def __repr__(self) -> str:
        return f"<RubricCriterion id={self.id} question={self.question_id} max={self.max_marks}>"
