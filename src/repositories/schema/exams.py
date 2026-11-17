"""Exams: an exam event that uses an approved question paper."""

from __future__ import annotations

import uuid
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Date, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.repositories.schema.base import AuditMixin, Base

if TYPE_CHECKING:
    from src.repositories.schema.grading import AnswerSheet
    from src.repositories.schema.papers import QuestionPaper


class Exam(Base, AuditMixin):
    """An exam event linked to the question paper it administers."""

    __tablename__ = "exams"

    title: Mapped[str] = mapped_column(String(300), nullable=False)
    subject: Mapped[str | None] = mapped_column(String(150), nullable=True)
    class_section: Mapped[str | None] = mapped_column(String(100), nullable=True)
    exam_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    total_marks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="draft")
    question_paper_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("question_papers.id"), nullable=False
    )

    question_paper: Mapped[QuestionPaper] = relationship(lazy="selectin")
    answer_sheets: Mapped[list[AnswerSheet]] = relationship(
        back_populates="exam",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Exam id={self.id} title={self.title!r} status={self.status!r}>"
