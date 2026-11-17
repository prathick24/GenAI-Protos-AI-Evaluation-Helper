"""Pydantic request/response models for exam endpoints (FR-004)."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.repositories.schema import Exam


class ExamCreate(BaseModel):
    """Request body for ``POST /exams``."""

    title: str = Field(min_length=1, max_length=300)
    question_paper_id: UUID
    subject: str | None = None
    class_section: str | None = None
    exam_date: date | None = None


class ExamRead(BaseModel):
    """An exam event linked to the approved question paper it administers."""

    id: UUID
    title: str
    subject: str | None = None
    class_section: str | None = None
    exam_date: date | None = None
    total_marks: int
    status: str
    question_paper_id: UUID
    created_at: datetime
    updated_at: datetime
    is_active: bool

    @classmethod
    def from_model(cls, exam: Exam) -> ExamRead:
        """Build the response model from a persisted :class:`Exam` row."""

        return cls(
            id=exam.id,
            title=exam.title,
            subject=exam.subject,
            class_section=exam.class_section,
            exam_date=exam.exam_date,
            total_marks=exam.total_marks,
            status=exam.status,
            question_paper_id=exam.question_paper_id,
            created_at=exam.created_at,
            updated_at=exam.updated_at,
            is_active=exam.is_active,
        )
