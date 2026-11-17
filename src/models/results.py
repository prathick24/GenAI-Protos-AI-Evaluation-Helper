"""Pydantic models for the exam marksheet and class analytics (FR-008)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class MarksheetRow(BaseModel):
    """One student's row in an exam's marksheet."""

    answer_sheet_id: UUID
    student_id: UUID | None = None
    student_name: str
    total_awarded_marks: float
    per_question: dict[str, float] = Field(default_factory=dict)


class Marksheet(BaseModel):
    """Consolidated per-student, per-question marks for an exam (FR-008)."""

    exam_id: UUID
    total_marks: int
    rows: list[MarksheetRow] = Field(default_factory=list)


class ClassAnalytics(BaseModel):
    """Deterministic class-level aggregates over an exam's stored marks (FR-008)."""

    exam_id: UUID
    average: float
    highest: float
    lowest: float
    distribution: dict[str, int] = Field(default_factory=dict)
    per_topic_strength: dict[str, float] = Field(default_factory=dict)
    hardest_questions: list[UUID] = Field(default_factory=list)
    easiest_questions: list[UUID] = Field(default_factory=list)
