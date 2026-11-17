"""Pydantic response models for answer sheet ingestion (FR-004)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.repositories.schema import AnswerSheet


class UploadResult(BaseModel):
    """The per-file outcome of an answer sheet batch upload."""

    file: str
    status: str
    answer_sheet_id: UUID | None = None
    student_name: str | None = None
    reason: str | None = None


class AnswerSheetRename(BaseModel):
    """Request body for renaming an uploaded answer sheet."""

    student_name: str | None = Field(default=None, max_length=200)


class AnswerSheetRead(BaseModel):
    """One student's answer sheet and its ingestion state."""

    id: UUID
    exam_id: UUID
    student_id: UUID | None = None
    student_name: str | None = None
    file_path: str | None = None
    input_type: str
    ocr_status: str
    sheet_status: str
    total_awarded_marks: float | None = None
    ocr_confidence: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime
    is_active: bool

    @classmethod
    def from_model(cls, sheet: AnswerSheet) -> AnswerSheetRead:
        """Build the response model from a persisted :class:`AnswerSheet` row."""

        return cls(
            id=sheet.id,
            exam_id=sheet.exam_id,
            student_id=sheet.student_id,
            student_name=sheet.student_name,
            file_path=sheet.file_path,
            input_type=sheet.input_type,
            ocr_status=sheet.ocr_status,
            sheet_status=sheet.sheet_status,
            total_awarded_marks=(
                float(sheet.total_awarded_marks)
                if sheet.total_awarded_marks is not None
                else None
            ),
            ocr_confidence=sheet.ocr_confidence,
            created_at=sheet.created_at,
            updated_at=sheet.updated_at,
            is_active=sheet.is_active,
        )
