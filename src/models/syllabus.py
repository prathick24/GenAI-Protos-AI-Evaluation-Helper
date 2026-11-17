"""Pydantic response models for syllabus endpoints (FR-001)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.repositories.schema import Syllabus


class SyllabusRead(BaseModel):
    """Representation of an ingested syllabus."""

    id: UUID
    title: str
    subject: str | None = None
    class_section: str | None = None
    file_path: str | None = None
    input_type: str
    ocr_status: str
    status: str
    version: int
    scope: list[str] | None = None
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime
    is_active: bool

    @classmethod
    def from_model(cls, syllabus: Syllabus, chunk_count: int = 0) -> SyllabusRead:
        """Build the response model from a persisted :class:`Syllabus` row."""

        return cls(
            id=syllabus.id,
            title=syllabus.title,
            subject=syllabus.subject,
            class_section=syllabus.class_section,
            file_path=syllabus.file_path,
            input_type=syllabus.input_type,
            ocr_status=syllabus.ocr_status,
            status=syllabus.status,
            version=syllabus.version,
            scope=syllabus.scope,
            chunk_count=chunk_count,
            created_at=syllabus.created_at,
            updated_at=syllabus.updated_at,
            is_active=syllabus.is_active,
        )


class UnitTopics(BaseModel):
    """A unit and the topics detected beneath it."""

    unit_no: str
    unit_name: str | None = None
    topics: list[str] = Field(default_factory=list)
