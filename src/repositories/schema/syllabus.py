"""Syllabus documents and their embedded content chunks."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.repositories.schema.base import AuditMixin, Base
from src.settings import EMBEDDING_DIM, InputType, OcrStatus, SyllabusStatus

if TYPE_CHECKING:
    from src.repositories.schema.papers import Question


class Syllabus(Base, AuditMixin):
    """An uploaded syllabus document and its ingestion state."""

    __tablename__ = "syllabus"

    title: Mapped[str] = mapped_column(String(300), nullable=False)
    subject: Mapped[str | None] = mapped_column(String(150), nullable=True)
    class_section: Mapped[str | None] = mapped_column(String(100), nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    input_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default=InputType.TYPED.value
    )
    ocr_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=OcrStatus.DONE.value
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=SyllabusStatus.PROCESSING.value
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    scope: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    chunks: Mapped[list[SyllabusChunk]] = relationship(
        back_populates="syllabus",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Syllabus id={self.id} title={self.title!r} status={self.status!r}>"


class SyllabusChunk(Base, AuditMixin):
    """A verbatim slice of the syllabus, tagged with unit/topic metadata."""

    __tablename__ = "syllabus_chunks"
    __table_args__ = (
        UniqueConstraint("syllabus_id", "chunk_index", name="uq_syllabus_chunks_syllabus_chunk"),
        Index("ix_syllabus_chunks_syllabus_unit", "syllabus_id", "unit_no"),
        Index("ix_syllabus_chunks_syllabus_topic", "syllabus_id", "topic_name"),
    )

    syllabus_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("syllabus.id"), nullable=False
    )
    unit_no: Mapped[str | None] = mapped_column(String(50), nullable=True)
    unit_name: Mapped[str | None] = mapped_column(String(300), nullable=True)
    topic_name: Mapped[str | None] = mapped_column(String(300), nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)

    syllabus: Mapped[Syllabus] = relationship(back_populates="chunks", lazy="selectin")
    questions: Mapped[list[Question]] = relationship(back_populates="source_chunk")

    def __repr__(self) -> str:
        return (
            f"<SyllabusChunk id={self.id} unit={self.unit_no!r} "
            f"index={self.chunk_index} topic={self.topic_name!r}>"
        )
