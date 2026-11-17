"""Pydantic request/response models for the paper chat assistant (FR-010)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from src.models.paper import (
    DifficultyMix,
    PaperValidationReport,
    QuestionPaperRead,
    QuestionTypeSpec,
)
from src.settings import ChangeOp


class ChangeOperation(BaseModel):
    """One paper change the assistant interpreted from the teacher's message.

    Fields are populated per ``op``: ``REWRITE``/``DROP`` use ``order_indices``,
    ``REORDER`` uses ``order``, ``SET_DURATION`` uses ``duration_minutes`` and
    ``ADJUST_BLUEPRINT`` uses ``question_types``/``difficulty_mix``.
    """

    op: ChangeOp
    order_indices: list[int] = Field(
        default_factory=list, description="Zero-based question positions to rewrite or drop"
    )
    order: list[int] = Field(
        default_factory=list,
        description="New order for REORDER, as the current zero-based positions",
    )
    duration_minutes: int | None = Field(default=None, gt=0)
    question_types: list[QuestionTypeSpec] | None = None
    difficulty_mix: DifficultyMix | None = None
    instructions: str | None = Field(
        default=None, description="The teacher's wording, passed to the LLM for rewrites"
    )


class ChangePlan(BaseModel):
    """The assistant's answer to a teacher message: a reply and the changes to apply."""

    reply: str = ""
    operations: list[ChangeOperation] = Field(default_factory=list)


class PaperChatRequest(BaseModel):
    """Request body for ``POST /question-papers/{paper_id}/chat``."""

    message: str = Field(min_length=1, max_length=4000)
    conversation_id: UUID | None = Field(
        default=None,
        description="Continue an existing conversation; a new one is created otherwise",
    )


class PaperChatResponse(BaseModel):
    """The assistant's reply plus the (re-validated) paper it changed."""

    conversation_id: UUID
    reply: str
    applied: list[str] = Field(default_factory=list)
    paper: QuestionPaperRead
    validation_report: PaperValidationReport


class ChatAskRequest(BaseModel):
    """Request body for the grounded Q&A endpoint ``POST /chat/ask`` (FR-010)."""

    question: str = Field(min_length=1, max_length=4000)
    exam_id: UUID | None = Field(
        default=None,
        description="Scope retrieval to this exam's paper and syllabus; all material otherwise",
    )
    conversation_id: UUID | None = Field(
        default=None,
        description="Continue an existing conversation; a new one is created otherwise",
    )


class Citation(BaseModel):
    """A source the answer was grounded in (syllabus chunk or paper question)."""

    source_type: str = Field(description="One of: syllabus, question")
    source_id: UUID
    unit_no: str | None = None
    topic_name: str | None = None
    snippet: str


class ChatAskResponse(BaseModel):
    """A grounded answer with the sources it cites."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    conversation_id: UUID
