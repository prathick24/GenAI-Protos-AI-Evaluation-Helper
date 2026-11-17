"""Traceability tables: error logs, model cost and agent runs.

These records are written best-effort through their own short-lived sessions so
a storage failure can never abort the primary request.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from src.repositories.schema.base import AuditMixin, Base
from src.settings import AgentRunStatus


class ErrorLog(Base, AuditMixin):
    """One recorded failure with enough context to reproduce it."""

    __tablename__ = "error_logs"

    level: Mapped[str] = mapped_column(String(20), nullable=False, default="ERROR")
    component: Mapped[str | None] = mapped_column(String(100), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    exception_type: Mapped[str | None] = mapped_column(String(200), nullable=True)
    endpoint: Mapped[str | None] = mapped_column(String(300), nullable=True)
    method: Mapped[str | None] = mapped_column(String(10), nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    def __repr__(self) -> str:
        return f"<ErrorLog id={self.id} level={self.level!r} component={self.component!r}>"


class ModelCost(Base, AuditMixin):
    """One LLM call's token usage, latency and outcome."""

    __tablename__ = "model_cost"

    task: Mapped[str | None] = mapped_column(String(100), nullable=True)
    agent_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    def __repr__(self) -> str:
        return f"<ModelCost id={self.id} task={self.task!r} total_tokens={self.total_tokens}>"


class AgentRun(Base, AuditMixin):
    """One agent task's lifecycle, from start to outcome."""

    __tablename__ = "agent_runs"

    exam_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("exams.id"), nullable=True
    )
    answer_sheet_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("answer_sheets.id"), nullable=True
    )
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    task: Mapped[str | None] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=AgentRunStatus.STARTED.value
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    input_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    output_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    def __repr__(self) -> str:
        return f"<AgentRun id={self.id} agent={self.agent_name!r} status={self.status!r}>"
