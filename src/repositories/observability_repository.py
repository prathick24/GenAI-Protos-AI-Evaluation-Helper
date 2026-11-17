"""Best-effort traceability writes: model cost, agent runs and error logs (FR-011).

Every write opens its own short-lived session so a storage failure is logged and
swallowed: recording can only ever add a log line, never abort the primary
request (including a streaming response). The LLM client reports each call to a
:class:`UsageRecorder`, which turns the outcome into one ``MODEL_COST`` row;
agents wrap their work in :func:`agent_run` so each task leaves one
``AGENT_RUNS`` row; failures are recorded with :func:`record_error`.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.repositories.database import SessionFactory
from src.repositories.schema import AgentRun, ErrorLog, ModelCost
from src.settings import AgentRunStatus
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.client.llm import LLMResult

logger = get_logger(__name__)


def _correlation_id() -> str | None:
    """Return the current request's correlation id, when there is one."""

    from src.middleware.correlation import get_correlation_id

    return get_correlation_id()


async def record_model_cost(
    *,
    task: str | None,
    agent_name: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_usd: float | None = None,
    latency_ms: int | None = None,
    success: bool = True,
    error_code: str | None = None,
) -> None:
    """Persist one ``MODEL_COST`` row, degrading to a log line on any failure."""

    try:
        async with SessionFactory() as session:
            session.add(
                ModelCost(
                    task=task,
                    agent_name=agent_name,
                    model=model,
                    provider=provider,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    success=success,
                    error_code=error_code,
                    correlation_id=_correlation_id(),
                )
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001 - recording must never break the request
        logger.warning(
            "Could not record model cost",
            extra={"task": task, "error": type(exc).__name__},
        )


async def record_agent_run(
    *,
    agent_name: str,
    task: str | None = None,
    status: str = AgentRunStatus.STARTED.value,
    exam_id: uuid.UUID | None = None,
    answer_sheet_id: uuid.UUID | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    input_summary: dict[str, Any] | None = None,
    output_summary: dict[str, Any] | None = None,
) -> None:
    """Persist one ``AGENT_RUNS`` row, degrading to a log line on any failure."""

    try:
        async with SessionFactory() as session:
            session.add(
                AgentRun(
                    exam_id=exam_id,
                    answer_sheet_id=answer_sheet_id,
                    agent_name=agent_name,
                    task=task,
                    status=status,
                    started_at=started_at,
                    ended_at=ended_at,
                    input_summary=input_summary,
                    output_summary=output_summary,
                )
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001 - recording must never break the request
        logger.warning(
            "Could not record agent run",
            extra={"agent": agent_name, "error": type(exc).__name__},
        )


async def record_error(
    *,
    message: str,
    level: str = "ERROR",
    component: str | None = None,
    exception_type: str | None = None,
    endpoint: str | None = None,
    method: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Persist one ``ERROR_LOGS`` row, degrading to a log line on any failure."""

    try:
        async with SessionFactory() as session:
            session.add(
                ErrorLog(
                    level=level,
                    component=component,
                    message=message,
                    exception_type=exception_type,
                    endpoint=endpoint,
                    method=method,
                    correlation_id=_correlation_id(),
                    detail=detail,
                )
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001 - recording must never break the request
        logger.warning(
            "Could not record error log",
            extra={"component": component, "error": type(exc).__name__},
        )


@asynccontextmanager
async def agent_run(
    agent_name: str,
    *,
    task: str | None = None,
    exam_id: uuid.UUID | None = None,
    answer_sheet_id: uuid.UUID | None = None,
    input_summary: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Record an ``AGENT_RUNS`` row spanning the enclosed agent task.

    Yields a mutable mapping the caller may fill with an output summary; the row
    is written once, as ``succeeded`` or ``failed``, when the block exits.
    """

    started_at = datetime.now(UTC)
    output_summary: dict[str, Any] = {}
    try:
        yield output_summary
    except BaseException:
        await record_agent_run(
            agent_name=agent_name,
            task=task,
            status=AgentRunStatus.FAILED.value,
            exam_id=exam_id,
            answer_sheet_id=answer_sheet_id,
            started_at=started_at,
            ended_at=datetime.now(UTC),
            input_summary=input_summary,
            output_summary=output_summary or None,
        )
        raise
    await record_agent_run(
        agent_name=agent_name,
        task=task,
        status=AgentRunStatus.SUCCEEDED.value,
        exam_id=exam_id,
        answer_sheet_id=answer_sheet_id,
        started_at=started_at,
        ended_at=datetime.now(UTC),
        input_summary=input_summary,
        output_summary=output_summary or None,
    )


class UsageRecorder:
    """Turns each LLM call's outcome into one ``MODEL_COST`` row (FR-011).

    Satisfies the LLM client's usage-sink contract. The ``task`` is the agent
    task name, which also identifies the agent that made the call.
    """

    async def record(
        self, *, task: str, result: LLMResult | None, error: BaseException | None
    ) -> None:
        """Record a successful or failed LLM call."""

        if result is not None:
            await record_model_cost(
                task=task,
                agent_name=task,
                model=result.model,
                provider=result.provider,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                latency_ms=result.latency_ms,
                success=True,
            )
            return

        await record_model_cost(
            task=task,
            agent_name=task,
            success=False,
            error_code=getattr(error, "error_code", None),
        )


_recorder = UsageRecorder()


def default_usage_recorder() -> UsageRecorder:
    """Return the process-wide usage recorder injected into the agents."""

    return _recorder


__all__ = [
    "UsageRecorder",
    "agent_run",
    "default_usage_recorder",
    "record_agent_run",
    "record_error",
    "record_model_cost",
]
