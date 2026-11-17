"""Data-access layer for answer sheets (FR-004)."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.schema import AnswerSheet


async def add(session: AsyncSession, sheet: AnswerSheet) -> AnswerSheet:
    """Persist a new answer sheet and flush so its generated id is available."""

    session.add(sheet)
    await session.flush()
    return sheet


async def get(session: AsyncSession, sheet_id: uuid.UUID) -> AnswerSheet | None:
    """Return an answer sheet by id (including soft-deleted ones)."""

    return await session.get(AnswerSheet, sheet_id)


async def list_for_exam(session: AsyncSession, exam_id: uuid.UUID) -> list[AnswerSheet]:
    """Return an exam's active answer sheets, oldest first."""

    stmt = (
        select(AnswerSheet)
        .where(AnswerSheet.exam_id == exam_id, AnswerSheet.is_active.is_(True))
        .order_by(AnswerSheet.created_at)
    )
    return list((await session.execute(stmt)).scalars().all())
