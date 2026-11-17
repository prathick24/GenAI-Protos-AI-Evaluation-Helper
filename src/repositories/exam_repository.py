"""Data-access layer for exams (FR-004)."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.schema import Exam


async def add(session: AsyncSession, exam: Exam) -> Exam:
    """Persist a new exam and flush so its generated id is available."""

    session.add(exam)
    await session.flush()
    return exam


async def get(session: AsyncSession, exam_id: uuid.UUID) -> Exam | None:
    """Return an exam by id (including soft-deleted ones)."""

    return await session.get(Exam, exam_id)


async def list_all(session: AsyncSession) -> list[Exam]:
    """Return all active exams, newest first."""

    stmt = select(Exam).where(Exam.is_active.is_(True)).order_by(Exam.created_at.desc())
    return list((await session.execute(stmt)).scalars().all())
