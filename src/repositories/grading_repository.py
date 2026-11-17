"""Data-access layer for segmented answers and marks (FR-005, FR-006)."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.repositories.schema import Answer, AnswerSheet, Mark


async def add_answers(session: AsyncSession, answers: list[Answer]) -> None:
    """Persist a sheet's segmented answers and flush so their ids are assigned."""

    session.add_all(answers)
    await session.flush()


async def get_answer(session: AsyncSession, answer_id: uuid.UUID) -> Answer | None:
    """Return an answer by id."""

    return await session.get(Answer, answer_id)


async def add_mark(session: AsyncSession, mark: Mark) -> Mark:
    """Persist a new mark and flush so its id is available."""

    session.add(mark)
    await session.flush()
    return mark


async def get_mark(session: AsyncSession, mark_id: uuid.UUID) -> Mark | None:
    """Return a mark by id, with its answer sheet and its marks eagerly loaded."""

    return await session.get(
        Mark,
        mark_id,
        options=[
            selectinload(Mark.answer)
            .selectinload(Answer.answer_sheet)
            .selectinload(AnswerSheet.answers)
            .selectinload(Answer.mark)
        ],
    )
