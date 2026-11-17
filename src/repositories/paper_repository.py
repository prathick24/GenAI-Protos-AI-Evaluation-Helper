"""Data-access layer for question papers, questions and rubrics (FR-002)."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.schema import Question, QuestionPaper


async def add_paper(session: AsyncSession, paper: QuestionPaper) -> QuestionPaper:
    """Persist a new question paper and flush so its id is available."""

    session.add(paper)
    await session.flush()
    return paper


async def add_questions(session: AsyncSession, questions: list[Question]) -> None:
    """Persist a paper's questions (and their rubric criteria)."""

    session.add_all(questions)
    await session.flush()


async def get_paper(session: AsyncSession, paper_id: uuid.UUID) -> QuestionPaper | None:
    """Return a question paper by id, with its questions eagerly loaded."""

    return await session.get(QuestionPaper, paper_id)


async def list_papers(session: AsyncSession) -> list[QuestionPaper]:
    """Return all active question papers, newest first."""

    stmt = (
        select(QuestionPaper)
        .where(QuestionPaper.is_active.is_(True))
        .order_by(QuestionPaper.created_at.desc())
    )
    return list((await session.execute(stmt)).scalars().all())
