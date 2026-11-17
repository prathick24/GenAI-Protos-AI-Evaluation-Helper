"""Data-access layer for syllabi and their embedded chunks (FR-001)."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.schema import Syllabus, SyllabusChunk


async def add(session: AsyncSession, syllabus: Syllabus) -> Syllabus:
    """Persist a new syllabus and flush so its generated id is available."""

    session.add(syllabus)
    await session.flush()
    return syllabus


async def add_chunks(session: AsyncSession, chunks: list[SyllabusChunk]) -> None:
    """Persist the chunks belonging to a syllabus."""

    session.add_all(chunks)
    await session.flush()


async def get(session: AsyncSession, syllabus_id: uuid.UUID) -> Syllabus | None:
    """Return a syllabus by id (including soft-deleted ones)."""

    return await session.get(Syllabus, syllabus_id)


async def list_all(session: AsyncSession) -> list[Syllabus]:
    """Return all active syllabi, newest first."""

    stmt = select(Syllabus).where(Syllabus.is_active.is_(True)).order_by(Syllabus.created_at.desc())
    return list((await session.execute(stmt)).scalars().all())


async def find_latest_active(
    session: AsyncSession,
    *,
    title: str,
    subject: str | None,
    class_section: str | None,
) -> Syllabus | None:
    """Return the newest active syllabus sharing an upload identity, if any.

    The identity is ``(title, subject, class_section)``; a re-upload of the same
    syllabus is turned into a new version of it.
    """

    subject_match = Syllabus.subject.is_(None) if subject is None else Syllabus.subject == subject
    class_match = (
        Syllabus.class_section.is_(None)
        if class_section is None
        else Syllabus.class_section == class_section
    )
    stmt = (
        select(Syllabus)
        .where(
            Syllabus.is_active.is_(True),
            Syllabus.title == title,
            subject_match,
            class_match,
        )
        .order_by(Syllabus.version.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalars().first()


async def chunk_count(session: AsyncSession, syllabus_id: uuid.UUID) -> int:
    """Return the number of chunks stored for a syllabus."""

    stmt = (
        select(func.count())
        .select_from(SyllabusChunk)
        .where(SyllabusChunk.syllabus_id == syllabus_id)
    )
    return int(await session.scalar(stmt) or 0)


async def chunk_counts(
    session: AsyncSession, syllabus_ids: list[uuid.UUID]
) -> dict[uuid.UUID, int]:
    """Return chunk counts for many syllabi in a single query."""

    if not syllabus_ids:
        return {}

    stmt = (
        select(SyllabusChunk.syllabus_id, func.count())
        .where(SyllabusChunk.syllabus_id.in_(syllabus_ids))
        .group_by(SyllabusChunk.syllabus_id)
    )
    rows = (await session.execute(stmt)).all()
    return {row[0]: int(row[1]) for row in rows}


async def list_chunks(session: AsyncSession, syllabus_id: uuid.UUID) -> list[SyllabusChunk]:
    """Return a syllabus's chunks ordered by their ingestion index."""

    stmt = (
        select(SyllabusChunk)
        .where(SyllabusChunk.syllabus_id == syllabus_id)
        .order_by(SyllabusChunk.chunk_index)
    )
    return list((await session.execute(stmt)).scalars().all())
