"""Syllabus routes (FR-001): upload, list, get, soft-delete and topics."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, Response, UploadFile, status

from src.models.syllabus import SyllabusRead, UnitTopics
from src.services.dependencies import SyllabusServiceDep

router = APIRouter(prefix="/syllabus", tags=["Syllabus"])


@router.post("/upload", response_model=SyllabusRead, status_code=status.HTTP_201_CREATED)
async def upload_syllabus(
    service: SyllabusServiceDep,
    file: Annotated[UploadFile, File(description="Syllabus PDF or text file")],
    title: Annotated[str | None, Form()] = None,
    subject: Annotated[str | None, Form()] = None,
    class_section: Annotated[str | None, Form()] = None,
    scope: Annotated[str | None, Form(description="Comma-separated in-scope unit numbers")] = None,
) -> SyllabusRead:
    """Upload a syllabus and index it (extract -> chunk -> embed)."""

    return await service.upload(
        upload=file,
        title=title,
        subject=subject,
        class_section=class_section,
        scope=[item.strip() for item in scope.split(",") if item.strip()] if scope else None,
    )


@router.get("", response_model=list[SyllabusRead])
async def list_syllabus(service: SyllabusServiceDep) -> list[SyllabusRead]:
    """List all active syllabi."""

    return await service.list()


@router.get("/{syllabus_id}", response_model=SyllabusRead)
async def get_syllabus(syllabus_id: uuid.UUID, service: SyllabusServiceDep) -> SyllabusRead:
    """Return a single syllabus."""

    return await service.get(syllabus_id)


@router.delete("/{syllabus_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_syllabus(syllabus_id: uuid.UUID, service: SyllabusServiceDep) -> Response:
    """Soft-delete a syllabus."""

    await service.delete(syllabus_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{syllabus_id}/topics", response_model=list[UnitTopics])
async def get_syllabus_topics(
    syllabus_id: uuid.UUID, service: SyllabusServiceDep
) -> list[UnitTopics]:
    """Return the unit/topic tree extracted from the syllabus."""

    return await service.topics(syllabus_id)
