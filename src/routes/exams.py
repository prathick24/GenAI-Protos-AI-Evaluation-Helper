"""Exam and answer sheet routes (FR-004)."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status

from src.models.answer_sheet import AnswerSheetRead, UploadResult
from src.models.exam import ExamCreate, ExamRead
from src.models.results import ClassAnalytics, Marksheet
from src.services.dependencies import AnswerSheetServiceDep, ExamServiceDep
from src.settings import UploadInputType

router = APIRouter(prefix="/exams")


@router.post("", response_model=ExamRead, status_code=status.HTTP_201_CREATED, tags=["Exams"])
async def create_exam(payload: ExamCreate, service: ExamServiceDep) -> ExamRead:
    """Create an exam linked to an approved question paper."""

    return await service.create(payload)


@router.get("", response_model=list[ExamRead], tags=["Exams"])
async def list_exams(service: ExamServiceDep) -> list[ExamRead]:
    """List all active exams."""

    return await service.list()


@router.get("/{exam_id}", response_model=ExamRead, tags=["Exams"])
async def get_exam(exam_id: uuid.UUID, service: ExamServiceDep) -> ExamRead:
    """Return a single exam."""

    return await service.get(exam_id)


@router.get("/{exam_id}/marksheet", response_model=Marksheet, tags=["Exams"])
async def get_marksheet(exam_id: uuid.UUID, service: ExamServiceDep) -> Marksheet:
    """Return the exam's per-student, per-question marksheet (FR-008)."""

    return await service.marksheet(exam_id)


@router.get("/{exam_id}/analytics", response_model=ClassAnalytics, tags=["Exams"])
async def get_analytics(exam_id: uuid.UUID, service: ExamServiceDep) -> ClassAnalytics:
    """Return deterministic class-level analytics for the exam (FR-008)."""

    return await service.analytics(exam_id)


@router.post(
    "/{exam_id}/answer-sheets/upload",
    response_model=list[UploadResult],
    status_code=status.HTTP_207_MULTI_STATUS,
    tags=["AnswerSheets"],
)
async def upload_answer_sheets(
    exam_id: uuid.UUID,
    service: AnswerSheetServiceDep,
    files: Annotated[list[UploadFile], File(description="Answer sheet files")],
    student_id: Annotated[uuid.UUID | None, Form()] = None,
    student_name: Annotated[str | None, Form(max_length=200)] = None,
    input_type: Annotated[UploadInputType, Form()] = UploadInputType.AUTO,
) -> list[UploadResult]:
    """Upload one or more answer sheets (typed or scanned) for an exam."""

    return await service.upload(
        exam_id,
        files=files,
        student_id=student_id,
        student_name=student_name,
        input_type=input_type,
    )


@router.get(
    "/{exam_id}/answer-sheets", response_model=list[AnswerSheetRead], tags=["AnswerSheets"]
)
async def list_answer_sheets(
    exam_id: uuid.UUID, service: AnswerSheetServiceDep
) -> list[AnswerSheetRead]:
    """List an exam's answer sheets."""

    return await service.list(exam_id)
