"""Answer sheet segmentation, grading and approval routes (FR-005, FR-006, FR-009)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter

from src.models.answer_sheet import AnswerSheetRead, AnswerSheetRename
from src.models.grading import AnswerSheetDetail, GradingResult, SegmentationResult
from src.services.dependencies import GradingServiceDep

router = APIRouter(prefix="/answer-sheets", tags=["AnswerSheets"])


@router.post("/{sheet_id}/segment", response_model=SegmentationResult)
async def segment_answer_sheet(
    sheet_id: uuid.UUID, service: GradingServiceDep
) -> SegmentationResult:
    """Segment a sheet into per-question answers aligned to the paper (FR-005)."""

    return await service.segment(sheet_id)


@router.post("/{sheet_id}/grade", response_model=GradingResult)
async def grade_answer_sheet(sheet_id: uuid.UUID, service: GradingServiceDep) -> GradingResult:
    """Grade a segmented sheet against each question's rubric (FR-006)."""

    return await service.grade(sheet_id)


@router.post("/{sheet_id}/approve", response_model=AnswerSheetRead)
async def approve_answer_sheet(
    sheet_id: uuid.UUID, service: GradingServiceDep
) -> AnswerSheetRead:
    """Approve a graded sheet, finalising the teacher's review (FR-009)."""

    return await service.approve(sheet_id)


@router.patch("/{sheet_id}", response_model=AnswerSheetRead)
async def rename_answer_sheet(
    sheet_id: uuid.UUID, payload: AnswerSheetRename, service: GradingServiceDep
) -> AnswerSheetRead:
    """Set or clear the student name shown for an answer sheet."""

    return await service.rename(sheet_id, payload)


@router.get("/{sheet_id}", response_model=AnswerSheetDetail)
async def get_answer_sheet(
    sheet_id: uuid.UUID, service: GradingServiceDep
) -> AnswerSheetDetail:
    """Return a sheet with its segmented answers, marks and feedback."""

    return await service.get_detail(sheet_id)
