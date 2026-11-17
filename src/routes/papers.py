"""Question paper routes (FR-002 generation + FR-003 validation, FR-003 approval)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query, Response, status

from src.models.paper import (
    GeneratePaperRequest,
    GeneratePaperResponse,
    QuestionPaperRead,
)
from src.services.dependencies import PaperServiceDep
from src.utils.paper_pdf import export_filename, render_paper_pdf

router = APIRouter(prefix="/question-papers", tags=["Question Papers"])


@router.post(
    "/generate", response_model=GeneratePaperResponse, status_code=status.HTTP_201_CREATED
)
async def generate_question_paper(
    payload: GeneratePaperRequest, service: PaperServiceDep
) -> GeneratePaperResponse:
    """Generate a question paper and validate it in one step (FR-002, FR-003).

    The response carries the persisted paper (``validated`` when every check
    passed, otherwise ``draft``) and the validation report.
    """

    return await service.generate(payload)


@router.get("", response_model=list[QuestionPaperRead])
async def list_question_papers(service: PaperServiceDep) -> list[QuestionPaperRead]:
    """List all active question papers."""

    return await service.list()


@router.get("/{paper_id}", response_model=QuestionPaperRead)
async def get_question_paper(paper_id: uuid.UUID, service: PaperServiceDep) -> QuestionPaperRead:
    """Return a single question paper with its questions."""

    return await service.get(paper_id)


@router.get("/{paper_id}/export.pdf", response_class=Response)
async def export_question_paper(
    paper_id: uuid.UUID,
    service: PaperServiceDep,
    include_answers: bool = Query(
        default=False,
        description="Include the teacher-only answer key and rubric.",
    ),
) -> Response:
    """Download a student (questions only) or teacher (with answers) PDF."""

    paper = await service.get(paper_id)
    content = render_paper_pdf(paper, include_answers=include_answers)
    filename = export_filename(paper, include_answers=include_answers)
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{paper_id}/approve", response_model=QuestionPaperRead)
async def approve_question_paper(
    paper_id: uuid.UUID, service: PaperServiceDep
) -> QuestionPaperRead:
    """Approve a validated question paper so it can be administered as an exam (FR-003)."""

    return await service.approve(paper_id)
