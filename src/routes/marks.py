"""Teacher mark-override routes (FR-009)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter

from src.models.grading import MarkOverrideRequest, MarkRead
from src.services.dependencies import GradingServiceDep

router = APIRouter(prefix="/marks", tags=["Marks"])


@router.patch("/{mark_id}/override", response_model=MarkRead)
async def override_mark(
    mark_id: uuid.UUID, payload: MarkOverrideRequest, service: GradingServiceDep
) -> MarkRead:
    """Override an AI-awarded mark, preserving the original for audit (FR-009)."""

    return await service.override(mark_id, payload)
