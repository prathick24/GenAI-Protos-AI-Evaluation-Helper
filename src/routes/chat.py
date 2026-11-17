"""Question paper chat routes (FR-010)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter

from src.models.chat import (
    ChatAskRequest,
    ChatAskResponse,
    PaperChatRequest,
    PaperChatResponse,
)
from src.services.dependencies import AssistantServiceDep, PaperChatServiceDep

router = APIRouter(prefix="/question-papers", tags=["Question Papers"])
ask_router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/{paper_id}/chat", response_model=PaperChatResponse)
async def chat_with_question_paper(
    paper_id: uuid.UUID, payload: PaperChatRequest, service: PaperChatServiceDep
) -> PaperChatResponse:
    """Change a paper from a teacher's chat message, then re-validate it (FR-010).

    The assistant interprets the message into a plan of changes; rewrites are
    authored by the generator agent while structural edits (drop, reorder,
    duration, blueprint) are applied deterministically. The blueprint is
    reconciled to what remains and the paper is re-validated, so it lands as
    ``validated`` or back in ``draft`` for the teacher to iterate.
    """

    return await service.chat(paper_id, payload)

@ask_router.post("/ask", response_model=ChatAskResponse)
async def ask_assistant(
    payload: ChatAskRequest, service: AssistantServiceDep
) -> ChatAskResponse:
    """Answer a teacher's question from the syllabus and papers, with citations (FR-010).

    Retrieval is scoped to the given exam (its paper and syllabus) or to all
    uploaded material, ranked lexically, and capped at the top K documents. The
    answer uses only those documents and cites them; when nothing relevant is
    found the reply states what the material does cover.
    """

    return await service.ask(payload)
