"""Dependency-injection wiring shared by route modules."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.database import get_session
from src.services.answer_sheet_service import AnswerSheetService
from src.services.assistant_service import AssistantService
from src.services.chat_service import PaperChatService
from src.services.exam_service import ExamService
from src.services.grading_service import GradingService
from src.services.paper_service import PaperService
from src.services.syllabus_service import SyllabusService

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_syllabus_service(session: SessionDep) -> SyllabusService:
    """Provide a request-scoped :class:`SyllabusService`."""

    return SyllabusService(session)


def get_paper_service(session: SessionDep) -> PaperService:
    """Provide a request-scoped :class:`PaperService`."""

    return PaperService(session)


def get_paper_chat_service(session: SessionDep) -> PaperChatService:
    """Provide a request-scoped :class:`PaperChatService`."""

    return PaperChatService(session)


def get_assistant_service(session: SessionDep) -> AssistantService:
    """Provide a request-scoped :class:`AssistantService`."""

    return AssistantService(session)


def get_exam_service(session: SessionDep) -> ExamService:
    """Provide a request-scoped :class:`ExamService`."""

    return ExamService(session)


def get_answer_sheet_service(session: SessionDep) -> AnswerSheetService:
    """Provide a request-scoped :class:`AnswerSheetService`."""

    return AnswerSheetService(session)


def get_grading_service(session: SessionDep) -> GradingService:
    """Provide a request-scoped :class:`GradingService`."""

    return GradingService(session)


SyllabusServiceDep = Annotated[SyllabusService, Depends(get_syllabus_service)]
PaperServiceDep = Annotated[PaperService, Depends(get_paper_service)]
PaperChatServiceDep = Annotated[PaperChatService, Depends(get_paper_chat_service)]
AssistantServiceDep = Annotated[AssistantService, Depends(get_assistant_service)]
ExamServiceDep = Annotated[ExamService, Depends(get_exam_service)]
AnswerSheetServiceDep = Annotated[AnswerSheetService, Depends(get_answer_sheet_service)]
GradingServiceDep = Annotated[GradingService, Depends(get_grading_service)]
