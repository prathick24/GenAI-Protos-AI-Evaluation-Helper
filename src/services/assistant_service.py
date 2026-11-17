"""Grounded Q&A assistant service (FR-010).

Retrieves a bounded, lexically-ranked set of syllabus chunks and paper questions
(scoped to an exam when one is given), asks the assistant agent to answer from
them, and persists the turn with citations, agent attribution and token usage.
Answers never use anything outside the retrieved context.
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.assistant_agent import AssistantAgent, ContextDocument
from src.models.chat import ChatAskRequest, ChatAskResponse, Citation
from src.repositories import chat_repository as chat_repo
from src.repositories import exam_repository as exam_repo
from src.repositories import paper_repository as paper_repo
from src.repositories import syllabus_repository as syllabus_repo
from src.repositories.observability_repository import agent_run
from src.repositories.schema import Exam, Question, QuestionPaper
from src.settings import DEFAULT_TOP_K, MessageRole
from src.utils.exceptions import NotFoundError
from src.utils.exceptions import error_codes as code
from src.utils.logger import get_logger

logger = get_logger(__name__)

_AGENT_NAME = "assistant"
_TASK = "assistant_ask"
_SNIPPET_LIMIT = 420
_SOURCE_ORDER = {"question": 0, "syllabus": 1}
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "what",
        "how", "why", "which", "who", "when", "where", "does", "do", "did",
        "for", "on", "with", "that", "this", "it", "be", "as", "by", "from",
        "at", "was", "were", "can", "could", "should", "would", "about",
    }
)


class AssistantService:
    """Answers a teacher's question from the uploaded syllabus and papers."""

    def __init__(self, session: AsyncSession, *, agent: AssistantAgent | None = None) -> None:
        self._session = session
        self._agent = agent or AssistantAgent()

    async def ask(self, payload: ChatAskRequest) -> ChatAskResponse:
        """Return a grounded, cited answer and persist the conversation turn."""

        exam = await self._exam(payload.exam_id)
        conversation = await self._conversation(payload, exam)
        await chat_repo.add_message(
            self._session,
            conversation,
            role=MessageRole.USER.value,
            content=payload.question,
        )

        available = await self._context(exam)
        selected = _rank(payload.question, available)
        tokens: int | None = None
        if selected:
            async with agent_run(
                _AGENT_NAME,
                task=_TASK,
                exam_id=payload.exam_id,
                input_summary={"documents": len(selected)},
            ) as run:
                outcome = await self._agent.compose(
                    question=payload.question, documents=selected
                )
                run["citations"] = len(outcome.cited)
            answer = outcome.answer
            tokens = outcome.tokens
            citations = [_citation(selected[index]) for index in outcome.cited]
        else:
            answer = _scope_reply(available)
            citations = []

        await chat_repo.add_message(
            self._session,
            conversation,
            role=MessageRole.ASSISTANT.value,
            content=answer,
            agent_name=_AGENT_NAME,
            tool_calls=[citation.model_dump(mode="json") for citation in citations],
            tokens=tokens,
        )
        await self._session.commit()

        logger.info(
            "Assistant turn answered",
            extra={
                "conversation_id": str(conversation.id),
                "exam_id": str(payload.exam_id) if payload.exam_id else None,
                "documents": len(selected),
                "citations": len(citations),
            },
        )
        return ChatAskResponse(
            question=payload.question,
            answer=answer,
            citations=citations,
            conversation_id=conversation.id,
        )

    async def _exam(self, exam_id: uuid.UUID | None) -> Exam | None:
        """Load the scoping exam, raising :class:`NotFoundError` when missing."""

        if exam_id is None:
            return None
        exam = await exam_repo.get(self._session, exam_id)
        if exam is None or not exam.is_active:
            raise NotFoundError(
                f"Exam {exam_id} not found", error_code=code.EXAM_NOT_FOUND
            )
        return exam

    async def _conversation(self, payload: ChatAskRequest, exam: Exam | None):
        """Resume the requested conversation, or start a new one."""

        if payload.conversation_id is not None:
            conversation = await chat_repo.get_conversation(
                self._session, payload.conversation_id
            )
            if conversation is None:
                raise NotFoundError(
                    f"Conversation {payload.conversation_id} not found",
                    error_code=code.NOT_FOUND,
                )
            return conversation
        title = payload.question.strip()[:80] or "Assistant"
        return await chat_repo.create_conversation(
            self._session, title=title, exam_id=payload.exam_id
        )

    async def _context(self, exam: Exam | None) -> list[ContextDocument]:
        """Load every candidate document for the retrieval scope."""

        documents: list[ContextDocument] = []
        if exam is not None:
            paper: QuestionPaper | None = exam.question_paper
            if paper is not None:
                documents.extend(await self._chunk_documents(paper.syllabus_id))
                documents.extend(_question_documents(list(paper.questions)))
            return documents

        for syllabus in await syllabus_repo.list_all(self._session):
            documents.extend(await self._chunk_documents(syllabus.id))
        for paper in await paper_repo.list_papers(self._session):
            documents.extend(
                _question_documents(
                    [question for question in paper.questions if question.is_active]
                )
            )
        return documents

    async def _chunk_documents(
        self, syllabus_id: uuid.UUID | None
    ) -> list[ContextDocument]:
        """Return the syllabus's chunks as context documents."""

        if syllabus_id is None:
            return []
        chunks = await syllabus_repo.list_chunks(self._session, syllabus_id)
        return [
            ContextDocument(
                source_type="syllabus",
                source_id=chunk.id,
                snippet=chunk.raw_text,
                unit_no=chunk.unit_no,
                topic_name=chunk.topic_name,
            )
            for chunk in chunks
            if chunk.raw_text
        ]


def _question_documents(questions: list[Question]) -> list[ContextDocument]:
    """Return paper questions as context documents."""

    return [
        ContextDocument(
            source_type="question",
            source_id=question.id,
            snippet=question.question_text,
            topic_name=question.topic,
        )
        for question in questions
        if question.question_text
    ]


def _rank(question: str, documents: list[ContextDocument]) -> list[ContextDocument]:
    """Return the top-K documents by lexical overlap with the question."""

    wanted = _tokens(question)
    if not wanted:
        return []
    scored: list[tuple[int, ContextDocument]] = []
    for document in documents:
        score = len(wanted & _tokens(_document_text(document)))
        if score:
            scored.append((score, document))
    scored.sort(
        key=lambda entry: (
            -entry[0],
            _SOURCE_ORDER.get(entry[1].source_type, 9),
            str(entry[1].source_id),
        )
    )
    return [document for _, document in scored[:DEFAULT_TOP_K]]


def _document_text(document: ContextDocument) -> str:
    """Return all searchable text for a document."""

    return " ".join(
        part
        for part in (document.snippet, document.topic_name, document.unit_no)
        if part
    )


def _tokens(text: str) -> set[str]:
    """Tokenise text into lowercase content words."""

    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS
    }


def _citation(document: ContextDocument) -> Citation:
    """Build the response citation for one used document."""

    return Citation(
        source_type=document.source_type,
        source_id=document.source_id,
        unit_no=document.unit_no,
        topic_name=document.topic_name,
        snippet=_snippet(document.snippet),
    )


def _scope_reply(documents: list[ContextDocument]) -> str:
    """Explain that the answer was not found and what the material covers."""

    if not documents:
        return (
            "I could not find that in the uploaded material. Upload a syllabus or "
            "generate a question paper first."
        )
    topics = sorted({document.topic_name for document in documents if document.topic_name})[:5]
    if topics:
        return (
            "I could not find that in the uploaded material. "
            f"I can answer questions about: {', '.join(topics)}."
        )
    return (
        "I could not find that in the uploaded material, and it does not name its "
        "topics."
    )


def _snippet(text: str) -> str:
    """Collapse whitespace and cap a snippet for display."""

    value = " ".join(text.split())
    return value[:_SNIPPET_LIMIT].rstrip()


__all__ = ["AssistantService"]
