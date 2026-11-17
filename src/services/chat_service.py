"""Chat-driven question paper changes (FR-010).

The teacher talks to the assistant in natural language; the router agent turns
each message into a :class:`ChangePlan` and the executors below apply it. The
LLM only ever *writes* question text (rewrites, and the questions needed to fit
a new blueprint); drops, reorders, duration and blueprint shape are applied
deterministically. Once the plan is applied the blueprint is reconciled to the
questions that remain and the paper is re-validated, so it lands as
``validated`` or back in ``draft`` for the teacher to iterate.
"""

from __future__ import annotations

import uuid
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.paper_chat_agent import PaperChatRouterAgent
from src.agents.paper_generation_agent import (
    ExistingQuestion,
    PaperGenerationAgent,
    QuestionSpec,
    RevisionRequest,
    RevisionTarget,
)
from src.agents.paper_validation_agent import PaperValidationAgent
from src.models.chat import (
    ChangeOperation,
    PaperChatRequest,
    PaperChatResponse,
)
from src.models.paper import Blueprint, QuestionPaperRead
from src.repositories import chat_repository as chat_repo
from src.repositories import paper_repository as paper_repo
from src.repositories import syllabus_repository as syllabus_repo
from src.repositories.schema import Conversation, Question, QuestionPaper
from src.services.paper_service import PaperService, _apply_revision, _plan_questions
from src.settings import ChangeOp, Difficulty, MessageRole, PaperStatus
from src.utils.exceptions import ConflictError, NotFoundError
from src.utils.exceptions import error_codes as code
from src.utils.logger import get_logger

logger = get_logger(__name__)

_AGENT_NAME = "paper_chat"


class PaperChatService:
    """Interprets a teacher's chat message and applies it to a question paper."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        router: PaperChatRouterAgent | None = None,
        generation_agent: PaperGenerationAgent | None = None,
        validation_agent: PaperValidationAgent | None = None,
    ) -> None:
        self._session = session
        self._agent = generation_agent or PaperGenerationAgent()
        self._router = router or PaperChatRouterAgent()
        self._papers = PaperService(
            session, agent=self._agent, validation_agent=validation_agent
        )

    async def chat(self, paper_id: uuid.UUID, payload: PaperChatRequest) -> PaperChatResponse:
        """Interpret ``payload.message`` and apply the resulting changes to the paper."""

        paper = await self._load(paper_id)
        conversation = await self._conversation(paper, payload)
        await chat_repo.add_message(
            self._session,
            conversation,
            role=MessageRole.USER.value,
            content=payload.message,
        )

        plan = await self._router.plan(paper=paper, message=payload.message)
        applied = await self._apply(paper, plan.operations)
        if applied:
            _reconcile_blueprint(paper, list(paper.questions))
        report = await self._papers.validate(paper.id)

        await chat_repo.add_message(
            self._session,
            conversation,
            role=MessageRole.ASSISTANT.value,
            content=plan.reply,
            agent_name=_AGENT_NAME,
            tool_calls=[operation.model_dump(mode="json") for operation in plan.operations],
        )
        await self._session.commit()

        logger.info(
            "Paper chat turn applied",
            extra={
                "paper_id": str(paper.id),
                "conversation_id": str(conversation.id),
                "operations": len(plan.operations),
                "applied": len(applied),
                "passed": report.passed,
            },
        )
        return PaperChatResponse(
            conversation_id=conversation.id,
            reply=plan.reply,
            applied=applied,
            paper=QuestionPaperRead.from_model(paper),
            validation_report=report,
        )

    async def _load(self, paper_id: uuid.UUID) -> QuestionPaper:
        """Load a mutable paper, rejecting missing or already-approved papers."""

        paper = await paper_repo.get_paper(self._session, paper_id)
        if paper is None or not paper.is_active:
            raise NotFoundError(
                f"Question paper {paper_id} not found",
                error_code=code.QUESTION_PAPER_NOT_FOUND,
            )
        if paper.status == PaperStatus.APPROVED.value:
            raise ConflictError(
                "An approved paper can no longer be changed.",
                error_code=code.CONFLICT,
            )
        return paper

    async def _conversation(
        self, paper: QuestionPaper, payload: PaperChatRequest
    ) -> Conversation:
        """Resume the requested conversation, or start a new one for this paper."""

        if payload.conversation_id is None:
            return await chat_repo.create_conversation(self._session, title=paper.title)

        conversation = await chat_repo.get_conversation(
            self._session, payload.conversation_id
        )
        if conversation is None:
            raise NotFoundError(
                f"Conversation {payload.conversation_id} not found",
                error_code=code.NOT_FOUND,
            )
        return conversation

    async def _apply(
        self, paper: QuestionPaper, operations: list[ChangeOperation]
    ) -> list[str]:
        """Apply every operation in order and return a summary of what changed."""

        applied: list[str] = []
        for operation in operations:
            note = await self._apply_one(paper, operation)
            if note:
                applied.append(note)
            if not paper.questions:
                raise ConflictError(
                    "This change would leave the paper with no questions.",
                    error_code=code.CONFLICT,
                )
        return applied

    async def _apply_one(
        self, paper: QuestionPaper, operation: ChangeOperation
    ) -> str | None:
        """Dispatch one operation to its executor."""

        if operation.op is ChangeOp.REWRITE:
            return await self._rewrite(paper, operation)
        if operation.op is ChangeOp.DROP:
            return await self._drop(paper, operation)
        if operation.op is ChangeOp.REORDER:
            return await self._reorder(paper, operation)
        if operation.op is ChangeOp.SET_DURATION:
            return self._set_duration(paper, operation)
        if operation.op is ChangeOp.ADJUST_BLUEPRINT:
            return await self._adjust_blueprint(paper, operation)
        return None

    async def _rewrite(
        self, paper: QuestionPaper, operation: ChangeOperation
    ) -> str | None:
        """Rewrite the selected questions with the generator agent (LLM)."""

        targets = _select(paper, operation.order_indices)
        if not targets or paper.syllabus_id is None:
            return None

        chunks = await syllabus_repo.list_chunks(self._session, paper.syllabus_id)
        by_chunk = {chunk.id: chunk for chunk in chunks}

        grouped: dict[uuid.UUID, list[Question]] = defaultdict(list)
        for question in targets:
            if question.source_chunk_id in by_chunk:
                grouped[question.source_chunk_id].append(question)

        reason = operation.instructions or "Reworded at the teacher's request"
        paper_questions = [
            ExistingQuestion(
                order_index=question.order_index,
                question_text=question.question_text,
                question_type=question.question_type,
                marks=int(question.marks),
                topic=question.topic,
            )
            for question in sorted(paper.questions, key=lambda item: item.order_index)
        ]
        rewritten = 0
        for chunk_id, group in grouped.items():
            chunk = by_chunk[chunk_id]
            revised = await self._agent.revise(
                RevisionRequest(
                    syllabus_id=paper.syllabus_id,
                    subject=paper.subject,
                    unit_no=chunk.unit_no,
                    unit_name=chunk.unit_name,
                    topic_name=None,
                    chunks=[chunk],
                    targets=[
                        RevisionTarget(
                            order_index=question.order_index,
                            question_text=question.question_text,
                            question_type=question.question_type,
                            marks=int(question.marks),
                            difficulty=question.difficulty or Difficulty.MEDIUM.value,
                            issues=[reason],
                        )
                        for question in group
                    ],
                    paper_questions=paper_questions,
                )
            )
            if not revised:
                continue
            for question, item in zip(group, revised, strict=True):
                _apply_revision(question, item)
                rewritten += 1

        return f"Rewrote {rewritten} question(s)" if rewritten else None

    async def _drop(
        self, paper: QuestionPaper, operation: ChangeOperation
    ) -> str | None:
        """Remove the selected questions and renumber what remains."""

        targets = _select(paper, operation.order_indices)
        if not targets:
            return None
        for question in targets:
            paper.questions.remove(question)
        await self._session.flush()
        survivors = sorted(paper.questions, key=lambda item: item.order_index)
        await _renumber(self._session, survivors)
        return f"Removed {len(targets)} question(s)"

    async def _reorder(
        self, paper: QuestionPaper, operation: ChangeOperation
    ) -> str | None:
        """Apply a new question order, ignoring orders that are not a permutation."""

        questions = sorted(paper.questions, key=lambda item: item.order_index)
        positions = [question.order_index for question in questions]
        if len(operation.order) != len(positions) or sorted(operation.order) != positions:
            return None

        by_index = {question.order_index: question for question in questions}
        ordered = [by_index[position] for position in operation.order]
        await _renumber(self._session, ordered)
        return "Reordered the questions"

    def _set_duration(
        self, paper: QuestionPaper, operation: ChangeOperation
    ) -> str | None:
        """Change the paper's duration."""

        if operation.duration_minutes is None:
            return None
        paper.duration_minutes = operation.duration_minutes
        return f"Set the duration to {operation.duration_minutes} minutes"

    async def _adjust_blueprint(
        self, paper: QuestionPaper, operation: ChangeOperation
    ) -> str | None:
        """Reshape the paper, keeping matching questions and generating the rest."""

        if not operation.question_types:
            return None

        blueprint = _merged_blueprint(paper, operation)
        specs = _plan_questions(blueprint)
        match_difficulty = blueprint.difficulty_mix is not None

        existing = sorted(paper.questions, key=lambda item: item.order_index)
        pool = list(existing)
        plan: list[Question | QuestionSpec] = []
        needed: list[QuestionSpec] = []
        for spec in specs:
            match = _take_match(pool, spec, match_difficulty=match_difficulty)
            if match is None:
                plan.append(spec)
                needed.append(spec)
            else:
                plan.append(match)

        generated = iter(await self._papers.generate_specs(paper, needed))
        ordered = [item if isinstance(item, Question) else next(generated, None) for item in plan]

        # Detach every current question, then re-attach the reshaped set in plan
        # order; re-attaching before the next flush keeps kept questions alive.
        for question in existing:
            paper.questions.remove(question)
        for question in ordered:
            if question is not None:
                paper.questions.append(question)
        await _renumber(self._session, list(paper.questions))
        return (
            f"Reshaped the paper to {blueprint.total_marks} marks "
            f"across {len(specs)} question(s)"
        )


async def _renumber(session: AsyncSession, questions: list[Question]) -> None:
    """Renumber ``questions`` sequentially, in the order they are given.

    The caller's list order is authoritative (a reorder passes the new order, a
    drop passes the survivors). Positions are moved through a unique negative
    range first so the ``(question_paper_id, order_index)`` constraint never
    trips mid-update.
    """

    for index, question in enumerate(questions):
        question.order_index = -(index + 1)
    await session.flush()
    for index, question in enumerate(questions):
        question.order_index = index


def _select(paper: QuestionPaper, order_indices: list[int]) -> list[Question]:
    """Return the paper's questions at ``order_indices``."""

    wanted = set(order_indices)
    return [
        question
        for question in sorted(paper.questions, key=lambda item: item.order_index)
        if question.order_index in wanted
    ]


def _take_match(
    pool: list[Question], spec: QuestionSpec, *, match_difficulty: bool
) -> Question | None:
    """Remove and return an existing question that already satisfies ``spec``."""

    for index, question in enumerate(pool):
        if question.question_type != spec.question_type.value:
            continue
        if int(question.marks) != spec.marks:
            continue
        if match_difficulty and (
            question.difficulty or Difficulty.MEDIUM.value
        ) != spec.difficulty.value:
            continue
        return pool.pop(index)
    return None


def _merged_blueprint(paper: QuestionPaper, operation: ChangeOperation) -> Blueprint:
    """Build the blueprint the paper should have after ``operation``."""

    current = dict(paper.blueprint or {})
    types = operation.question_types or []
    mix = operation.difficulty_mix
    return Blueprint.model_validate(
        {
            "total_marks": sum(spec.count * spec.marks_each for spec in types),
            "duration_minutes": paper.duration_minutes,
            "question_types": [spec.model_dump(mode="json") for spec in types],
            "difficulty_mix": (
                mix.model_dump(mode="json")
                if mix is not None
                else current.get("difficulty_mix")
            ),
            "topic_weightage": current.get("topic_weightage"),
        }
    )


def _reconcile_blueprint(paper: QuestionPaper, questions: list[Question]) -> None:
    """Rewrite the paper's blueprint so it matches the questions that now exist.

    The teacher's instruction is the new specification, so the counts, marks and
    difficulty distribution are recomputed from the paper itself; re-validation
    then focuses on the substantive checks (grounding, coverage, duplication).
    """

    blueprint = dict(paper.blueprint or {})
    counts: dict[str, int] = defaultdict(int)
    marks: dict[str, int] = {}
    difficulties = {level.value: 0 for level in Difficulty}
    for question in questions:
        counts[question.question_type] += 1
        marks.setdefault(question.question_type, int(question.marks))
        level = question.difficulty or Difficulty.MEDIUM.value
        difficulties[level] = difficulties.get(level, 0) + 1

    # ``total_marks`` is derived from the per-type counts so the blueprint stays
    # internally consistent (a type's questions share one marks value).
    total = sum(counts[kind] * marks[kind] for kind in counts)
    blueprint["question_types"] = [
        {"type": kind, "count": counts[kind], "marks_each": marks[kind]} for kind in counts
    ]
    blueprint["total_marks"] = total
    if blueprint.get("difficulty_mix") is not None:
        blueprint["difficulty_mix"] = difficulties

    paper.blueprint = blueprint
    paper.total_marks = total


__all__ = ["PaperChatService"]
