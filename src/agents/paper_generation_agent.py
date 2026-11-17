"""Question paper generation agent (FR-002).

The agent turns a *section* of verbatim syllabus text plus a list of required
question specs into typed questions. It uses the configured LLM when available
and otherwise falls back to a deterministic offline stub, so the pipeline can be
exercised without network access. Both paths emit the same JSON shape, which is
validated before anything is persisted.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from src.client import llm
from src.prompt import paper_generation as prompt
from src.repositories.observability_repository import default_usage_recorder
from src.repositories.schema import SyllabusChunk
from src.settings import PAPER_GENERATION_TEMPERATURE, Difficulty, QuestionType
from src.utils.exceptions import AppError
from src.utils.exceptions import error_codes as code
from src.utils.logger import get_logger

logger = get_logger(__name__)

_AGENT_NAME = "paper_generation"
_REVISION_TASK = "paper_revision"
_MCQ_OPTION_COUNT = 4


@dataclass(slots=True)
class QuestionSpec:
    """One required question: format, marks and difficulty."""

    question_type: QuestionType
    marks: int
    difficulty: Difficulty


@dataclass(slots=True)
class GenerationRequest:
    """Everything the agent needs for one generation call."""

    syllabus_id: uuid.UUID
    subject: str | None
    unit_no: str | None
    unit_name: str | None
    topic_name: str | None
    chunks: list[SyllabusChunk]
    specs: list[QuestionSpec]


@dataclass(slots=True)
class RevisionTarget:
    """An existing question to rewrite, with the reasons it was flagged."""

    order_index: int
    question_text: str
    question_type: str
    marks: int
    difficulty: str
    issues: list[str]


@dataclass(slots=True)
class ExistingQuestion:
    """A sibling question the rewrite must stay distinct from."""

    order_index: int
    question_text: str
    question_type: str
    marks: int
    topic: str | None = None


@dataclass(slots=True)
class RevisionRequest:
    """A request to rewrite specific questions in place.

    ``paper_questions`` carries every other question already on the paper so a
    rewrite can avoid duplicating their wording or idea.
    """

    syllabus_id: uuid.UUID
    subject: str | None
    unit_no: str | None
    unit_name: str | None
    topic_name: str | None
    chunks: list[SyllabusChunk]
    targets: list[RevisionTarget]
    paper_questions: list[ExistingQuestion] = field(default_factory=list)


@dataclass(slots=True)
class GeneratedRubric:
    """A single scoring criterion produced alongside a question."""

    criterion_text: str
    max_marks: Decimal


@dataclass(slots=True)
class GeneratedQuestion:
    """A validated question ready to be persisted."""

    question_text: str
    question_type: str
    marks: Decimal
    difficulty: str
    topic: str | None
    options: list[str] | None
    correct_option: str | None
    model_answer: str
    source_chunk_id: uuid.UUID | None
    rubrics: list[GeneratedRubric] = field(default_factory=list)


class PaperGenerationAgent:
    """Writes questions from raw syllabus chunks."""

    async def generate(self, request: GenerationRequest) -> list[GeneratedQuestion]:
        """Generate and validate one batch of questions for ``request``."""

        if llm.is_available():
            payload = await self._generate_with_llm(request)
        else:
            logger.warning(
                "No LLM key configured; using the offline stub generator",
                extra={"task": _AGENT_NAME},
            )
            payload = _stub_payload(request)
        return _build_questions(request, payload)

    async def _generate_with_llm(self, request: GenerationRequest) -> dict[str, Any]:
        """Call the LLM and return the parsed JSON payload."""

        system, user = prompt.build_prompt(request=request)
        result = await llm.complete_json(
            system=system,
            user=user,
            task=_AGENT_NAME,
            temperature=PAPER_GENERATION_TEMPERATURE,
            sink=default_usage_recorder(),
        )
        logger.info(
            "Paper generation LLM call",
            extra={
                "task": _AGENT_NAME,
                "model": result.model,
                "total_tokens": result.total_tokens,
                "latency_ms": result.latency_ms,
            },
        )
        return llm.extract_json_object(result.text)

    async def revise(self, request: RevisionRequest) -> list[GeneratedQuestion]:
        """Rewrite the targeted questions so they address their flagged issues.

        Returns an empty list when no LLM is configured: the offline path cannot
        author new wording, so the caller should stop trying to repair.
        """

        if not request.targets:
            return []
        if not llm.is_available():
            logger.warning(
                "No LLM key configured; skipping question revision",
                extra={"task": _REVISION_TASK},
            )
            return []

        system, user = prompt.build_revision_prompt(request=request)
        result = await llm.complete_json(
            system=system,
            user=user,
            task=_REVISION_TASK,
            temperature=PAPER_GENERATION_TEMPERATURE,
            sink=default_usage_recorder(),
        )
        logger.info(
            "Paper revision LLM call",
            extra={
                "task": _REVISION_TASK,
                "model": result.model,
                "total_tokens": result.total_tokens,
                "latency_ms": result.latency_ms,
            },
        )
        payload = llm.extract_json_object(result.text)
        return _build_questions(_as_generation_request(request), payload)


def _as_generation_request(request: RevisionRequest) -> GenerationRequest:
    """Adapt a revision request onto the shared question-building contract."""

    return GenerationRequest(
        syllabus_id=request.syllabus_id,
        subject=request.subject,
        unit_no=request.unit_no,
        unit_name=request.unit_name,
        topic_name=request.topic_name,
        chunks=request.chunks,
        specs=[
            QuestionSpec(
                question_type=QuestionType(target.question_type),
                marks=target.marks,
                difficulty=Difficulty(target.difficulty),
            )
            for target in request.targets
        ],
    )


def _build_questions(
    request: GenerationRequest, payload: dict[str, Any]
) -> list[GeneratedQuestion]:
    """Validate the raw payload and map it onto the requested specs."""

    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or len(raw_questions) != len(request.specs):
        raise AppError(
            "The generation model returned an unexpected number of questions.",
            error_code=code.PAPER_GENERATION_FAILED,
            detail={"expected": len(request.specs), "received": len(raw_questions or [])},
        )

    chunk_ids = {chunk.id for chunk in request.chunks}
    default_chunk_id = request.chunks[0].id if request.chunks else None
    return [
        _parse_question(raw, spec, request, chunk_ids, default_chunk_id)
        for raw, spec in zip(raw_questions, request.specs, strict=True)
    ]


def _parse_question(
    raw: object,
    spec: QuestionSpec,
    request: GenerationRequest,
    chunk_ids: set[uuid.UUID],
    default_chunk_id: uuid.UUID | None,
) -> GeneratedQuestion:
    """Validate one raw question dict against its spec."""

    if not isinstance(raw, dict):
        raise _failed("A generated question was not a JSON object.")

    question_text = str(raw.get("question_text") or "").strip()
    if not question_text:
        raise _failed("A generated question was missing its text.")

    model_answer = str(raw.get("model_answer") or "").strip()
    if not model_answer:
        raise _failed("A generated question was missing its model answer.")

    options, correct_option = _parse_options(raw, spec)
    source_chunk_id = _parse_source_chunk_id(raw, chunk_ids) or default_chunk_id
    topic = str(raw.get("topic") or request.topic_name or request.unit_name or "").strip() or None

    return GeneratedQuestion(
        question_text=question_text,
        question_type=spec.question_type.value,
        marks=Decimal(spec.marks),
        difficulty=spec.difficulty.value,
        topic=topic,
        options=options,
        correct_option=correct_option,
        model_answer=model_answer,
        source_chunk_id=source_chunk_id,
        rubrics=_parse_rubrics(raw.get("rubrics"), spec),
    )


def _parse_options(raw: dict[str, Any], spec: QuestionSpec) -> tuple[list[str] | None, str | None]:
    """Validate MCQ options; non-MCQ questions carry neither options nor answer."""

    if spec.question_type is not QuestionType.MCQ:
        return None, None

    options = raw.get("options")
    if not isinstance(options, list) or len(options) < 2:
        raise _failed("A multiple-choice question was missing its options.")
    cleaned = [str(option).strip() for option in options if str(option).strip()]
    if len(cleaned) < 2:
        raise _failed("A multiple-choice question had too few usable options.")

    correct_option = str(raw.get("correct_option") or "").strip()
    if correct_option not in cleaned:
        raise _failed("A multiple-choice question's correct option was not among its options.")
    return cleaned, correct_option


def _parse_source_chunk_id(raw: dict[str, Any], chunk_ids: set[uuid.UUID]) -> uuid.UUID | None:
    """Resolve the cited chunk id, requiring it to reference a supplied chunk."""

    value = raw.get("source_chunk_id")
    if not value:
        return None
    try:
        candidate = uuid.UUID(str(value))
    except ValueError:
        return None
    return candidate if candidate in chunk_ids else None


def _parse_rubrics(raw_rubrics: object, spec: QuestionSpec) -> list[GeneratedRubric]:
    """Validate the rubric criteria, defaulting to one full-marks criterion."""

    rubrics: list[GeneratedRubric] = []
    if isinstance(raw_rubrics, list):
        for item in raw_rubrics:
            if not isinstance(item, dict):
                continue
            text = str(item.get("criterion_text") or "").strip()
            if not text:
                continue
            rubrics.append(
                GeneratedRubric(
                    criterion_text=text,
                    max_marks=_to_decimal(item.get("max_marks"), spec.marks),
                )
            )
    if rubrics:
        return rubrics

    logger.warning(
        "A generated question had no rubric; using a single full-marks criterion",
        extra={"task": _AGENT_NAME},
    )
    return [
        GeneratedRubric(
            criterion_text="Award full marks for a correct answer.",
            max_marks=Decimal(spec.marks),
        )
    ]


def _to_decimal(value: object, fallback: int) -> Decimal:
    """Coerce a rubric mark value to Decimal, falling back when unusable."""

    try:
        return Decimal(str(value))
    except (ArithmeticError, ValueError, TypeError):
        return Decimal(fallback)


def _failed(message: str) -> AppError:
    """Build a paper-generation failure error."""

    return AppError(message, error_code=code.PAPER_GENERATION_FAILED)


def _stub_payload(request: GenerationRequest) -> dict[str, Any]:
    """Deterministically build a valid payload without calling any LLM.

    Questions are grounded in the section's verbatim text so the offline path
    exercises the full pipeline (parsing, persistence, traceability).
    """

    topic = (
        request.topic_name
        or request.unit_name
        or (f"Unit {request.unit_no}" if request.unit_no else "this syllabus")
    )
    questions: list[dict[str, Any]] = []
    for index, spec in enumerate(request.specs):
        chunk = request.chunks[index % len(request.chunks)] if request.chunks else None
        basis = _first_sentence(chunk.raw_text) if chunk else topic
        chunk_id = str(chunk.id) if chunk else ""
        if spec.question_type is QuestionType.MCQ:
            questions.append(
                {
                    "question_text": f"Which statement about {topic} is correct?",
                    "options": [
                        basis,
                        f"{topic} is not covered in this syllabus.",
                        "The opposite of the syllabus statement.",
                        "None of the above.",
                    ],
                    "correct_option": basis,
                    "model_answer": basis,
                    "topic": topic,
                    "source_chunk_id": chunk_id,
                    "rubrics": [
                        {
                            "criterion_text": f"Selects the correct statement about {topic}.",
                            "max_marks": spec.marks,
                        }
                    ],
                }
            )
        else:
            questions.append(
                {
                    "question_text": (
                        f"Explain {topic} as described in the syllabus "
                        f"({spec.difficulty.value} aspect {index + 1})."
                    ),
                    "options": None,
                    "correct_option": None,
                    "model_answer": basis,
                    "topic": topic,
                    "source_chunk_id": chunk_id,
                    "rubrics": [
                        {
                            "criterion_text": f"Explains {topic} accurately.",
                            "max_marks": spec.marks,
                        }
                    ],
                }
            )
    return {"questions": questions}


def _first_sentence(text: str) -> str:
    """Return the first sentence of ``text`` (used as the stub's answer)."""

    for separator in (".", "?", "!"):
        head = text.split(separator, 1)[0].strip()
        if head:
            return f"{head}{separator}"
    return text.strip()


__all__ = [
    "ExistingQuestion",
    "GenerationRequest",
    "GeneratedQuestion",
    "GeneratedRubric",
    "PaperGenerationAgent",
    "QuestionSpec",
    "RevisionRequest",
    "RevisionTarget",
]
