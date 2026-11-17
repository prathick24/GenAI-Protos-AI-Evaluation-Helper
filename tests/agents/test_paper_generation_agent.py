"""Tests for the question paper generation agent (FR-002)."""

from __future__ import annotations

import json
import uuid
from decimal import Decimal

import pytest

from src.agents import paper_generation_agent as agent_module
from src.agents.paper_generation_agent import (
    ExistingQuestion,
    GenerationRequest,
    PaperGenerationAgent,
    QuestionSpec,
    RevisionRequest,
    RevisionTarget,
)
from src.client import llm
from src.prompt import paper_generation as prompt
from src.repositories.schema import SyllabusChunk
from src.settings import Difficulty, QuestionType
from src.utils.exceptions import AppError
from src.utils.exceptions import error_codes as code


def _chunk(**overrides: object) -> SyllabusChunk:
    defaults: dict[str, object] = {
        "id": uuid.uuid4(),
        "syllabus_id": uuid.uuid4(),
        "chunk_index": 0,
        "raw_text": "Force equals mass times acceleration.",
        "unit_no": "1",
        "unit_name": "Mechanics",
        "topic_name": "Newton's Laws",
    }
    defaults.update(overrides)
    return SyllabusChunk(**defaults)


def _request(
    specs: list[QuestionSpec],
    chunks: list[SyllabusChunk] | None = None,
    **overrides: object,
) -> GenerationRequest:
    defaults: dict[str, object] = {
        "syllabus_id": uuid.uuid4(),
        "subject": "Science",
        "unit_no": "1",
        "unit_name": "Mechanics",
        "topic_name": None,
        "chunks": [_chunk()] if chunks is None else chunks,
        "specs": specs,
    }
    defaults.update(overrides)
    return GenerationRequest(**defaults)


def _use_llm(monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]) -> None:
    """Force the LLM path and return ``payload`` as the model's JSON reply."""

    monkeypatch.setattr(agent_module.llm, "is_available", lambda: True)

    async def _complete(**_kwargs: object) -> llm.LLMResult:
        return llm.LLMResult(
            text=json.dumps(payload), model="test-model", provider="test", total_tokens=7
        )

    monkeypatch.setattr(agent_module.llm, "complete_json", _complete)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to the deterministic offline stub."""

    monkeypatch.setattr(agent_module.llm, "is_available", lambda: False)


# --------------------------------------------------------------------------- #
# Offline stub path
# --------------------------------------------------------------------------- #
async def test_stub_generates_one_question_per_spec() -> None:
    chunks = [_chunk()]
    specs = [
        QuestionSpec(QuestionType.SHORT, 5, Difficulty.EASY),
        QuestionSpec(QuestionType.MCQ, 2, Difficulty.HARD),
    ]

    result = await PaperGenerationAgent().generate(_request(specs, chunks))

    assert len(result) == 2
    short, mcq = result
    assert short.question_type == "short"
    assert short.marks == Decimal(5)
    assert short.difficulty == "easy"
    assert short.topic == "Mechanics"
    assert short.source_chunk_id == chunks[0].id
    assert short.model_answer == "Force equals mass times acceleration."
    assert short.rubrics[0].max_marks == Decimal(5)

    assert mcq.question_type == "mcq"
    assert mcq.marks == Decimal(2)
    assert mcq.difficulty == "hard"
    assert mcq.options is not None
    assert len(mcq.options) == 4
    assert mcq.correct_option in mcq.options


async def test_stub_without_chunks_uses_topic_only() -> None:
    specs = [QuestionSpec(QuestionType.SHORT, 1, Difficulty.MEDIUM)]

    result = await PaperGenerationAgent().generate(_request(specs, []))

    assert result[0].source_chunk_id is None
    assert result[0].topic == "Mechanics"


# --------------------------------------------------------------------------- #
# LLM path
# --------------------------------------------------------------------------- #
async def test_llm_path_parses_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunk()
    _use_llm(
        monkeypatch,
        {
            "questions": [
                {
                    "question_text": "State Newton's second law.",
                    "options": None,
                    "correct_option": None,
                    "model_answer": "F = ma",
                    "topic": "Newton's Laws",
                    "source_chunk_id": str(chunk.id),
                    "rubrics": [{"criterion_text": "States F = ma", "max_marks": 5}],
                }
            ]
        },
    )

    result = await PaperGenerationAgent().generate(
        _request([QuestionSpec(QuestionType.SHORT, 5, Difficulty.MEDIUM)], [chunk])
    )

    assert result[0].question_text == "State Newton's second law."
    assert result[0].source_chunk_id == chunk.id
    assert result[0].rubrics[0].criterion_text == "States F = ma"


async def test_llm_path_rejects_wrong_question_count(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_llm(monkeypatch, {"questions": []})

    with pytest.raises(AppError) as excinfo:
        await PaperGenerationAgent().generate(
            _request([QuestionSpec(QuestionType.SHORT, 5, Difficulty.EASY)])
        )

    assert excinfo.value.error_code == code.PAPER_GENERATION_FAILED


async def test_llm_path_rejects_missing_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_llm(monkeypatch, {"questions": [{"question_text": "Q?"}]})

    with pytest.raises(AppError):
        await PaperGenerationAgent().generate(
            _request([QuestionSpec(QuestionType.SHORT, 5, Difficulty.EASY)])
        )


async def test_llm_path_rejects_mcq_without_options(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_llm(
        monkeypatch,
        {"questions": [{"question_text": "Q?", "options": None, "model_answer": "A"}]},
    )

    with pytest.raises(AppError):
        await PaperGenerationAgent().generate(
            _request([QuestionSpec(QuestionType.MCQ, 1, Difficulty.EASY)])
        )


async def test_llm_path_falls_back_to_default_source_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunk()
    _use_llm(
        monkeypatch,
        {
            "questions": [
                {
                    "question_text": "Q?",
                    "model_answer": "A",
                    "source_chunk_id": str(uuid.uuid4()),
                }
            ]
        },
    )

    result = await PaperGenerationAgent().generate(
        _request([QuestionSpec(QuestionType.SHORT, 1, Difficulty.EASY)], [chunk])
    )

    assert result[0].source_chunk_id == chunk.id


async def test_llm_path_defaults_missing_rubric(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_llm(monkeypatch, {"questions": [{"question_text": "Q?", "model_answer": "A"}]})

    result = await PaperGenerationAgent().generate(
        _request([QuestionSpec(QuestionType.SHORT, 3, Difficulty.EASY)])
    )

    assert result[0].rubrics[0].max_marks == Decimal(3)
    assert "full marks" in result[0].rubrics[0].criterion_text.lower()


# --------------------------------------------------------------------------- #
# Prompt contract
# --------------------------------------------------------------------------- #
def test_build_prompt_includes_verbatim_text_and_chunk_ids() -> None:
    chunk = _chunk(raw_text="Momentum is mass times velocity.")
    request = _request([QuestionSpec(QuestionType.SHORT, 3, Difficulty.MEDIUM)], [chunk])

    system, user = prompt.build_prompt(request=request)

    assert "Science" in system
    assert "Momentum is mass times velocity." in user
    assert str(chunk.id) in user
    assert "type=short marks=3 difficulty=medium" in user
    assert "Never ask what the syllabus lists" in user


def test_revision_prompt_includes_other_paper_questions() -> None:
    chunk = _chunk()
    request = RevisionRequest(
        syllabus_id=uuid.uuid4(),
        subject="Science",
        unit_no="1",
        unit_name="Mechanics",
        topic_name=None,
        chunks=[chunk],
        targets=[
            RevisionTarget(
                order_index=0,
                question_text="Which of the following is NOT a force?",
                question_type="mcq",
                marks=1,
                difficulty="easy",
                issues=["Near-duplicate of question 2"],
            )
        ],
        paper_questions=[
            ExistingQuestion(
                order_index=1,
                question_text="Define momentum and state its SI unit.",
                question_type="short",
                marks=3,
                topic="Momentum",
            )
        ],
    )

    _, user = prompt.build_revision_prompt(request=request)

    assert "WHOLE QUESTION PAPER" in user
    assert "Define momentum and state its SI unit." in user
    assert "Near-duplicate of question 2" in user
    assert "clearly distinct" in user
    assert "Never ask what the syllabus lists" in user
