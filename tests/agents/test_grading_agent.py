"""Tests for the rubric-based grading agent (FR-006)."""

from __future__ import annotations

import json
import uuid

import pytest

from src.agents import grading_agent as agent_module
from src.agents.grading_agent import GradingAgent, GradingTarget, RubricTarget
from src.client import llm
from src.prompt import grading as prompt
from src.utils.exceptions import LLMError

_ANSWER = "Force equals mass times acceleration."
_NO_CONTENT = "No relevant content in the student's answer."


def _target(**overrides: object) -> GradingTarget:
    defaults: dict[str, object] = {
        "question_text": "State Newton's second law.",
        "question_type": "short",
        "model_answer": _ANSWER,
        "max_marks": 5.0,
        "answer_text": _ANSWER,
        "is_answered": True,
        "rubrics": [RubricTarget(uuid.uuid4(), "States the law", 5.0)],
    }
    defaults.update(overrides)
    return GradingTarget(**defaults)


def _use_llm(monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]) -> None:
    """Force the LLM path and return ``payload`` as the model's JSON reply."""

    monkeypatch.setattr(agent_module.llm, "is_available", lambda: True)

    async def _complete(**_kwargs: object) -> llm.LLMResult:
        return llm.LLMResult(
            text=json.dumps(payload), model="test-model", provider="test", total_tokens=9
        )

    monkeypatch.setattr(agent_module.llm, "complete_json", _complete)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to the deterministic offline heuristic."""

    monkeypatch.setattr(agent_module.llm, "is_available", lambda: False)


# --------------------------------------------------------------------------- #
# Offline heuristic
# --------------------------------------------------------------------------- #
async def test_offline_awards_full_marks_for_a_complete_answer() -> None:
    outcome = await GradingAgent().grade(subject="Science", target=_target())

    assert outcome.awarded_marks == 5.0
    assert outcome.max_marks == 5.0
    assert outcome.evidence == _ANSWER
    assert outcome.confidence == 0.5


async def test_offline_awards_partial_marks_for_partial_coverage() -> None:
    outcome = await GradingAgent().grade(
        subject="Science", target=_target(answer_text="Force equals mass.")
    )

    assert outcome.awarded_marks == 3.0  # 3 of 5 model-answer terms present
    assert outcome.evidence == "Force equals mass."


async def test_offline_defaults_a_single_criterion_when_no_rubric() -> None:
    outcome = await GradingAgent().grade(subject=None, target=_target(rubrics=[]))

    assert len(outcome.rubric_scores) == 1
    assert outcome.rubric_scores[0].criterion_id is None
    assert "model answer" in outcome.rubric_scores[0].criterion_text


async def test_blank_answer_scores_zero_without_calling_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(**_kwargs: object) -> llm.LLMResult:
        raise AssertionError("the model must not be called for a blank answer")

    monkeypatch.setattr(agent_module.llm, "is_available", lambda: True)
    monkeypatch.setattr(agent_module.llm, "complete_json", _boom)

    outcome = await GradingAgent().grade(
        subject="Science", target=_target(answer_text="   ", is_answered=False)
    )

    assert outcome.awarded_marks == 0.0
    assert outcome.evidence == _NO_CONTENT
    assert outcome.confidence == 1.0
    assert outcome.rubric_scores[0].awarded_marks == 0.0


# --------------------------------------------------------------------------- #
# MCQ option matching
# --------------------------------------------------------------------------- #
def _mcq_target(answer: str, model_answer: str) -> GradingTarget:
    return _target(
        question_type="mcq",
        max_marks=1.0,
        answer_text=answer,
        model_answer=model_answer,
        rubrics=[RubricTarget(uuid.uuid4(), "Select the correct option.", 1.0)],
    )


@pytest.mark.parametrize(
    ("answer", "model_answer"),
    [
        ("Super computer", "Supercomputer"),
        ("M.AY", "MAX"),
        ("Randomware", "Ransomware"),
        ("Control Unit", "The Control Unit decodes instructions."),
        ("Word processing", "Word processing is not an OS function."),
    ],
)
async def test_mcq_matching_tolerates_spacing_and_ocr_slips(
    answer: str, model_answer: str
) -> None:
    outcome = await GradingAgent().grade(
        subject="Science", target=_mcq_target(answer, model_answer)
    )

    assert outcome.awarded_marks == 1.0
    assert outcome.evidence == answer


@pytest.mark.parametrize(
    ("answer", "model_answer"),
    [
        ("Group By", "The WHERE clause filters rows in a SELECT query."),
        ("Star", "Stack"),
    ],
)
async def test_mcq_matching_rejects_a_different_option(
    answer: str, model_answer: str
) -> None:
    outcome = await GradingAgent().grade(
        subject="Science", target=_mcq_target(answer, model_answer)
    )

    assert outcome.awarded_marks < 1.0


# --------------------------------------------------------------------------- #
# LLM path
# --------------------------------------------------------------------------- #
async def test_llm_path_parses_the_awarded_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_llm(
        monkeypatch,
        {
            "rubric_scores": [{"criterion": "States the law", "awarded_marks": 4}],
            "evidence": _ANSWER,
            "reasoning": "Correctly stated.",
            "confidence": 0.9,
        },
    )

    outcome = await GradingAgent().grade(subject="Science", target=_target())

    assert outcome.awarded_marks == 4.0
    assert outcome.confidence == 0.9
    assert outcome.evidence == _ANSWER
    assert outcome.reasoning == "Correctly stated."


async def test_llm_path_clamps_each_criterion_to_its_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_llm(
        monkeypatch,
        {"rubric_scores": [{"awarded_marks": 99}], "evidence": _ANSWER},
    )

    outcome = await GradingAgent().grade(subject="Science", target=_target())

    assert outcome.awarded_marks == 5.0
    assert outcome.rubric_scores[0].awarded_marks == 5.0


async def test_llm_path_resplits_when_scores_exceed_the_question_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target(
        max_marks=5.0,
        rubrics=[
            RubricTarget(uuid.uuid4(), "First part", 5.0),
            RubricTarget(uuid.uuid4(), "Second part", 5.0),
        ],
    )
    _use_llm(
        monkeypatch,
        {
            "rubric_scores": [{"awarded_marks": 5}, {"awarded_marks": 5}],
            "evidence": _ANSWER,
        },
    )

    outcome = await GradingAgent().grade(subject="Science", target=target)

    assert outcome.awarded_marks == 5.0
    assert [score.awarded_marks for score in outcome.rubric_scores] == [2.5, 2.5]


async def test_llm_path_replaces_non_verbatim_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_llm(
        monkeypatch,
        {"rubric_scores": [{"awarded_marks": 5}], "evidence": "A paraphrase not in the answer."},
    )

    outcome = await GradingAgent().grade(subject="Science", target=_target())

    assert outcome.evidence in _ANSWER


async def test_llm_failure_falls_back_to_the_offline_heuristic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module.llm, "is_available", lambda: True)

    async def _boom(**_kwargs: object) -> llm.LLMResult:
        raise LLMError("provider unavailable")

    monkeypatch.setattr(agent_module.llm, "complete_json", _boom)

    outcome = await GradingAgent().grade(subject="Science", target=_target())

    assert outcome.awarded_marks == 5.0
    assert outcome.confidence == 0.5


# --------------------------------------------------------------------------- #
# Prompt contract
# --------------------------------------------------------------------------- #
def test_build_prompt_includes_the_rubric_and_the_verbatim_answer() -> None:
    system, user = prompt.build_prompt(subject="Science", target=_target())

    assert "Science" in system
    assert "RUBRIC" in user
    assert "States the law" in user
    assert _ANSWER in user
    assert "OUTPUT CONTRACT" in user
