"""Tests for the per-student feedback agent (FR-008)."""

from __future__ import annotations

import json

import pytest

from src.agents import feedback_agent as agent_module
from src.agents.feedback_agent import (
    FeedbackAgent,
    FeedbackItem,
    FeedbackTarget,
)
from src.client import llm
from src.utils.exceptions import LLMError


def _target(**overrides: object) -> FeedbackTarget:
    defaults: dict[str, object] = {
        "total_awarded_marks": 7.0,
        "total_max_marks": 10.0,
        "student_name": "Asha",
        "subject": "Science",
        "items": [
            FeedbackItem(
                question_text="State the law.",
                topic="Motion",
                max_marks=5.0,
                awarded_marks=5.0,
            ),
            FeedbackItem(
                question_text="Define momentum.",
                topic="Forces",
                max_marks=5.0,
                awarded_marks=2.0,
            ),
        ],
    }
    defaults.update(overrides)
    return FeedbackTarget(**defaults)


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
    """Default every test to the deterministic offline writer."""

    monkeypatch.setattr(agent_module.llm, "is_available", lambda: False)


# --------------------------------------------------------------------------- #
# Offline writer
# --------------------------------------------------------------------------- #
async def test_offline_writes_grounded_feedback() -> None:
    outcome = await FeedbackAgent().compose(target=_target())

    assert "Asha scored 7 out of 10 marks (70%)" in outcome.overall_comments
    assert "Motion" in outcome.strengths
    assert "Forces" in outcome.improvements


async def test_offline_mentions_unanswered_questions() -> None:
    target = _target(
        items=[
            FeedbackItem(
                question_text="State the law.",
                topic="Motion",
                max_marks=5.0,
                awarded_marks=0.0,
                is_answered=False,
            )
        ]
    )

    outcome = await FeedbackAgent().compose(target=target)

    assert "left unanswered" in outcome.overall_comments
    assert "left blank" in outcome.improvements


async def test_offline_handles_a_sheet_without_topics() -> None:
    target = _target(items=[])

    outcome = await FeedbackAgent().compose(target=target)

    assert "no marks were recorded" in outcome.strengths
    assert outcome.improvements


# --------------------------------------------------------------------------- #
# LLM path
# --------------------------------------------------------------------------- #
async def test_llm_path_uses_the_model_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_llm(
        monkeypatch,
        {
            "overall_comments": "A solid effort.",
            "strengths": "Motion.",
            "improvements": "Forces.",
        },
    )

    outcome = await FeedbackAgent().compose(target=_target())

    assert outcome.overall_comments == "A solid effort."
    assert outcome.strengths == "Motion."
    assert outcome.improvements == "Forces."


async def test_llm_blank_fields_fall_back_to_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_llm(monkeypatch, {"overall_comments": "", "strengths": "Motion."})

    outcome = await FeedbackAgent().compose(target=_target())

    assert outcome.overall_comments.startswith("Asha scored")
    assert outcome.strengths == "Motion."
    assert "Forces" in outcome.improvements


async def test_llm_failure_falls_back_to_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module.llm, "is_available", lambda: True)

    async def _boom(**_kwargs: object) -> llm.LLMResult:
        raise LLMError("boom", error_code="AEH-9999")

    monkeypatch.setattr(agent_module.llm, "complete_json", _boom)

    outcome = await FeedbackAgent().compose(target=_target())

    assert outcome.overall_comments.startswith("Asha scored")
