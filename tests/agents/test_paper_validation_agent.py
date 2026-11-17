"""Tests for the question paper validation agent (FR-003)."""

from __future__ import annotations

from src.agents.paper_validation_agent import ValidationTarget
from src.prompt import paper_validation as prompt


def _target(**overrides: object) -> ValidationTarget:
    defaults: dict[str, object] = {
        "index": 0,
        "question_text": "Which of the following is NOT a main characteristic of a computer?",
        "question_type": "mcq",
        "marks": 1,
        "topic": "Computer characteristics",
    }
    defaults.update(overrides)
    return ValidationTarget(**defaults)


def test_prompt_includes_mcq_options() -> None:
    target = _target(options=["Speed", "Accuracy", "Reliability", "Versatility"])

    _, user = prompt.build_prompt(subject="Computer Science", questions=[target])

    assert "options: A) Speed | B) Accuracy | C) Reliability | D) Versatility" in user
    assert "judge the question together with those options" in user


def test_prompt_omits_options_when_absent() -> None:
    target = _target(question_type="short", topic=None, options=None)

    _, user = prompt.build_prompt(subject="Science", questions=[target])

    assert "options:" not in user
