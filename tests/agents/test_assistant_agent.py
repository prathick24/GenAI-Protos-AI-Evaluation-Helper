"""Tests for the grounded Q&A assistant agent (FR-010)."""

from __future__ import annotations

import json
import uuid

import pytest

from src.agents import assistant_agent as agent_module
from src.agents.assistant_agent import AssistantAgent, ContextDocument
from src.client import llm
from src.utils.exceptions import LLMError


def _documents() -> list[ContextDocument]:
    return [
        ContextDocument(
            source_type="syllabus",
            source_id=uuid.uuid4(),
            snippet="Newton's second law states that force equals mass times acceleration.",
            unit_no="1",
            topic_name="Mechanics",
        ),
        ContextDocument(
            source_type="question",
            source_id=uuid.uuid4(),
            snippet="State Newton's second law and give one example.",
            topic_name="Mechanics",
        ),
    ]


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
    """Default every test to the deterministic offline answer."""

    monkeypatch.setattr(agent_module.llm, "is_available", lambda: False)


# --------------------------------------------------------------------------- #
# Offline path
# --------------------------------------------------------------------------- #
async def test_offline_quotes_the_top_documents_with_citations() -> None:
    outcome = await AssistantAgent().compose(
        question="What is Newton's second law?", documents=_documents()
    )

    assert outcome.answer.startswith("Based on the uploaded material:")
    assert "[1]" in outcome.answer
    assert "[2]" in outcome.answer
    assert outcome.cited == [0, 1]


async def test_offline_caps_citations_at_two_documents() -> None:
    documents = _documents() + [
        ContextDocument("question", uuid.uuid4(), "Define momentum.")
    ]

    outcome = await AssistantAgent().compose(question="momentum", documents=documents)

    assert outcome.cited == [0, 1]


async def test_offline_says_nothing_was_found_without_documents() -> None:
    outcome = await AssistantAgent().compose(question="What is covered?", documents=[])

    assert "could not find" in outcome.answer
    assert outcome.cited == []


# --------------------------------------------------------------------------- #
# LLM path
# --------------------------------------------------------------------------- #
async def test_llm_path_maps_one_based_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_llm(
        monkeypatch,
        {"answer": "Force equals mass times acceleration [1].", "citations": [1, 2]},
    )

    outcome = await AssistantAgent().compose(
        question="What is Newton's second law?", documents=_documents()
    )

    assert outcome.answer == "Force equals mass times acceleration [1]."
    assert outcome.cited == [0, 1]
    assert outcome.tokens == 9


async def test_llm_out_of_range_citations_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_llm(monkeypatch, {"answer": "Made up answer.", "citations": [9]})

    outcome = await AssistantAgent().compose(
        question="What is Newton's second law?", documents=_documents()
    )

    assert outcome.answer.startswith("Based on the uploaded material:")
    assert outcome.cited == [0, 1]


async def test_llm_without_citations_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_llm(monkeypatch, {"answer": "Unsupported answer.", "citations": []})

    outcome = await AssistantAgent().compose(
        question="What is Newton's second law?", documents=_documents()
    )

    assert outcome.answer.startswith("Based on the uploaded material:")


async def test_llm_failure_falls_back_to_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module.llm, "is_available", lambda: True)

    async def _boom(**_kwargs: object) -> llm.LLMResult:
        raise LLMError("boom", error_code="AEH-9999")

    monkeypatch.setattr(agent_module.llm, "complete_json", _boom)

    outcome = await AssistantAgent().compose(
        question="What is Newton's second law?", documents=_documents()
    )

    assert outcome.answer.startswith("Based on the uploaded material:")
