"""Grounded Q&A assistant agent (FR-010).

Answers a teacher's question from a bounded set of retrieved context documents
and cites the ones it used. When no LLM is configured a deterministic writer
quotes the top documents, so the endpoint still returns grounded, cited answers
offline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.client import llm
from src.prompt import assistant as prompt
from src.repositories.observability_repository import default_usage_recorder
from src.settings import CHAT_ASSISTANT_TEMPERATURE
from src.utils.exceptions import AppError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_AGENT_NAME = "assistant"
_SNIPPET_LIMIT = 420
_OFFLINE_PREFIX = "Based on the uploaded material: "


@dataclass(slots=True)
class ContextDocument:
    """One retrievable source the assistant may answer from."""

    source_type: str
    source_id: uuid.UUID
    snippet: str
    unit_no: str | None = None
    topic_name: str | None = None


@dataclass(slots=True)
class AssistantAnswer:
    """The grounded answer text, the (zero-based) documents it cites and usage."""

    answer: str
    cited: list[int] = field(default_factory=list)
    tokens: int | None = None


class AssistantAgent:
    """Writes a grounded, cited answer for one teacher question."""

    async def compose(
        self, *, question: str, documents: list[ContextDocument]
    ) -> AssistantAnswer:
        """Return the answer for ``question`` using only ``documents``."""

        if not documents:
            return AssistantAnswer(answer=_scope_reply(documents))

        fallback = _offline_answer(documents)
        if not llm.is_available():
            logger.warning(
                "No LLM key configured; using the offline grounded answer",
                extra={"task": _AGENT_NAME},
            )
            return fallback

        try:
            system, user = prompt.build_prompt(question=question, documents=documents)
            result = await llm.complete_json(
                system=system,
                user=user,
                task=_AGENT_NAME,
                temperature=CHAT_ASSISTANT_TEMPERATURE,
                sink=default_usage_recorder(),
            )
            payload = llm.extract_json_object(result.text)
        except AppError as exc:
            logger.warning(
                "Assistant LLM call failed; using the offline grounded answer",
                extra={"task": _AGENT_NAME, "error": exc.error_code},
            )
            return fallback

        logger.info(
            "Assistant LLM call",
            extra={
                "task": _AGENT_NAME,
                "model": result.model,
                "total_tokens": result.total_tokens,
                "latency_ms": result.latency_ms,
            },
        )
        return _parse(payload, documents, fallback, tokens=result.total_tokens)


def _offline_answer(documents: list[ContextDocument]) -> AssistantAnswer:
    """Quote the strongest documents as a deterministic, cited answer."""

    cited = list(range(min(2, len(documents))))
    parts = [
        f"{_trim(documents[index].snippet)} [{index + 1}]" for index in cited
    ]
    return AssistantAnswer(answer=_OFFLINE_PREFIX + " ".join(parts), cited=cited)


def _parse(
    payload: dict[str, object],
    documents: list[ContextDocument],
    fallback: AssistantAnswer,
    *,
    tokens: int | None,
) -> AssistantAnswer:
    """Map the model's JSON onto an answer, falling back to grounded text.

    A non-empty answer with no valid citation cannot be trusted to be grounded,
    so the deterministic answer is used instead.
    """

    answer = _text(payload.get("answer"))
    cited = _citation_indices(payload.get("citations"), len(documents))
    if not answer or not cited:
        return fallback
    return AssistantAnswer(answer=answer, cited=cited, tokens=tokens)


def _citation_indices(value: object, limit: int) -> list[int]:
    """Return the unique, in-range zero-based indices from a JSON citation list."""

    if not isinstance(value, list):
        return []
    indices: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            continue
        index = item - 1
        if 0 <= index < limit and index not in indices:
            indices.append(index)
    return indices


def _scope_reply(documents: list[ContextDocument]) -> str:
    """Return the scope reply used when no context was retrieved."""

    topics = sorted(
        {document.topic_name for document in documents if document.topic_name}
    )[:5]
    if topics:
        return (
            "I could not find that in the uploaded material. "
            f"I can answer questions about: {', '.join(topics)}."
        )
    return (
        "I could not find that in the uploaded material. Upload a syllabus or "
        "generate a question paper first."
    )


def _trim(text: str) -> str:
    """Collapse whitespace and cap a snippet for display."""

    value = " ".join(text.split())
    return value[:_SNIPPET_LIMIT].rstrip()


def _text(value: object) -> str:
    """Return a stripped string from a JSON value, or an empty string."""

    return str(value).strip() if value is not None else ""


__all__ = ["AssistantAgent", "AssistantAnswer", "ContextDocument"]
