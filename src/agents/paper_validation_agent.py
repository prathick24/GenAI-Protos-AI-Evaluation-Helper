"""Question paper validation agent (FR-003).

The authoritative checks — coverage, marks, difficulty mix, source grounding and
near-duplicate detection — are deterministic and live in the service layer. This
agent contributes only the *advisory* judgement checks: whether a question's
wording is ambiguous/clarity-poor, and whether it duplicates an earlier one.

When no LLM is configured the review is skipped and an empty list is returned,
so validation still runs on the deterministic checks alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.client import llm
from src.prompt import paper_validation as prompt
from src.repositories.observability_repository import default_usage_recorder
from src.settings import PAPER_VALIDATION_TEMPERATURE
from src.utils.exceptions import AppError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_AGENT_NAME = "paper_validation"


@dataclass(slots=True)
class ValidationTarget:
    """One question submitted for advisory review."""

    index: int
    question_text: str
    question_type: str
    marks: int
    topic: str | None
    options: list[str] | None = None


@dataclass(slots=True)
class QuestionReview:
    """The agent's advisory verdict for one question."""

    index: int
    ambiguous: bool
    duplicate_of: int | None
    notes: str | None


class PaperValidationAgent:
    """Reviews questions for ambiguous wording and duplication."""

    async def review(
        self, *, subject: str | None, questions: list[ValidationTarget]
    ) -> list[QuestionReview]:
        """Return one advisory review per question, or an empty list when skipped."""

        if not questions:
            return []
        if not llm.is_available():
            logger.warning(
                "No LLM key configured; skipping the advisory validation review",
                extra={"task": _AGENT_NAME},
            )
            return []

        system, user = prompt.build_prompt(subject=subject, questions=questions)
        indices = {target.index for target in questions}
        try:
            result = await llm.complete_json(
                system=system,
                user=user,
                task=_AGENT_NAME,
                temperature=PAPER_VALIDATION_TEMPERATURE,
                sink=default_usage_recorder(),
            )
            payload = llm.extract_json_object(result.text)
        except AppError as exc:
            logger.warning(
                "Advisory validation review unavailable",
                extra={"task": _AGENT_NAME, "error": exc.error_code},
            )
            return []

        logger.info(
            "Paper validation LLM call",
            extra={
                "task": _AGENT_NAME,
                "model": result.model,
                "total_tokens": result.total_tokens,
                "latency_ms": result.latency_ms,
            },
        )
        return _parse_reviews(payload, indices=indices)


def _parse_reviews(payload: dict[str, Any], *, indices: set[int]) -> list[QuestionReview]:
    """Leniently map the raw payload onto reviews, ignoring unusable entries."""

    raw = payload.get("reviews")
    if not isinstance(raw, list):
        logger.warning(
            "Advisory validation review returned no 'reviews' list",
            extra={"task": _AGENT_NAME},
        )
        return []

    reviews: list[QuestionReview] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        index = _to_index(item.get("index"), indices)
        if index is None:
            continue
        duplicate_of = _to_index(item.get("duplicate_of"), indices)
        reviews.append(
            QuestionReview(
                index=index,
                ambiguous=_to_bool(item.get("ambiguous")),
                duplicate_of=duplicate_of if duplicate_of != index else None,
                notes=str(item.get("notes") or "").strip() or None,
            )
        )
    return reviews


def _to_index(value: object, indices: set[int]) -> int | None:
    """Coerce ``value`` to a known question index, or ``None``."""

    if isinstance(value, bool):
        return None
    try:
        candidate = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return candidate if candidate in indices else None


def _to_bool(value: object) -> bool:
    """Coerce a JSON value to a strict boolean."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


__all__ = [
    "PaperValidationAgent",
    "QuestionReview",
    "ValidationTarget",
]
