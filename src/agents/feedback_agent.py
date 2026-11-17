"""Per-student feedback agent (FR-008).

Writes the qualitative ``FEEDBACK`` row from the verified marks: an overall
comment, the strongest topics and the topics that need work. Every statement is
derived from the questions and marks actually awarded; when no LLM is configured
a deterministic writer produces the same grounded feedback offline, so every
graded sheet still leaves feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.client import llm
from src.prompt import feedback as prompt
from src.repositories.observability_repository import default_usage_recorder
from src.settings import FEEDBACK_TEMPERATURE
from src.utils.exceptions import AppError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_AGENT_NAME = "student_feedback"


@dataclass(slots=True)
class FeedbackItem:
    """One marked question as the feedback writer sees it."""

    question_text: str
    topic: str | None
    max_marks: float
    awarded_marks: float
    is_answered: bool = True


@dataclass(slots=True)
class FeedbackTarget:
    """Everything the feedback agent may use: the marked questions and totals."""

    total_awarded_marks: float
    total_max_marks: float
    student_name: str | None = None
    subject: str | None = None
    items: list[FeedbackItem] = field(default_factory=list)


@dataclass(slots=True)
class FeedbackOutcome:
    """The feedback text ready to be persisted."""

    overall_comments: str
    strengths: str
    improvements: str


class FeedbackAgent:
    """Writes grounded per-student feedback from the awarded marks."""

    async def compose(self, *, target: FeedbackTarget) -> FeedbackOutcome:
        """Return feedback for one student's marked sheet.

        The deterministic writer runs first; it is returned when no LLM is
        configured or the call fails, and fills any field the model leaves blank.
        """

        fallback = _offline_outcome(target)
        if not llm.is_available():
            logger.warning(
                "No LLM key configured; using the offline feedback writer",
                extra={"task": _AGENT_NAME},
            )
            return fallback

        try:
            system, user = prompt.build_prompt(target=target)
            result = await llm.complete_json(
                system=system,
                user=user,
                task=_AGENT_NAME,
                temperature=FEEDBACK_TEMPERATURE,
                sink=default_usage_recorder(),
            )
            payload = llm.extract_json_object(result.text)
        except AppError as exc:
            logger.warning(
                "Feedback LLM call failed; using the offline writer",
                extra={"task": _AGENT_NAME, "error": exc.error_code},
            )
            return fallback

        logger.info(
            "Student feedback LLM call",
            extra={
                "task": _AGENT_NAME,
                "model": result.model,
                "total_tokens": result.total_tokens,
                "latency_ms": result.latency_ms,
            },
        )
        return _merge(payload, fallback)


def _offline_outcome(target: FeedbackTarget) -> FeedbackOutcome:
    """Write grounded feedback deterministically from the awarded marks."""

    name = target.student_name or "The student"
    percentage = _percentage(target)
    topics = _topic_scores(target)
    unanswered = sum(1 for item in target.items if not item.is_answered)

    overall = (
        f"{name} scored {_marks(target.total_awarded_marks)} out of "
        f"{_marks(target.total_max_marks)} marks ({percentage:.0f}%)."
    )
    if unanswered:
        overall += f" {unanswered} question(s) were left unanswered."

    if topics:
        best_topic, (best_ratio, best_awarded, best_max) = max(
            topics.items(), key=lambda entry: (entry[1][0], entry[1][1] - entry[1][0])
        )
        worst_topic, (worst_ratio, worst_awarded, worst_max) = min(
            topics.items(), key=lambda entry: (entry[1][0], -entry[1][1])
        )
        strengths = (
            f"Strongest topic: {best_topic}, scoring {_marks(best_awarded)} of "
            f"{_marks(best_max)} marks ({best_ratio:.0%})."
        )
        improvements = (
            f"Most marks were lost in {worst_topic}, scoring {_marks(worst_awarded)} of "
            f"{_marks(worst_max)} marks ({worst_ratio:.0%})."
        )
        if unanswered:
            improvements = (
                f"Answer every question first; {unanswered} were left blank. "
                + improvements
            )
    else:
        strengths = "No topic could be scored because no marks were recorded."
        improvements = "Review the unanswered questions and the rubric for each one."

    return FeedbackOutcome(
        overall_comments=overall, strengths=strengths, improvements=improvements
    )


def _merge(payload: dict[str, object], fallback: FeedbackOutcome) -> FeedbackOutcome:
    """Take the model's text, filling any blank field from the deterministic writer."""

    return FeedbackOutcome(
        overall_comments=_text(payload.get("overall_comments")) or fallback.overall_comments,
        strengths=_text(payload.get("strengths")) or fallback.strengths,
        improvements=_text(payload.get("improvements")) or fallback.improvements,
    )


def _topic_scores(target: FeedbackTarget) -> dict[str, tuple[float, float, float]]:
    """Aggregate ``(ratio, awarded, max)`` per topic across the marked questions."""

    totals: dict[str, list[float]] = {}
    for item in target.items:
        topic = (item.topic or "general").strip() or "general"
        entry = totals.setdefault(topic, [0.0, 0.0])
        entry[0] += item.awarded_marks
        entry[1] += item.max_marks
    return {
        topic: (
            (awarded / maximum) if maximum > 0 else 0.0,
            awarded,
            maximum,
        )
        for topic, (awarded, maximum) in totals.items()
    }


def _percentage(target: FeedbackTarget) -> float:
    """Return the awarded percentage, or zero when there is no maximum."""

    if target.total_max_marks <= 0:
        return 0.0
    return target.total_awarded_marks / target.total_max_marks * 100


def _text(value: object) -> str:
    """Return a stripped string from a JSON value, or an empty string."""

    return str(value).strip() if value is not None else ""


def _marks(value: float) -> str:
    """Format a mark value without a trailing ``.0``."""

    return f"{value:g}"


__all__ = [
    "FeedbackAgent",
    "FeedbackItem",
    "FeedbackOutcome",
    "FeedbackTarget",
]
