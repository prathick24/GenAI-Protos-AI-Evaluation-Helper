"""Independent mark verification agent (FR-007).

A second, independent agent re-checks every mark before the teacher sees it. It
is handed the question, the model answer, the rubric, the student's answer, the
evidence span and the awarded mark — but not the grader's reasoning — and returns
one verdict: ``agreed``, ``flagged`` or ``adjusted``, with notes.

A disagreement is never applied silently: the awarded mark is left untouched and
the verdict is surfaced on the mark for the teacher to review. When no LLM is
configured a deterministic consistency check stands in, so every mark still
carries a verdict offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.agents.grading_agent import CriterionScore
from src.client import llm
from src.prompt import verification as prompt
from src.repositories.observability_repository import default_usage_recorder
from src.settings import GRADING_MARKS_PRECISION, VerifierStatus
from src.utils.exceptions import AppError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_AGENT_NAME = "mark_verification"
# Marks are compared at the stored precision so float noise never flags a mark.
_TOLERANCE = 10 ** (-GRADING_MARKS_PRECISION) / 2
_AGREED_BLANK = "The question was left unanswered, so the zero mark is correct."


@dataclass(slots=True)
class VerificationTarget:
    """Everything the verifier may see about one awarded mark."""

    question_text: str
    question_type: str
    model_answer: str | None
    max_marks: float
    answer_text: str | None
    is_answered: bool
    awarded_marks: float
    evidence: str | None = None
    rubric_scores: list[CriterionScore] = field(default_factory=list)


@dataclass(slots=True)
class VerificationOutcome:
    """The verifier's verdict for one mark, ready to be persisted."""

    status: str
    notes: str
    suggested_marks: float | None = None


class MarkVerificationAgent:
    """Independently re-checks the awarded mark for one answer."""

    async def verify(
        self, *, subject: str | None, target: VerificationTarget
    ) -> VerificationOutcome:
        """Return an agreement verdict for one mark.

        A blank answer is trivially worth zero, so it is accepted without a
        model call. Otherwise the configured LLM is asked for an independent
        view, falling back to the deterministic consistency check when no LLM is
        available or the call fails.
        """

        if not target.is_answered or not (target.answer_text or "").strip():
            return VerificationOutcome(status=VerifierStatus.AGREED.value, notes=_AGREED_BLANK)

        if not llm.is_available():
            logger.warning(
                "No LLM key configured; using the offline verification check",
                extra={"task": _AGENT_NAME},
            )
            return _offline_outcome(target)

        try:
            system, user = prompt.build_prompt(subject=subject, target=target)
            result = await llm.complete_json(
                system=system,
                user=user,
                task=_AGENT_NAME,
                temperature=0.0,
                sink=default_usage_recorder(),
            )
            payload = llm.extract_json_object(result.text)
        except AppError as exc:
            logger.warning(
                "Verification LLM call failed; using the offline check",
                extra={"task": _AGENT_NAME, "error": exc.error_code},
            )
            return _offline_outcome(target)

        logger.info(
            "Mark verification LLM call",
            extra={
                "task": _AGENT_NAME,
                "model": result.model,
                "total_tokens": result.total_tokens,
                "latency_ms": result.latency_ms,
            },
        )
        return _parse_outcome(payload, target)


def _offline_outcome(target: VerificationTarget) -> VerificationOutcome:
    """Derive a verdict from the mark's own internal consistency."""

    awarded = target.awarded_marks
    rubric_total = sum(score.awarded_marks for score in target.rubric_scores)
    if awarded < 0 or awarded > target.max_marks + _TOLERANCE:
        return VerificationOutcome(
            status=VerifierStatus.FLAGGED.value,
            notes=(
                f"Offline verifier: the awarded mark {awarded:g} is outside the "
                f"question's range of 0-{target.max_marks:g}."
            ),
        )
    if target.rubric_scores and abs(rubric_total - awarded) > _TOLERANCE:
        return VerificationOutcome(
            status=VerifierStatus.FLAGGED.value,
            notes=(
                f"Offline verifier: the awarded mark {awarded:g} does not match the rubric "
                f"scores, which total {rubric_total:g}."
            ),
            suggested_marks=round(rubric_total, GRADING_MARKS_PRECISION),
        )
    if target.evidence and target.evidence not in (target.answer_text or ""):
        return VerificationOutcome(
            status=VerifierStatus.FLAGGED.value,
            notes=(
                "Offline verifier: the cited evidence does not appear verbatim in the "
                "student's answer."
            ),
        )
    return VerificationOutcome(
        status=VerifierStatus.AGREED.value,
        notes="Offline verifier: the mark is internally consistent with the rubric and evidence.",
    )


def _parse_outcome(payload: dict[str, object], target: VerificationTarget) -> VerificationOutcome:
    """Validate the model's verdict, clamping any suggested mark to the range."""

    status = _to_status(payload.get("verdict"))
    suggested = _to_float(payload.get("suggested_marks"))
    if status is VerifierStatus.ADJUSTED and suggested is None:
        # An "adjusted" verdict without a replacement mark is really a flag.
        status = VerifierStatus.FLAGGED
    if suggested is not None:
        suggested = round(min(max(suggested, 0.0), target.max_marks), GRADING_MARKS_PRECISION)

    notes = str(payload.get("notes") or "").strip()
    if not notes:
        notes = _default_note(status, suggested)
    elif status is VerifierStatus.ADJUSTED and suggested is not None:
        notes = f"{notes} (suggested mark: {suggested:g})"

    return VerificationOutcome(status=status.value, notes=notes, suggested_marks=suggested)


def _default_note(status: VerifierStatus, suggested: float | None) -> str:
    """Build a fallback note when the model returned none."""

    if status is VerifierStatus.AGREED:
        return "The verifier independently reached the same mark."
    if status is VerifierStatus.ADJUSTED and suggested is not None:
        return f"The verifier would award {suggested:g} instead."
    return "The verifier disputes the mark; please review."


def _to_status(value: object) -> VerifierStatus:
    """Coerce a JSON value to a known verdict, defaulting to ``agreed``."""

    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in set(VerifierStatus):
            return VerifierStatus(candidate)
    return VerifierStatus.AGREED


def _to_float(value: object) -> float | None:
    """Coerce a JSON value to a float, or ``None`` when unusable."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


__all__ = ["MarkVerificationAgent", "VerificationOutcome", "VerificationTarget"]
