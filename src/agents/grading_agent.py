"""Rubric-based grading agent (FR-006).

The agent marks one student answer against its question's rubric. It scores each
criterion, sums the scores into an awarded mark (never above the question's
maximum) and quotes the exact span of the student's answer that justifies the
mark. When no LLM is configured it falls back to a deterministic heuristic so
the grading pipeline still produces grounded marks offline.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field

from src.client import llm
from src.prompt import grading as prompt
from src.repositories.observability_repository import default_usage_recorder
from src.settings import GRADING_MARKS_PRECISION, SHEET_GRADING_TEMPERATURE
from src.utils.exceptions import AppError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_AGENT_NAME = "sheet_grading"
_NO_CONTENT = "No relevant content in the student's answer."
_DEFAULT_CRITERION = "Award marks for an answer consistent with the model answer."
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(slots=True)
class RubricTarget:
    """One scoring criterion the answer is marked against."""

    criterion_id: uuid.UUID | None
    criterion_text: str
    max_marks: float


@dataclass(slots=True)
class GradingTarget:
    """Everything the agent needs to mark one answer."""

    question_text: str
    question_type: str
    model_answer: str | None
    max_marks: float
    answer_text: str | None
    is_answered: bool
    rubrics: list[RubricTarget] = field(default_factory=list)


@dataclass(slots=True)
class CriterionScore:
    """The marks the agent awarded for one rubric criterion."""

    criterion_id: uuid.UUID | None
    criterion_text: str
    max_marks: float
    awarded_marks: float


@dataclass(slots=True)
class GradingOutcome:
    """A validated mark for one answer, ready to be persisted."""

    awarded_marks: float
    max_marks: float
    rubric_scores: list[CriterionScore]
    evidence: str
    reasoning: str
    confidence: float


class GradingAgent:
    """Marks answers against their rubric and cites the evidence."""

    async def grade(self, *, subject: str | None, target: GradingTarget) -> GradingOutcome:
        """Return the mark for one answer.

        A blank answer is awarded 0 without calling any model; an unconfigured or
        failing LLM falls back to the deterministic heuristic so a sheet can
        always be graded (and re-graded) offline.
        """

        if not target.is_answered or not (target.answer_text or "").strip():
            return _zero_outcome(target)

        if target.question_type.lower() == "mcq" and _matches_mcq(
            target.answer_text, target.model_answer
        ):
            return _mcq_outcome(target)

        if not llm.is_available():
            logger.warning(
                "No LLM key configured; using the offline grading heuristic",
                extra={"task": _AGENT_NAME},
            )
            return _offline_outcome(target)

        try:
            system, user = prompt.build_prompt(subject=subject, target=target)
            result = await llm.complete_json(
                system=system,
                user=user,
                task=_AGENT_NAME,
                temperature=SHEET_GRADING_TEMPERATURE,
                sink=default_usage_recorder(),
            )
            payload = llm.extract_json_object(result.text)
        except AppError as exc:
            logger.warning(
                "Grading LLM call failed; using the offline heuristic",
                extra={"task": _AGENT_NAME, "error": exc.error_code},
            )
            return _offline_outcome(target)

        logger.info(
            "Sheet grading LLM call",
            extra={
                "task": _AGENT_NAME,
                "model": result.model,
                "total_tokens": result.total_tokens,
                "latency_ms": result.latency_ms,
            },
        )
        return _parse_outcome(payload, target)


def _rubrics(target: GradingTarget) -> list[RubricTarget]:
    """Return the target's rubric, defaulting to one full-marks criterion."""

    if target.rubrics:
        return target.rubrics
    return [
        RubricTarget(
            criterion_id=None,
            criterion_text=_DEFAULT_CRITERION,
            max_marks=target.max_marks,
        )
    ]


def _matches_mcq(answer: str | None, model_answer: str | None) -> bool:
    """Return whether an MCQ answer matches the correct option leniently.

    Handwriting OCR introduces spacing, punctuation and single-character slips
    (``Super computer``, ``M.AY`` for MAX, ``Randomware`` for Ransomware), so the
    comparison ignores case/spacing and allows one edit for short options.
    """

    if not answer or not model_answer:
        return False
    answer_norm = _normalise_option(answer)
    model_norm = _normalise_option(model_answer)
    if len(answer_norm) < 3 or not model_norm:
        return False
    if answer_norm in model_norm:
        return True
    if len(answer_norm) > 40:
        return False
    return any(
        len(token) >= 3 and _edit_distance(answer_norm, token) <= 1
        for token in _TOKEN_RE.findall(model_answer.lower())
    )


def _normalise_option(text: str) -> str:
    """Lowercase an option and drop all non-alphanumeric characters."""

    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _edit_distance(left: str, right: str) -> int:
    """Return the Levenshtein distance between two short strings."""

    previous = list(range(len(right) + 1))
    for row, char in enumerate(left, start=1):
        current = [row]
        for column, other in enumerate(right, start=1):
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (char != other),
                )
            )
        previous = current
    return previous[-1]


def _mcq_outcome(target: GradingTarget) -> GradingOutcome:
    """Build the full-marks outcome for an MCQ that matches its option."""

    rubrics = _rubrics(target)
    awards = _split_marks(target.max_marks, [rubric.max_marks for rubric in rubrics])
    awards = [
        min(award, rubric.max_marks) for award, rubric in zip(awards, rubrics, strict=True)
    ]
    return GradingOutcome(
        awarded_marks=_round(sum(awards)),
        max_marks=target.max_marks,
        rubric_scores=[
            CriterionScore(rubric.criterion_id, rubric.criterion_text, rubric.max_marks, award)
            for rubric, award in zip(rubrics, awards, strict=True)
        ],
        evidence=(target.answer_text or "").strip(),
        reasoning=(
            "The answer matches the correct option after normalising case, spacing "
            "and minor OCR errors."
        ),
        confidence=1.0,
    )


def _zero_outcome(target: GradingTarget) -> GradingOutcome:
    """Build the explicit 0-mark outcome for a blank/absent answer."""

    rubrics = _rubrics(target)
    return GradingOutcome(
        awarded_marks=0.0,
        max_marks=target.max_marks,
        rubric_scores=[
            CriterionScore(rubric.criterion_id, rubric.criterion_text, rubric.max_marks, 0.0)
            for rubric in rubrics
        ],
        evidence=_NO_CONTENT,
        reasoning="The student left this question unanswered, so no marks can be awarded.",
        confidence=1.0,
    )


def _offline_outcome(target: GradingTarget) -> GradingOutcome:
    """Build a deterministic mark from the answer's overlap with the model answer."""

    rubrics = _rubrics(target)
    answer = target.answer_text or ""
    coverage = _coverage(answer, target.model_answer)
    awards = _split_marks(target.max_marks * coverage, [rubric.max_marks for rubric in rubrics])
    awards = [
        min(award, rubric.max_marks) for award, rubric in zip(awards, rubrics, strict=True)
    ]
    return GradingOutcome(
        awarded_marks=_round(sum(awards)),
        max_marks=target.max_marks,
        rubric_scores=[
            CriterionScore(rubric.criterion_id, rubric.criterion_text, rubric.max_marks, award)
            for rubric, award in zip(rubrics, awards, strict=True)
        ],
        evidence=_best_evidence(answer, target.model_answer) or _NO_CONTENT,
        reasoning=(
            f"Offline heuristic: about {coverage:.0%} of the model answer's key terms "
            "appear in the student's answer."
        ),
        confidence=0.5,
    )


def _parse_outcome(payload: dict[str, object], target: GradingTarget) -> GradingOutcome:
    """Validate the model's mark, clamping it to the rubric and the question cap.

    The awarded mark can never exceed the question's maximum, and every criterion
    is clamped to its own maximum so ``rubric_scores`` always sum to
    ``awarded_marks``. Evidence that is not a verbatim span of the answer is
    replaced with the closest verbatim span, so no invented quote is ever stored.
    """

    rubrics = _rubrics(target)
    raw = payload.get("rubric_scores")
    scores: list[float] = []
    for index, rubric in enumerate(rubrics):
        item = raw[index] if isinstance(raw, list) and index < len(raw) else None
        value = _to_float(item.get("awarded_marks")) if isinstance(item, dict) else 0.0
        scores.append(_clamp(value, 0.0, rubric.max_marks))

    if sum(scores) > target.max_marks:
        scores = [
            min(share, rubric.max_marks)
            for share, rubric in zip(
                _split_marks(target.max_marks, scores), rubrics, strict=True
            )
        ]

    answer = target.answer_text or ""
    evidence = str(payload.get("evidence") or "").strip()
    if not _is_verbatim(evidence, answer):
        evidence = _best_evidence(answer, target.model_answer) or _NO_CONTENT

    return GradingOutcome(
        awarded_marks=_round(sum(scores)),
        max_marks=target.max_marks,
        rubric_scores=[
            CriterionScore(
                rubric.criterion_id, rubric.criterion_text, rubric.max_marks, _round(score)
            )
            for rubric, score in zip(rubrics, scores, strict=True)
        ],
        evidence=evidence,
        reasoning=(
            str(payload.get("reasoning") or "").strip()
            or "Marks awarded against the question's rubric."
        ),
        confidence=_clamp(_to_float(payload.get("confidence"), 0.5), 0.0, 1.0),
    )


def _split_marks(total: float, weights: list[float]) -> list[float]:
    """Split ``total`` across ``weights`` using the largest-remainder method."""

    weights = [max(weight, 0.0) for weight in weights]
    if not weights:
        return []
    scale = 10**GRADING_MARKS_PRECISION
    units = int(round(total * scale))
    weight_total = sum(weights)
    if units <= 0 or weight_total <= 0:
        return [0.0 for _ in weights]

    raw = [units * weight / weight_total for weight in weights]
    floors = [int(math.floor(value)) for value in raw]
    remainder = units - sum(floors)
    ranked = sorted(range(len(weights)), key=lambda index: raw[index] - floors[index], reverse=True)
    for index in ranked[:remainder]:
        floors[index] += 1
    return [unit / scale for unit in floors]


def _coverage(answer: str, model_answer: str | None) -> float:
    """Return the fraction of the model answer's terms present in ``answer``."""

    answer_tokens = _tokens(answer)
    model_tokens = _tokens(model_answer or "")
    if not model_tokens:
        return 1.0 if answer_tokens else 0.0
    return min(len(answer_tokens & model_tokens) / len(model_tokens), 1.0)


def _best_evidence(answer: str, model_answer: str | None) -> str | None:
    """Return the answer span that best supports the mark (always verbatim)."""

    sentences = _sentences(answer)
    if not sentences:
        return answer.strip() or None
    if model_answer:
        model_tokens = _tokens(model_answer)
        return max(sentences, key=lambda sentence: len(_tokens(sentence) & model_tokens))
    return max(sentences, key=len)


def _sentences(text: str) -> list[str]:
    """Split ``text`` into trimmed spans that each remain a substring of it."""

    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _is_verbatim(candidate: str, answer: str) -> bool:
    """Return whether ``candidate`` is a non-empty verbatim span of ``answer``."""

    return bool(candidate) and candidate in answer


def _tokens(text: str) -> set[str]:
    """Return the lowercased word set of ``text``."""

    return set(_TOKEN_RE.findall(text.lower()))


def _to_float(value: object, default: float = 0.0) -> float:
    """Coerce a JSON value to a float, falling back when unusable."""

    if isinstance(value, bool):
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""

    return min(max(value, low), high)


def _round(value: float) -> float:
    """Round a mark to the configured precision."""

    return round(value, GRADING_MARKS_PRECISION)


__all__ = [
    "CriterionScore",
    "GradingAgent",
    "GradingOutcome",
    "GradingTarget",
    "RubricTarget",
]
