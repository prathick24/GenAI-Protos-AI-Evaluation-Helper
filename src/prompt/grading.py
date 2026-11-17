"""Prompt contract for the rubric-based grading agent (FR-006).

Follows the project's prompt-engineering contract: Role, Mission, Context,
Inputs, Constraints, Critical Rules, Output Contract, Error Handling. The agent
sees only the question, its rubric, the model answer and the student's own
answer; the evidence it returns must be copied verbatim from that answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.grading_agent import GradingTarget, RubricTarget


def build_prompt(*, subject: str | None, target: GradingTarget) -> tuple[str, str]:
    """Return the ``(system, user)`` prompts for one grading call."""

    topic = subject or "the subject"
    system = (
        f"You are a fair and exacting {topic} examiner.\n"
        "You mark a student's answer against a rubric, awarding each criterion "
        "only the marks the answer has actually earned.\n"
        "Every piece of evidence you quote is copied verbatim from the student's "
        "answer.\n"
        "You always reply with a single valid JSON object and nothing else."
    )
    return system, _user_prompt(target)


def _user_prompt(target: GradingTarget) -> str:
    """Build the user prompt: mission, inputs and the output contract."""

    return "\n".join(
        [
            "MISSION",
            f"Mark the student's answer to the question below against its rubric, for a "
            f"maximum of {_marks(target.max_marks)} marks.",
            "",
            "QUESTION",
            target.question_text,
            "",
            "MODEL ANSWER",
            target.model_answer or "(none provided)",
            "",
            "RUBRIC",
            _rubric_block(target.rubrics),
            "",
            "STUDENT ANSWER",
            (target.answer_text or "(blank)"),
            "",
            "CONSTRAINTS",
            "- Use ONLY the rubric and the student's answer; do not add outside knowledge.",
            "- Award each criterion between 0 and its maximum marks, in 0.5-mark steps.",
            "- The awarded marks must never exceed the question's maximum.",
            "- If the answer is blank or irrelevant, award 0 for every criterion and say so.",
            "",
            "CRITICAL RULES",
            "- 'evidence' MUST be a short span copied character-for-character from the "
            "student's answer; never paraphrase it, and never invent it.",
            "- If nothing in the answer is relevant, set evidence to "
            '"No relevant content in the student\'s answer."',
            "- 'reasoning' explains, in one or two sentences, why those marks were awarded.",
            "- 'confidence' is your certainty in this mark, between 0 and 1.",
            "",
            "OUTPUT CONTRACT",
            'Return JSON: {"rubric_scores": [{"criterion": string, '
            '"awarded_marks": number}], "evidence": string, "reasoning": string, '
            '"confidence": number}',
            f"Return exactly {len(target.rubrics)} entries in rubric_scores, in the order "
            "the criteria are listed above.",
            "",
            "ERROR HANDLING",
            "If you are unsure, award the lower mark rather than the higher one and explain "
            "your doubt in 'reasoning'.",
        ]
    )


def _rubric_block(rubrics: list[RubricTarget]) -> str:
    """Render each criterion as a single numbered line."""

    lines = []
    for index, rubric in enumerate(rubrics):
        lines.append(f"[{index + 1}] ({_marks(rubric.max_marks)} marks) {rubric.criterion_text}")
    return "\n".join(lines)


def _marks(value: float) -> str:
    """Format a mark value without a trailing ``.0``."""

    return f"{value:g}"
