"""Prompt contract for the independent mark verification agent (FR-007).

Follows the project's prompt-engineering contract: Role, Mission, Context,
Inputs, Constraints, Critical Rules, Output Contract, Error Handling. The
verifier is deliberately shown the question, model answer, rubric, the student's
answer, the evidence span and the *awarded mark* — but never the grader's
reasoning chain — so its verdict is reached independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.verification_agent import VerificationTarget


def build_prompt(*, subject: str | None, target: VerificationTarget) -> tuple[str, str]:
    """Return the ``(system, user)`` prompts for one verification call."""

    topic = subject or "the subject"
    system = (
        f"You are an independent {topic} moderator double-checking another "
        "examiner's mark.\n"
        "You re-derive the fair mark from the rubric and the student's answer "
        "alone, then say whether you agree with the mark you were shown.\n"
        "You always reply with a single valid JSON object and nothing else."
    )
    return system, _user_prompt(target)


def _user_prompt(target: VerificationTarget) -> str:
    """Build the user prompt: mission, inputs and the output contract."""

    return "\n".join(
        [
            "MISSION",
            "Independently re-check the awarded mark for the answer below and report whether "
            "you agree with it, or would adjust it.",
            "",
            "QUESTION",
            target.question_text,
            "",
            "MODEL ANSWER",
            target.model_answer or "(none provided)",
            "",
            "RUBRIC",
            _rubric_block(target),
            "",
            "STUDENT ANSWER",
            (target.answer_text or "(blank)"),
            "",
            "AWARDED MARK",
            f"{_marks(target.awarded_marks)} of {_marks(target.max_marks)} marks",
            "",
            "EVIDENCE THE EXAMINER CITED",
            target.evidence or "(none)",
            "",
            "CONSTRAINTS",
            "- Use ONLY the rubric and the student's answer; do not add outside knowledge.",
            "- Reach your own view of the mark before comparing it with the awarded mark.",
            "- Do not rewrite the answer and do not explain another examiner's thinking.",
            "",
            "CRITICAL RULES",
            "- 'verdict' is 'agreed' when the awarded mark is fair, 'flagged' when it is "
            "clearly wrong but you cannot give an exact replacement, and 'adjusted' when you "
            "can state the mark it should have been.",
            "- 'suggested_marks' is required for 'adjusted' and must be between 0 and the "
            "question's maximum; set it to null otherwise.",
            "- 'notes' is one or two sentences justifying your verdict.",
            "",
            "OUTPUT CONTRACT",
            'Return JSON: {"verdict": "agreed" | "flagged" | "adjusted", '
            '"suggested_marks": number | null, "notes": string}',
            "",
            "ERROR HANDLING",
            "If you are unsure, return 'agreed' with a note recording the doubt rather than "
            "raising a false alarm.",
        ]
    )


def _rubric_block(target: VerificationTarget) -> str:
    """Render each criterion with the mark the examiner awarded it."""

    lines = []
    for index, score in enumerate(target.rubric_scores):
        lines.append(
            f"[{index + 1}] {score.criterion_text} "
            f"(awarded {_marks(score.awarded_marks)} of {_marks(score.max_marks)})"
        )
    return "\n".join(lines) or "(no rubric criteria)"


def _marks(value: float) -> str:
    """Format a mark value without a trailing ``.0``."""

    return f"{value:g}"
