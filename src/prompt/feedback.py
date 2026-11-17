"""Prompt contract for the per-student feedback agent (FR-008).

Follows the project's prompt-engineering contract: Role, Mission, Context,
Inputs, Constraints, Critical Rules, Output Contract, Error Handling. The agent
sees only the student's per-question marks and totals, so every statement it
writes is grounded in the marks actually awarded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.feedback_agent import FeedbackTarget


def build_prompt(*, target: FeedbackTarget) -> tuple[str, str]:
    """Return the ``(system, user)`` prompts for one feedback call."""

    topic = target.subject or "the subject"
    system = (
        f"You are a supportive but honest {topic} teacher writing feedback for one "
        "student.\n"
        "You comment only on the marks the student actually earned: name the topics "
        "they did well on and the ones that need work.\n"
        "You never invent questions, topics or marks that are not in the input.\n"
        "You always reply with a single valid JSON object and nothing else."
    )
    return system, _user_prompt(target)


def _user_prompt(target: FeedbackTarget) -> str:
    """Build the user prompt: mission, the marked questions and the contract."""

    name = target.student_name or "The student"
    return "\n".join(
        [
            "MISSION",
            f"Write per-student feedback for {name} based on the marked questions below.",
            "",
            "RESULT",
            f"{_marks(target.total_awarded_marks)} of {_marks(target.total_max_marks)} marks "
            f"awarded ({_percentage(target):.0f}%).",
            "",
            "MARKED QUESTIONS",
            _item_block(target),
            "",
            "CONSTRAINTS",
            "- Use ONLY the questions, topics and marks shown above.",
            "- If the student left a question unanswered, say so plainly.",
            "- Do not add new questions, topics or predicted grades.",
            "",
            "CRITICAL RULES",
            "- 'strengths' names the topic(s) where the student scored best, with the mark.",
            "- 'improvements' names the topic(s) where the student lost the most marks and "
            "what to practise next.",
            "- 'overall_comments' is two or three sentences that reference the actual result.",
            "- Never use generic filler such as 'keep up the good work'.",
            "",
            "OUTPUT CONTRACT",
            'Return JSON: {"overall_comments": string, "strengths": string, '
            '"improvements": string}',
            "",
            "ERROR HANDLING",
            "If the marks are too few to identify a pattern, say what the marks show and "
            "recommend reviewing the unanswered questions.",
        ]
    )


def _item_block(target: FeedbackTarget) -> str:
    """Render each marked question as a single line."""

    lines = []
    for index, item in enumerate(target.items):
        topic = item.topic or "general"
        status = "" if item.is_answered else " (unanswered)"
        lines.append(
            f"[{index + 1}] {topic}: {_marks(item.awarded_marks)} of "
            f"{_marks(item.max_marks)} marks{status} — {item.question_text}"
        )
    return "\n".join(lines) or "(no marked questions)"


def _marks(value: float) -> str:
    """Format a mark value without a trailing ``.0``."""

    return f"{value:g}"


def _percentage(target: FeedbackTarget) -> float:
    """Return the awarded percentage, or zero when there is no maximum."""

    if target.total_max_marks <= 0:
        return 0.0
    return target.total_awarded_marks / target.total_max_marks * 100
