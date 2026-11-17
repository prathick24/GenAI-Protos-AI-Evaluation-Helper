"""Prompt contract for the question paper validation agent (FR-003).

Follows the project's prompt-engineering contract: Role, Mission, Context,
Inputs, Constraints, Critical Rules, Output Contract, Error Handling. Only the
*advisory* checks (ambiguity and phrasing-level duplication) are delegated to
the LLM; the authoritative checks are deterministic and never call a model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.paper_validation_agent import ValidationTarget


def build_prompt(
    *, subject: str | None, questions: list[ValidationTarget]
) -> tuple[str, str]:
    """Return the ``(system, user)`` prompts for one validation review call."""

    topic = subject or "the subject"
    system = (
        f"You are a meticulous {topic} exam moderator.\n"
        "You review a set of exam questions for ambiguous wording and for "
        "questions that ask essentially the same thing.\n"
        "You always reply with a single valid JSON object and nothing else."
    )
    return system, _user_prompt(questions)


def _user_prompt(questions: list[ValidationTarget]) -> str:
    """Build the user prompt: mission, the questions and the contract."""

    return "\n".join(
        [
            "MISSION",
            f"Review the {len(questions)} exam question(s) below and report, per question, "
            "whether its wording is ambiguous or unanswerable as written, and whether it is a "
            "near-duplicate of an earlier question.",
            "",
            "QUESTIONS",
            _questions_block(questions),
            "",
            "CONSTRAINTS",
            "- Judge only the wording shown; do not rewrite the questions.",
            "- Multiple-choice questions include their options; judge the question together "
            "with those options and assume they are part of the question.",
            "- A question is ambiguous when a student could reasonably interpret it in more "
            "than one way, or when it cannot be answered from standard subject knowledge.",
            "- Report a duplicate only when a question asks essentially the same thing as an "
            "earlier question in this list; cite that earlier question's index.",
            "",
            "CRITICAL RULES",
            f"- Return exactly {len(questions)} reviews, one for each question index.",
            "- Use the exact index shown for each question.",
            "- duplicate_of must be the index of an earlier question, or null.",
            "",
            "OUTPUT CONTRACT",
            'Return JSON: {"reviews": [{"index": number, "ambiguous": boolean, '
            '"duplicate_of": number | null, "notes": string}]}',
            "",
            "ERROR HANDLING",
            "If you are unsure, prefer reporting the question as clear (ambiguous = false); "
            "only flag genuine problems.",
        ]
    )


def _questions_block(questions: list[ValidationTarget]) -> str:
    """Render each question (and any options) as an indexed entry."""

    lines = []
    for target in questions:
        topic = f" [{target.topic}]" if target.topic else ""
        lines.append(
            f"[{target.index}] ({target.question_type}, {target.marks} mark(s)){topic} "
            f"{target.question_text}"
        )
        if target.options:
            rendered = " | ".join(
                f"{chr(65 + position)}) {option}"
                for position, option in enumerate(target.options)
            )
            lines.append(f"    options: {rendered}")
    return "\n".join(lines)
