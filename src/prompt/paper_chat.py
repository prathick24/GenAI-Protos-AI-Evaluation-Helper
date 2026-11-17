"""Prompt contract for the paper chat router (FR-010).

The router is the only LLM step that turns a teacher's free-text message into a
structured :class:`~src.models.chat.ChangePlan`. It never authors the questions
itself — wording changes are delegated to the generation agent, and the
executors apply the plan deterministically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.repositories.schema import QuestionPaper


def build_prompt(*, paper: QuestionPaper, message: str) -> tuple[str, str]:
    """Return the ``(system, user)`` prompts for one chat-routing call."""

    subject = paper.subject or "the subject"
    system = (
        f"You are the assistant inside a {subject} question-paper editor.\n"
        "You translate a teacher's instruction into a small, ordered list of "
        "structured operations that change the paper.\n"
        "You answer follow-up questions about the paper using only the paper shown.\n"
        "You always reply with a single valid JSON object and nothing else."
    )
    return system, _user_prompt(paper, message)


def _user_prompt(paper: QuestionPaper, message: str) -> str:
    """Build the user prompt: paper context, the instruction and the contract."""

    return "\n".join(
        [
            "MISSION",
            "Read the teacher's message and reply with a short confirmation plus the "
            "operations that carry it out. Use an empty operations list when the teacher "
            "is only asking a question.",
            "",
            "PAPER",
            _paper_block(paper),
            "",
            "EXISTING QUESTIONS",
            _questions_block(paper),
            "",
            "TEACHER MESSAGE",
            message.strip(),
            "",
            "AVAILABLE OPERATIONS",
            '- {"op": "rewrite", "order_indices": [number], "instructions": string} — '
            "rewrite the wording of the listed questions; the generator agent does the writing.",
            '- {"op": "drop", "order_indices": [number]} — remove the listed questions.',
            '- {"op": "reorder", "order": [number]} — set a new order, listing every current '
            "question position exactly once.",
            '- {"op": "set_duration", "duration_minutes": number} — change the paper duration.',
            '- {"op": "adjust_blueprint", "question_types": [{"type": "mcq"|"short"|"long", '
            '"count": number, "marks_each": number}], "difficulty_mix": {"easy": number, '
            '"medium": number, "hard": number} | null} — change the paper shape; the executor '
            "keeps matching questions and adds or removes the rest to fit.",
            "",
            "CONSTRAINTS",
            "- Use only the operations above; never invent a question's text yourself.",
            "- Question positions are the zero-based values shown in EXISTING QUESTIONS.",
            "- Leave a field out when it does not apply to the chosen operation.",
            "- The question_types of adjust_blueprint must sum to the intended total marks.",
            "",
            "CRITICAL RULES",
            "- Return one JSON object with a non-empty reply and an operations list.",
            "- Order the operations the way they should be applied.",
            "- If the teacher asks something you cannot do with these operations, explain that "
            "in the reply and return an empty operations list.",
            "",
            "OUTPUT CONTRACT",
            'Return JSON: {"reply": string, "operations": [{"op": string, '
            '"order_indices": [number], "order": [number], "duration_minutes": number, '
            '"question_types": [{"type": string, "count": number, "marks_each": number}], '
            '"difficulty_mix": {"easy": number, "medium": number, "hard": number} | null, '
            '"instructions": string}]}',
            "",
            "ERROR HANDLING",
            "If the message is ambiguous, ask a clarifying question in the reply and return an "
            "empty operations list.",
        ]
    )


def _paper_block(paper: QuestionPaper) -> str:
    """Render the paper's identifying fields and blueprint."""

    blueprint = paper.blueprint or {}
    types = blueprint.get("question_types") or []
    mix = blueprint.get("difficulty_mix")
    return "\n".join(
        [
            f"title: {paper.title}",
            f"subject: {paper.subject or 'unknown'}",
            f"duration_minutes: {paper.duration_minutes}",
            f"total_marks: {paper.total_marks}",
            "question_types: "
            + (
                ", ".join(
                    f"{spec.get('type')} x{spec.get('count')} @ {spec.get('marks_each')} marks"
                    for spec in types
                )
                or "unspecified"
            ),
            f"difficulty_mix: {mix if mix is not None else 'unspecified'}",
        ]
    )


def _questions_block(paper: QuestionPaper) -> str:
    """Render each question as a single indexed line."""

    questions = sorted(paper.questions, key=lambda item: item.order_index)
    if not questions:
        return "(none)"
    return "\n".join(
        f"[{question.order_index}] ({question.question_type}, {int(question.marks)} mark(s), "
        f"{question.difficulty or 'medium'}) {question.question_text}"
        for question in questions
    )
