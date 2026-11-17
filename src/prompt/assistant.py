"""Prompt contract for the grounded Q&A assistant (FR-010).

The assistant answers a teacher's question using only the numbered context
documents it is given (syllabus chunks and paper questions). It must cite the
documents it used and refuse to guess when the context does not cover the
question.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.assistant_agent import ContextDocument

_SCOPE_INSTRUCTION = (
    "If the CONTEXT does not contain enough information to answer, reply with a "
    "short sentence saying the information could not be found in the uploaded "
    "material and name the topics the context does cover. Return an empty "
    "citations list in that case."
)


def build_prompt(*, question: str, documents: list[ContextDocument]) -> tuple[str, str]:
    """Return the ``(system, user)`` prompts for one grounded answer."""

    system = (
        "You are the assistant inside an AI evaluation helper for teachers.\n"
        "You answer questions about syllabi and question papers using ONLY the "
        "CONTEXT provided.\n"
        "Never use outside knowledge and never invent facts.\n"
        "Cite every claim with the bracketed number of the document it came from, "
        "for example [1].\n"
        f"{_SCOPE_INSTRUCTION}\n"
        "You always reply with a single valid JSON object and nothing else."
    )
    return system, _user_prompt(question, documents)


def _user_prompt(question: str, documents: list[ContextDocument]) -> str:
    """Build the user prompt: numbered context, the question and the contract."""

    return "\n".join(
        [
            "MISSION",
            "Answer the teacher's question from the numbered CONTEXT below and cite "
            "the documents you used.",
            "",
            "CONTEXT",
            _context_block(documents),
            "",
            "QUESTION",
            question.strip(),
            "",
            "OUTPUT",
            "Reply with JSON of the form:",
            '{"answer": "<grounded answer with [n] citations>", "citations": [<numbers>]}',
            "Use only document numbers that appear in CONTEXT.",
        ]
    )


def _context_block(documents: list[ContextDocument]) -> str:
    """Render the numbered context documents."""

    if not documents:
        return "(no documents available)"

    lines: list[str] = []
    for index, document in enumerate(documents, start=1):
        labels = [document.source_type]
        if document.unit_no:
            labels.append(f"unit {document.unit_no}")
        if document.topic_name:
            labels.append(document.topic_name)
        lines.append(f"[{index}] ({', '.join(labels)}) {document.snippet}")
    return "\n".join(lines)
