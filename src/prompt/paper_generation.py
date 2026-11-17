"""Prompt contract for the question paper generation agent (FR-002).

Follows the project's prompt-engineering contract: Role, Mission, Context,
Inputs, Constraints, Rules, Critical Rules, Output Contract, Error Handling.
The context is always *verbatim* syllabus text — never a summary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.paper_generation_agent import GenerationRequest, RevisionRequest
    from src.repositories.schema import SyllabusChunk

# Question types accepted by the output contract.
_MCQ_OPTION_COUNT = 4


def build_prompt(*, request: GenerationRequest) -> tuple[str, str]:
    """Return the ``(system, user)`` prompts for one generation call."""

    subject = request.subject or "the subject"
    system = (
        f"You are an experienced {subject} examiner and assessment designer.\n"
        "You write fair exam questions using ONLY the supplied syllabus text.\n"
        "You always reply with a single valid JSON object and nothing else."
    )
    return system, _user_prompt(request)


def build_revision_prompt(*, request: RevisionRequest) -> tuple[str, str]:
    """Return the ``(system, user)`` prompts for one question-revision call."""

    subject = request.subject or "the subject"
    system = (
        f"You are an experienced {subject} examiner revising exam questions.\n"
        "You rewrite questions so they no longer fail validation, using ONLY the "
        "supplied syllabus text.\n"
        "You always reply with a single valid JSON object and nothing else."
    )
    return system, _revision_user_prompt(request)


def _user_prompt(request: GenerationRequest) -> str:
    """Build the user prompt: mission, verbatim context, inputs and contract."""

    return "\n".join(
        [
            "MISSION",
            f"Write exactly {len(request.specs)} exam question(s) "
            f"for {_target_label(request)}, grounded strictly in the syllabus text below.",
            "",
            _context_block(request),
            "",
            "REQUIRED QUESTIONS (produce one question per line, in this order)",
            _specs_block(request),
            "",
            "CONSTRAINTS",
            "- Use only facts stated in the syllabus text above; add no outside knowledge.",
            "- Ask only about subject-matter concepts described in the text; never ask about "
            "the syllabus document itself (its title, class, subject, unit or page structure).",
            "- Never ask what the syllabus lists, mentions or omits (for example 'Which of "
            "the following is NOT listed in the syllabus?'); test the concept directly.",
            "- Each question must be self-contained, unambiguous and answerable from the text.",
            f"- Multiple-choice questions need exactly {_MCQ_OPTION_COUNT} options and a "
            "correct_option that is one of those options.",
            "- Keep the wording natural for the requested question type and difficulty.",
            "",
            "CRITICAL RULES",
            f"- Return exactly {len(request.specs)} questions, in the requested order.",
            "- Every question needs a non-empty model_answer and at least one rubric criterion.",
            "- Each rubric criterion states what earns the marks; the criteria marks sum to the "
            "question's marks.",
            "- source_chunk_id must be one of the chunk ids listed in the context above.",
            "",
            "OUTPUT CONTRACT",
            'Return JSON: {"questions": [{"question_text": string, "options": [string] | null, '
            '"correct_option": string | null, "model_answer": string, "topic": string, '
            '"source_chunk_id": string, "rubrics": [{"criterion_text": string, '
            '"max_marks": number}]}]}',
            "",
            "ERROR HANDLING",
            "If the text is thin for a requested question, still produce the best possible "
            "question that is fully supported by the text.",
        ]
    )


def _target_label(request: GenerationRequest) -> str:
    """Describe the unit/topic the questions are being written for."""

    parts = []
    if request.unit_name:
        parts.append(f"Unit {request.unit_no} ({request.unit_name})" if request.unit_no
                     else request.unit_name)
    elif request.unit_no:
        parts.append(f"Unit {request.unit_no}")
    if request.topic_name:
        parts.append(f"topic '{request.topic_name}'")
    return " for ".join(parts) if parts else "this syllabus"


def _context_block(request: GenerationRequest) -> str:
    """Render the verbatim syllabus chunks with their ids."""

    return _render_context(request.chunks)


def _render_context(chunks: list[SyllabusChunk]) -> str:
    """Render verbatim syllabus chunks with their ids."""

    lines = ["SYLLABUS TEXT (verbatim — do not summarise)"]
    for chunk in chunks:
        lines.append(f"[chunk {chunk.id}]")
        lines.append(chunk.raw_text.strip())
    return "\n".join(lines)


def _revision_user_prompt(request: RevisionRequest) -> str:
    """Build the user prompt for rewriting flagged questions."""

    return "\n".join(
        [
            "MISSION",
            f"Rewrite exactly {len(request.targets)} question(s) that failed validation "
            f"for {_revision_target_label(request)}, so each one passes.",
            "",
            _render_context(request.chunks),
            "",
            *_existing_questions_block(request),
            "QUESTIONS TO REWRITE (one output question per item, in this order)",
            _revision_targets_block(request),
            "",
            "CONSTRAINTS",
            "- Use only facts stated in the syllabus text above; add no outside knowledge.",
            "- Keep each question's type, marks and difficulty exactly as listed above.",
            "- Directly resolve the listed issues; do not repeat the same wording or idea.",
            "- The rewrites replace the listed questions. Each one must be clearly distinct "
            "in both wording and idea from EVERY question above, including the questions "
            "being replaced and the other rewrites in this same reply.",
            "- Ask only about subject-matter concepts described in the text; never ask about "
            "the syllabus document itself (its title, class, subject, unit or page structure).",
            "- Never ask what the syllabus lists, mentions or omits (for example 'Which of "
            "the following is NOT listed in the syllabus?'); test the concept directly.",
            f"- Multiple-choice questions need exactly {_MCQ_OPTION_COUNT} options and a "
            "correct_option that is one of those options.",
            "",
            "CRITICAL RULES",
            f"- Return exactly {len(request.targets)} questions, in the requested order.",
            "- Every question needs a non-empty model_answer and at least one rubric criterion.",
            "- source_chunk_id must be one of the chunk ids listed in the context above.",
            "",
            "OUTPUT CONTRACT",
            'Return JSON: {"questions": [{"question_text": string, "options": [string] | null, '
            '"correct_option": string | null, "model_answer": string, "topic": string, '
            '"source_chunk_id": string, "rubrics": [{"criterion_text": string, '
            '"max_marks": number}]}]}',
            "",
            "ERROR HANDLING",
            "If the text is thin, still produce the best possible question fully supported by it.",
        ]
    )


def _existing_questions_block(request: RevisionRequest) -> list[str]:
    """Render the rest of the paper so rewrites can avoid duplicating it."""

    if not request.paper_questions:
        return []
    lines = ["WHOLE QUESTION PAPER (do not duplicate any of these)"]
    for index, question in enumerate(request.paper_questions, start=1):
        topic = f" - {question.topic}" if question.topic else ""
        lines.append(
            f"{index}. [{question.question_type}] {question.marks} marks{topic}\n"
            f"   {question.question_text}"
        )
    lines.append("")
    return lines


def _revision_targets_block(request: RevisionRequest) -> str:
    """Render the flagged questions and their issues as a numbered list."""

    lines = []
    for index, target in enumerate(request.targets, start=1):
        issues = "; ".join(target.issues) or "flagged by validation"
        lines.append(
            f"{index}. type={target.question_type} marks={target.marks} "
            f"difficulty={target.difficulty}\n"
            f"   current: {target.question_text}\n"
            f"   issues: {issues}"
        )
    return "\n".join(lines)


def _revision_target_label(request: RevisionRequest) -> str:
    """Describe the unit/topic the questions are being revised for."""

    if request.unit_name:
        return (
            f"Unit {request.unit_no} ({request.unit_name})"
            if request.unit_no
            else request.unit_name
        )
    if request.unit_no:
        return f"Unit {request.unit_no}"
    return "this syllabus"



def _specs_block(request: GenerationRequest) -> str:
    """Render the required question specs as a numbered list."""

    return "\n".join(
        f"{i}. type={spec.question_type.value} marks={spec.marks} "
        f"difficulty={spec.difficulty.value}"
        for i, spec in enumerate(request.specs, start=1)
    )
