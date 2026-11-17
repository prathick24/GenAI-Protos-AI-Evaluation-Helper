"""Tests for grouped question-paper PDF export."""

from __future__ import annotations

import uuid

from src.models.paper import QuestionRead
from src.utils.paper_pdf import _section_letter, group_questions


def _question(
    *,
    order_index: int,
    question_type: str,
    marks: float,
    text: str | None = None,
) -> QuestionRead:
    return QuestionRead(
        id=uuid.uuid4(),
        order_index=order_index,
        question_text=text or f"Question {order_index + 1}",
        question_type=question_type,
        marks=marks,
        options=["A", "B", "C", "D"] if question_type == "mcq" else None,
        correct_option="A" if question_type == "mcq" else None,
        model_answer=f"Answer {order_index + 1}",
        rubric_criteria=[],
    )


def test_group_questions_orders_by_marks_then_type_and_renumbers() -> None:
    questions = [
        _question(order_index=0, question_type="long", marks=5),
        _question(order_index=1, question_type="mcq", marks=1),
        _question(order_index=2, question_type="short", marks=2),
        _question(order_index=3, question_type="mcq", marks=1),
    ]

    sections = group_questions(questions)

    assert [section.label for section in sections] == [
        "Section A - 1 mark (Multiple Choice)",
        "Section B - 2 marks (Short Answer)",
        "Section C - 5 marks (Long Answer)",
    ]
    assert [number for section in sections for number, _ in section.questions] == [1, 2, 3, 4]
    assert [question.order_index for _, question in sections[0].questions] == [1, 3]


def test_group_questions_keeps_same_marks_different_types_separate() -> None:
    questions = [
        _question(order_index=0, question_type="short", marks=2),
        _question(order_index=1, question_type="mcq", marks=2),
    ]

    sections = group_questions(questions)

    assert [section.label for section in sections] == [
        "Section A - 2 marks (Multiple Choice)",
        "Section B - 2 marks (Short Answer)",
    ]


def test_section_letter_handles_rollover() -> None:
    assert _section_letter(0) == "A"
    assert _section_letter(25) == "Z"
    assert _section_letter(26) == "AA"
    assert _section_letter(27) == "AB"
