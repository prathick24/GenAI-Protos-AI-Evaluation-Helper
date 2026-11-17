"""Render a generated question paper as a student or teacher PDF.

The student copy contains only the paper; the teacher copy appends the model
answers and rubric criteria. Questions are grouped into sections by marks and
question type so the exported layout matches the grouped UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.models.paper import QuestionPaperRead, QuestionRead

_PAGE_WIDTH = 595.0
_PAGE_HEIGHT = 842.0
_MARGIN = 48.0
_BOTTOM = _PAGE_HEIGHT - _MARGIN
_WIDTH = _PAGE_WIDTH - 2 * _MARGIN

_TYPE_ORDER = {"mcq": 0, "short": 1, "long": 2}
_TYPE_LABEL = {
    "mcq": "Multiple Choice",
    "short": "Short Answer",
    "long": "Long Answer",
}

_TRANSLATIONS = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00ad": "-",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2026": "...",
        "\u2192": "->",
        "\u2190": "<-",
        "\u00b0": " deg",
        "\u00d7": "x",
        "\u00f7": "/",
        "\u00b1": "+/-",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u00a0": " ",
        **{chr(0x2080 + i): str(i) for i in range(10)},
        **{chr(0x2070 + i): str(i) for i in range(10)},
        "\u00b2": "2",
        "\u00b3": "3",
        "\u00b9": "1",
    }
)


@dataclass(slots=True)
class PaperSection:
    """A group of questions with the same marks and question type."""

    letter: str
    label: str
    marks: float
    question_type: str
    questions: list[tuple[int, QuestionRead]]


def group_questions(questions: list[QuestionRead]) -> list[PaperSection]:
    """Group questions by marks then type, numbering them continuously.

    Sections are ordered by marks (ascending) and, within the same marks, by the
    conventional MCQ -> short -> long order. This keeps a 2-mark MCQ separate
    from a 2-mark short answer while still presenting the paper in mark order.
    """

    groups: dict[tuple[float, str], list[QuestionRead]] = {}
    for question in questions:
        key = (float(question.marks), question.question_type)
        groups.setdefault(key, []).append(question)

    ordered = sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            _TYPE_ORDER.get(item[0][1], len(_TYPE_ORDER)),
            item[0][1],
        ),
    )

    sections: list[PaperSection] = []
    number = 1
    for index, ((marks, question_type), rows) in enumerate(ordered):
        numbered: list[tuple[int, QuestionRead]] = []
        for question in sorted(rows, key=lambda item: item.order_index):
            numbered.append((number, question))
            number += 1

        letter = _section_letter(index)
        marks_text = f"{marks:g} mark" if marks == 1 else f"{marks:g} marks"
        type_label = _TYPE_LABEL.get(
            question_type, question_type.replace("_", " ").title()
        )
        sections.append(
            PaperSection(
                letter=letter,
                label=f"Section {letter} - {marks_text} ({type_label})",
                marks=marks,
                question_type=question_type,
                questions=numbered,
            )
        )
    return sections


def export_filename(paper: QuestionPaperRead, *, include_answers: bool) -> str:
    """Return a safe download filename for the requested copy."""

    base = re.sub(r"[^A-Za-z0-9._-]+", "_", paper.title).strip("._-")
    base = (base or "question-paper")[:80]
    suffix = "teacher" if include_answers else "student"
    return f"{base}_{suffix}.pdf"


def render_paper_pdf(paper: QuestionPaperRead, *, include_answers: bool) -> bytes:
    """Render ``paper`` to PDF bytes.

    ``include_answers`` adds a teacher-only answer key and rubric after the
    question paper. PyMuPDF is imported lazily so the rest of the API can boot
    without the optional ``ml`` extras installed.
    """

    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - exercised only without ml extras
        raise RuntimeError(
            "PDF export requires the optional 'ml' dependencies (pymupdf)."
        ) from exc

    sections = group_questions(list(paper.questions))
    writer = _Writer(pymupdf)
    try:
        writer.text(paper.title, size=17, bold=True)
        writer.text(
            f"Subject: {paper.subject or '-'}    Class: {paper.class_section or '-'}",
            size=10,
        )
        writer.text(
            f"Time: {paper.duration_minutes or '-'} minutes    "
            f"Maximum marks: {paper.total_marks}",
            size=10,
        )
        if include_answers:
            writer.text(
                "Teacher copy - includes answer key and rubric",
                size=9,
                bold=True,
                color=(0.55, 0.0, 0.0),
            )
        writer.space(4)
        writer.rule()
        writer.text(
            "General instructions: All questions are compulsory. "
            "Marks for each question are shown in brackets.",
            size=9,
        )
        writer.space(8)

        _write_questions(writer, sections)

        if include_answers:
            writer.new_page()
            writer.text("Answer Key and Rubric", size=16, bold=True)
            writer.text(
                "Teacher copy - do not distribute to students.",
                size=9,
                color=(0.55, 0.0, 0.0),
            )
            writer.space(6)
            _write_answers(writer, sections)

        writer.footer(
            f"{paper.title} - {'Teacher' if include_answers else 'Student'} copy"
        )
        return writer.to_bytes()
    finally:
        writer.close()


def _write_questions(writer: _Writer, sections: list[PaperSection]) -> None:
    for section in sections:
        writer.space(6)
        writer.text(section.label, size=12, bold=True)
        writer.space(2)
        for number, question in section.questions:
            height = writer.measure(
                f"{number}. {question.question_text}  [{question.marks:g}]",
                size=11,
            )
            for index, option in enumerate(question.options or []):
                height += writer.measure(
                    f"{chr(65 + index)}. {option}", size=10, indent=14
                )
            writer.ensure(min(height + 5, writer.usable_height))
            writer.text(
                f"{number}. {question.question_text}  [{question.marks:g}]",
                size=11,
            )
            if question.options:
                for index, option in enumerate(question.options):
                    writer.text(
                        f"{chr(65 + index)}. {option}",
                        size=10,
                        indent=14,
                    )
            writer.space(5)


def _write_answers(writer: _Writer, sections: list[PaperSection]) -> None:
    for section in sections:
        writer.space(6)
        writer.text(section.label, size=12, bold=True)
        writer.space(2)
        for number, question in section.questions:
            height = writer.measure(f"{number}. {question.question_text}", size=10.5)
            if question.correct_option:
                height += writer.measure(
                    f"Correct option: {question.correct_option}", size=10
                )
            height += writer.measure(
                "Model answer: " + (question.model_answer or "Not provided."),
                size=10,
            )
            if question.rubric_criteria:
                height += writer.measure("Rubric:", size=10)
                for criterion in question.rubric_criteria:
                    height += writer.measure(
                        f"- {criterion.criterion_text} ({criterion.max_marks:g})",
                        size=9.5,
                        indent=14,
                    )
            writer.ensure(min(height + 6, writer.usable_height))
            writer.text(f"{number}. {question.question_text}", size=10.5, bold=True)
            if question.correct_option:
                writer.text(f"Correct option: {question.correct_option}", size=10)
            writer.text(
                "Model answer: " + (question.model_answer or "Not provided."),
                size=10,
            )
            if question.rubric_criteria:
                writer.text("Rubric:", size=10, bold=True)
                for criterion in sorted(
                    question.rubric_criteria, key=lambda item: item.order_index
                ):
                    writer.text(
                        f"- {criterion.criterion_text} ({criterion.max_marks:g})",
                        size=9.5,
                        indent=14,
                    )
            writer.space(6)


def _section_letter(index: int) -> str:
    """Return A, B, ..., Z, AA, AB, ... for a zero-based section index."""

    letters: list[str] = []
    value = index
    while True:
        value, remainder = divmod(value, 26)
        letters.append(chr(65 + remainder))
        if value == 0:
            break
        value -= 1
    return "".join(reversed(letters))


def _clean(text: str) -> str:
    """Replace characters unsupported by the built-in PDF fonts."""

    translated = text.translate(_TRANSLATIONS)
    return translated.encode("latin-1", "replace").decode("latin-1")


class _Writer:
    """Small top-down text layout helper for a PyMuPDF document."""

    def __init__(self, pymupdf_module: Any) -> None:
        self._pymupdf = pymupdf_module
        self._doc = pymupdf_module.open()
        self._page = self._doc.new_page(width=_PAGE_WIDTH, height=_PAGE_HEIGHT)
        self._y = _MARGIN

    @property
    def usable_height(self) -> float:
        return _BOTTOM - _MARGIN

    def new_page(self) -> None:
        self._page = self._doc.new_page(width=_PAGE_WIDTH, height=_PAGE_HEIGHT)
        self._y = _MARGIN

    def measure(self, text: str, *, size: float = 10, indent: float = 0) -> float:
        font = self._pymupdf.Font("helv")
        lines = _wrap(_clean(text), font, size, _WIDTH - indent)
        return len(lines or [""]) * size * 1.35

    def ensure(self, height: float) -> None:
        if self._y + height > _BOTTOM:
            self.new_page()

    def text(
        self,
        text: str,
        *,
        size: float = 10,
        bold: bool = False,
        indent: float = 0,
        gap: float = 4,
        color: tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        fontname = "hebo" if bold else "helv"
        font = self._pymupdf.Font(fontname)
        lines = _wrap(_clean(text), font, size, _WIDTH - indent)
        line_height = size * 1.35
        for line in lines or [""]:
            self.ensure(line_height)
            self._page.insert_text(
                (_MARGIN + indent, self._y + size * 0.8),
                line,
                fontsize=size,
                fontname=fontname,
                color=color,
            )
            self._y += line_height
        self._y += gap

    def space(self, height: float) -> None:
        self._y += height
        if self._y > _BOTTOM:
            self.new_page()

    def rule(self) -> None:
        self.ensure(4)
        self._page.draw_line(
            (_MARGIN, self._y),
            (_PAGE_WIDTH - _MARGIN, self._y),
            color=(0.55, 0.55, 0.55),
            width=0.6,
        )
        self._y += 4

    def footer(self, label: str) -> None:
        for index, page in enumerate(self._doc):
            page.insert_text(
                (_MARGIN, _PAGE_HEIGHT - 26),
                _clean(f"{label} - Page {index + 1}"),
                fontsize=8,
                fontname="helv",
                color=(0.4, 0.4, 0.4),
            )

    def to_bytes(self) -> bytes:
        return self._doc.tobytes()

    def close(self) -> None:
        self._doc.close()


def _wrap(text: str, font: Any, size: float, width: float) -> list[str]:
    """Greedily wrap ``text`` to ``width`` using ``font`` metrics."""

    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph.strip():
            lines.append("")
            continue

        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if font.text_length(candidate, fontsize=size) <= width:
                current = candidate
                continue

            if current:
                lines.append(current)
            if font.text_length(word, fontsize=size) <= width:
                current = word
                continue

            chunk = ""
            for character in word:
                candidate = chunk + character
                if font.text_length(candidate, fontsize=size) <= width:
                    chunk = candidate
                else:
                    lines.append(chunk)
                    chunk = character
            current = chunk
        if current:
            lines.append(current)
    return lines
