"""Tests for syllabus text extraction and chunking."""

from __future__ import annotations

import pytest

from src.utils import documents
from src.utils.exceptions import FileProcessingError

SAMPLE = (
    "Unit 1 Mechanics\n"
    "1.1 Newton's Laws\n"
    "Force equals mass times acceleration.\n"
    "1.2 Friction\n"
    "Friction opposes motion.\n"
    "Unit 2 Optics\n"
    "2.1 Lenses\n"
    "Convex lenses converge light.\n"
)


def test_extract_pages_txt(tmp_path):
    path = tmp_path / "syllabus.txt"
    path.write_text("Hello", encoding="utf-8")
    assert documents.extract_pages(path) == [(1, "Hello")]


def test_extract_pages_unsupported_extension(tmp_path):
    path = tmp_path / "syllabus.docx"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(FileProcessingError):
        documents.extract_pages(path)


def test_chunk_syllabus_tags_units_and_topics():
    chunks = documents.chunk_syllabus([(1, SAMPLE)])
    assert [(chunk.unit_no, chunk.topic_name) for chunk in chunks] == [
        ("1", "Newton's Laws"),
        ("1", "Friction"),
        ("2", "Lenses"),
    ]
    assert chunks[0].unit_name == "Mechanics"
    assert chunks[0].raw_text == "Force equals mass times acceleration."
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_chunk_syllabus_without_headings_is_single_chunk():
    chunks = documents.chunk_syllabus([(1, "Just some text.\nMore text.")])
    assert len(chunks) == 1
    assert chunks[0].unit_no is None
    assert chunks[0].raw_text == "Just some text.\nMore text."


def test_chunk_syllabus_ignores_blank_pages():
    assert documents.chunk_syllabus([(1, "   "), (2, "")]) == []


def test_chunk_syllabus_keeps_numbered_list_items_inside_the_topic():
    text = (
        "Unit 1 Matter\n"
        "1.1 Atoms\n"
        "Atoms are the smallest unit of matter.\n"
        "1) Protons are positively charged.\n"
        "2) Electrons are negatively charged.\n"
    )

    chunks = documents.chunk_syllabus([(1, text)])

    assert [(chunk.unit_no, chunk.topic_name) for chunk in chunks] == [("1", "Atoms")]
    assert chunks[0].raw_text == (
        "Atoms are the smallest unit of matter.\n"
        "1) Protons are positively charged.\n"
        "2) Electrons are negatively charged."
    )


def test_chunk_syllabus_accepts_question_style_headings():
    text = "Unit 2 Living\n2.3 How do Organisms Reproduce?\nThey reproduce sexually.\n"

    chunks = documents.chunk_syllabus([(1, text)])

    assert [(chunk.unit_no, chunk.topic_name) for chunk in chunks] == [
        ("2", "How do Organisms Reproduce?")
    ]
    assert chunks[0].raw_text == "They reproduce sexually."


def test_split_text_short_input_unchanged():
    assert documents._split_text("short", limit=100) == ["short"]


def test_split_text_respects_limit():
    text = "\n".join(["x" * 60 for _ in range(10)])
    pieces = documents._split_text(text, limit=100)
    assert len(pieces) > 1
    assert all(len(piece) <= 100 for piece in pieces)
