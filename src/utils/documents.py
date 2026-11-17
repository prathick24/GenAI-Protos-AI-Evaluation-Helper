"""Text extraction and unit/topic chunking for syllabus documents.

Extraction turns an uploaded PDF/text file into ``(page_no, text)`` pairs.
Chunking splits that text into verbatim slices tagged with the syllabus's own
unit and topic headings, so retrieval can later filter by ``unit_no`` while the
generation agent still receives the *raw* text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.settings import MAX_SYLLABUS_CHUNK_CHARS
from src.utils.exceptions import FileProcessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# "Unit 3", "UNIT-3", "Chapter IV: Kinematics" -> unit number + optional name.
_UNIT_RE = re.compile(
    r"^\s*(?:unit|chapter)\s*[-:.\u2013]?\s*(\d{1,3}|[ivxlcdm]{1,7})\b\s*[-:.\u2013]?\s*(.*)$",
    re.IGNORECASE,
)
# "1.2 Newton's laws" / "3) Friction" -> a numbered topic heading.
_TOPIC_RE = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2})*\s*[.)]?\s+(\S.{0,118})$")

# Lines longer than this are treated as body text, never as a heading.
_MAX_HEADING_LEN = 120
# Sentence punctuation marks a body line, e.g. the list item "1) Protons are positive.".
_SENTENCE_PUNCTUATION = re.compile(r"[.;!](\s|$)")


@dataclass(slots=True)
class ChunkDraft:
    """A verbatim syllabus slice plus the metadata used for retrieval."""

    chunk_index: int
    unit_no: str | None
    unit_name: str | None
    topic_name: str | None
    page_no: int | None
    raw_text: str


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Extract ``(page_number, text)`` pairs from a supported document."""

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [(1, path.read_text(encoding="utf-8", errors="replace"))]
    if suffix == ".pdf":
        return _extract_pdf(path)
    raise FileProcessingError(f"Unsupported document type: {suffix or 'unknown'}")


def _extract_pdf(path: Path) -> list[tuple[int, str]]:
    """Extract text per page from a PDF using pypdf (an ML extra)."""

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise FileProcessingError(
            "PDF parsing requires the ML extras. Install them with: uv sync --extra ml"
        ) from exc

    try:
        reader = PdfReader(str(path))
        return [
            (page_no, page.extract_text() or "")
            for page_no, page in enumerate(reader.pages, start=1)
        ]
    except Exception as exc:  # noqa: BLE001 - surface any parser failure uniformly
        raise FileProcessingError(f"Could not read PDF '{path.name}': {exc}") from exc


def _is_heading(line: str) -> bool:
    """Return True when a line can be a unit or topic heading.

    Numbered list items inside body text (e.g. ``1) Protons are positive.``)
    otherwise look exactly like topic headings, which truncates the enclosing
    topic and invents a topic named after a sentence. A real heading is short
    and does not read as a complete sentence.
    """

    return len(line) <= _MAX_HEADING_LEN and _SENTENCE_PUNCTUATION.search(line) is None


def chunk_syllabus(pages: list[tuple[int, str]]) -> list[ChunkDraft]:
    """Split page text into chunks delimited by unit and topic headings."""

    chunks: list[ChunkDraft] = []
    buffer: list[str] = []
    index = 0
    page_no: int | None = None
    unit_no: str | None = None
    unit_name: str | None = None
    topic: str | None = None

    def flush() -> None:
        nonlocal buffer, index, page_no
        text = "\n".join(buffer).strip()
        if text:
            chunks.append(ChunkDraft(index, unit_no, unit_name, topic, page_no, text))
            index += 1
        buffer = []
        page_no = None

    for current_page, text in pages:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            unit_match = _UNIT_RE.match(stripped)
            if unit_match and _is_heading(stripped):
                flush()
                unit_no = _normalise_unit_no(unit_match.group(1))
                unit_name = unit_match.group(2).strip() or None
                topic = None
                continue

            topic_match = _TOPIC_RE.match(stripped)
            if topic_match and _is_heading(stripped):
                flush()
                topic = topic_match.group(1).strip()
                continue

            if page_no is None:
                page_no = current_page
            buffer.append(stripped)

    flush()
    return _split_oversized(chunks)


def _normalise_unit_no(token: str) -> str:
    """Return a stable unit identifier (``"3"`` or ``"IV"``)."""

    return token.upper() if token.isalpha() else str(int(token))


def _split_oversized(chunks: list[ChunkDraft]) -> list[ChunkDraft]:
    """Split chunks exceeding the size limit, preserving metadata and order."""

    result: list[ChunkDraft] = []
    index = 0
    for chunk in chunks:
        for piece in _split_text(chunk.raw_text):
            result.append(
                ChunkDraft(
                    chunk_index=index,
                    unit_no=chunk.unit_no,
                    unit_name=chunk.unit_name,
                    topic_name=chunk.topic_name,
                    page_no=chunk.page_no,
                    raw_text=piece,
                )
            )
            index += 1
    return result


def _split_text(text: str, limit: int = MAX_SYLLABUS_CHUNK_CHARS) -> list[str]:
    """Greedily pack lines into slices of at most ``limit`` characters."""

    if len(text) <= limit:
        return [text]

    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines():
        if current and size + len(line) + 1 > limit:
            pieces.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        pieces.append("\n".join(current))
    return pieces
