"""Business logic for syllabus ingestion and retrieval (FR-001).

Ingestion pipeline: store the upload -> extract text -> chunk by unit/topic ->
embed locally -> persist the syllabus and its chunks.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.client import embedding
from src.models.syllabus import SyllabusRead, UnitTopics
from src.repositories import syllabus_repository as repo
from src.repositories.schema import Syllabus, SyllabusChunk
from src.settings import (
    ALLOWED_SYLLABUS_EXTENSIONS,
    MAX_SYLLABUS_FILE_MB,
    SYLLABUS_STORAGE_SUBDIR,
    InputType,
    OcrStatus,
    SyllabusStatus,
    settings,
)
from src.utils import documents
from src.utils.exceptions import AppError, FileProcessingError, NotFoundError
from src.utils.exceptions import error_codes as code
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyllabusService:
    """Orchestrates syllabus ingestion and read operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upload(
        self,
        *,
        upload: UploadFile,
        title: str | None = None,
        subject: str | None = None,
        class_section: str | None = None,
        scope: list[str] | None = None,
    ) -> SyllabusRead:
        """Ingest an uploaded syllabus, returning its indexed representation."""

        _validate_filename(upload.filename)
        data = await upload.read()
        _validate_size(data)

        filename = upload.filename or "syllabus.txt"
        file_path = _persist_bytes(data, filename)
        title_value = title or Path(filename).stem or "Untitled syllabus"
        prior = await repo.find_latest_active(
            self._session,
            title=title_value,
            subject=subject,
            class_section=class_section,
        )
        syllabus = Syllabus(
            title=title_value,
            subject=subject,
            class_section=class_section,
            file_path=file_path,
            input_type=InputType.TYPED.value,
            ocr_status=OcrStatus.DONE.value,
            status=SyllabusStatus.PROCESSING.value,
            version=prior.version + 1 if prior is not None else 1,
            scope=scope,
        )
        await repo.add(self._session, syllabus)

        try:
            drafts, vectors = _build_chunks(Path(file_path))
        except AppError:
            syllabus.status = SyllabusStatus.FAILED.value
            await self._session.commit()
            logger.warning("Syllabus ingestion failed", extra={"syllabus_id": str(syllabus.id)})
            raise

        chunks = [
            SyllabusChunk(
                syllabus_id=syllabus.id,
                unit_no=draft.unit_no,
                unit_name=draft.unit_name,
                topic_name=draft.topic_name,
                chunk_index=draft.chunk_index,
                page_no=draft.page_no,
                raw_text=draft.raw_text,
                embedding=vector,
            )
            for draft, vector in zip(drafts, vectors, strict=True)
        ]
        await repo.add_chunks(self._session, chunks)
        syllabus.status = SyllabusStatus.READY.value
        if prior is not None:
            # Supersede the previous version only once the new one is indexed.
            prior.is_active = False
        await self._session.commit()

        logger.info(
            "Syllabus ingested",
            extra={"syllabus_id": str(syllabus.id), "chunks": len(chunks)},
        )
        return SyllabusRead.from_model(syllabus, len(chunks))

    async def list(self) -> list[SyllabusRead]:
        """Return all active syllabi with their chunk counts."""

        rows = await repo.list_all(self._session)
        counts = await repo.chunk_counts(self._session, [row.id for row in rows])
        return [SyllabusRead.from_model(row, counts.get(row.id, 0)) for row in rows]

    async def get(self, syllabus_id: uuid.UUID) -> SyllabusRead:
        """Return a single syllabus or raise :class:`NotFoundError`."""

        row = await self._require(syllabus_id)
        count = await repo.chunk_count(self._session, syllabus_id)
        return SyllabusRead.from_model(row, count)

    async def delete(self, syllabus_id: uuid.UUID) -> None:
        """Soft-delete a syllabus by clearing its ``is_active`` flag."""

        row = await self._require(syllabus_id)
        row.is_active = False
        await self._session.commit()
        logger.info("Syllabus soft-deleted", extra={"syllabus_id": str(syllabus_id)})

    async def topics(self, syllabus_id: uuid.UUID) -> list[UnitTopics]:
        """Return the unit/topic tree derived from the syllabus's chunks."""

        await self._require(syllabus_id)
        chunks = await repo.list_chunks(self._session, syllabus_id)
        return _build_unit_tree(chunks)

    async def _require(self, syllabus_id: uuid.UUID) -> Syllabus:
        """Load a syllabus, raising :class:`NotFoundError` when absent or deleted."""

        row = await repo.get(self._session, syllabus_id)
        if row is None or not row.is_active:
            raise NotFoundError(
                f"Syllabus {syllabus_id} not found",
                error_code=code.SYLLABUS_NOT_FOUND,
            )
        return row


def _build_chunks(
    path: Path,
) -> tuple[list[documents.ChunkDraft], list[list[float] | None]]:
    """Extract, chunk and embed a stored syllabus file."""

    pages = documents.extract_pages(path)
    if not any(text.strip() for _, text in pages):
        raise FileProcessingError(
            "No extractable text found. The document appears to be scanned; OCR is not "
            "configured yet, so upload a text-based PDF or a .txt file."
        )

    drafts = documents.chunk_syllabus(pages)
    if not drafts:
        raise FileProcessingError("No content could be extracted from the syllabus.")

    return drafts, _embed([draft.raw_text for draft in drafts])


def _embed(texts: list[str]) -> list[list[float] | None]:
    """Embed chunk texts locally, degrading to null vectors when unavailable."""

    if not embedding.is_available():
        logger.warning(
            "Chunks stored without embeddings; retrieval will be limited",
            extra={"hint": "uv sync --extra ml"},
        )
        return [None] * len(texts)
    return list(embedding.embed_texts(texts))


def _build_unit_tree(chunks: list[SyllabusChunk]) -> list[UnitTopics]:
    """Group chunk metadata into ordered units and their distinct topics."""

    nodes: dict[str, UnitTopics] = {}
    order: list[str] = []
    for chunk in chunks:
        key = chunk.unit_no or ""
        node = nodes.get(key)
        if node is None:
            node = UnitTopics(unit_no=key, unit_name=chunk.unit_name, topics=[])
            nodes[key] = node
            order.append(key)
        elif node.unit_name is None and chunk.unit_name:
            node.unit_name = chunk.unit_name
        if chunk.topic_name and chunk.topic_name not in node.topics:
            node.topics.append(chunk.topic_name)
    return [nodes[key] for key in order]


def _validate_filename(filename: str | None) -> None:
    """Reject uploads without a name or with an unsupported extension."""

    if not filename:
        raise FileProcessingError("A file name is required.")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SYLLABUS_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_SYLLABUS_EXTENSIONS))
        raise FileProcessingError(f"Unsupported file type '{suffix}'. Allowed: {allowed}.")


def _validate_size(data: bytes) -> None:
    """Reject empty files and files above the configured limit."""

    if not data:
        raise FileProcessingError("The uploaded file is empty.")
    if len(data) > MAX_SYLLABUS_FILE_MB * 1024 * 1024:
        raise FileProcessingError(f"File exceeds the {MAX_SYLLABUS_FILE_MB} MB limit.")


def _persist_bytes(data: bytes, filename: str) -> str:
    """Write the upload to the storage directory and return its path."""

    suffix = Path(filename).suffix.lower()
    directory = Path(settings.storage_dir) / SYLLABUS_STORAGE_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{uuid.uuid4()}{suffix}"
    target.write_bytes(data)
    return str(target)
