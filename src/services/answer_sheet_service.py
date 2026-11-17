"""Business logic for answer sheet ingestion and OCR (FR-004).

Ingestion pipeline per uploaded file: validate -> store on disk -> decide
``typed`` vs ``scanned`` (auto) -> extract text (typed) or OCR (scanned) ->
persist the ``ANSWER_SHEETS`` row with its ``input_type`` / ``ocr_status`` and
confidence. Per-file failures never abort the batch: each file yields its own
result and OCR failures are flagged on the sheet so it can be retried.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.client import ocr
from src.models.answer_sheet import AnswerSheetRead, UploadResult
from src.repositories import answer_sheet_repository as repo
from src.repositories import exam_repository as exam_repo
from src.repositories.schema import AnswerSheet, Exam
from src.settings import (
    ALLOWED_ANSWER_SHEET_EXTENSIONS,
    ANSWER_SHEET_STORAGE_SUBDIR,
    IMAGE_ANSWER_SHEET_EXTENSIONS,
    MAX_ANSWER_SHEET_FILE_MB,
    InputType,
    OcrStatus,
    SheetStatus,
    UploadInputType,
    settings,
)
from src.utils import documents
from src.utils.exceptions import AppError, FileProcessingError, NotFoundError, OcrError
from src.utils.exceptions import error_codes as code
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnswerSheetService:
    """Orchestrates answer sheet upload, text extraction/OCR and listing."""

    def __init__(self, session: AsyncSession, *, provider: ocr.OcrProvider | None = None) -> None:
        self._session = session
        self._ocr = provider or ocr.get_provider()

    async def upload(
        self,
        exam_id: uuid.UUID,
        *,
        files: list[UploadFile],
        student_id: uuid.UUID | None = None,
        student_name: str | None = None,
        input_type: UploadInputType = UploadInputType.AUTO,
    ) -> list[UploadResult]:
        """Ingest a batch of answer sheets, returning a per-file result each."""

        exam = await self._require(exam_id)
        name = _clean_student_name(student_name)
        results = [
            await self._ingest(exam, upload, student_id, name, input_type) for upload in files
        ]
        await self._session.commit()
        stored = sum(1 for result in results if result.status == "uploaded")
        logger.info(
            "Answer sheets uploaded",
            extra={"exam_id": str(exam_id), "files": len(files), "stored": stored},
        )
        return results

    async def list(self, exam_id: uuid.UUID) -> list[AnswerSheetRead]:
        """Return an exam's answer sheets."""

        await self._require(exam_id)
        rows = await repo.list_for_exam(self._session, exam_id)
        return [AnswerSheetRead.from_model(row) for row in rows]

    async def _ingest(
        self,
        exam: Exam,
        upload: UploadFile,
        student_id: uuid.UUID | None,
        student_name: str | None,
        requested: UploadInputType,
    ) -> UploadResult:
        """Ingest a single file, converting any failure into an error result."""

        name = upload.filename or "answer_sheet"
        try:
            _validate_filename(upload.filename)
            data = await upload.read()
            _validate_size(data)
        except FileProcessingError as exc:
            return UploadResult(
                file=name, status="error", student_name=student_name, reason=exc.message
            )

        file_path = _persist_bytes(data, name)
        source = Path(file_path)
        resolved = _resolve_input_type(source, requested)
        sheet = AnswerSheet(
            exam_id=exam.id,
            student_id=student_id,
            student_name=student_name,
            file_path=file_path,
            input_type=resolved.value,
            ocr_status=OcrStatus.PENDING.value,
            sheet_status=SheetStatus.UPLOADED.value,
        )

        try:
            text, confidence = await asyncio.to_thread(_read, source, resolved, self._ocr)
        except AppError as exc:
            sheet.ocr_status = OcrStatus.FAILED.value
            sheet.sheet_status = SheetStatus.FAILED.value
            sheet.ocr_confidence = {"error": exc.message}
            await repo.add(self._session, sheet)
            logger.warning(
                "Answer sheet ingestion failed",
                extra={"exam_id": str(exam.id), "file": name, "reason": exc.message},
            )
            return UploadResult(
                file=name,
                status="error",
                answer_sheet_id=sheet.id,
                student_name=student_name,
                reason=exc.message,
            )

        _write_text(source, text)
        sheet.ocr_status = OcrStatus.DONE.value
        sheet.ocr_confidence = confidence
        await repo.add(self._session, sheet)
        return UploadResult(
            file=name,
            status="uploaded",
            answer_sheet_id=sheet.id,
            student_name=student_name,
        )

    async def _require(self, exam_id: uuid.UUID) -> Exam:
        """Load an exam, raising :class:`NotFoundError` when absent or deleted."""

        exam = await exam_repo.get(self._session, exam_id)
        if exam is None or not exam.is_active:
            raise NotFoundError(
                f"Exam {exam_id} not found",
                error_code=code.EXAM_NOT_FOUND,
            )
        return exam


def _resolve_input_type(path: Path, requested: UploadInputType) -> InputType:
    """Return the sheet's input type, inferring it from the file for ``auto``."""

    if requested is not UploadInputType.AUTO:
        return InputType(requested.value)
    suffix = path.suffix.lower()
    if suffix in IMAGE_ANSWER_SHEET_EXTENSIONS:
        return InputType.SCANNED
    if suffix == ".txt":
        return InputType.TYPED
    # A PDF is "typed" when it carries an extractable text layer.
    try:
        pages = documents.extract_pages(path)
    except AppError:
        return InputType.SCANNED
    return InputType.TYPED if any(text.strip() for _, text in pages) else InputType.SCANNED


def _read(path: Path, input_type: InputType, provider: ocr.OcrProvider) -> tuple[str, dict | None]:
    """Extract (typed) or OCR (scanned) a stored sheet's text and confidence."""

    if input_type is InputType.TYPED:
        pages = documents.extract_pages(path)
        text = "\n".join(page_text for _, page_text in pages).strip()
        if not text:
            raise FileProcessingError("No extractable text found in the answer sheet.")
        return text, None

    result = provider.recognise(path)
    if not result.text.strip():
        raise OcrError(f"No text could be recognised in '{path.name}'.")
    return result.text, result.confidence_map()


def _clean_student_name(value: str | None) -> str | None:
    """Return a trimmed student name, treating blank input as absent."""

    if value is None:
        return None
    return value.strip() or None


def _validate_filename(filename: str | None) -> None:
    """Reject uploads without a name or with an unsupported extension."""

    if not filename:
        raise FileProcessingError("A file name is required.")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_ANSWER_SHEET_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_ANSWER_SHEET_EXTENSIONS))
        raise FileProcessingError(f"Unsupported file type '{suffix}'. Allowed: {allowed}.")


def _validate_size(data: bytes) -> None:
    """Reject empty files and files above the configured limit."""

    if not data:
        raise FileProcessingError("The uploaded file is empty.")
    if len(data) > MAX_ANSWER_SHEET_FILE_MB * 1024 * 1024:
        raise FileProcessingError(f"File exceeds the {MAX_ANSWER_SHEET_FILE_MB} MB limit.")


def _persist_bytes(data: bytes, filename: str) -> str:
    """Write the upload to the storage directory and return its path."""

    suffix = Path(filename).suffix.lower()
    directory = Path(settings.storage_dir) / ANSWER_SHEET_STORAGE_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{uuid.uuid4()}{suffix}"
    target.write_bytes(data)
    return str(target)


def _write_text(path: Path, text: str) -> None:
    """Persist the recognised/extracted text beside the upload for segmentation."""

    path.with_name(f"{path.name}.ocr.txt").write_text(text, encoding="utf-8")
