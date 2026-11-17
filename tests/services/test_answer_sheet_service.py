"""Tests for the answer sheet ingestion service (FR-004)."""

from __future__ import annotations

import io
import uuid
from datetime import UTC, datetime

import pytest
from fastapi import UploadFile

from src.client import ocr
from src.repositories.schema import AnswerSheet, Exam
from src.services import answer_sheet_service
from src.services.answer_sheet_service import (
    AnswerSheetService,
    _resolve_input_type,
    _validate_filename,
    _validate_size,
)
from src.settings import InputType, OcrStatus, SheetStatus, UploadInputType
from src.utils.exceptions import FileProcessingError, NotFoundError, OcrError
from src.utils.exceptions import error_codes as code

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _FakeSession:
    """Minimal AsyncSession stand-in that stamps audit fields and records writes."""

    def __init__(self, *, exam: Exam | None = None) -> None:
        self._exam = exam
        self.commits = 0
        self.flushes = 0
        self.added: list[object] = []

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        self.flushes += 1
        for item in self.added:
            _apply_defaults(item)

    async def commit(self) -> None:
        self.commits += 1

    async def get(self, model: object, key: object) -> object | None:
        if model is Exam:
            return self._exam
        return None


def _apply_defaults(row: object) -> None:
    """Emulate the column defaults SQLAlchemy applies at INSERT time."""

    if getattr(row, "id", None) is None:
        row.id = uuid.uuid4()
    for attr, value in (("created_at", _NOW), ("updated_at", _NOW), ("is_active", True)):
        if hasattr(row, attr) and getattr(row, attr) is None:
            setattr(row, attr, value)


def _exam(**overrides: object) -> Exam:
    defaults: dict[str, object] = {
        "id": uuid.uuid4(),
        "title": "Midterm exam",
        "total_marks": 50,
        "status": "draft",
        "question_paper_id": uuid.uuid4(),
        "created_at": _NOW,
        "updated_at": _NOW,
        "is_active": True,
    }
    defaults.update(overrides)
    return Exam(**defaults)


class _StubProvider:
    """OCR provider returning a fixed result (or raising a preset error)."""

    name = "stub"

    def __init__(
        self,
        *,
        text: str = "Recognised answer",
        confidence: float | None = 0.9,
        error: Exception | None = None,
    ) -> None:
        self._text = text
        self._confidence = confidence
        self._error = error

    def recognise(self, path: object) -> ocr.OcrResult:
        if self._error is not None:
            raise self._error
        return ocr.OcrResult(text=self._text, provider=self.name, confidence=self._confidence)


def _upload(filename: str | None, data: bytes = b"answer text") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


@pytest.fixture(autouse=True)
def _storage(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect answer sheet storage into a per-test temporary directory."""

    monkeypatch.setattr(answer_sheet_service.settings, "storage_dir", str(tmp_path))


async def test_typed_text_is_extracted_without_ocr() -> None:
    exam = _exam()
    session = _FakeSession(exam=exam)
    service = AnswerSheetService(session, provider=_StubProvider(error=OcrError("unused")))

    results = await service.upload(
        exam.id, files=[_upload("sheet.txt", b"F = m a")], student_id=None
    )

    assert len(results) == 1
    assert results[0].status == "uploaded"
    sheet = session.added[0]
    assert isinstance(sheet, AnswerSheet)
    assert sheet.input_type == InputType.TYPED.value
    assert sheet.ocr_status == OcrStatus.DONE.value
    assert sheet.ocr_confidence is None
    assert session.commits == 1


async def test_upload_records_a_trimmed_student_name() -> None:
    exam = _exam()
    session = _FakeSession(exam=exam)
    service = AnswerSheetService(session, provider=_StubProvider())

    results = await service.upload(
        exam.id,
        files=[_upload("sheet.txt", b"F = m a")],
        student_name="  Asha  ",
    )

    sheet = session.added[0]
    assert sheet.student_name == "Asha"
    assert results[0].student_name == "Asha"


async def test_blank_student_name_is_stored_as_absent() -> None:
    exam = _exam()
    session = _FakeSession(exam=exam)
    service = AnswerSheetService(session, provider=_StubProvider())

    await service.upload(exam.id, files=[_upload("sheet.txt", b"F = m a")], student_name="  ")

    assert session.added[0].student_name is None


async def test_scanned_image_is_ocrd_and_flagged() -> None:
    exam = _exam()
    session = _FakeSession(exam=exam)
    provider = _StubProvider(text="Handwritten", confidence=0.75)
    service = AnswerSheetService(session, provider=provider)

    results = await service.upload(exam.id, files=[_upload("sheet.png", b"\x89PNG")])

    assert results[0].status == "uploaded"
    sheet = session.added[0]
    assert sheet.input_type == InputType.SCANNED.value
    assert sheet.ocr_status == OcrStatus.DONE.value
    assert sheet.ocr_confidence == {"provider": "stub", "mean": 0.75}


async def test_ocr_failure_marks_the_sheet_and_does_not_abort() -> None:
    exam = _exam()
    session = _FakeSession(exam=exam)
    provider = _StubProvider(error=OcrError("engine unavailable"))
    service = AnswerSheetService(session, provider=provider)

    results = await service.upload(exam.id, files=[_upload("sheet.png", b"\x89PNG")])

    assert results[0].status == "error"
    assert results[0].reason == "engine unavailable"
    sheet = session.added[0]
    assert sheet.ocr_status == OcrStatus.FAILED.value
    assert sheet.sheet_status == SheetStatus.FAILED.value
    assert sheet.ocr_confidence == {"error": "engine unavailable"}
    assert session.commits == 1


async def test_empty_ocr_output_is_treated_as_failure() -> None:
    exam = _exam()
    session = _FakeSession(exam=exam)
    service = AnswerSheetService(session, provider=_StubProvider(text="   "))

    results = await service.upload(exam.id, files=[_upload("sheet.png", b"\x89PNG")])

    assert results[0].status == "error"
    assert session.added[0].ocr_status == OcrStatus.FAILED.value


async def test_a_bad_file_does_not_abort_the_batch() -> None:
    exam = _exam()
    session = _FakeSession(exam=exam)
    service = AnswerSheetService(session, provider=_StubProvider())

    results = await service.upload(
        exam.id,
        files=[_upload("sheet.exe", b"binary"), _upload("sheet.txt", b"F = m a")],
    )

    assert [result.status for result in results] == ["error", "uploaded"]
    assert len(session.added) == 1  # only the valid file was persisted


async def test_empty_file_is_rejected() -> None:
    exam = _exam()
    session = _FakeSession(exam=exam)
    service = AnswerSheetService(session, provider=_StubProvider())

    results = await service.upload(exam.id, files=[_upload("sheet.txt", b"")])

    assert results[0].status == "error"
    assert session.added == []


async def test_upload_raises_when_exam_missing() -> None:
    session = _FakeSession(exam=None)

    with pytest.raises(NotFoundError) as excinfo:
        await AnswerSheetService(session, provider=_StubProvider()).upload(
            uuid.uuid4(), files=[_upload("sheet.txt")]
        )

    assert excinfo.value.error_code == code.EXAM_NOT_FOUND


async def test_list_returns_answer_sheets(monkeypatch: pytest.MonkeyPatch) -> None:
    exam = _exam()
    rows = [AnswerSheet(exam_id=exam.id, input_type=InputType.TYPED.value, **_audit())]

    async def _list_for_exam(session: object, exam_id: uuid.UUID) -> list[AnswerSheet]:
        return rows

    monkeypatch.setattr(answer_sheet_service.repo, "list_for_exam", _list_for_exam)

    result = await AnswerSheetService(_FakeSession(exam=exam)).list(exam.id)

    assert len(result) == 1
    assert result[0].exam_id == exam.id


def _audit() -> dict[str, object]:
    return {
        "id": uuid.uuid4(),
        "created_at": _NOW,
        "updated_at": _NOW,
        "is_active": True,
        "ocr_status": OcrStatus.DONE.value,
        "sheet_status": SheetStatus.UPLOADED.value,
    }


def test_resolve_input_type_honours_an_explicit_choice(tmp_path) -> None:
    path = tmp_path / "sheet.txt"
    path.write_text("answer", encoding="utf-8")

    assert _resolve_input_type(path, UploadInputType.SCANNED) is InputType.SCANNED
    assert _resolve_input_type(path, UploadInputType.TYPED) is InputType.TYPED


def test_resolve_input_type_infers_from_the_extension(tmp_path) -> None:
    text = tmp_path / "sheet.txt"
    text.write_text("answer", encoding="utf-8")
    image = tmp_path / "sheet.jpg"
    image.write_bytes(b"\x89JPG")

    assert _resolve_input_type(text, UploadInputType.AUTO) is InputType.TYPED
    assert _resolve_input_type(image, UploadInputType.AUTO) is InputType.SCANNED


def test_resolve_input_type_falls_back_to_scanned_for_unreadable_pdf(tmp_path) -> None:
    pdf = tmp_path / "sheet.pdf"
    pdf.write_bytes(b"%PDF-1.4 not really")

    assert _resolve_input_type(pdf, UploadInputType.AUTO) is InputType.SCANNED


def test_validate_filename_rejects_unknown_types() -> None:
    with pytest.raises(FileProcessingError):
        _validate_filename("sheet.docx")
    with pytest.raises(FileProcessingError):
        _validate_filename(None)

    _validate_filename("sheet.PNG")  # uppercase extensions are accepted


def test_validate_size_rejects_empty_and_oversized() -> None:
    with pytest.raises(FileProcessingError):
        _validate_size(b"")
    with pytest.raises(FileProcessingError):
        _validate_size(b"x" * (answer_sheet_service.MAX_ANSWER_SHEET_FILE_MB * 1024 * 1024 + 1))
