"""Tests for the syllabus ingestion service (FR-001)."""

from __future__ import annotations

import io
import uuid
from datetime import UTC, datetime

import pytest
from fastapi import UploadFile

from src.repositories.schema import Syllabus, SyllabusChunk
from src.services import syllabus_service
from src.services.syllabus_service import (
    SyllabusService,
    _build_unit_tree,
    _validate_filename,
    _validate_size,
)
from src.settings import SyllabusStatus
from src.utils.exceptions import FileProcessingError, NotFoundError
from src.utils.exceptions import error_codes as code

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_INGESTED = "Unit 1 Mechanics\n1.1 Newton's Laws\nForce equals mass times acceleration.\n"


class _FakeSession:
    """Minimal AsyncSession stand-in that records commits and flushes."""

    def __init__(self, *, get_result: Syllabus | None = None) -> None:
        self.commits = 0
        self.flushes = 0
        self._get_result = get_result
        self.added: list[object] = []

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def add_all(self, objs: list[object]) -> None:
        self.added.extend(objs)

    async def flush(self) -> None:
        self.flushes += 1

    async def commit(self) -> None:
        self.commits += 1

    async def get(self, model: object, key: object) -> Syllabus | None:
        return self._get_result


def _syllabus(**overrides: object) -> Syllabus:
    defaults: dict[str, object] = {
        "id": uuid.uuid4(),
        "created_at": _NOW,
        "updated_at": _NOW,
        "version": 1,
        "is_active": True,
        "title": "Physics",
        "input_type": "typed",
        "ocr_status": "done",
        "status": SyllabusStatus.READY.value,
    }
    defaults.update(overrides)
    return Syllabus(**defaults)


def _chunk(**overrides: object) -> SyllabusChunk:
    defaults: dict[str, object] = {
        "syllabus_id": uuid.uuid4(),
        "chunk_index": 0,
        "raw_text": "text",
    }
    defaults.update(overrides)
    return SyllabusChunk(**defaults)


@pytest.fixture
def no_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the local embedding client to report as unavailable."""

    monkeypatch.setattr(syllabus_service.embedding, "is_available", lambda: False)


@pytest.fixture
def storage_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(syllabus_service.settings, "storage_dir", str(tmp_path))
    return tmp_path


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def test_validate_filename_accepts_supported_extension() -> None:
    _validate_filename("syllabus.txt")


def test_validate_filename_rejects_missing_name() -> None:
    with pytest.raises(FileProcessingError):
        _validate_filename(None)


def test_validate_filename_rejects_unsupported_extension() -> None:
    with pytest.raises(FileProcessingError):
        _validate_filename("syllabus.docx")


def test_validate_size_accepts_small_file() -> None:
    _validate_size(b"some content")


def test_validate_size_rejects_empty_file() -> None:
    with pytest.raises(FileProcessingError):
        _validate_size(b"")


def test_validate_size_rejects_oversized_file() -> None:
    with pytest.raises(FileProcessingError):
        _validate_size(b"x" * (20 * 1024 * 1024 + 1))


# --------------------------------------------------------------------------- #
# Unit tree
# --------------------------------------------------------------------------- #
def test_build_unit_tree_groups_units_and_dedups_topics() -> None:
    chunks = [
        _chunk(unit_no="1", unit_name="Mechanics", topic_name="Newton's Laws"),
        _chunk(unit_no="1", unit_name="Mechanics", topic_name="Friction"),
        _chunk(unit_no="1", unit_name="Mechanics", topic_name="Friction"),
        _chunk(unit_no="2", unit_name="Optics", topic_name="Lenses"),
    ]

    tree = _build_unit_tree(chunks)

    assert [node.unit_no for node in tree] == ["1", "2"]
    assert tree[0].unit_name == "Mechanics"
    assert tree[0].topics == ["Newton's Laws", "Friction"]
    assert tree[1].topics == ["Lenses"]


def test_build_unit_tree_backfills_unit_name_from_later_chunk() -> None:
    chunks = [
        _chunk(unit_no="1", unit_name=None, topic_name="Intro"),
        _chunk(unit_no="1", unit_name="Mechanics", topic_name="Friction"),
    ]

    tree = _build_unit_tree(chunks)

    assert tree[0].unit_name == "Mechanics"


def test_build_unit_tree_without_unit_numbers_uses_single_bucket() -> None:
    tree = _build_unit_tree([_chunk(unit_no=None, topic_name="Misc")])

    assert len(tree) == 1
    assert tree[0].unit_no == ""
    assert tree[0].topics == ["Misc"]


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
async def test_upload_ingests_and_marks_ready(
    storage_dir,
    no_embeddings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_add(session: _FakeSession, syllabus: Syllabus) -> Syllabus:
        syllabus.id = uuid.uuid4()
        syllabus.created_at = _NOW
        syllabus.updated_at = _NOW
        syllabus.version = 1
        syllabus.is_active = True
        session.add(syllabus)
        return syllabus

    async def _no_prior(session: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(syllabus_service.repo, "add", _fake_add)
    monkeypatch.setattr(syllabus_service.repo, "find_latest_active", _no_prior)
    session = _FakeSession()
    service = SyllabusService(session)
    upload = UploadFile(file=io.BytesIO(_INGESTED.encode()), filename="syllabus.txt")

    result = await service.upload(upload=upload, title="Physics", scope=["1"])

    assert result.status == SyllabusStatus.READY.value
    assert result.chunk_count == 1
    assert result.title == "Physics"
    assert result.scope == ["1"]
    assert session.commits == 1
    assert any(isinstance(item, SyllabusChunk) for item in session.added)


async def test_upload_versions_and_deactivates_the_prior_syllabus(
    storage_dir,
    no_embeddings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = _syllabus(version=3)

    async def _prior(session: object, **kwargs: object) -> Syllabus:
        return prior

    async def _fake_add(session: _FakeSession, syllabus: Syllabus) -> Syllabus:
        syllabus.id = uuid.uuid4()
        syllabus.created_at = _NOW
        syllabus.updated_at = _NOW
        syllabus.is_active = True
        session.add(syllabus)
        return syllabus

    monkeypatch.setattr(syllabus_service.repo, "add", _fake_add)
    monkeypatch.setattr(syllabus_service.repo, "find_latest_active", _prior)
    session = _FakeSession()
    service = SyllabusService(session)
    upload = UploadFile(file=io.BytesIO(_INGESTED.encode()), filename="syllabus.txt")

    result = await service.upload(upload=upload, title="Physics", scope=["1"])

    assert result.version == 4
    assert prior.is_active is False


async def test_upload_failure_leaves_the_prior_syllabus_active(
    storage_dir,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = _syllabus(version=2)

    async def _prior(session: object, **kwargs: object) -> Syllabus:
        return prior

    async def _fake_add(session: _FakeSession, syllabus: Syllabus) -> Syllabus:
        syllabus.id = uuid.uuid4()
        session.add(syllabus)
        return syllabus

    def _raise(path: object) -> None:
        raise FileProcessingError("No extractable text found.")

    monkeypatch.setattr(syllabus_service.repo, "add", _fake_add)
    monkeypatch.setattr(syllabus_service.repo, "find_latest_active", _prior)
    monkeypatch.setattr(syllabus_service, "_build_chunks", _raise)
    session = _FakeSession()
    service = SyllabusService(session)
    upload = UploadFile(file=io.BytesIO(b"data"), filename="syllabus.txt")

    with pytest.raises(FileProcessingError):
        await service.upload(upload=upload, title="Physics")

    assert prior.is_active is True


async def test_upload_marks_failed_when_extraction_fails(
    storage_dir,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_add(session: _FakeSession, syllabus: Syllabus) -> Syllabus:
        syllabus.id = uuid.uuid4()
        session.add(syllabus)
        return syllabus

    def _raise(path: object) -> None:
        raise FileProcessingError("No extractable text found.")

    async def _no_prior(session: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(syllabus_service.repo, "add", _fake_add)
    monkeypatch.setattr(syllabus_service.repo, "find_latest_active", _no_prior)
    monkeypatch.setattr(syllabus_service, "_build_chunks", _raise)
    session = _FakeSession()
    service = SyllabusService(session)
    upload = UploadFile(file=io.BytesIO(b"data"), filename="syllabus.txt")

    with pytest.raises(FileProcessingError):
        await service.upload(upload=upload)

    syllabus = next(item for item in session.added if isinstance(item, Syllabus))
    assert syllabus.status == SyllabusStatus.FAILED.value
    assert session.commits == 1


async def test_upload_rejects_unsupported_extension() -> None:
    service = SyllabusService(_FakeSession())
    upload = UploadFile(file=io.BytesIO(b"data"), filename="syllabus.docx")

    with pytest.raises(FileProcessingError):
        await service.upload(upload=upload)


# --------------------------------------------------------------------------- #
# Read / delete
# --------------------------------------------------------------------------- #
async def test_get_raises_not_found_when_missing() -> None:
    service = SyllabusService(_FakeSession(get_result=None))

    with pytest.raises(NotFoundError) as excinfo:
        await service.get(uuid.uuid4())

    assert excinfo.value.error_code == code.SYLLABUS_NOT_FOUND


async def test_get_returns_syllabus_with_chunk_count(monkeypatch: pytest.MonkeyPatch) -> None:
    syllabus = _syllabus()
    session = _FakeSession(get_result=syllabus)

    async def _count(inner_session: object, syllabus_id: uuid.UUID) -> int:
        return 3

    monkeypatch.setattr(syllabus_service.repo, "chunk_count", _count)
    result = await SyllabusService(session).get(syllabus.id)

    assert result.id == syllabus.id
    assert result.chunk_count == 3


async def test_delete_soft_deletes_syllabus() -> None:
    syllabus = _syllabus()
    session = _FakeSession(get_result=syllabus)

    await SyllabusService(session).delete(syllabus.id)

    assert syllabus.is_active is False
    assert session.commits == 1


async def test_delete_raises_not_found_when_missing() -> None:
    with pytest.raises(NotFoundError):
        await SyllabusService(_FakeSession(get_result=None)).delete(uuid.uuid4())


# --------------------------------------------------------------------------- #
# Topics
# --------------------------------------------------------------------------- #
async def test_topics_returns_unit_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    syllabus = _syllabus()
    session = _FakeSession(get_result=syllabus)

    async def _chunks(inner_session: object, syllabus_id: uuid.UUID) -> list[SyllabusChunk]:
        return [
            _chunk(syllabus_id=syllabus.id, unit_no="1", topic_name="Newton's Laws"),
            _chunk(syllabus_id=syllabus.id, unit_no="2", topic_name="Lenses"),
        ]

    monkeypatch.setattr(syllabus_service.repo, "list_chunks", _chunks)
    tree = await SyllabusService(session).topics(syllabus.id)

    assert [node.unit_no for node in tree] == ["1", "2"]
    assert tree[0].topics == ["Newton's Laws"]


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #
async def test_list_returns_syllabi_with_chunk_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = _syllabus(), _syllabus(title="Chemistry")
    session = _FakeSession()

    async def _all(inner_session: object) -> list[Syllabus]:
        return [first, second]

    async def _counts(inner_session: object, ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
        return {first.id: 2, second.id: 5}

    monkeypatch.setattr(syllabus_service.repo, "list_all", _all)
    monkeypatch.setattr(syllabus_service.repo, "chunk_counts", _counts)
    result = await SyllabusService(session).list()

    assert [item.chunk_count for item in result] == [2, 5]


# --------------------------------------------------------------------------- #
# Ingestion helpers
# --------------------------------------------------------------------------- #
def test_build_chunks_extracts_chunks_and_null_vectors(
    tmp_path,
    no_embeddings: None,
) -> None:
    path = tmp_path / "syllabus.txt"
    path.write_text(_INGESTED, encoding="utf-8")

    drafts, vectors = syllabus_service._build_chunks(path)

    assert len(drafts) == 1
    assert vectors == [None]


def test_build_chunks_rejects_empty_document(tmp_path, no_embeddings: None) -> None:
    path = tmp_path / "syllabus.txt"
    path.write_text("   \n", encoding="utf-8")

    with pytest.raises(FileProcessingError):
        syllabus_service._build_chunks(path)


def test_embed_uses_model_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(syllabus_service.embedding, "is_available", lambda: True)
    monkeypatch.setattr(
        syllabus_service.embedding, "embed_texts", lambda texts: [[0.1, 0.2] for _ in texts]
    )

    assert syllabus_service._embed(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
