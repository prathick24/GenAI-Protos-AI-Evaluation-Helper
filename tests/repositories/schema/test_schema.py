"""Tests for the ORM schema metadata (no database connection required)."""

from __future__ import annotations

import pytest

from src.repositories.schema import (
    AgentRun,
    Answer,
    AnswerSheet,
    Base,
    Conversation,
    ConversationMessage,
    ErrorLog,
    Exam,
    Feedback,
    Mark,
    ModelCost,
    Question,
    QuestionPaper,
    RubricCriterion,
    Student,
    Syllabus,
    SyllabusChunk,
    User,
)
from src.settings import EMBEDDING_DIM

EXPECTED_TABLES = {
    "users",
    "students",
    "syllabus",
    "syllabus_chunks",
    "exams",
    "question_papers",
    "questions",
    "rubric_criteria",
    "answer_sheets",
    "answers",
    "marks",
    "feedback",
    "conversations",
    "conversation_messages",
    "error_logs",
    "model_cost",
    "agent_runs",
}

MODEL_CLASSES = [
    User,
    Student,
    Syllabus,
    SyllabusChunk,
    Exam,
    QuestionPaper,
    Question,
    RubricCriterion,
    AnswerSheet,
    Answer,
    Mark,
    Feedback,
    Conversation,
    ConversationMessage,
    ErrorLog,
    ModelCost,
    AgentRun,
]


def test_metadata_declares_all_seventeen_tables() -> None:
    assert set(Base.metadata.tables.keys()) == EXPECTED_TABLES


def test_every_model_is_mapped_to_a_table() -> None:
    for model in MODEL_CLASSES:
        assert model.__tablename__ in EXPECTED_TABLES
        assert model.__table__ is Base.metadata.tables[model.__tablename__]


@pytest.mark.parametrize("model", MODEL_CLASSES)
def test_audit_columns_present(model: type) -> None:
    columns = model.__table__.columns.keys()
    for column in ("id", "created_at", "created_by", "updated_at", "updated_by", "is_active"):
        assert column in columns, f"{model.__name__} is missing '{column}'"


def test_syllabus_chunk_embedding_dimension() -> None:
    embedding = SyllabusChunk.__table__.columns["embedding"]
    assert embedding.type.dim == EMBEDDING_DIM


def test_marks_answer_id_is_unique() -> None:
    assert Mark.__table__.columns["answer_id"].unique is True


def test_syllabus_chunk_unique_constraint_on_syllabus_and_index() -> None:
    constraints = {c.name for c in SyllabusChunk.__table__.constraints}
    assert "uq_syllabus_chunks_syllabus_chunk" in constraints


def test_question_paper_syllabus_foreign_key() -> None:
    fk_targets = {fk.target_fullname for fk in QuestionPaper.__table__.foreign_keys}
    assert "syllabus.id" in fk_targets


def test_question_source_chunk_foreign_key() -> None:
    fk_targets = {fk.target_fullname for fk in Question.__table__.foreign_keys}
    assert "syllabus_chunks.id" in fk_targets
