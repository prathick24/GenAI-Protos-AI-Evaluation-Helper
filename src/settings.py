"""Application settings and shared constants for the AI Evaluation Helper.

Configuration is loaded from environment variables / `.env` and validated with
Pydantic Settings. Business constants (status vocabularies, limits, dimensions)
live here so no other layer defines them locally.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
APP_NAME = "AI Evaluation Helper"
APP_VERSION = "0.1.0"
API_V1_PREFIX = "/api/v1"

# --------------------------------------------------------------------------- #
# Embeddings / retrieval
# --------------------------------------------------------------------------- #
# Output dimensionality of sentence-transformers/all-MiniLM-L6-v2.
EMBEDDING_DIM = 384
DEFAULT_TOP_K = 5

# --------------------------------------------------------------------------- #
# Upload limits
# --------------------------------------------------------------------------- #
MAX_SYLLABUS_FILE_MB = 20
MAX_ANSWER_SHEET_FILE_MB = 20
ALLOWED_SYLLABUS_EXTENSIONS = frozenset({".pdf", ".txt"})
ALLOWED_ANSWER_SHEET_EXTENSIONS = frozenset({".pdf", ".txt", ".png", ".jpg", ".jpeg"})

# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
# Largest slice of raw syllabus text kept in one chunk before it is split.
MAX_SYLLABUS_CHUNK_CHARS = 1500
# Sub-directory (under STORAGE_DIR) where uploaded syllabus files are persisted.
SYLLABUS_STORAGE_SUBDIR = "syllabus"

# --------------------------------------------------------------------------- #
# Answer sheet ingestion (FR-004)
# --------------------------------------------------------------------------- #
# Sub-directory (under STORAGE_DIR) where uploaded answer sheets are persisted.
ANSWER_SHEET_STORAGE_SUBDIR = "answer_sheets"
# Extensions that are always treated as scanned images and routed through OCR.
IMAGE_ANSWER_SHEET_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})

# --------------------------------------------------------------------------- #
# Question paper generation (FR-002)
# --------------------------------------------------------------------------- #
# Raw syllabus text budget per generation call; oversized units are split into
# sections (the sub-topic loop) that each stay within this budget.
MAX_PAPER_PROMPT_CHARS = 12000
# Default marking length for a generated question paper.
PAPER_GENERATION_TEMPERATURE = 0.3

# --------------------------------------------------------------------------- #
# Question paper validation (FR-003)
# --------------------------------------------------------------------------- #
# Temperature for the advisory (LLM) validation review; the authoritative
# checks are deterministic and never call the LLM.
PAPER_VALIDATION_TEMPERATURE = 0.0
# Two questions whose normalised word sets overlap at or above this ratio are
# reported as near-duplicates.
PAPER_DUPLICATE_SIMILARITY_THRESHOLD = 0.8
# When the first validation fails, the failing questions are sent back to the
# generator to be rewritten and then re-validated, at most this many times.
PAPER_REPAIR_MAX_ATTEMPTS = 2

# --------------------------------------------------------------------------- #
# Chat assistant (FR-010)
# --------------------------------------------------------------------------- #
# Temperature for the intent router that turns a teacher's message into a plan.
CHAT_ROUTER_TEMPERATURE = 0.0
# Temperature for the grounded Q&A writer; the spec fixes it at <= 0.2.
CHAT_ASSISTANT_TEMPERATURE = 0.2

# --------------------------------------------------------------------------- #
# Answer sheet grading (FR-006)
# --------------------------------------------------------------------------- #
# Temperature for rubric-based grading; the spec bounds it to 0.0-0.2.
SHEET_GRADING_TEMPERATURE = 0.1
# Marks are awarded to this many decimal places (cents).
GRADING_MARKS_PRECISION = 2
# Temperature for the per-student feedback writer (FR-008).
FEEDBACK_TEMPERATURE = 0.3

# LLM call behaviour: bounded exponential backoff, honouring Retry-After.
LLM_MAX_ATTEMPTS = 5
LLM_BACKOFF_BASE_SECONDS = 2.0
LLM_BACKOFF_MAX_SECONDS = 10.0
LLM_TIMEOUT_SECONDS = 60.0
LLM_MAX_TOKENS = 4096


# --------------------------------------------------------------------------- #
# Status vocabularies (persisted as plain strings)
# --------------------------------------------------------------------------- #
class SyllabusStatus(StrEnum):
    """Lifecycle of a syllabus ingestion."""

    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class InputType(StrEnum):
    """How a document's content was obtained."""

    TYPED = "typed"
    SCANNED = "scanned"


class UploadInputType(StrEnum):
    """Input type requested at upload time; ``auto`` infers it from the file."""

    TYPED = "typed"
    SCANNED = "scanned"
    AUTO = "auto"


class OcrStatus(StrEnum):
    """OCR processing state for a scanned document."""

    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class PaperStatus(StrEnum):
    """Lifecycle of a generated question paper."""

    DRAFT = "draft"
    VALIDATED = "validated"
    APPROVED = "approved"
    FAILED = "failed"


class QuestionType(StrEnum):
    """Supported question formats in a blueprint."""

    MCQ = "mcq"
    SHORT = "short"
    LONG = "long"


class Difficulty(StrEnum):
    """Difficulty bucket of a generated question."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ValidationStatus(StrEnum):
    """Per-question validation state (populated fully by FR-003)."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class SheetStatus(StrEnum):
    """Lifecycle of an answer sheet."""

    UPLOADED = "uploaded"
    QUEUED = "queued"
    GRADING = "grading"
    GRADED = "graded"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    FAILED = "failed"


class VerifierStatus(StrEnum):
    """Independent verification verdict for a mark."""

    AGREED = "agreed"
    FLAGGED = "flagged"
    ADJUSTED = "adjusted"


class GradedBy(StrEnum):
    """Who produced a mark: the grading agent or a teacher override."""

    AGENT = "agent"
    TEACHER = "teacher"


class MessageRole(StrEnum):
    """Role of a chat message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChangeOp(StrEnum):
    """A single paper change the chat assistant can interpret and apply (FR-010).

    Text authoring ops (``REWRITE``) and blueprint reconciliation additions go
    through the LLM; the remaining ops are applied deterministically.
    """

    REWRITE = "rewrite"
    DROP = "drop"
    REORDER = "reorder"
    SET_DURATION = "set_duration"
    ADJUST_BLUEPRINT = "adjust_blueprint"


class AgentRunStatus(StrEnum):
    """Lifecycle of a single agent task."""

    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Runtime configuration loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: str = "local"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database (PostgreSQL 18 + pgvector, async driver)
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/aeh"

    # LLM (free-tier, OpenAI-compatible). Blank key enables the local stub provider.
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.2

    # Embeddings (local)
    embedding_model: str = "all-MiniLM-L6-v2"

    # OCR (local default, vision model optional)
    ocr_provider: str = "local"
    ocr_language: str = "en"
    # OpenAI-compatible vision endpoint used by the "cloud" provider. The
    # defaults target a local Ollama server; point these at any vision API.
    ocr_vision_base_url: str = "http://localhost:11434/v1"
    ocr_vision_api_key: str = "ollama"
    ocr_vision_model: str = "qwen2.5vl:3b"
    ocr_vision_api: str = "ollama"
    ocr_vision_timeout_seconds: float = 600.0

    # Retrieval / grading
    top_k: int = DEFAULT_TOP_K
    grade_concurrency: int = 1

    # Storage
    storage_dir: str = "./storage"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""

    return Settings()


settings = get_settings()
