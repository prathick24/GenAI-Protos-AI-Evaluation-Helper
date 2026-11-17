"""Domain-specific exception classes.

All application errors derive from :class:`AppError`, which carries an HTTP
status code, a stable error code and an optional structured detail payload.
"""

from __future__ import annotations

from typing import Any

from src.utils.exceptions import error_codes as code


class AppError(Exception):
    """Base class for all application errors."""

    status_code: int = 500
    error_code: str = code.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        status_code: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if error_code is not None:
            self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code
        self.detail = detail


class NotFoundError(AppError):
    """A requested resource does not exist."""

    status_code = 404
    error_code = code.NOT_FOUND


class RequestValidationError(AppError):
    """The request payload failed validation."""

    status_code = 422
    error_code = code.VALIDATION_ERROR


class ConflictError(AppError):
    """The request conflicts with the current state of the resource."""

    status_code = 409
    error_code = code.CONFLICT


class DatabaseError(AppError):
    """The data store is unavailable or the query failed."""

    status_code = 503
    error_code = code.DATABASE_ERROR


class FileProcessingError(AppError):
    """An uploaded file could not be read, parsed or validated."""

    status_code = 400
    error_code = code.FILE_PROCESSING_ERROR


class OcrError(AppError):
    """OCR / handwriting recognition failed for a document."""

    status_code = 422
    error_code = code.OCR_ERROR


class LLMError(AppError):
    """The LLM provider returned an error or was unreachable."""

    status_code = 502
    error_code = code.LLM_ERROR


class RateLimitError(LLMError):
    """The LLM provider rate-limited the request."""

    status_code = 429
    error_code = code.LLM_RATE_LIMITED
