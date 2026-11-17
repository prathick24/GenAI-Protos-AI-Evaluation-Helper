"""Exception handling: error codes, custom exceptions, and error responses."""

from __future__ import annotations

from src.utils.exceptions import error_codes
from src.utils.exceptions.error_responses import ErrorResponse
from src.utils.exceptions.exceptions import (
    AppError,
    ConflictError,
    DatabaseError,
    FileProcessingError,
    LLMError,
    NotFoundError,
    OcrError,
    RateLimitError,
    RequestValidationError,
)

__all__ = [
    "AppError",
    "ConflictError",
    "DatabaseError",
    "ErrorResponse",
    "FileProcessingError",
    "LLMError",
    "NotFoundError",
    "OcrError",
    "RateLimitError",
    "RequestValidationError",
    "error_codes",
]
