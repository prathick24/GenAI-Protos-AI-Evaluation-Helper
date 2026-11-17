"""Standardized error response models returned by the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.utils.exceptions.exceptions import AppError


class ErrorResponse(BaseModel):
    """Consistent error body for every failed request."""

    error: str
    message: str
    correlation_id: str | None = None
    detail: dict[str, Any] | None = None

    @classmethod
    def from_error(cls, error: AppError, correlation_id: str | None = None) -> ErrorResponse:
        """Build a response from an :class:`AppError`."""

        return cls(
            error=error.error_code,
            message=error.message,
            correlation_id=correlation_id,
            detail=error.detail,
        )
