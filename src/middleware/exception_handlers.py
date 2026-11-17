"""Exception handlers mapping application errors to standard error responses.

No stack traces, secrets or internal traces are ever returned to the client.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from fastapi.responses import JSONResponse

from src.middleware.correlation import get_correlation_id
from src.repositories.observability_repository import record_error
from src.utils.exceptions import AppError, ErrorResponse
from src.utils.exceptions import error_codes as code
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _correlation_id(request: Request) -> str | None:
    return getattr(request.state, "correlation_id", None) or get_correlation_id()


async def _record_failure(request: Request, *, message: str, **detail: object) -> None:
    """Best-effort write of one ``ERROR_LOGS`` row for a failed request (FR-011)."""

    await record_error(
        message=message,
        component=request.url.path,
        endpoint=request.url.path,
        method=request.method,
        detail=detail or None,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register handlers for application, validation and unhandled errors."""

    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        correlation_id = _correlation_id(request)
        logger.error(
            "Application error",
            extra={
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "correlation_id": correlation_id,
                "path": request.url.path,
                "method": request.method,
            },
        )
        await _record_failure(
            request,
            message=exc.message,
            error_code=exc.error_code,
            status_code=exc.status_code,
        )
        body = ErrorResponse.from_error(exc, correlation_id)
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(FastAPIRequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: FastAPIRequestValidationError
    ) -> JSONResponse:
        correlation_id = _correlation_id(request)
        body = ErrorResponse(
            error=code.VALIDATION_ERROR,
            message="Request validation failed.",
            correlation_id=correlation_id,
            detail={"errors": jsonable_encoder(exc.errors())},
        )
        await _record_failure(request, message="Request validation failed.", status_code=422)
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        correlation_id = _correlation_id(request)
        logger.exception(
            "Unhandled exception",
            extra={"correlation_id": correlation_id, "path": request.url.path},
        )
        await _record_failure(
            request,
            message=str(exc) or type(exc).__name__,
            exception_type=type(exc).__name__,
            status_code=500,
        )
        body = ErrorResponse(
            error=code.INTERNAL_ERROR,
            message="An unexpected error occurred.",
            correlation_id=correlation_id,
        )
        return JSONResponse(status_code=500, content=body.model_dump())
