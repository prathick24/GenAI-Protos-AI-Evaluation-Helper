"""Centralized JSON structured logging.

Every log record is emitted as a single JSON object so logs are machine
parseable and privacy-safe. Custom fields passed via ``extra=`` are included
automatically (e.g. ``correlation_id``, ``task_id``, ``latency_ms``).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

# LogRecord attributes that are part of the logging machinery and must not be
# copied into the JSON payload as custom fields.
_RESERVED_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)

# Third-party loggers that are noisy at INFO/DEBUG during normal operation.
_NOISY_LOGGERS = ("httpx", "httpcore", "uvicorn.access", "asyncio", "multipart")


class JsonFormatter(logging.Formatter):
    """Render a :class:`logging.LogRecord` as a compact JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "filename": record.filename,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            payload["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value),
                "stacktrace": self.formatException(record.exc_info),
            }

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger.

    Safe to call more than once; existing handlers are replaced.
    """

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str = "aeh") -> logging.Logger:
    """Return a named logger."""

    return logging.getLogger(name)


logger = get_logger()
