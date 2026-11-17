"""OpenAI-compatible LLM client.

Talks to any OpenAI-compatible ``/chat/completions`` endpoint (the default is a
free-tier provider). Calls retry on transient failures with bounded exponential
backoff and honour ``Retry-After`` on rate limits. When no API key is configured
the client reports itself unavailable so callers can fall back to an offline stub.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from src.settings import (
    LLM_BACKOFF_BASE_SECONDS,
    LLM_BACKOFF_MAX_SECONDS,
    LLM_MAX_ATTEMPTS,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT_SECONDS,
    settings,
)
from src.utils.exceptions import AppError, LLMError, RateLimitError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_RETRYABLE_STATUS = frozenset({500, 502, 503, 504})
_RATE_LIMIT_STATUS = 429
# A Retry-After longer than this is not worth blocking the request on.
_MAX_RETRY_AFTER_SECONDS = 60.0


@dataclass(slots=True)
class LLMResult:
    """The text output of one LLM call plus its usage metadata."""

    text: str
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int | None = None


def is_available() -> bool:
    """Return whether a real LLM endpoint is configured."""

    return bool(settings.llm_api_key.strip())


class UsageSink(Protocol):
    """Receives the outcome of every LLM call so usage can be recorded (FR-011)."""

    async def record(
        self, *, task: str, result: LLMResult | None, error: BaseException | None
    ) -> None: ...


async def complete_json(
    *,
    system: str,
    user: str,
    task: str,
    temperature: float | None = None,
    max_tokens: int = LLM_MAX_TOKENS,
    sink: UsageSink | None = None,
) -> LLMResult:
    """Run a chat completion, reporting the outcome to ``sink`` when given.

    The caller is expected to have checked :func:`is_available` first; this
    raises :class:`LLMError` when no endpoint is configured. ``sink`` is a
    best-effort traceability hook (FR-011) and never affects the result.
    """

    try:
        result = await _complete(
            system=system,
            user=user,
            task=task,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except AppError as exc:
        await _emit(sink, task=task, result=None, error=exc)
        raise
    await _emit(sink, task=task, result=result, error=None)
    return result


async def _complete(
    *,
    system: str,
    user: str,
    task: str,
    temperature: float | None = None,
    max_tokens: int = LLM_MAX_TOKENS,
) -> LLMResult:
    """Perform the chat completion with retries (see :func:`complete_json`)."""

    if not is_available():
        raise LLMError("No LLM API key is configured; install one or use the offline stub.")

    payload = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature if temperature is None else temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"

    last_error: Exception | None = None
    async with _new_client() as client:
        for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
            started = time.perf_counter()
            try:
                response = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "LLM request failed",
                    extra={"task": task, "attempt": attempt, "error": type(exc).__name__},
                )
                await _sleep_backoff(attempt)
                continue

            latency_ms = int((time.perf_counter() - started) * 1000)

            if response.status_code == _RATE_LIMIT_STATUS:
                delay = _retry_after_seconds(response)
                if delay is not None and delay <= _MAX_RETRY_AFTER_SECONDS:
                    logger.warning(
                        "LLM rate limited; retrying",
                        extra={"task": task, "attempt": attempt, "retry_after_s": delay},
                    )
                    await asyncio.sleep(delay)
                    continue
                raise RateLimitError(
                    "The LLM provider rate-limited the request. Please retry shortly.",
                    detail={"task": task},
                )

            if response.status_code in _RETRYABLE_STATUS:
                last_error = LLMError(
                    f"LLM provider returned {response.status_code}.", detail={"task": task}
                )
                logger.warning(
                    "LLM transient error",
                    extra={"task": task, "attempt": attempt, "status": response.status_code},
                )
                await _sleep_backoff(attempt)
                continue

            if response.status_code >= 400:
                raise LLMError(
                    f"LLM request rejected with status {response.status_code}.",
                    detail={"task": task, "status": response.status_code},
                )

            return _parse_response(response, latency_ms=latency_ms)

    raise LLMError(
        "The LLM provider is unreachable after several attempts.",
        detail={"task": task, "error": str(last_error)},
    )


async def _emit(
    sink: UsageSink | None,
    *,
    task: str,
    result: LLMResult | None,
    error: BaseException | None,
) -> None:
    """Report a call outcome to ``sink``, swallowing any recording failure.

    Traceability recording (FR-011) is best-effort: a broken sink must never
    turn a successful LLM call into a failed one, so exceptions are logged and
    ignored.
    """

    if sink is None:
        return
    try:
        await sink.record(task=task, result=result, error=error)
    except Exception as exc:  # noqa: BLE001 - recording must never break the call
        logger.warning(
            "Usage sink failed",
            extra={"task": task, "error": type(exc).__name__},
        )


def extract_json_object(text: str) -> dict[str, object]:
    """Parse a JSON object out of an LLM response, tolerating code fences.

    Models frequently wrap JSON in ``` fences or add a sentence around it, so
    the outermost ``{...}`` span is extracted before parsing.
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise LLMError("The LLM response did not contain a JSON object.")

    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError("The LLM response was not valid JSON.", detail={"error": str(exc)}) from exc

    if not isinstance(parsed, dict):
        raise LLMError("The LLM response was not a JSON object.")
    return parsed


def _new_client() -> httpx.AsyncClient:
    """Create the HTTP client used for LLM calls (patchable in tests)."""

    return httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS)


async def _sleep_backoff(attempt: int) -> None:
    """Sleep for a bounded exponential backoff before the next attempt."""

    delay = min(LLM_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), LLM_BACKOFF_MAX_SECONDS)
    await asyncio.sleep(delay)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Return the ``Retry-After`` delay in seconds, when present and numeric."""

    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_response(response: httpx.Response, *, latency_ms: int) -> LLMResult:
    """Extract the message text and token usage from a chat-completions body."""

    try:
        body = response.json()
        choice = body["choices"][0]
        text = choice["message"]["content"] or ""
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError("The LLM response had an unexpected shape.") from exc

    usage = body.get("usage") or {}
    return LLMResult(
        text=text,
        model=body.get("model") or settings.llm_model,
        provider=settings.llm_base_url,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        latency_ms=latency_ms,
    )
