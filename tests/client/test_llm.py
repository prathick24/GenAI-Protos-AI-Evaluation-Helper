"""Tests for the OpenAI-compatible LLM client."""

from __future__ import annotations

import httpx
import pytest

from src.client import llm
from src.utils.exceptions import LLMError, RateLimitError

_OK_BODY = {
    "model": "test-model",
    "choices": [{"message": {"content": '{"questions": []}'}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


@pytest.fixture(autouse=True)
def _configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend an API key is present and make backoff instantaneous."""

    monkeypatch.setattr(llm.settings, "llm_api_key", "test-key")
    monkeypatch.setattr(llm.settings, "llm_base_url", "https://example.test/v1")

    async def _no_sleep(_attempt: int) -> None:
        return None

    monkeypatch.setattr(llm, "_sleep_backoff", _no_sleep)


def _use_handler(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    monkeypatch.setattr(
        llm, "_new_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def test_is_available_false_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm.settings, "llm_api_key", "   ")
    assert llm.is_available() is False


def test_is_available_true_with_key() -> None:
    assert llm.is_available() is True


def test_complete_json_returns_text_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_handler(monkeypatch, lambda request: httpx.Response(200, json=_OK_BODY))

    result = __import__("asyncio").run(
        llm.complete_json(system="s", user="u", task="t")
    )

    assert result.text == '{"questions": []}'
    assert result.total_tokens == 15
    assert result.model == "test-model"
    assert result.latency_ms is not None


def test_complete_json_retries_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json=_OK_BODY)

    _use_handler(monkeypatch, handler)

    result = __import__("asyncio").run(llm.complete_json(system="s", user="u", task="t"))

    assert calls["count"] == 2
    assert result.total_tokens == 15


def test_complete_json_raises_on_long_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_handler(
        monkeypatch,
        lambda request: httpx.Response(429, headers={"retry-after": "120"}, json={}),
    )

    with pytest.raises(RateLimitError):
        __import__("asyncio").run(llm.complete_json(system="s", user="u", task="t"))


def test_complete_json_raises_on_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_handler(monkeypatch, lambda request: httpx.Response(400, json={"error": "bad"}))

    with pytest.raises(LLMError):
        __import__("asyncio").run(llm.complete_json(system="s", user="u", task="t"))


def test_extract_json_object_handles_fences_and_noise() -> None:
    text = 'Here you go:\n```json\n{"questions": [{"a": 1}]}\n```\nThanks!'
    assert llm.extract_json_object(text) == {"questions": [{"a": 1}]}


def test_extract_json_object_plain_object() -> None:
    assert llm.extract_json_object('  {"ok": true}  ') == {"ok": True}


def test_extract_json_object_without_json_raises() -> None:
    with pytest.raises(LLMError):
        llm.extract_json_object("no json here")


def test_extract_json_object_malformed_raises() -> None:
    with pytest.raises(LLMError):
        llm.extract_json_object("{not valid json}")
