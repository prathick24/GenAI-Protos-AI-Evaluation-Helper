"""Tests for the pluggable OCR client (FR-004)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.client import ocr
from src.settings import settings
from src.utils.exceptions import OcrError


class _FakeEngine:
    """Stand-in for RapidOCR returning a fixed result structure."""

    def __init__(self, scores: tuple[float, ...] = (0.9, 0.7)) -> None:
        self._scores = scores

    def __call__(self, image: object) -> tuple[list[list[object]], None]:
        return (
            [
                [[[0, 0], [1, 0], [1, 1], [0, 1]], "Hello", self._scores[0]],
                [[[0, 1], [1, 1], [1, 2], [0, 2]], "World", self._scores[1]],
            ],
            None,
        )


class _BoomEngine:
    """Stand-in engine that always fails."""

    def __call__(self, image: object) -> None:
        raise RuntimeError("engine exploded")


class _FakeResponse:
    """Minimal stand-in for an httpx response."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        body: dict[str, object] | None = None,
        text: str = "Section A\n1. RAM",
    ) -> None:
        self.status_code = status_code
        self._body = body or {"choices": [{"message": {"content": text}}]}

    def json(self) -> dict[str, object]:
        return self._body


class _FakeClient:
    """Minimal stand-in for the vision HTTP client."""

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.request: dict[str, object] = {}

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
    ) -> _FakeResponse:
        self.request = {"url": url, "json": json, "headers": headers}
        return self.response


def test_confidence_map_reports_provider_and_mean() -> None:
    result = ocr.OcrResult(text="hi", provider="local", confidence=0.81234)

    assert result.confidence_map() == {"provider": "local", "mean": 0.8123}


def test_confidence_map_without_confidence_keeps_provider() -> None:
    result = ocr.OcrResult(text="hi", provider="local")

    assert result.confidence_map() == {"provider": "local"}


def test_get_provider_defaults_to_local() -> None:
    assert isinstance(ocr.get_provider("local"), ocr.LocalOcrProvider)


def test_get_provider_selects_cloud() -> None:
    assert isinstance(ocr.get_provider("cloud"), ocr.CloudVisionOcrProvider)


def test_get_provider_falls_back_for_unknown_name() -> None:
    assert isinstance(ocr.get_provider("nope"), ocr.LocalOcrProvider)


def test_cloud_provider_uses_the_ollama_native_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(
        _FakeResponse(body={"message": {"content": "Section A\n1. RAM"}})
    )
    monkeypatch.setattr(settings, "ocr_vision_api", "ollama")
    monkeypatch.setattr(ocr, "_vision_client", lambda: client)
    monkeypatch.setattr(ocr, "_page_pngs", lambda path: [(b"png-bytes", "image/png")])

    result = ocr.CloudVisionOcrProvider().recognise(Path("sheet.pdf"))

    assert result.text == "Section A\n1. RAM"
    assert str(client.request["url"]).endswith("/api/chat")
    payload = client.request["json"]
    assert isinstance(payload, dict)
    assert payload["options"]["num_ctx"] == 16384  # type: ignore[index]
    assert payload["messages"][1]["images"] == ["cG5nLWJ5dGVz"]  # type: ignore[index]


def test_cloud_provider_sends_the_page_to_an_openai_vision_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(_FakeResponse())
    monkeypatch.setattr(settings, "ocr_vision_api", "openai")
    monkeypatch.setattr(ocr, "_vision_client", lambda: client)
    monkeypatch.setattr(ocr, "_page_pngs", lambda path: [(b"png-bytes", "image/png")])

    result = ocr.CloudVisionOcrProvider().recognise(Path("sheet.pdf"))

    assert result.text == "Section A\n1. RAM"
    assert result.provider == "cloud"
    assert str(client.request["url"]).endswith("/chat/completions")
    payload = client.request["json"]
    assert isinstance(payload, dict)
    image_part = payload["messages"][1]["content"][1]  # type: ignore[index]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert "Authorization" in client.request["headers"]  # type: ignore[operator]


def test_cloud_provider_reports_a_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(_FakeResponse(status_code=404))
    monkeypatch.setattr(ocr, "_vision_client", lambda: client)
    monkeypatch.setattr(ocr, "_page_pngs", lambda path: [(b"png-bytes", "image/png")])

    with pytest.raises(OcrError, match="ollama pull"):
        ocr.CloudVisionOcrProvider().recognise(Path("sheet.pdf"))


def test_local_provider_parses_engine_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "_engine", lambda: _FakeEngine())

    result = ocr.LocalOcrProvider().recognise(Path("sheet.png"))

    assert result.text == "Hello\nWorld"
    assert result.provider == "local"
    assert result.confidence == pytest.approx(0.8)


def test_local_provider_wraps_engine_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "_engine", lambda: _BoomEngine())

    with pytest.raises(OcrError):
        ocr.LocalOcrProvider().recognise(Path("sheet.png"))


def test_local_provider_requires_the_ml_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    ocr._engine.cache_clear()
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)

    with pytest.raises(OcrError):
        ocr.LocalOcrProvider().recognise(Path("sheet.png"))

    ocr._engine.cache_clear()


def test_is_available_returns_a_bool() -> None:
    assert isinstance(ocr.is_available(), bool)


def test_is_available_requires_the_pdf_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ocr.importlib.util,
        "find_spec",
        lambda name: object() if name == "rapidocr_onnxruntime" else None,
    )

    assert ocr.is_available() is False
