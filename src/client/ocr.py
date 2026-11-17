"""Pluggable OCR client for scanned answer sheets (FR-004).

Typed PDFs and text files are extracted directly by :mod:`src.utils.documents`;
only scanned images reach this client. The provider is selected by
``settings.ocr_provider`` and defaults to a **local-first** engine (RapidOCR on
ONNX Runtime) so the core demo runs offline with zero cost. The ``cloud``
provider is an OpenAI-compatible vision model (a local Ollama server by default)
and is the right choice for handwriting.
"""

from __future__ import annotations

import base64
import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import httpx

from src.settings import settings
from src.utils.exceptions import OcrError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class OcrResult:
    """Recognised text for one document, with a confidence summary."""

    text: str
    provider: str
    confidence: float | None = None

    def confidence_map(self) -> dict[str, object]:
        """Return a JSONB-serialisable summary of recognition confidence."""

        payload: dict[str, object] = {"provider": self.provider}
        if self.confidence is not None:
            payload["mean"] = round(self.confidence, 4)
        return payload


class OcrProvider(Protocol):
    """Interface every OCR backend implements."""

    name: str

    def recognise(self, path: Path) -> OcrResult:
        """Return the recognised text for the document at ``path``."""
        ...


class LocalOcrProvider:
    """Local-first provider backed by RapidOCR (an ML extra)."""

    name = "local"

    def recognise(self, path: Path) -> OcrResult:
        """Run the local OCR engine over ``path`` and collect the recognised lines."""

        engine = _engine()
        try:
            images = _page_images(path)
            lines: list[str] = []
            scores: list[float] = []
            for image in images:
                raw, _ = engine(image)
                for entry in raw or []:
                    text, score = str(entry[1]), float(entry[2])
                    if text.strip():
                        lines.append(text)
                        scores.append(score)
        except Exception as exc:  # noqa: BLE001 - surface any engine failure uniformly
            raise OcrError(f"OCR failed for '{path.name}': {exc}") from exc

        mean = sum(scores) / len(scores) if scores else None
        return OcrResult(text="\n".join(lines).strip(), provider=self.name, confidence=mean)


def _page_images(path: Path) -> list[str | bytes]:
    """Return the OCR inputs for ``path``, rendering each PDF page to PNG bytes."""

    if path.suffix.lower() != ".pdf":
        return [str(path)]
    return _pdf_page_pngs(path)


def _pdf_page_pngs(path: Path) -> list[bytes]:
    """Render every page of a PDF to a PNG at 200 DPI."""

    import pymupdf

    images: list[bytes] = []
    with pymupdf.open(path) as document:
        for page in document:
            images.append(page.get_pixmap(dpi=200).tobytes("png"))
    return images


_IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _page_pngs(path: Path) -> list[tuple[bytes, str]]:
    """Return ``(png_or_image_bytes, mime_type)`` for each page of ``path``."""

    if path.suffix.lower() == ".pdf":
        return [(image, "image/png") for image in _pdf_page_pngs(path)]
    mime = _IMAGE_MIME_TYPES.get(path.suffix.lower(), "image/png")
    return [(path.read_bytes(), mime)]


_VISION_SYSTEM_PROMPT = (
    "You are a meticulous OCR engine for exam answer sheets. Transcribe the "
    "text exactly as it appears, including handwriting, question numbers and "
    "answers. Preserve line breaks. Never answer, explain, summarise or correct "
    "the text. If a word is illegible, transcribe your best guess."
)
_VISION_USER_PROMPT = (
    "Transcribe every word visible on this page, exactly as written. Output only "
    "the transcription."
)


class CloudVisionOcrProvider:
    """OpenAI-compatible vision model (local Ollama by default) for handwriting."""

    name = "cloud"

    def recognise(self, path: Path) -> OcrResult:
        """Transcribe each page with the configured vision model."""

        logger.info(
            "Transcribing with vision OCR",
            extra={"model": settings.ocr_vision_model},
        )
        try:
            pages = _page_pngs(path)
            text = "\n".join(_transcribe_page(image, mime) for image, mime in pages)
        except OcrError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any engine failure uniformly
            raise OcrError(f"OCR failed for '{path.name}': {exc}") from exc
        return OcrResult(text=text.strip(), provider=self.name)


def _transcribe_page(image: bytes, mime: str) -> str:
    """Send one rendered page to the configured vision API and return its text."""

    if settings.ocr_vision_api.strip().lower() == "ollama":
        return _transcribe_with_ollama(image)
    return _transcribe_with_openai(image, mime)


def _transcribe_with_ollama(image: bytes) -> str:
    """Use Ollama's native ``/api/chat`` so the context window can be raised."""

    base = settings.ocr_vision_base_url.rstrip("/")
    for suffix in ("/v1", "/api"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    payload = {
        "model": settings.ocr_vision_model,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 4096},
        "messages": [
            {"role": "system", "content": _VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _VISION_USER_PROMPT,
                "images": [base64.b64encode(image).decode()],
            },
        ],
    }
    body = _post_vision(f"{base}/api/chat", payload, headers={})
    try:
        content = body["message"]["content"] or ""
    except (KeyError, TypeError) as exc:
        raise OcrError("The vision OCR response had an unexpected shape.") from exc
    return str(content).strip()


def _transcribe_with_openai(image: bytes, mime: str) -> str:
    """Use the OpenAI-compatible ``/chat/completions`` contract."""

    encoded = base64.b64encode(image).decode()
    payload = {
        "model": settings.ocr_vision_model,
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [
            {"role": "system", "content": _VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    },
                ],
            },
        ],
    }
    headers = {"Authorization": f"Bearer {settings.ocr_vision_api_key}"}
    url = f"{settings.ocr_vision_base_url.rstrip('/')}/chat/completions"
    body = _post_vision(url, payload, headers=headers)
    try:
        content = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise OcrError("The vision OCR response had an unexpected shape.") from exc
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content).strip()


def _post_vision(
    url: str, payload: dict[str, object], *, headers: dict[str, str]
) -> dict[str, object]:
    """POST a vision request, translating transport and status failures."""

    headers = {"Content-Type": "application/json", **headers}
    try:
        with _vision_client() as client:
            response = client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise OcrError(f"The vision OCR endpoint is unreachable: {exc}") from exc

    if response.status_code == 404:
        raise OcrError(
            f"Vision OCR model '{settings.ocr_vision_model}' is not available at "
            f"{settings.ocr_vision_base_url}. Pull it with: "
            f"ollama pull {settings.ocr_vision_model}"
        )
    if response.status_code >= 400:
        raise OcrError(
            f"The vision OCR request failed with status {response.status_code}."
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise OcrError("The vision OCR response was not valid JSON.") from exc
    if not isinstance(body, dict):
        raise OcrError("The vision OCR response had an unexpected shape.")
    return body


def _vision_client() -> httpx.Client:
    """Create the HTTP client used for vision OCR calls (patchable in tests)."""

    return httpx.Client(timeout=settings.ocr_vision_timeout_seconds)


@lru_cache(maxsize=1)
def _engine() -> Any:
    """Load and cache the local OCR engine, or fail with an actionable message."""

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise OcrError(
            "Local OCR requires RapidOCR on ONNX Runtime. Install it with: "
            "uv sync --extra ml"
        ) from exc

    logger.info("Loading local OCR engine", extra={"engine": "rapidocr"})
    return RapidOCR()


def is_available() -> bool:
    """Return whether the local OCR engine and its PDF renderer can be imported."""

    return all(
        importlib.util.find_spec(module) is not None
        for module in ("rapidocr_onnxruntime", "pymupdf")
    )


def get_provider(name: str | None = None) -> OcrProvider:
    """Return the OCR provider named by configuration (default: local)."""

    selected = (name or settings.ocr_provider or LocalOcrProvider.name).strip().lower()
    if selected in {CloudVisionOcrProvider.name, "vision", "ollama"}:
        return CloudVisionOcrProvider()
    if selected != LocalOcrProvider.name:
        logger.warning(
            "Unknown OCR provider; falling back to the local engine",
            extra={"provider": selected},
        )
    return LocalOcrProvider()
