"""External and local model clients (LLM, embeddings, OCR)."""

from __future__ import annotations

from src.client import embedding, llm, ocr

__all__ = ["embedding", "llm", "ocr"]
