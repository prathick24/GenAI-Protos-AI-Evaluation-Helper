"""Local sentence-embedding client.

The embedding model (``all-MiniLM-L6-v2``) runs on the local CPU through
sentence-transformers, so it needs no API key and no network access. The model
is loaded once and cached; if the ML extras are not installed the client
reports itself unavailable and callers store chunks without vectors.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> Any | None:
    """Load and cache the embedding model, or ``None`` when unavailable."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence-transformers is not installed; embeddings are disabled",
            extra={"hint": "uv sync --extra ml"},
        )
        return None

    logger.info("Loading embedding model", extra={"model": settings.embedding_model})
    return SentenceTransformer(settings.embedding_model)


def is_available() -> bool:
    """Return whether the local embedding model can be used."""

    return _load_model() is not None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one 384-dimensional embedding per input text."""

    model = _load_model()
    if model is None:
        raise RuntimeError("Embedding model is unavailable; install the ML extras.")
    vectors = model.encode(texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]
