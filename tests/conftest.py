"""Shared test fixtures.

Traceability writes (FR-011) open their own short-lived sessions against the
real engine. Unit tests run without a database, so the recorder's session
factory is replaced with an in-memory no-op by default: recording still runs
end-to-end but never touches a database or blocks on a connection. A test that
wants to assert on the writes overrides ``observability_repository.SessionFactory``
itself.
"""

from __future__ import annotations

import pytest

from src.repositories import observability_repository


class _NullSession:
    """A stand-in session whose writes are discarded."""

    async def __aenter__(self) -> _NullSession:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def add(self, obj: object) -> None:
        return None

    async def commit(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _isolate_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep best-effort traceability writes out of the test database."""

    def _factory() -> _NullSession:
        return _NullSession()

    monkeypatch.setattr(observability_repository, "SessionFactory", _factory)
