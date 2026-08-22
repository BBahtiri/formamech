"""Backend protocol for interchangeable S7 FEM implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from piginot.fem.models import FemAttempt, FemRequest


@runtime_checkable
class FemBackend(Protocol):
    backend_name: str

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        """Return one visible outcome for every request."""


__all__ = ("FemBackend",)
