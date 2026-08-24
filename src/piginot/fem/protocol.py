"""Backend protocol for interchangeable S7 FEM implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from piginot.fem.models import FemAttempt, FemRequest


@runtime_checkable
class FemBackend(Protocol):
    backend_name: str

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        """Return one visible outcome for every request."""


@runtime_checkable
class IndependentSolverAdapter(Protocol):
    """Optional boundary for verification with a separately implemented solver.

    An adapter translates canonical :class:`FemRequest` values to an independent
    solver and translates every outcome back to :class:`FemAttempt`, preserving
    input order and failed attempts.  ``adapter_id`` must identify the adapter and
    independent solver version used for provenance.  Core intentionally provides
    this protocol only; an external package owns any concrete implementation.
    """

    adapter_id: str

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        """Return one independently computed outcome for every request."""


__all__ = ("FemBackend", "IndependentSolverAdapter")
