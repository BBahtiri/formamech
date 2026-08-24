"""Contract tests for the optional independent-solver seam."""

from __future__ import annotations

import pytest

from piginot.fem.backend import ScikitFemBackend
from piginot.fem.models import FemAttempt, FemRequest
from piginot.fem.protocol import FemBackend, IndependentSolverAdapter


class _DummyIndependentAdapter:
    adapter_id = "dummy-independent-solver-v1"

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        assert requests == ()
        return ()


class _MissingSolveMethod:
    adapter_id = "incomplete-independent-solver-v1"


class _MissingAdapterIdentity:
    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        return ()


def test_structural_adapter_can_solve_canonical_fem_requests() -> None:
    adapter = _DummyIndependentAdapter()

    assert isinstance(adapter, IndependentSolverAdapter)
    assert adapter.solve_many(()) == ()


def test_adapter_requires_both_identity_and_solve_method() -> None:
    assert not isinstance(_MissingSolveMethod(), IndependentSolverAdapter)
    assert not isinstance(_MissingAdapterIdentity(), IndependentSolverAdapter)


def test_core_primary_backend_is_not_an_independent_adapter() -> None:
    backend = ScikitFemBackend()

    assert isinstance(backend, FemBackend)
    assert not isinstance(backend, IndependentSolverAdapter)
    with pytest.raises(TypeError, match="Protocols cannot be instantiated"):
        IndependentSolverAdapter()
