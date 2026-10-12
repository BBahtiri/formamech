"""S22 warm start: the solver can start from a given state; defaults unchanged."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from piginot.fem import ScikitFemBackend
from piginot.fem.linearization import FemLinearization
from tests.fem.test_nonlinear_solver import _refined_unit_square, _traction_request


def _request(order: int = 2):  # type: ignore[no-untyped-def]
    return replace(_traction_request(element_order=order), mesh=_refined_unit_square())


def test_the_cold_solve_is_unchanged_by_the_new_parameters() -> None:
    request = _request()
    backend = ScikitFemBackend()
    cold = backend.solve_many((request,))[0]
    assert cold.success and cold.result is not None
    steps = request.config.load_steps
    zero = np.zeros(FemLinearization(request).basis.N)
    explicit = backend.solve_from(
        request, zero, tuple(s / steps for s in range(1, steps + 1))
    )
    assert explicit.result_hash == cold.result_hash


@pytest.mark.parametrize("order", [1, 2])
def test_a_warm_start_converges_to_the_cold_solution_in_few_iterations(
    order: int,
) -> None:
    request = _request(order)
    backend = ScikitFemBackend()
    cold = backend.solve_many((request,))[0]
    assert cold.result is not None
    linearization = FemLinearization(request)
    solution = np.asarray(cold.result.displacement)
    x, y = linearization.dof_coordinates.T
    guess = solution + 1e-3 * np.column_stack(
        (x * np.sin(np.pi * y), y * np.cos(np.pi * x))
    )
    warm = backend.solve_from(request, linearization.vector(guess))
    assert warm.success and warm.result is not None
    assert len(warm.result.convergence_history) <= 4
    assert len(warm.result.convergence_history) < len(cold.result.convergence_history)
    difference = np.max(np.abs(np.asarray(warm.result.displacement) - solution))
    assert difference < 1e-9
    assert warm.result.total_potential_energy == pytest.approx(
        cold.result.total_potential_energy, rel=1e-12
    )
    continued = backend.solve_from(
        request, 0.5 * linearization.vector(guess), (0.5, 0.75, 1.0)
    )
    assert continued.success and continued.result is not None
    assert {e.load_step for e in continued.result.convergence_history} == {1, 2, 3}


def test_a_failing_warm_start_is_a_visible_attempt() -> None:
    request = _request(1)
    linearization = FemLinearization(request)
    x, _ = linearization.dof_coordinates.T
    folded = linearization.vector(np.column_stack((-3.0 * x, np.zeros_like(x))))
    attempt = ScikitFemBackend().solve_from(request, folded)
    assert not attempt.success
    assert attempt.failure_code is not None


@pytest.mark.parametrize("factors", [(), (0.5,), (0.75, 0.5, 1.0), (0.0, 1.0), (1.5,)])
def test_bad_load_factors_are_refused(factors: tuple[float, ...]) -> None:
    request = _request(1)
    zero = np.zeros(FemLinearization(request).basis.N)
    with pytest.raises(ValueError):
        ScikitFemBackend().solve_from(request, zero, factors)


def test_a_wrongly_sized_start_is_refused() -> None:
    with pytest.raises(ValueError, match="DOFs"):
        ScikitFemBackend().solve_from(_request(1), np.zeros(3))
