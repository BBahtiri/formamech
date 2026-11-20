"""Tests for Feature D: exact discrete DWR adjoint estimators for smooth QoIs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from piginot.certify import (
    CertificateVerdict,
    ReactionForceQoi,
    RegionalAverageStressQoi,
    estimate_qoi_dwr,
)
from piginot.fem import ScikitFemBackend
from piginot.fem.linearization import FemLinearization
from tests.fem.test_nonlinear_solver import _refined_unit_square, _traction_request


def _solved_nonlinear_problem(order: int = 2) -> tuple[FemLinearization, np.ndarray]:
    request = replace(
        _traction_request(element_order=order), mesh=_refined_unit_square()
    )
    attempt = ScikitFemBackend().solve_many((request,))[0]
    assert attempt.success, attempt.message
    assert attempt.result is not None
    linearization = FemLinearization(request)
    solution = np.asarray(attempt.result.displacement, dtype=np.float64)
    return linearization, solution


def _smooth_perturbation(linearization: FemLinearization, scale: float) -> np.ndarray:
    x, y = linearization.dof_coordinates.T
    return scale * np.column_stack(
        (x * np.sin(np.pi * y + 0.3), y * np.cos(np.pi * x + 0.2))
    )


def test_qoi_free_gradients_match_central_finite_differences() -> None:
    linearization, solution = _solved_nonlinear_problem(order=1)
    perturbed = solution + _smooth_perturbation(linearization, 2e-2)
    vec = linearization.with_exact_dirichlet(linearization.vector(perturbed))
    state = linearization.state(vec)

    bc_name = next(iter(sorted(linearization.constraints.by_name)))
    qois = (
        ReactionForceQoi(name="rxn", boundary_condition_name=bc_name),
        RegionalAverageStressQoi(name="avg_p11", component=(0, 0)),
        RegionalAverageStressQoi(name="avg_frob", component=None),
    )

    rng = np.random.default_rng(123)
    direction_free = rng.standard_normal(linearization.constraints.free.size)
    direction_free /= np.linalg.norm(direction_free)
    direction_full = np.zeros(linearization.basis.N, dtype=np.float64)
    direction_full[linearization.constraints.free] = direction_free

    eps = 1e-6
    plus_state = linearization.state(vec + eps * direction_full)
    minus_state = linearization.state(vec - eps * direction_full)

    for qoi in qois:
        grad = qoi.free_gradient(linearization, state)
        analytic_dir_deriv = float(grad @ direction_free)
        fd_dir_deriv = (
            qoi.evaluate(linearization, plus_state)
            - qoi.evaluate(linearization, minus_state)
        ) / (2.0 * eps)
        assert analytic_dir_deriv == pytest.approx(fd_dir_deriv, rel=1e-6, abs=1e-8)


def test_dwr_effectivity_is_near_unity_for_reaction_and_regional_stress() -> None:
    linearization, u_fem = _solved_nonlinear_problem(order=2)
    fem_state = linearization.state(linearization.vector(u_fem))
    u_surrogate = u_fem + _smooth_perturbation(linearization, 1e-3)

    bc_name = next(iter(sorted(linearization.constraints.by_name)))
    qois = (
        ReactionForceQoi(name="rxn", boundary_condition_name=bc_name),
        RegionalAverageStressQoi(name="avg_p11", component=(0, 0)),
        RegionalAverageStressQoi(name="avg_frob", component=None),
    )

    for qoi in qois:
        q_true = qoi.evaluate(linearization, fem_state)
        est = estimate_qoi_dwr(linearization, u_surrogate, qoi)
        assert est.verdict is CertificateVerdict.OK
        assert est.qoi_value is not None
        assert est.dwr_estimate is not None
        assert est.newton_qoi_change is not None
        assert est.adjoint_residual_norm is not None
        assert est.adjoint_residual_norm < 1e-10
        true_err = q_true - est.qoi_value
        assert abs(true_err) > 1e-9
        effectivity = est.dwr_estimate / true_err
        assert 0.95 <= effectivity <= 1.05
        # Corrected state moves quadratically closer to the FEM QoI.
        assert est.corrected_qoi_value is not None
        assert abs(q_true - est.corrected_qoi_value) < 0.05 * abs(true_err)


def test_dwr_returns_inadmissible_on_folded_displacement() -> None:
    linearization, u_fem = _solved_nonlinear_problem(order=1)
    # Invert every element: u_x = -2 x => F_11 = -1 => det F < 0.
    xy = linearization.dof_coordinates
    folded = np.column_stack((-2.0 * xy[:, 0], np.zeros_like(xy[:, 1])))
    bc_name = next(iter(sorted(linearization.constraints.by_name)))
    est = estimate_qoi_dwr(
        linearization,
        folded,
        ReactionForceQoi(name="rxn", boundary_condition_name=bc_name),
    )
    assert est.verdict is CertificateVerdict.INADMISSIBLE
    assert est.guards.get("admissible") is False
    assert est.dwr_estimate is None
