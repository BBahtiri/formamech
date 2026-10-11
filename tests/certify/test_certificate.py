"""S21 certificate: the Newton-step estimate matches the true discrete error."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

from piginot.certify import CertificateVerdict, certify, symmetric_inertia
from piginot.fem import ScikitFemBackend
from piginot.fem.linearization import FemLinearization
from tests.fem.test_nonlinear_solver import _refined_unit_square, _traction_request


def _problem(order: int) -> tuple[FemLinearization, np.ndarray]:
    request = replace(
        _traction_request(element_order=order), mesh=_refined_unit_square()
    )
    attempt = ScikitFemBackend().solve_many((request,))[0]
    assert attempt.success, attempt.message
    assert attempt.result is not None
    linearization = FemLinearization(request)
    solution = np.asarray(attempt.result.displacement, dtype=np.float64)
    return linearization, solution


def _perturbation(linearization: FemLinearization, scale: float) -> np.ndarray:
    """A smooth field vanishing on the constrained boundaries (x=0 for u_x, y=0 for u_y)."""

    x, y = linearization.dof_coordinates.T
    return scale * np.column_stack((x * np.sin(np.pi * y + 0.3), y * np.cos(np.pi * x)))


@pytest.fixture(scope="module", params=[1, 2], ids=["P1", "P2"])
def problem(request: Any) -> tuple[FemLinearization, np.ndarray]:
    return _problem(request.param)


def test_at_the_converged_solution_the_estimate_vanishes(
    problem: tuple[FemLinearization, np.ndarray],
) -> None:
    linearization, solution = problem
    certificate = certify(linearization, solution)
    assert certificate.verdict is CertificateVerdict.OK
    assert certificate.eta_energy is not None and certificate.eta_energy < 1e-8
    assert certificate.dirichlet_deviation < 1e-12


@pytest.mark.parametrize("scale", [1e-4, 1e-3, 1e-2])
def test_the_estimate_tracks_the_true_energy_norm_error(
    problem: tuple[FemLinearization, np.ndarray], scale: float
) -> None:
    linearization, solution = problem
    surrogate = solution + _perturbation(linearization, scale)
    certificate = certify(linearization, surrogate)
    assert certificate.verdict is CertificateVerdict.OK
    error = linearization.vector(surrogate) - linearization.vector(solution)
    exact = linearization.state(linearization.vector(solution))
    true_norm = linearization.energy_norm(error, exact)
    assert certificate.eta_energy is not None
    effectivity = certificate.eta_energy / true_norm
    assert 0.9 < effectivity < 1.1
    assert certificate.indicators["energy_ratio"] == pytest.approx(1.0, abs=0.1)
    assert certificate.indicators["p_l2_change"] > 0
    assert certificate.guards == {
        "admissible": True,
        "increment_solved": True,
        "positive_curvature": True,
        "positive_definite": True,
        "step_admissible": True,
        "energy_decrease": True,
    }


def test_the_estimate_is_monotone_in_the_error(
    problem: tuple[FemLinearization, np.ndarray],
) -> None:
    linearization, solution = problem
    etas = [
        certify(linearization, solution + _perturbation(linearization, s)).eta_energy
        for s in (1e-4, 1e-3, 1e-2)
    ]
    assert etas == sorted(etas)


def test_a_folded_state_is_inadmissible(
    problem: tuple[FemLinearization, np.ndarray],
) -> None:
    linearization, solution = problem
    x, y = linearization.dof_coordinates.T
    folded = solution + np.column_stack((-3.0 * x, np.zeros_like(y)))
    certificate = certify(linearization, folded)
    assert certificate.verdict is CertificateVerdict.INADMISSIBLE
    assert certificate.eta_energy is None
    assert "NEGATIVE_JACOBIAN" in certificate.reason


def test_dirichlet_deviation_is_reported_and_corrected(
    problem: tuple[FemLinearization, np.ndarray],
) -> None:
    linearization, solution = problem
    shifted = solution.copy()
    shifted[:, 0] += 1e-6  # breaks u_x = 0 on the left edge
    certificate = certify(linearization, shifted)
    assert certificate.dirichlet_deviation == pytest.approx(1e-6)
    assert certificate.verdict is CertificateVerdict.OK


def test_the_certificate_document_is_deterministic(
    problem: tuple[FemLinearization, np.ndarray],
) -> None:
    linearization, solution = problem
    surrogate = solution + _perturbation(linearization, 1e-3)
    first = certify(linearization, surrogate).document()
    assert first == certify(linearization, surrogate).document()
    assert first["verdict"] == "OK"


def test_wrong_shapes_are_refused(problem: tuple[FemLinearization, np.ndarray]) -> None:
    linearization, solution = problem
    with pytest.raises(ValueError, match="shape"):
        certify(linearization, solution[:-1])


def test_inertia_detects_definiteness() -> None:
    spd = csr_matrix(np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]))
    assert symmetric_inertia(spd).positive_definite is True
    indefinite = csr_matrix(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert symmetric_inertia(indefinite).positive_definite is False
    assert symmetric_inertia(diags([1.0, -1.0, 2.0])).non_positive == 1
    assert symmetric_inertia(csr_matrix((0, 0))).positive_definite is True
    singular = csr_matrix(np.zeros((2, 2)))
    assert symmetric_inertia(singular).positive_definite is False


# --- guard branches --------------------------------------------------------

from piginot.certify import estimator as estimator_module  # noqa: E402
from piginot.certify import guards as guard_module  # noqa: E402
from piginot.certify.guards import Inertia  # noqa: E402
from piginot.fem.models import FemFailureCode, FemSolveFailure  # noqa: E402


def test_an_unsolvable_increment_is_indefinite(
    problem: tuple[FemLinearization, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    linearization, solution = problem

    def singular(_state: object) -> np.ndarray:
        raise FemSolveFailure(FemFailureCode.CONSTRAINT_RANK_LOSS, "singular")

    monkeypatch.setattr(linearization, "newton_increment", singular)
    certificate = certify(linearization, solution)
    assert certificate.verdict is CertificateVerdict.INDEFINITE
    assert certificate.guards["increment_solved"] is False


def test_an_inadmissible_step_is_indefinite(
    problem: tuple[FemLinearization, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    linearization, solution = problem
    real_state = linearization.state
    calls = []

    def second_fails(vector: np.ndarray) -> object:
        calls.append(1)
        if len(calls) == 2:
            raise FemSolveFailure(FemFailureCode.NEGATIVE_JACOBIAN, "folded")
        return real_state(vector)

    monkeypatch.setattr(linearization, "state", second_fails)
    certificate = certify(linearization, solution + _perturbation(linearization, 1e-3))
    assert certificate.verdict is CertificateVerdict.INDEFINITE
    assert certificate.guards["step_admissible"] is False


def test_hidden_inertia_is_refused_and_global_constraints_skip_the_test(
    problem: tuple[FemLinearization, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    linearization, solution = problem
    surrogate = solution + _perturbation(linearization, 1e-3)
    monkeypatch.setattr(
        estimator_module, "symmetric_inertia", lambda matrix: Inertia(5, None)
    )
    hidden = certify(linearization, surrogate)
    assert hidden.verdict is CertificateVerdict.INDEFINITE
    assert "positive_definite" in hidden.reason
    monkeypatch.setattr(
        FemLinearization, "has_global_constraints", property(lambda self: True)
    )
    projected = certify(linearization, surrogate)
    assert projected.guards["positive_definite"] is None
    assert projected.verdict is CertificateVerdict.OK


def test_campaign_qoi_functions_become_indicators(
    problem: tuple[FemLinearization, np.ndarray],
) -> None:
    linearization, solution = problem

    def peak(lin: FemLinearization, state: object) -> float:
        return float(np.max(np.abs(lin.recovered_stress(state)[:, 0, 0])))  # type: ignore[arg-type]

    certificate = certify(
        linearization,
        solution + _perturbation(linearization, 1e-3),
        qoi_functions={"peak": peak},
    )
    assert certificate.indicators["qoi:peak"] > 0.0


def test_layout_round_trips(problem: tuple[FemLinearization, np.ndarray]) -> None:
    linearization, solution = problem
    assert np.array_equal(linearization.nodal(linearization.vector(solution)), solution)


def test_pivoting_hides_the_inertia(monkeypatch: pytest.MonkeyPatch) -> None:
    class Factor:
        perm_r = np.array([1, 0])
        perm_c = np.array([0, 1])

    monkeypatch.setattr(guard_module, "splu", lambda *args, **kwargs: Factor())
    assert symmetric_inertia(diags([1.0, 2.0])).positive_definite is None
