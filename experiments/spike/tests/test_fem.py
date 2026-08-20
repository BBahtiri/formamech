from __future__ import annotations

import math

import numpy as np

from experiments.spike.fem import (
    DEFAULT_E,
    DEFAULT_NU,
    first_piola_plane_stress,
    lame_parameters,
    plane_stress_f33,
    plane_stress_f33_derivative,
    plane_stress_tangent,
    solve_uniaxial_square,
)


def _closure_residual(f2: np.ndarray, f33: float) -> float:
    mu, lam = lame_parameters(DEFAULT_E, DEFAULT_NU)
    j2 = np.linalg.det(f2)
    return float(mu * (f33**2 - 1.0) + lam * math.log(j2 * f33))


def _independent_reduced_energy(f2: np.ndarray) -> float:
    mu, lam = lame_parameters(DEFAULT_E, DEFAULT_NU)
    f33 = float(plane_stress_f33(f2))
    log_j = math.log(float(np.linalg.det(f2)) * f33)
    trace_c = float(np.sum(f2**2)) + f33**2
    return 0.5 * mu * (trace_c - 3.0) - mu * log_j + 0.5 * lam * log_j**2


def test_plane_stress_closure_is_positive_and_has_correct_ift_derivative() -> None:
    states = (
        np.eye(2),
        np.diag([1.1, 0.94]),
        np.array([[1.02, 0.17], [0.03, 0.98]]),
    )
    mu, lam = lame_parameters(DEFAULT_E, DEFAULT_NU)

    for f2 in states:
        f33 = float(plane_stress_f33(f2))
        assert f33 > 0.0
        assert abs(_closure_residual(f2, f33)) / (mu + abs(lam)) < 1e-12

        analytic = plane_stress_f33_derivative(f2)
        finite_difference = np.empty((2, 2))
        step = 1e-6
        for i in range(2):
            for j in range(2):
                perturbation = np.zeros((2, 2))
                perturbation[i, j] = step
                upper = float(plane_stress_f33(f2 + perturbation))
                lower = float(plane_stress_f33(f2 - perturbation))
                finite_difference[i, j] = (upper - lower) / (2.0 * step)
        np.testing.assert_allclose(analytic, finite_difference, rtol=1e-6, atol=1e-8)


def test_first_piola_and_consistent_tangent_match_independent_differences() -> None:
    f2 = np.array([[1.08, 0.09], [-0.04, 0.96]])
    stress = first_piola_plane_stress(f2)
    tangent = plane_stress_tangent(f2)
    step = 2e-6

    energy_gradient = np.empty((2, 2))
    stress_gradient = np.empty((2, 2, 2, 2))
    for k in range(2):
        for capital_k in range(2):
            perturbation = np.zeros((2, 2))
            perturbation[k, capital_k] = step
            energy_gradient[k, capital_k] = (
                _independent_reduced_energy(f2 + perturbation)
                - _independent_reduced_energy(f2 - perturbation)
            ) / (2.0 * step)
            stress_gradient[:, :, k, capital_k] = (
                first_piola_plane_stress(f2 + perturbation)
                - first_piola_plane_stress(f2 - perturbation)
            ) / (2.0 * step)

    np.testing.assert_allclose(stress, energy_gradient, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(tangent, stress_gradient, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(first_piola_plane_stress(np.eye(2)), 0.0, atol=1e-12)


def _lateral_stretch_oracle(axial_stretch: float) -> float:
    mu, lam = lame_parameters(DEFAULT_E, DEFAULT_NU)

    def residual(value: float) -> float:
        return mu * (value**2 - 1.0) + lam * math.log(axial_stretch * value**2)

    lower, upper = 0.2, 2.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if residual(midpoint) > 0.0:
            upper = midpoint
        else:
            lower = midpoint
    return 0.5 * (lower + upper)


def test_uniaxial_square_matches_closed_form_to_one_part_per_million() -> None:
    axial_stretch = 1.10
    result = solve_uniaxial_square(axial_stretch=axial_stretch, nx=6, ny=6)
    lateral_stretch = _lateral_stretch_oracle(axial_stretch)

    assert result.converged, result.failure_reason
    assert abs(result.lateral_stretch - lateral_stretch) < 1e-8
    assert result.relative_l2_u <= 1e-6
    assert result.relative_l2_p <= 1e-6
    assert result.max_error <= 1e-6
