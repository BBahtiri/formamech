"""Unit tests for S8 discretization-error estimation primitives."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from piginot.fem.error_estimation import (
    ConvergenceOrderSource,
    calibrate_error_estimator,
    observed_loglog_order,
    recovery_stress_estimate,
    richardson_fine_grid_error,
)


def test_observed_loglog_order_recovers_exact_power_law() -> None:
    mesh_sizes = (0.5, 0.25, 0.125, 0.0625)
    errors = tuple(2.5 * mesh_size**2 for mesh_size in mesh_sizes)

    result = observed_loglog_order(mesh_sizes, errors)

    assert result.order == pytest.approx(2.0)
    assert result.coefficient == pytest.approx(2.5)
    assert result.r_squared == pytest.approx(1.0)
    assert result.pairwise_orders == pytest.approx((2.0, 2.0, 2.0))
    assert result.sample_count == 4


def test_observed_order_result_is_immutable() -> None:
    result = observed_loglog_order((0.5, 0.25), (0.25, 0.0625))

    with pytest.raises(FrozenInstanceError):
        result.order = 1.0  # type: ignore[misc]


def test_observed_order_avoids_overflow_in_pairwise_ratios() -> None:
    result = observed_loglog_order(
        (1.0, 1.0e-308),
        (1.0e308, 1.0e-308),
    )

    assert result.order == pytest.approx(2.0)
    assert result.pairwise_orders == pytest.approx((2.0,))


@pytest.mark.parametrize(
    ("mesh_sizes", "errors"),
    (
        ((0.5,), (0.25,)),
        ((0.5, 0.25), (0.25,)),
        ((0.5, 0.0), (0.25, 0.0625)),
        ((0.5, 0.25), (0.25, math.inf)),
        ((0.5, 0.5), (0.25, 0.0625)),
        ((0.25, 0.5), (0.0625, 0.25)),
    ),
)
def test_observed_loglog_order_rejects_invalid_series(
    mesh_sizes: tuple[float, ...],
    errors: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        observed_loglog_order(mesh_sizes, errors)


def test_richardson_uses_theoretical_order_for_fine_grid_error() -> None:
    result = richardson_fine_grid_error(
        coarse_fine_difference=0.03,
        refinement_ratio=2.0,
        theoretical_order=2.0,
    )

    assert result.fine_grid_error == pytest.approx(0.01)
    assert result.denominator == pytest.approx(3.0)
    assert result.order == 2.0
    assert result.order_source is ConvergenceOrderSource.THEORETICAL


def test_richardson_accepts_observed_order_result() -> None:
    observed = observed_loglog_order(
        (0.5, 0.25, 0.125),
        (0.125, 0.015625, 0.001953125),
    )

    result = richardson_fine_grid_error(
        coarse_fine_difference=0.07,
        refinement_ratio=2.0,
        observed_order=observed,
    )

    assert result.fine_grid_error == pytest.approx(0.01)
    assert result.order == pytest.approx(3.0)
    assert result.order_source is ConvergenceOrderSource.OBSERVED


def test_richardson_is_stable_for_a_refinement_ratio_close_to_one() -> None:
    refinement_ratio = 1.0 + 1.0e-12
    difference = 1.0e-12

    result = richardson_fine_grid_error(
        difference,
        refinement_ratio,
        theoretical_order=1.0,
    )

    expected = difference / math.expm1(math.log(refinement_ratio))
    assert result.fine_grid_error == pytest.approx(expected, rel=1.0e-12)


@pytest.mark.parametrize(
    "kwargs",
    (
        {
            "coarse_fine_difference": 0.0,
            "refinement_ratio": 2.0,
            "theoretical_order": 2.0,
        },
        {
            "coarse_fine_difference": math.nan,
            "refinement_ratio": 2.0,
            "theoretical_order": 2.0,
        },
        {
            "coarse_fine_difference": 0.03,
            "refinement_ratio": 1.0,
            "theoretical_order": 2.0,
        },
        {
            "coarse_fine_difference": 0.03,
            "refinement_ratio": 2.0,
            "theoretical_order": 0.0,
        },
        {
            "coarse_fine_difference": 0.03,
            "refinement_ratio": 2.0,
        },
        {
            "coarse_fine_difference": 0.03,
            "refinement_ratio": 2.0,
            "theoretical_order": 2.0,
            "observed_order": 1.9,
        },
    ),
)
def test_richardson_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        richardson_fine_grid_error(**kwargs)  # type: ignore[arg-type]


def test_calibration_records_bracketing_and_conservatism_without_fabrication() -> None:
    result = calibrate_error_estimator(
        estimated_errors=(1.0, 1.0, 1.0, 1.0),
        true_errors=(0.75, 1.25, 2.1, 0.4),
        safety_factor=2.0,
    )

    assert result.population_size == 4
    assert result.safety_factor == 2.0
    assert result.bracketed_count == 2
    assert result.bracketing_rate == pytest.approx(0.5)
    assert result.conservative_count == 3
    assert result.conservative_rate == pytest.approx(0.75)
    assert result.effectivity_indices == pytest.approx(
        (1.0 / 0.75, 1.0 / 1.25, 1.0 / 2.1, 1.0 / 0.4)
    )
    assert result.lower_bounds == (0.5, 0.5, 0.5, 0.5)
    assert result.upper_bounds == (2.0, 2.0, 2.0, 2.0)


@pytest.mark.parametrize(
    ("estimated_errors", "true_errors", "safety_factor"),
    (
        ((), (), 1.0),
        ((1.0,), (1.0, 2.0), 1.0),
        ((0.0,), (1.0,), 1.0),
        ((1.0,), (-1.0,), 1.0),
        ((math.inf,), (1.0,), 1.0),
        ((1.0,), (1.0,), 0.99),
        ((1.0,), (1.0,), math.nan),
        ((1.0e-308,), (1.0,), 1.0e308),
    ),
)
def test_calibration_rejects_invalid_population_or_factor(
    estimated_errors: tuple[float, ...],
    true_errors: tuple[float, ...],
    safety_factor: float,
) -> None:
    with pytest.raises(ValueError):
        calibrate_error_estimator(
            estimated_errors,
            true_errors,
            safety_factor=safety_factor,
        )


def test_recovery_stress_estimator_uses_all_tensor_components_and_weights() -> None:
    raw = np.zeros((2, 2, 2, 2), dtype=np.float64)
    raw[:, :, 0, 0] = ((1.0, 2.0), (3.0, 4.0))
    recovered = np.zeros_like(raw)
    weights = np.asarray(((0.25, 0.75), (0.5, 1.5)))

    result = recovery_stress_estimate(raw, recovered, weights)

    assert result.estimate_l2 == pytest.approx(math.sqrt(31.75))
    assert result.raw_stress_l2 == pytest.approx(math.sqrt(31.75))
    assert result.relative_estimate == pytest.approx(1.0)
    assert result.integrated_weight == pytest.approx(3.0)
    assert result.quadrature_point_count == 4


def test_recovery_stress_estimator_reports_zero_reference_without_dividing() -> None:
    raw = np.zeros((2, 1, 3, 3), dtype=np.float64)
    recovered = np.ones_like(raw)

    result = recovery_stress_estimate(raw, recovered, ((0.5,), (0.5,)))

    assert result.estimate_l2 == pytest.approx(3.0)
    assert result.raw_stress_l2 == 0.0
    assert result.relative_estimate is None


@pytest.mark.parametrize(
    ("raw", "recovered", "weights"),
    (
        (np.zeros((2, 2, 2)), np.zeros((2, 1, 2)), np.ones((2, 2))),
        (np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), np.ones((3, 2))),
        (np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2))),
        (np.zeros((2, 2, 0)), np.zeros((2, 2, 0)), np.ones((2, 2))),
        (np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), np.asarray(((1.0, 0.0),) * 2)),
        (np.full((2, 2, 2), math.nan), np.zeros((2, 2, 2)), np.ones((2, 2))),
        (np.full((1, 1, 1), 1.0e308), np.zeros((1, 1, 1)), np.ones((1, 1))),
    ),
)
def test_recovery_stress_estimator_rejects_invalid_shapes_or_values(
    raw: np.ndarray,
    recovered: np.ndarray,
    weights: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        recovery_stress_estimate(raw, recovered, weights)
