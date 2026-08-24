"""Pure error-estimation primitives for S8 FEM verification."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ConvergenceOrderSource(StrEnum):
    """Provenance of the order used by Richardson extrapolation."""

    THEORETICAL = "theoretical"
    OBSERVED = "observed"


@dataclass(frozen=True, slots=True)
class ObservedOrderResult:
    """Least-squares fit of ``error = coefficient * h**order``."""

    order: float
    coefficient: float
    r_squared: float
    pairwise_orders: tuple[float, ...]
    sample_count: int


@dataclass(frozen=True, slots=True)
class RichardsonEstimate:
    """Fine-grid error inferred from a coarse/fine solution difference."""

    fine_grid_error: float
    coarse_fine_difference: float
    refinement_ratio: float
    order: float
    order_source: ConvergenceOrderSource
    denominator: float


@dataclass(frozen=True, slots=True)
class EstimatorCalibration:
    """Population statistics for a declared multiplicative safety factor."""

    population_size: int
    safety_factor: float
    bracketed_count: int
    bracketing_rate: float
    conservative_count: int
    conservative_rate: float
    effectivity_indices: tuple[float, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RecoveryStressEstimate:
    """Quadrature-weighted distance between raw and recovered stress fields."""

    estimate_l2: float
    raw_stress_l2: float
    relative_estimate: float | None
    integrated_weight: float
    quadrature_point_count: int


def observed_loglog_order(
    mesh_sizes: Sequence[float],
    errors: Sequence[float],
) -> ObservedOrderResult:
    """Fit an observed convergence order over a coarse-to-fine sequence."""

    sizes = _positive_series(mesh_sizes, "mesh_sizes")
    error_values = _positive_series(errors, "errors")
    if len(sizes) < 2:
        raise ValueError("at least two refinement levels are required")
    if len(sizes) != len(error_values):
        raise ValueError("mesh_sizes and errors must have equal length")
    if any(coarse <= fine for coarse, fine in zip(sizes, sizes[1:], strict=False)):
        raise ValueError("mesh_sizes must be strictly decreasing")

    log_sizes = tuple(math.log(value) for value in sizes)
    log_errors = tuple(math.log(value) for value in error_values)
    mean_size = math.fsum(log_sizes) / len(log_sizes)
    mean_error = math.fsum(log_errors) / len(log_errors)
    centered_sizes = tuple(value - mean_size for value in log_sizes)
    denominator = math.fsum(value * value for value in centered_sizes)
    order = (
        math.fsum(
            size_offset * (error - mean_error)
            for size_offset, error in zip(centered_sizes, log_errors, strict=True)
        )
        / denominator
    )
    intercept = mean_error - order * mean_size
    try:
        coefficient = math.exp(intercept)
    except OverflowError as error:
        raise ValueError("fitted convergence coefficient is not finite") from error
    if not math.isfinite(order) or not math.isfinite(coefficient):
        raise ValueError("fitted convergence model is not finite")

    residual_sum = math.fsum(
        (error - (intercept + order * size)) ** 2
        for size, error in zip(log_sizes, log_errors, strict=True)
    )
    total_sum = math.fsum((error - mean_error) ** 2 for error in log_errors)
    r_squared = 1.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    r_squared = min(1.0, max(0.0, r_squared))
    pairwise_orders = tuple(
        (coarse_error - fine_error) / (coarse_size - fine_size)
        for coarse_size, fine_size, coarse_error, fine_error in zip(
            log_sizes,
            log_sizes[1:],
            log_errors,
            log_errors[1:],
            strict=False,
        )
    )
    return ObservedOrderResult(
        order=order,
        coefficient=coefficient,
        r_squared=r_squared,
        pairwise_orders=pairwise_orders,
        sample_count=len(sizes),
    )


def richardson_fine_grid_error(
    coarse_fine_difference: float,
    refinement_ratio: float,
    *,
    theoretical_order: float | None = None,
    observed_order: ObservedOrderResult | float | None = None,
) -> RichardsonEstimate:
    """Estimate fine-grid error as ``difference / (ratio**order - 1)``."""

    if (theoretical_order is None) == (observed_order is None):
        raise ValueError("provide exactly one of theoretical_order or observed_order")
    difference = _positive_scalar(coarse_fine_difference, "coarse_fine_difference")
    ratio = _positive_scalar(refinement_ratio, "refinement_ratio")
    if ratio <= 1.0:
        raise ValueError("refinement_ratio must be greater than one")

    if theoretical_order is not None:
        order = _positive_scalar(theoretical_order, "theoretical_order")
        source = ConvergenceOrderSource.THEORETICAL
    else:
        observed_value = (
            observed_order.order
            if isinstance(observed_order, ObservedOrderResult)
            else observed_order
        )
        order = _positive_scalar(observed_value, "observed_order")
        source = ConvergenceOrderSource.OBSERVED

    try:
        denominator = math.expm1(order * math.log(ratio))
    except OverflowError as error:
        raise ValueError("Richardson denominator is not finite") from error
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Richardson denominator must be finite and positive")
    estimate = difference / denominator
    if not math.isfinite(estimate) or estimate <= 0.0:
        raise ValueError("Richardson estimate must be finite and positive")
    return RichardsonEstimate(
        fine_grid_error=estimate,
        coarse_fine_difference=difference,
        refinement_ratio=ratio,
        order=order,
        order_source=source,
        denominator=denominator,
    )


def calibrate_error_estimator(
    estimated_errors: Sequence[float],
    true_errors: Sequence[float],
    *,
    safety_factor: float,
) -> EstimatorCalibration:
    """Evaluate a declared symmetric multiplicative calibration interval.

    Each raw estimate defines ``[estimate / safety_factor, estimate *
    safety_factor]``.  Conservatism is measured against the upper endpoint; the
    factor is supplied by the caller and is never inferred from this population.
    """

    estimates = _positive_series(estimated_errors, "estimated_errors")
    truths = _positive_series(true_errors, "true_errors")
    if not estimates:
        raise ValueError("calibration population must not be empty")
    if len(estimates) != len(truths):
        raise ValueError("estimated_errors and true_errors must have equal length")
    factor = _positive_scalar(safety_factor, "safety_factor")
    if factor < 1.0:
        raise ValueError("safety_factor must be at least one")

    lower_bounds = tuple(estimate / factor for estimate in estimates)
    upper_bounds = tuple(estimate * factor for estimate in estimates)
    if not all(
        math.isfinite(bound) and bound > 0.0 for bound in (*lower_bounds, *upper_bounds)
    ):
        raise ValueError("calibrated bounds must be finite and positive")
    effectivity = tuple(
        estimate / truth for estimate, truth in zip(estimates, truths, strict=True)
    )
    if not all(math.isfinite(value) and value > 0.0 for value in effectivity):
        raise ValueError("effectivity indices must be finite and positive")
    bracketed = sum(
        lower <= truth <= upper
        for lower, truth, upper in zip(
            lower_bounds,
            truths,
            upper_bounds,
            strict=True,
        )
    )
    conservative = sum(
        upper >= truth for upper, truth in zip(upper_bounds, truths, strict=True)
    )
    population_size = len(truths)
    return EstimatorCalibration(
        population_size=population_size,
        safety_factor=factor,
        bracketed_count=bracketed,
        bracketing_rate=bracketed / population_size,
        conservative_count=conservative,
        conservative_rate=conservative / population_size,
        effectivity_indices=effectivity,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )


def recovery_stress_estimate(
    raw_stress: ArrayLike,
    recovered_stress: ArrayLike,
    quadrature_weights: ArrayLike,
) -> RecoveryStressEstimate:
    """Return the weighted Frobenius L2 gap between raw and recovered stress."""

    raw = _finite_array(raw_stress, "raw_stress")
    recovered = _finite_array(recovered_stress, "recovered_stress")
    weights = _finite_array(quadrature_weights, "quadrature_weights")
    if raw.shape != recovered.shape:
        raise ValueError("raw_stress and recovered_stress must have equal shape")
    if weights.ndim == 0 or weights.size == 0:
        raise ValueError("quadrature_weights must be a non-empty array")
    if np.any(weights <= 0.0):
        raise ValueError("quadrature_weights must be positive")
    if raw.ndim <= weights.ndim or raw.shape[: weights.ndim] != weights.shape:
        raise ValueError("stress leading dimensions must match quadrature_weights")
    if any(size == 0 for size in raw.shape[weights.ndim :]):
        raise ValueError("stress component dimensions must not be empty")

    component_axes = tuple(range(weights.ndim, raw.ndim))
    try:
        with np.errstate(over="raise", invalid="raise"):
            difference_density = np.sum(
                np.square(raw - recovered),
                axis=component_axes,
            )
            raw_density = np.sum(np.square(raw), axis=component_axes)
            squared_estimate = float(np.sum(weights * difference_density))
            squared_raw_norm = float(np.sum(weights * raw_density))
            integrated_weight = float(np.sum(weights))
    except FloatingPointError as error:
        raise ValueError("weighted stress norm is not finite") from error
    if not all(
        math.isfinite(value)
        for value in (squared_estimate, squared_raw_norm, integrated_weight)
    ):
        raise ValueError("weighted stress norm is not finite")

    estimate = math.sqrt(squared_estimate)
    raw_norm = math.sqrt(squared_raw_norm)
    relative = None if raw_norm == 0.0 else estimate / raw_norm
    return RecoveryStressEstimate(
        estimate_l2=estimate,
        raw_stress_l2=raw_norm,
        relative_estimate=relative,
        integrated_weight=integrated_weight,
        quadrature_point_count=int(weights.size),
    )


def _positive_series(values: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite positive scalars") from error
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain finite positive scalars")
    return result


def _positive_scalar(value: object, name: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite positive scalar") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return result


def _finite_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite rectangular array") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


__all__ = (
    "ConvergenceOrderSource",
    "EstimatorCalibration",
    "ObservedOrderResult",
    "RecoveryStressEstimate",
    "RichardsonEstimate",
    "calibrate_error_estimator",
    "observed_loglog_order",
    "recovery_stress_estimate",
    "richardson_fine_grid_error",
)
