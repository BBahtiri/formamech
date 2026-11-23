"""Fail-closed reducer for the finite S11 architecture gate."""

from __future__ import annotations

import math
from dataclasses import dataclass

from piginot.contracts.state_machine import GateStatus as GateStatus

_PERMUTATION_LIMIT = 1.0e-5
_RESAMPLING_LIMIT = 0.02
_ORDER_LIMIT = 1.0e-6
_MINIMUM_COVERAGE_PERCENT = 80.0


@dataclass(frozen=True, slots=True)
class S11GateMeasurements:
    """Complete finite evidence consumed by the G11 reducer."""

    permutation_count: int
    permutation_max_relative_change: float | None
    resampling_low_count: int
    resampling_high_count: int
    resampling_relative_change: float | None
    node_order_relative_change: float | None
    query_batch_relative_change: float | None
    reachable_parameter_count: int
    required_parameter_count: int
    mask_mutant_fired: bool | None
    pickle_refused_before_deserialization: bool | None
    u_shape_verified: bool | None
    p_shape_verified: bool | None
    served_stress_is_predicted: bool | None
    code_coverage_percent: float | None

    def __post_init__(self) -> None:
        for name in (
            "permutation_count",
            "resampling_low_count",
            "resampling_high_count",
            "reachable_parameter_count",
            "required_parameter_count",
        ):
            _non_negative_integer(getattr(self, name), name)
        for name in (
            "permutation_max_relative_change",
            "resampling_relative_change",
            "node_order_relative_change",
            "query_batch_relative_change",
        ):
            _optional_non_negative_metric(getattr(self, name), name)
        for name in (
            "mask_mutant_fired",
            "pickle_refused_before_deserialization",
            "u_shape_verified",
            "p_shape_verified",
            "served_stress_is_predicted",
        ):
            _optional_boolean(getattr(self, name), name)
        _optional_coverage(self.code_coverage_percent)


@dataclass(frozen=True, slots=True)
class S11GateCheck:
    """One measured G11 check and its fixed decision rule."""

    check_id: str
    status: GateStatus
    measured: object
    threshold: object
    comparison: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class S11GateEvaluation:
    """Ordered G11 checks with a fail-closed overall verdict."""

    checks: tuple[S11GateCheck, ...]
    verdict: GateStatus

    def by_id(self, check_id: str) -> S11GateCheck:
        for check in self.checks:
            if check.check_id == check_id:
                return check
        raise KeyError(check_id)


def evaluate_s11_gate(measurements: S11GateMeasurements) -> S11GateEvaluation:
    """Evaluate all required G11 checks; every non-PASS blocks S12."""

    if type(measurements) is not S11GateMeasurements:
        raise TypeError("measurements must be exact S11GateMeasurements")
    checks = (
        _permutation_check(measurements),
        _resampling_check(measurements),
        _metric_check(
            "G11.node-order",
            measurements.node_order_relative_change,
            _ORDER_LIMIT,
        ),
        _metric_check(
            "G11.query-batch",
            measurements.query_batch_relative_change,
            _ORDER_LIMIT,
        ),
        _reachability_check(measurements),
        _boolean_check("G11.mask", measurements.mask_mutant_fired),
        _boolean_check(
            "G11.safetensors",
            measurements.pickle_refused_before_deserialization,
        ),
        _heads_check(measurements),
        _coverage_check(measurements.code_coverage_percent),
    )
    verdict = (
        GateStatus.PASS
        if all(check.status is GateStatus.PASS for check in checks)
        else GateStatus.FAIL
    )
    return S11GateEvaluation(checks, verdict)


def _permutation_check(measurements: S11GateMeasurements) -> S11GateCheck:
    measured = {
        "random_permutations": measurements.permutation_count,
        "max_relative_change": measurements.permutation_max_relative_change,
    }
    threshold = {
        "random_permutations": ">= 100",
        "max_relative_change": "< 1e-5",
    }
    if measurements.permutation_max_relative_change is None:
        return S11GateCheck(
            "G11.permutation",
            GateStatus.NOT_RUN,
            measured,
            threshold,
            "all",
            "permutation metric is missing",
        )
    passed = (
        measurements.permutation_count >= 100
        and measurements.permutation_max_relative_change < _PERMUTATION_LIMIT
    )
    return S11GateCheck(
        "G11.permutation",
        GateStatus.PASS if passed else GateStatus.FAIL,
        measured,
        threshold,
        "all",
    )


def _resampling_check(measurements: S11GateMeasurements) -> S11GateCheck:
    measured = {
        "low_boundary_points": measurements.resampling_low_count,
        "high_boundary_points": measurements.resampling_high_count,
        "relative_change": measurements.resampling_relative_change,
    }
    threshold = {
        "low_boundary_points": 320,
        "high_boundary_points": 480,
        "relative_change": "< 0.02",
    }
    if measurements.resampling_relative_change is None:
        return S11GateCheck(
            "G11.resampling",
            GateStatus.NOT_RUN,
            measured,
            threshold,
            "all",
            "resampling metric is missing",
        )
    passed = (
        measurements.resampling_low_count == 320
        and measurements.resampling_high_count == 480
        and measurements.resampling_relative_change < _RESAMPLING_LIMIT
    )
    return S11GateCheck(
        "G11.resampling",
        GateStatus.PASS if passed else GateStatus.FAIL,
        measured,
        threshold,
        "all",
    )


def _metric_check(
    check_id: str,
    value: float | None,
    limit: float,
) -> S11GateCheck:
    threshold = f"< {limit:g}"
    if value is None:
        return S11GateCheck(
            check_id,
            GateStatus.NOT_RUN,
            None,
            threshold,
            "lt",
            "metric is missing",
        )
    return S11GateCheck(
        check_id,
        GateStatus.PASS if value < limit else GateStatus.FAIL,
        value,
        threshold,
        "lt",
    )


def _reachability_check(measurements: S11GateMeasurements) -> S11GateCheck:
    measured = {
        "reachable_parameters": measurements.reachable_parameter_count,
        "required_parameters": measurements.required_parameter_count,
    }
    threshold = "reachable_parameters == required_parameters > 0"
    if measurements.required_parameter_count == 0:
        return S11GateCheck(
            "G11.reachability",
            GateStatus.NOT_RUN,
            measured,
            threshold,
            "eq",
            "no parameter population was measured",
        )
    passed = (
        measurements.reachable_parameter_count == measurements.required_parameter_count
    )
    return S11GateCheck(
        "G11.reachability",
        GateStatus.PASS if passed else GateStatus.FAIL,
        measured,
        threshold,
        "eq",
    )


def _boolean_check(check_id: str, value: bool | None) -> S11GateCheck:
    if value is None:
        return S11GateCheck(
            check_id,
            GateStatus.NOT_RUN,
            None,
            True,
            "eq",
            "boolean evidence is missing",
        )
    return S11GateCheck(
        check_id,
        GateStatus.PASS if value else GateStatus.FAIL,
        value,
        True,
        "eq",
    )


def _heads_check(measurements: S11GateMeasurements) -> S11GateCheck:
    measured = {
        "u_shape_verified": measurements.u_shape_verified,
        "p_shape_verified": measurements.p_shape_verified,
        "served_stress_is_predicted": measurements.served_stress_is_predicted,
    }
    if any(value is None for value in measured.values()):
        return S11GateCheck(
            "G11.heads",
            GateStatus.NOT_RUN,
            measured,
            {name: True for name in measured},
            "all",
            "head evidence is incomplete",
        )
    return S11GateCheck(
        "G11.heads",
        GateStatus.PASS if all(measured.values()) else GateStatus.FAIL,
        measured,
        {name: True for name in measured},
        "all",
    )


def _coverage_check(value: float | None) -> S11GateCheck:
    threshold = ">= 80.0"
    if value is None:
        return S11GateCheck(
            "G11.coverage",
            GateStatus.NOT_RUN,
            None,
            threshold,
            "gte",
            "coverage evidence is missing",
        )
    return S11GateCheck(
        "G11.coverage",
        GateStatus.PASS if value >= _MINIMUM_COVERAGE_PERCENT else GateStatus.FAIL,
        value,
        threshold,
        "gte",
    )


def _non_negative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _optional_non_negative_metric(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-negative number or None")
    metric = float(value)
    if not math.isfinite(metric) or metric < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number or None")
    return metric


def _optional_boolean(value: object, label: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean or None")
    return value


def _optional_coverage(value: object) -> float | None:
    if value is None:
        return None
    coverage = _optional_non_negative_metric(value, "code_coverage_percent")
    if coverage is None or coverage > 100.0:
        raise ValueError("code_coverage_percent must lie in [0, 100]")
    return coverage


__all__ = (
    "GateStatus",
    "S11GateCheck",
    "S11GateEvaluation",
    "S11GateMeasurements",
    "evaluate_s11_gate",
)
