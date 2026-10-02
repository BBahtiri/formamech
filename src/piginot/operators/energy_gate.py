"""Fail-closed reducer for the amended G11 gate of the energy-form operator.

The energy-form operator receives no boundary point cloud, so the S11
boundary-cloud checks (permutation, resampling and the boundary-mask mutant)
cannot be measured on it. ADR 0018 fixes the replacement checks before any
measurement: exact Dirichlet values over the full parameter box at several
boundary refinements (``G11.lifting``) and a mutant that removes the lifting
and must fire (``G11.lifting-mutant``). ``G11.heads`` is amended to require
that the served stress is the constitutive stress of the predicted
displacement gradient and that no independent stress head exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from piginot.operators.gate import GateStatus, S11GateCheck, S11GateEvaluation

ENERGY_G11_CHECK_IDS = (
    "G11.node-order",
    "G11.query-batch",
    "G11.reachability",
    "G11.safetensors",
    "G11.heads",
    "G11.lifting",
    "G11.lifting-mutant",
    "G11.coverage",
)
ENERGY_G11_LIFTING_POINTS_PER_EDGE = (320, 480, 4_800)
ENERGY_G11_MINIMUM_PARAMETER_VECTORS = 64

_ORDER_LIMIT = 1.0e-6
_STRESS_IDENTITY_LIMIT = 1.0e-10
_LIFTING_LIMIT = 1.0e-12
_MUTANT_CONTROL_LIMIT = 1.0e-12
_MUTANT_FIRE_LIMIT = 1.0e-3
_MINIMUM_COVERAGE_PERCENT = 80.0


@dataclass(frozen=True, slots=True)
class EnergyG11Measurements:
    """Complete finite evidence consumed by the amended G11 reducer."""

    node_order_relative_change: float | None
    query_batch_relative_change: float | None
    reachable_parameter_count: int
    required_parameter_count: int
    pickle_refused_before_deserialization: bool | None
    u_shape_verified: bool | None
    p_shape_verified: bool | None
    displacement_only_output: bool | None
    served_stress_energy_gradient_relative_difference: float | None
    lifting_parameter_vector_count: int
    lifting_points_per_edge: tuple[int, ...]
    lifting_max_relative_error: float | None
    lifting_mutant_control_relative_error: float | None
    lifting_mutant_relative_error: float | None
    code_coverage_percent: float | None

    def __post_init__(self) -> None:
        for name in (
            "reachable_parameter_count",
            "required_parameter_count",
            "lifting_parameter_vector_count",
        ):
            _non_negative_integer(getattr(self, name), name)
        if type(self.lifting_points_per_edge) is not tuple or any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in self.lifting_points_per_edge
        ):
            raise ValueError("lifting_points_per_edge must be positive integers")
        for name in (
            "node_order_relative_change",
            "query_batch_relative_change",
            "served_stress_energy_gradient_relative_difference",
            "lifting_max_relative_error",
            "lifting_mutant_control_relative_error",
            "lifting_mutant_relative_error",
        ):
            _optional_non_negative_metric(getattr(self, name), name)
        for name in (
            "pickle_refused_before_deserialization",
            "u_shape_verified",
            "p_shape_verified",
            "displacement_only_output",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean or None")
        coverage = _optional_non_negative_metric(
            self.code_coverage_percent, "code_coverage_percent"
        )
        if coverage is not None and coverage > 100.0:
            raise ValueError("code_coverage_percent must lie in [0, 100]")


def evaluate_energy_g11_gate(
    measurements: EnergyG11Measurements,
) -> S11GateEvaluation:
    """Evaluate every required amended G11 check; any non-PASS fails the gate."""

    if type(measurements) is not EnergyG11Measurements:
        raise TypeError("measurements must be exact EnergyG11Measurements")
    checks = (
        _metric_check(
            "G11.node-order", measurements.node_order_relative_change, _ORDER_LIMIT
        ),
        _metric_check(
            "G11.query-batch", measurements.query_batch_relative_change, _ORDER_LIMIT
        ),
        _reachability_check(measurements),
        _boolean_check(
            "G11.safetensors", measurements.pickle_refused_before_deserialization
        ),
        _heads_check(measurements),
        _lifting_check(measurements),
        _mutant_check(measurements),
        _coverage_check(measurements.code_coverage_percent),
    )
    verdict = (
        GateStatus.PASS
        if all(check.status is GateStatus.PASS for check in checks)
        else GateStatus.FAIL
    )
    return S11GateEvaluation(checks, verdict)


def _metric_check(check_id: str, value: float | None, limit: float) -> S11GateCheck:
    threshold = f"< {limit:g}"
    if value is None:
        return S11GateCheck(
            check_id, GateStatus.NOT_RUN, None, threshold, "lt", "metric is missing"
        )
    return S11GateCheck(
        check_id,
        GateStatus.PASS if value < limit else GateStatus.FAIL,
        value,
        threshold,
        "lt",
    )


def _reachability_check(measurements: EnergyG11Measurements) -> S11GateCheck:
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
        check_id, GateStatus.PASS if value else GateStatus.FAIL, value, True, "eq"
    )


def _heads_check(measurements: EnergyG11Measurements) -> S11GateCheck:
    measured: dict[str, object] = {
        "u_shape_verified": measurements.u_shape_verified,
        "p_shape_verified": measurements.p_shape_verified,
        "displacement_only_output": measurements.displacement_only_output,
        "served_stress_energy_gradient_relative_difference": (
            measurements.served_stress_energy_gradient_relative_difference
        ),
    }
    threshold: dict[str, object] = {
        "u_shape_verified": True,
        "p_shape_verified": True,
        "displacement_only_output": True,
        "served_stress_energy_gradient_relative_difference": (
            f"< {_STRESS_IDENTITY_LIMIT:g}"
        ),
    }
    if any(value is None for value in measured.values()):
        return S11GateCheck(
            "G11.heads",
            GateStatus.NOT_RUN,
            measured,
            threshold,
            "all",
            "head evidence is incomplete",
        )
    difference = measurements.served_stress_energy_gradient_relative_difference
    assert difference is not None
    passed = (
        measurements.u_shape_verified is True
        and measurements.p_shape_verified is True
        and measurements.displacement_only_output is True
        and difference < _STRESS_IDENTITY_LIMIT
    )
    return S11GateCheck(
        "G11.heads",
        GateStatus.PASS if passed else GateStatus.FAIL,
        measured,
        threshold,
        "all",
    )


def _lifting_check(measurements: EnergyG11Measurements) -> S11GateCheck:
    measured: dict[str, object] = {
        "parameter_vectors": measurements.lifting_parameter_vector_count,
        "points_per_edge": list(measurements.lifting_points_per_edge),
        "max_relative_error": measurements.lifting_max_relative_error,
    }
    threshold: dict[str, object] = {
        "parameter_vectors": f">= {ENERGY_G11_MINIMUM_PARAMETER_VECTORS}",
        "points_per_edge": list(ENERGY_G11_LIFTING_POINTS_PER_EDGE),
        "max_relative_error": f"< {_LIFTING_LIMIT:g}",
    }
    if measurements.lifting_max_relative_error is None:
        return S11GateCheck(
            "G11.lifting",
            GateStatus.NOT_RUN,
            measured,
            threshold,
            "all",
            "lifting metric is missing",
        )
    passed = (
        measurements.lifting_parameter_vector_count
        >= ENERGY_G11_MINIMUM_PARAMETER_VECTORS
        and measurements.lifting_points_per_edge == ENERGY_G11_LIFTING_POINTS_PER_EDGE
        and measurements.lifting_max_relative_error < _LIFTING_LIMIT
    )
    return S11GateCheck(
        "G11.lifting",
        GateStatus.PASS if passed else GateStatus.FAIL,
        measured,
        threshold,
        "all",
    )


def _mutant_check(measurements: EnergyG11Measurements) -> S11GateCheck:
    control = measurements.lifting_mutant_control_relative_error
    mutant = measurements.lifting_mutant_relative_error
    measured = {"control_relative_error": control, "mutant_relative_error": mutant}
    threshold = {
        "control_relative_error": f"< {_MUTANT_CONTROL_LIMIT:g}",
        "mutant_relative_error": f"> {_MUTANT_FIRE_LIMIT:g}",
    }
    if control is None or mutant is None:
        return S11GateCheck(
            "G11.lifting-mutant",
            GateStatus.NOT_RUN,
            measured,
            threshold,
            "all",
            "mutant evidence is incomplete",
        )
    fired = control < _MUTANT_CONTROL_LIMIT and mutant > _MUTANT_FIRE_LIMIT
    return S11GateCheck(
        "G11.lifting-mutant",
        GateStatus.PASS if fired else GateStatus.FAIL,
        measured,
        threshold,
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


__all__ = (
    "ENERGY_G11_CHECK_IDS",
    "ENERGY_G11_LIFTING_POINTS_PER_EDGE",
    "ENERGY_G11_MINIMUM_PARAMETER_VECTORS",
    "EnergyG11Measurements",
    "evaluate_energy_g11_gate",
)
