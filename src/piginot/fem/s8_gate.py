"""Measured S8 gate reduction with explicit sign-off and CI blockers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from piginot.contracts.specs import FrozenMap
from piginot.fem.manufactured import MANUFACTURED_CASE_NAMES
from piginot.fem.mms_convergence import run_all_mms_convergence_studies
from piginot.fem.mms_estimators import calibrate_mms_estimators
from piginot.fem.s8_physical import run_s8_physical_verification
from piginot.fem.seeded_defects import run_seeded_defect_suite

_ORDER_TOLERANCE = 0.15
_STRESS_ORDER_TOLERANCE = 0.2
_PATCH_LIMIT = 1.0e-12
_OBJECTIVITY_LIMIT = 1.0e-10
_ANGULAR_LIMIT = 1.0e-10
_ENERGY_LIMIT = 1.0e-8
_REACTION_LIMIT = 1.0e-9
_SMALL_STRAIN_LIMIT = 5.0e-3
_ESTIMATOR_BRACKETING_MINIMUM = 0.95
_ESTIMATOR_CONSERVATIVE_MINIMUM = 0.90
_SEEDED_DEFECT_COUNT = 6
_FEM_COVERAGE_MINIMUM = 85.0


class GateStatus(StrEnum):
    """Allowed status values in the step-gate evidence schema."""

    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"


@dataclass(frozen=True, slots=True)
class S8GateMeasurements:
    """Finite measured population consumed by the G8 reducer."""

    displacement_l2_orders: Mapping[str, float]
    first_piola_l2_orders: Mapping[str, float]
    patch_max_error: float
    objectivity_invariant_change: float
    angular_skew_ratio: float
    energy_relative_error: float
    reaction_relative_error: float
    small_strain_relative_error: float
    estimator_bracketing_rate: float
    estimator_conservative_rate: float
    seeded_defects_detected: int
    coverage_percent: float | None

    def __post_init__(self) -> None:
        displacement: FrozenMap[float] = FrozenMap.copy_of(self.displacement_l2_orders)
        stress: FrozenMap[float] = FrozenMap.copy_of(self.first_piola_l2_orders)
        expected_keys = {
            f"{case_name}:p{order}"
            for case_name in MANUFACTURED_CASE_NAMES
            for order in (1, 2)
        }
        if set(displacement) != expected_keys or set(stress) != expected_keys:
            raise ValueError(
                "order measurements must cover all four MMS cases at P1/P2"
            )
        scalar_values = (
            *displacement.values(),
            *stress.values(),
            self.patch_max_error,
            self.objectivity_invariant_change,
            self.angular_skew_ratio,
            self.energy_relative_error,
            self.reaction_relative_error,
            self.small_strain_relative_error,
            self.estimator_bracketing_rate,
            self.estimator_conservative_rate,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in scalar_values):
            raise ValueError("S8 measurements must be finite and non-negative")
        if (
            self.estimator_bracketing_rate > 1.0
            or self.estimator_conservative_rate > 1.0
        ):
            raise ValueError("estimator rates must be between 0 and 1")
        if self.seeded_defects_detected < 0:
            raise ValueError("seeded_defects_detected must be non-negative")
        if self.coverage_percent is not None and (
            not math.isfinite(self.coverage_percent)
            or not 0.0 <= self.coverage_percent <= 100.0
        ):
            raise ValueError("coverage_percent must be between 0 and 100")
        object.__setattr__(self, "displacement_l2_orders", displacement)
        object.__setattr__(self, "first_piola_l2_orders", stress)


@dataclass(frozen=True, slots=True)
class S8GateCheck:
    """One required check with its measurement and proposed threshold."""

    check_id: str
    status: GateStatus
    measured: object
    threshold: object
    comparison: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class S8GateEvaluation:
    """Reduced gate state; PASS requires checks, sign-off, and exact-head CI."""

    checks: tuple[S8GateCheck, ...]
    thresholds_signed_off: bool
    exact_head_ci_complete: bool
    verdict: GateStatus
    verdict_reason: str


def collect_s8_gate_measurements(
    *, coverage_percent: float | None = None
) -> S8GateMeasurements:
    """Execute the required numerical population and retain every gate statistic."""

    studies = run_all_mms_convergence_studies()
    physical = run_s8_physical_verification()
    estimators = calibrate_mms_estimators()
    seeded_defects = run_seeded_defect_suite()
    displacement_orders = {
        f"{study.case_name}:p{study.element_order}": study.displacement_l2_order.order
        for study in studies
    }
    stress_orders = {
        f"{study.case_name}:p{study.element_order}": study.first_piola_l2_order.order
        for study in studies
    }
    return S8GateMeasurements(
        displacement_l2_orders=displacement_orders,
        first_piola_l2_orders=stress_orders,
        patch_max_error=physical.patch_max_error,
        objectivity_invariant_change=physical.objectivity_invariant_change,
        angular_skew_ratio=physical.angular_skew_ratio,
        energy_relative_error=physical.energy_relative_error,
        reaction_relative_error=physical.reaction_relative_error,
        small_strain_relative_error=physical.small_strain_relative_error,
        estimator_bracketing_rate=estimators.minimum_validation_bracketing_rate,
        estimator_conservative_rate=(estimators.minimum_validation_conservative_rate),
        seeded_defects_detected=seeded_defects.detected_count,
        coverage_percent=coverage_percent,
    )


def evaluate_s8_gate(
    measurements: S8GateMeasurements,
    *,
    thresholds_signed_off: bool,
    exact_head_ci_complete: bool,
) -> S8GateEvaluation:
    """Apply the finite G8 table without treating NOT_RUN as success."""

    checks = _evaluated_checks(measurements)
    if not thresholds_signed_off:
        checks = tuple(
            S8GateCheck(
                check_id=check.check_id,
                status=GateStatus.NOT_RUN,
                measured=check.measured,
                threshold=check.threshold,
                comparison=check.comparison,
                reason="proposed threshold awaits explicit human sign-off",
            )
            for check in checks
        )
    blockers: list[str] = []
    if not thresholds_signed_off:
        blockers.append("human sign-off is missing")
    if any(check.status is not GateStatus.PASS for check in checks):
        blockers.append("one or more required checks are not PASS")
    if not exact_head_ci_complete:
        blockers.append("exact-head CI matrix evidence is missing")
    verdict = GateStatus.FAIL if blockers else GateStatus.PASS
    reason = (
        "all required G8 evidence is complete" if not blockers else "; ".join(blockers)
    )
    return S8GateEvaluation(
        checks=checks,
        thresholds_signed_off=thresholds_signed_off,
        exact_head_ci_complete=exact_head_ci_complete,
        verdict=verdict,
        verdict_reason=reason,
    )


def _evaluated_checks(measurements: S8GateMeasurements) -> tuple[S8GateCheck, ...]:
    p1_orders: FrozenMap[float] = FrozenMap.copy_of(
        {
            key: value
            for key, value in measurements.displacement_l2_orders.items()
            if key.endswith(":p1")
        }
    )
    p2_orders: FrozenMap[float] = FrozenMap.copy_of(
        {
            key: value
            for key, value in measurements.displacement_l2_orders.items()
            if key.endswith(":p2")
        }
    )
    stress_orders: FrozenMap[float] = FrozenMap.copy_of(
        measurements.first_piola_l2_orders
    )
    return (
        _check(
            "G8.order-p1",
            p1_orders,
            "max |observed - 2| <= 0.15",
            "lte",
            max(abs(value - 2.0) for value in p1_orders.values()) <= _ORDER_TOLERANCE,
        ),
        _check(
            "G8.order-p2",
            p2_orders,
            "max |observed - 3| <= 0.15",
            "lte",
            max(abs(value - 3.0) for value in p2_orders.values()) <= _ORDER_TOLERANCE,
        ),
        _check(
            "G8.order-stress",
            stress_orders,
            "max |observed - element order| <= 0.2",
            "lte",
            max(
                abs(value - float(int(key[-1]))) for key, value in stress_orders.items()
            )
            <= _STRESS_ORDER_TOLERANCE,
        ),
        _check(
            "G8.patch",
            measurements.patch_max_error,
            _PATCH_LIMIT,
            "lte",
            measurements.patch_max_error <= _PATCH_LIMIT,
        ),
        _check(
            "G8.objectivity",
            measurements.objectivity_invariant_change,
            _OBJECTIVITY_LIMIT,
            "lte",
            measurements.objectivity_invariant_change <= _OBJECTIVITY_LIMIT,
        ),
        _check(
            "G8.angular",
            measurements.angular_skew_ratio,
            _ANGULAR_LIMIT,
            "lt",
            measurements.angular_skew_ratio < _ANGULAR_LIMIT,
        ),
        _check(
            "G8.energy",
            measurements.energy_relative_error,
            _ENERGY_LIMIT,
            "lt",
            measurements.energy_relative_error < _ENERGY_LIMIT,
        ),
        _check(
            "G8.reaction",
            measurements.reaction_relative_error,
            _REACTION_LIMIT,
            "lt",
            measurements.reaction_relative_error < _REACTION_LIMIT,
        ),
        _check(
            "G8.small-strain",
            measurements.small_strain_relative_error,
            _SMALL_STRAIN_LIMIT,
            "lt",
            measurements.small_strain_relative_error < _SMALL_STRAIN_LIMIT,
        ),
        _check(
            "G8.estimator-brackets",
            measurements.estimator_bracketing_rate,
            _ESTIMATOR_BRACKETING_MINIMUM,
            "gte",
            measurements.estimator_bracketing_rate >= _ESTIMATOR_BRACKETING_MINIMUM,
        ),
        _check(
            "G8.estimator-conservative",
            measurements.estimator_conservative_rate,
            _ESTIMATOR_CONSERVATIVE_MINIMUM,
            "gte",
            measurements.estimator_conservative_rate >= _ESTIMATOR_CONSERVATIVE_MINIMUM,
        ),
        _check(
            "G8.seeded-defects",
            measurements.seeded_defects_detected,
            _SEEDED_DEFECT_COUNT,
            "eq",
            measurements.seeded_defects_detected == _SEEDED_DEFECT_COUNT,
        ),
        _coverage_check(measurements.coverage_percent),
    )


def _check(
    check_id: str,
    measured: object,
    threshold: object,
    comparison: str,
    passes: bool,
) -> S8GateCheck:
    return S8GateCheck(
        check_id=check_id,
        status=GateStatus.PASS if passes else GateStatus.FAIL,
        measured=measured,
        threshold=threshold,
        comparison=comparison,
    )


def _coverage_check(coverage_percent: float | None) -> S8GateCheck:
    if coverage_percent is None:
        return S8GateCheck(
            check_id="G8.coverage",
            status=GateStatus.NOT_RUN,
            measured=None,
            threshold=_FEM_COVERAGE_MINIMUM,
            comparison="gte",
            reason="piginot.fem coverage measurement was not supplied",
        )
    return _check(
        "G8.coverage",
        coverage_percent,
        _FEM_COVERAGE_MINIMUM,
        "gte",
        coverage_percent >= _FEM_COVERAGE_MINIMUM,
    )


__all__ = (
    "GateStatus",
    "S8GateCheck",
    "S8GateEvaluation",
    "S8GateMeasurements",
    "collect_s8_gate_measurements",
    "evaluate_s8_gate",
)
