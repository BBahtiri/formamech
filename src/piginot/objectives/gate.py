"""Fail-closed reducer for the finite S12 objective-verification gate."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from piginot.objectives.verification import (
    PHYSICS_TERM_IDS,
    VERIFIED_GRADIENT_TERM_IDS,
)

_EXACT_LIMIT = 1.0e-8
_FINITE_DIFFERENCE_LIMIT = 1.0e-5
_ANISOTROPY_LIMIT = 1.0e-12
_MINIMUM_COVERAGE_PERCENT = 85.0


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    NOT_RUN = "NOT_RUN"


@dataclass(frozen=True, slots=True)
class S12GateMeasurements:
    """Complete finite measurement set consumed by G12."""

    exact_max_residuals: Mapping[str, float]
    perturbation_fired_terms: Mapping[str, tuple[str, ...]]
    finite_difference_relative_errors: Mapping[str, float]
    detached_terms: tuple[str, ...]
    reachable_parameter_names: tuple[str, ...]
    required_parameter_names: tuple[str, ...]
    anisotropy_relative_change: float
    coordinate_anisotropy_mutant_fired: bool
    fem_solve_success: bool
    weak_strong_difference: float
    s8_discretization_error_bound: float
    constitutive_residual_shape: tuple[int, int]
    constitutive_rejects_out_of_plane: bool
    holdout_refused: bool
    holdout_access_mutant_fired: bool
    code_coverage_percent: float | None

    def __post_init__(self) -> None:
        exact = _metric_mapping(self.exact_max_residuals, "exact_max_residuals")
        finite_difference = _metric_mapping(
            self.finite_difference_relative_errors,
            "finite_difference_relative_errors",
        )
        perturbations = _tuple_mapping(self.perturbation_fired_terms)
        object.__setattr__(self, "exact_max_residuals", MappingProxyType(exact))
        object.__setattr__(
            self,
            "finite_difference_relative_errors",
            MappingProxyType(finite_difference),
        )
        object.__setattr__(
            self,
            "perturbation_fired_terms",
            MappingProxyType(perturbations),
        )
        for name in ("detached_terms", "reachable_parameter_names", "required_parameter_names"):
            value = getattr(self, name)
            if type(value) is not tuple or any(
                not isinstance(item, str) or not item for item in value
            ):
                raise ValueError(f"{name} must contain non-empty names")
        for name in (
            "coordinate_anisotropy_mutant_fired",
            "fem_solve_success",
            "constitutive_rejects_out_of_plane",
            "holdout_refused",
            "holdout_access_mutant_fired",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        for name in ("anisotropy_relative_change", "weak_strong_difference"):
            _non_negative_metric(getattr(self, name), name)
        bound = _non_negative_metric(
            self.s8_discretization_error_bound,
            "s8_discretization_error_bound",
        )
        if bound <= 0.0:
            raise ValueError("s8_discretization_error_bound must be positive")
        shape = self.constitutive_residual_shape
        if (
            type(shape) is not tuple
            or len(shape) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
        ):
            raise ValueError("constitutive_residual_shape must contain two positive integers")
        if self.code_coverage_percent is not None:
            coverage = _non_negative_metric(
                self.code_coverage_percent, "code_coverage_percent"
            )
            if coverage > 100.0:
                raise ValueError("code_coverage_percent must lie in [0, 100]")


@dataclass(frozen=True, slots=True)
class S12GateCheck:
    check_id: str
    status: GateStatus
    measured: object
    threshold: object
    comparison: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class S12GateEvaluation:
    checks: tuple[S12GateCheck, ...]
    verdict: GateStatus

    def by_id(self, check_id: str) -> S12GateCheck:
        for check in self.checks:
            if check.check_id == check_id:
                return check
        raise KeyError(check_id)


def evaluate_s12_gate(measurements: S12GateMeasurements) -> S12GateEvaluation:
    if type(measurements) is not S12GateMeasurements:
        raise TypeError("measurements must be exact S12GateMeasurements")
    checks = (
        _exact_check(measurements),
        _perturbed_check(measurements),
        _finite_difference_check(measurements),
        _reachability_check(measurements),
        _anisotropy_check(measurements),
        _weak_strong_check(measurements),
        _constitutive_shape_check(measurements),
        _role_isolation_check(measurements),
        _coverage_check(measurements.code_coverage_percent),
    )
    verdict = (
        GateStatus.PASS
        if all(check.status is GateStatus.PASS for check in checks)
        else GateStatus.FAIL
    )
    return S12GateEvaluation(checks, verdict)


def _exact_check(measurements: S12GateMeasurements) -> S12GateCheck:
    complete = set(measurements.exact_max_residuals) == set(PHYSICS_TERM_IDS)
    passed = complete and all(
        value < _EXACT_LIMIT for value in measurements.exact_max_residuals.values()
    )
    return S12GateCheck(
        "G12.exact",
        GateStatus.PASS if passed else GateStatus.FAIL,
        measurements.exact_max_residuals,
        {"required_terms": PHYSICS_TERM_IDS, "max_residual": "< 1e-8"},
        "all",
    )


def _perturbed_check(measurements: S12GateMeasurements) -> S12GateCheck:
    complete = set(measurements.perturbation_fired_terms) == set(PHYSICS_TERM_IDS)
    passed = complete and all(
        measurements.perturbation_fired_terms.get(term) == (term,)
        for term in PHYSICS_TERM_IDS
    )
    return S12GateCheck(
        "G12.perturbed",
        GateStatus.PASS if passed else GateStatus.FAIL,
        measurements.perturbation_fired_terms,
        "each required physics fixture fires exactly its named term",
        "all",
    )


def _finite_difference_check(measurements: S12GateMeasurements) -> S12GateCheck:
    complete = set(measurements.finite_difference_relative_errors) == set(
        VERIFIED_GRADIENT_TERM_IDS
    )
    passed = (
        complete
        and not measurements.detached_terms
        and all(
            value < _FINITE_DIFFERENCE_LIMIT
            for value in measurements.finite_difference_relative_errors.values()
        )
    )
    measured = {
        "relative_errors": measurements.finite_difference_relative_errors,
        "detached_terms": measurements.detached_terms,
    }
    return S12GateCheck(
        "G12.fd",
        GateStatus.PASS if passed else GateStatus.FAIL,
        measured,
        {
            "required_terms": VERIFIED_GRADIENT_TERM_IDS,
            "max_relative_error": "< 1e-5",
            "detached_terms": (),
        },
        "all",
    )


def _reachability_check(measurements: S12GateMeasurements) -> S12GateCheck:
    passed = (
        bool(measurements.required_parameter_names)
        and measurements.reachable_parameter_names
        == measurements.required_parameter_names
    )
    return S12GateCheck(
        "G12.reachability",
        GateStatus.PASS if passed else GateStatus.FAIL,
        {
            "reachable": measurements.reachable_parameter_names,
            "required": measurements.required_parameter_names,
        },
        "reachable == required and required is non-empty",
        "eq",
    )


def _anisotropy_check(measurements: S12GateMeasurements) -> S12GateCheck:
    passed = (
        measurements.anisotropy_relative_change < _ANISOTROPY_LIMIT
        and measurements.coordinate_anisotropy_mutant_fired
    )
    return S12GateCheck(
        "G12.anisotropy",
        GateStatus.PASS if passed else GateStatus.FAIL,
        {
            "relative_change": measurements.anisotropy_relative_change,
            "COORDINATE_ANISOTROPY_mutant_fired": (
                measurements.coordinate_anisotropy_mutant_fired
            ),
        },
        {"relative_change": "< 1e-12", "mutant_fired": True},
        "all",
    )


def _weak_strong_check(measurements: S12GateMeasurements) -> S12GateCheck:
    passed = (
        measurements.fem_solve_success
        and measurements.weak_strong_difference
        <= measurements.s8_discretization_error_bound
    )
    return S12GateCheck(
        "G12.weak-strong",
        GateStatus.PASS if passed else GateStatus.FAIL,
        {
            "fem_solve_success": measurements.fem_solve_success,
            "difference": measurements.weak_strong_difference,
        },
        {"maximum": measurements.s8_discretization_error_bound},
        "lte",
    )


def _constitutive_shape_check(measurements: S12GateMeasurements) -> S12GateCheck:
    passed = (
        measurements.constitutive_residual_shape == (2, 2)
        and measurements.constitutive_rejects_out_of_plane
    )
    return S12GateCheck(
        "G12.constitutive-shape",
        GateStatus.PASS if passed else GateStatus.FAIL,
        {
            "residual_shape": measurements.constitutive_residual_shape,
            "rejects_out_of_plane": measurements.constitutive_rejects_out_of_plane,
        },
        {"residual_shape": (2, 2), "rejects_out_of_plane": True},
        "all",
    )


def _role_isolation_check(measurements: S12GateMeasurements) -> S12GateCheck:
    passed = measurements.holdout_refused and measurements.holdout_access_mutant_fired
    return S12GateCheck(
        "G12.role-isolation",
        GateStatus.PASS if passed else GateStatus.FAIL,
        {
            "holdout_refused": measurements.holdout_refused,
            "HOLDOUT_ACCESS_mutant_fired": measurements.holdout_access_mutant_fired,
        },
        {"holdout_refused": True, "mutant_fired": True},
        "all",
    )


def _coverage_check(value: float | None) -> S12GateCheck:
    if value is None:
        return S12GateCheck(
            "G12.coverage",
            GateStatus.NOT_RUN,
            None,
            ">= 85.0",
            "gte",
            "piginot.objectives coverage measurement was not supplied",
        )
    return S12GateCheck(
        "G12.coverage",
        GateStatus.PASS if value >= _MINIMUM_COVERAGE_PERCENT else GateStatus.FAIL,
        value,
        ">= 85.0",
        "gte",
    )


def _metric_mapping(value: Mapping[str, float], label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result: dict[str, float] = {}
    for key, metric in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        result[key] = _non_negative_metric(metric, label)
    return result


def _tuple_mapping(value: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise TypeError("perturbation_fired_terms must be a mapping")
    result: dict[str, tuple[str, ...]] = {}
    for key, names in value.items():
        if not isinstance(key, str) or not key or type(names) is not tuple:
            raise ValueError("perturbation evidence must map names to tuples")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("perturbation fired terms must be non-empty strings")
        result[key] = names
    return result


def _non_negative_metric(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    metric = float(value)
    if not math.isfinite(metric) or metric < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return metric


__all__ = (
    "GateStatus",
    "S12GateCheck",
    "S12GateEvaluation",
    "S12GateMeasurements",
    "evaluate_s12_gate",
)
