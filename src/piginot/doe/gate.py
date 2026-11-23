"""Fail-closed reducer for the finite S9 gate evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

from piginot.contracts.state_machine import GateStatus as GateStatus

_REQUIRED_PROPERTY_TRIALS = 1000
_MINIMUM_CODE_COVERAGE_PERCENT = 90.0


def _non_negative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _proportion(value: object, label: str) -> float:
    number = _finite(value, label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return number


def _identifier_tuple(values: object, label: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{label} must be a tuple")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{label} must contain non-empty text")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class DesignCoverageDiagnostics:
    """Non-aggregated DOE diagnostics recorded beside the G9 verdict.

    These measurements deliberately do not produce a synthetic percentage used by
    ``G9.coverage``. That named check is the S9 source-package coverage floor from
    the global gate policy.
    """

    empirical_fill_distance: float
    minimum_stratum_count: int
    boundary_coverage: float
    stratum_coverage: float
    interaction_coverage: float
    difficult_region_coverage: float
    uncovered_region_ids: tuple[str, ...]
    unresolved_region_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        fill_distance = _finite(self.empirical_fill_distance, "empirical_fill_distance")
        if fill_distance < 0.0:
            raise ValueError("empirical_fill_distance must be non-negative")
        object.__setattr__(self, "empirical_fill_distance", fill_distance)
        object.__setattr__(
            self,
            "minimum_stratum_count",
            _non_negative_integer(self.minimum_stratum_count, "minimum_stratum_count"),
        )
        for field_name in (
            "boundary_coverage",
            "stratum_coverage",
            "interaction_coverage",
            "difficult_region_coverage",
        ):
            object.__setattr__(
                self,
                field_name,
                _proportion(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "uncovered_region_ids",
            _identifier_tuple(self.uncovered_region_ids, "uncovered_region_ids"),
        )
        object.__setattr__(
            self,
            "unresolved_region_ids",
            _identifier_tuple(self.unresolved_region_ids, "unresolved_region_ids"),
        )


@dataclass(frozen=True, slots=True)
class ReplacementEvidence:
    """Measurements proving that the replacement rule was exercised."""

    failures_injected: int
    failures_recorded: int
    deterministic_rule_matches: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "failures_injected",
            _non_negative_integer(self.failures_injected, "failures_injected"),
        )
        object.__setattr__(
            self,
            "failures_recorded",
            _non_negative_integer(self.failures_recorded, "failures_recorded"),
        )
        if not isinstance(self.deterministic_rule_matches, bool):
            raise ValueError("deterministic_rule_matches must be boolean")


@dataclass(frozen=True, slots=True)
class S9GateMeasurements:
    """Complete finite evidence consumed by the G9 reducer."""

    immutable_rewrite_raised: bool | None
    one_role_trials: int
    one_role_violations: int
    leakage_trials: int
    leakage_violations: int
    holdout_visibility_violations: int
    node_split_mutant_fired: bool | None
    commitment_file_order_verified: bool | None
    commitment_git_order_verified: bool | None
    reveal_verified: bool | None
    replacement: ReplacementEvidence | None
    code_coverage_percent: float | None
    design_diagnostics: DesignCoverageDiagnostics

    def __post_init__(self) -> None:
        for field_name in (
            "immutable_rewrite_raised",
            "node_split_mutant_fired",
            "commitment_file_order_verified",
            "commitment_git_order_verified",
            "reveal_verified",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{field_name} must be boolean or None")
        for field_name in (
            "one_role_trials",
            "one_role_violations",
            "leakage_trials",
            "leakage_violations",
            "holdout_visibility_violations",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_negative_integer(getattr(self, field_name), field_name),
            )
        if self.one_role_violations > self.one_role_trials:
            raise ValueError("one_role_violations cannot exceed one_role_trials")
        if self.leakage_violations > self.leakage_trials:
            raise ValueError("leakage_violations cannot exceed leakage_trials")
        if self.replacement is not None and not isinstance(
            self.replacement, ReplacementEvidence
        ):
            raise ValueError("replacement must be ReplacementEvidence or None")
        if self.code_coverage_percent is not None:
            coverage = _finite(self.code_coverage_percent, "code_coverage_percent")
            if not 0.0 <= coverage <= 100.0:
                raise ValueError("code_coverage_percent must lie in [0, 100]")
            object.__setattr__(self, "code_coverage_percent", coverage)
        if not isinstance(self.design_diagnostics, DesignCoverageDiagnostics):
            raise ValueError("design_diagnostics must be DesignCoverageDiagnostics")


@dataclass(frozen=True, slots=True)
class S9GateCheck:
    """One required G9 check and its measured value."""

    check_id: str
    status: GateStatus
    measured: object
    threshold: object
    comparison: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class S9GateEvaluation:
    """Reduced S9 verdict plus the non-aggregated design diagnostics."""

    checks: tuple[S9GateCheck, ...]
    verdict: GateStatus
    verdict_reason: str
    design_diagnostics: DesignCoverageDiagnostics

    def by_id(self, check_id: str) -> S9GateCheck:
        for check in self.checks:
            if check.check_id == check_id:
                return check
        raise KeyError(check_id)


def evaluate_s9_gate(measurements: S9GateMeasurements) -> S9GateEvaluation:
    """Apply the exact S9 table without allowing missing evidence to pass."""

    if not isinstance(measurements, S9GateMeasurements):
        raise TypeError("measurements must be S9GateMeasurements")
    checks = (
        _optional_boolean_check("G9.immutable", measurements.immutable_rewrite_raised),
        _property_check(
            "G9.one-role",
            measurements.one_role_trials,
            measurements.one_role_violations,
        ),
        _property_check(
            "G9.leakage",
            measurements.leakage_trials,
            measurements.leakage_violations,
        ),
        _zero_check("G9.visibility", measurements.holdout_visibility_violations),
        _optional_boolean_check(
            "G9.no-node-split", measurements.node_split_mutant_fired
        ),
        _commitment_check(measurements),
        _replacement_check(measurements.replacement),
        _coverage_check(measurements.code_coverage_percent),
    )
    blockers = tuple(
        check.check_id for check in checks if check.status is not GateStatus.PASS
    )
    return S9GateEvaluation(
        checks=checks,
        verdict=GateStatus.FAIL if blockers else GateStatus.PASS,
        verdict_reason=(
            "all eight required G9 checks passed"
            if not blockers
            else "required checks are not PASS: " + ", ".join(blockers)
        ),
        design_diagnostics=measurements.design_diagnostics,
    )


def _optional_boolean_check(check_id: str, value: bool | None) -> S9GateCheck:
    if value is None:
        return S9GateCheck(
            check_id,
            GateStatus.NOT_RUN,
            None,
            True,
            "eq",
            "required boolean evidence is missing",
        )
    return S9GateCheck(
        check_id,
        GateStatus.PASS if value else GateStatus.FAIL,
        value,
        True,
        "eq",
    )


def _property_check(check_id: str, trials: int, violations: int) -> S9GateCheck:
    measured = {"trials": trials, "violations": violations}
    threshold = {"minimum_trials": _REQUIRED_PROPERTY_TRIALS, "violations": 0}
    if trials < _REQUIRED_PROPERTY_TRIALS:
        return S9GateCheck(
            check_id,
            GateStatus.NOT_RUN,
            measured,
            threshold,
            "gte trials and eq violations",
            "property-test population is incomplete",
        )
    return S9GateCheck(
        check_id,
        GateStatus.PASS if violations == 0 else GateStatus.FAIL,
        measured,
        threshold,
        "gte trials and eq violations",
    )


def _zero_check(check_id: str, value: int) -> S9GateCheck:
    return S9GateCheck(
        check_id,
        GateStatus.PASS if value == 0 else GateStatus.FAIL,
        value,
        0,
        "eq",
    )


def _commitment_check(measurements: S9GateMeasurements) -> S9GateCheck:
    parts = {
        "file_mtime_precedes": measurements.commitment_file_order_verified,
        "git_commit_precedes": measurements.commitment_git_order_verified,
        "reveal_verifies": measurements.reveal_verified,
    }
    if any(value is None for value in parts.values()):
        return S9GateCheck(
            "G9.commitment-order",
            GateStatus.NOT_RUN,
            parts,
            {key: True for key in parts},
            "all true",
            "commitment-order or reveal evidence is missing",
        )
    return S9GateCheck(
        "G9.commitment-order",
        GateStatus.PASS if all(parts.values()) else GateStatus.FAIL,
        parts,
        {key: True for key in parts},
        "all true",
    )


def _replacement_check(evidence: ReplacementEvidence | None) -> S9GateCheck:
    threshold = {
        "failures_injected": ">= 1",
        "failures_recorded": "equals injected",
        "deterministic_rule_matches": True,
    }
    if evidence is None or evidence.failures_injected == 0:
        return S9GateCheck(
            "G9.replacement",
            GateStatus.NOT_RUN,
            evidence,
            threshold,
            "all",
            "no injected sample failure exercised the replacement rule",
        )
    measured = {
        "failures_injected": evidence.failures_injected,
        "failures_recorded": evidence.failures_recorded,
        "deterministic_rule_matches": evidence.deterministic_rule_matches,
    }
    passed = (
        evidence.failures_recorded == evidence.failures_injected
        and evidence.deterministic_rule_matches
    )
    return S9GateCheck(
        "G9.replacement",
        GateStatus.PASS if passed else GateStatus.FAIL,
        measured,
        threshold,
        "all",
    )


def _coverage_check(coverage: float | None) -> S9GateCheck:
    if coverage is None:
        return S9GateCheck(
            "G9.coverage",
            GateStatus.NOT_RUN,
            None,
            _MINIMUM_CODE_COVERAGE_PERCENT,
            "gte",
            "S9 source-package coverage was not measured",
        )
    return S9GateCheck(
        "G9.coverage",
        (
            GateStatus.PASS
            if coverage >= _MINIMUM_CODE_COVERAGE_PERCENT
            else GateStatus.FAIL
        ),
        coverage,
        _MINIMUM_CODE_COVERAGE_PERCENT,
        "gte",
    )
