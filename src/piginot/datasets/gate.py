"""Fail-closed reducer for the finite S10 gate evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

_MINIMUM_CODE_COVERAGE_PERCENT = 85.0


class GateStatus(StrEnum):
    """Status vocabulary required by the project gate schema."""

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    NOT_RUN = "NOT_RUN"


def _non_negative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _optional_boolean(value: object, label: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean or None")
    return value


def _optional_coverage(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("code_coverage_percent must be a finite number")
    coverage = float(value)
    if not math.isfinite(coverage):
        raise ValueError("code_coverage_percent must be a finite number")
    if not 0.0 <= coverage <= 100.0:
        raise ValueError("code_coverage_percent must lie in [0, 100]")
    return coverage


@dataclass(frozen=True, slots=True)
class S10GateMeasurements:
    """Complete finite evidence consumed by the G10 reducer."""

    sample_count: int
    fem_result_trace_count: int
    sampler_stream_trace_count: int
    ambiguous_sample_trace_count: int
    cp_manifest_binding_verified: bool | None
    request_plan_binding_verified: bool | None
    fem_result_store_resolution_verified: bool | None
    projection_field_count: int
    projection_error_artifact_count: int
    source_verified_projection_count: int
    drop_projection_error_mutant_fired: bool | None
    physical_integral_comparison_count: int
    source_verified_integral_comparison_count: int
    stress_node_pairing_mutant_fired: bool | None
    normalization_statistic_count: int
    normalization_non_train_source_count: int
    normalization_non_active_source_count: int
    normalization_snapshot_binding_verified: bool | None
    normalize_on_all_splits_mutant_fired: bool | None
    failed_fem_case_count: int
    represented_failed_fem_case_count: int
    collocation_replay_count: int
    byte_identical_replay_count: int
    collocation_request_binding_verified: bool | None
    code_coverage_percent: float | None

    def __post_init__(self) -> None:
        integer_fields = (
            "sample_count",
            "fem_result_trace_count",
            "sampler_stream_trace_count",
            "ambiguous_sample_trace_count",
            "projection_field_count",
            "projection_error_artifact_count",
            "source_verified_projection_count",
            "physical_integral_comparison_count",
            "source_verified_integral_comparison_count",
            "normalization_statistic_count",
            "normalization_non_train_source_count",
            "normalization_non_active_source_count",
            "failed_fem_case_count",
            "represented_failed_fem_case_count",
            "collocation_replay_count",
            "byte_identical_replay_count",
        )
        for field_name in integer_fields:
            object.__setattr__(
                self,
                field_name,
                _non_negative_integer(getattr(self, field_name), field_name),
            )
        boolean_fields = (
            "cp_manifest_binding_verified",
            "request_plan_binding_verified",
            "fem_result_store_resolution_verified",
            "drop_projection_error_mutant_fired",
            "stress_node_pairing_mutant_fired",
            "normalization_snapshot_binding_verified",
            "normalize_on_all_splits_mutant_fired",
            "collocation_request_binding_verified",
        )
        for field_name in boolean_fields:
            object.__setattr__(
                self,
                field_name,
                _optional_boolean(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "code_coverage_percent",
            _optional_coverage(self.code_coverage_percent),
        )


@dataclass(frozen=True, slots=True)
class S10GateCheck:
    """One required G10 check and its measured value."""

    check_id: str
    status: GateStatus
    measured: object
    threshold: object
    comparison: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class S10GateEvaluation:
    """Reduced G10 verdict."""

    checks: tuple[S10GateCheck, ...]
    verdict: GateStatus
    verdict_reason: str

    def by_id(self, check_id: str) -> S10GateCheck:
        for check in self.checks:
            if check.check_id == check_id:
                return check
        raise KeyError(check_id)


def evaluate_s10_gate(measurements: S10GateMeasurements) -> S10GateEvaluation:
    """Apply the exact S10 table without allowing absent evidence to pass."""

    if not isinstance(measurements, S10GateMeasurements):
        raise TypeError("measurements must be S10GateMeasurements")
    checks = (
        _traceability_check(measurements),
        _projection_error_check(measurements),
        _cross_mesh_check(measurements),
        _normalization_check(measurements),
        _failed_visible_check(measurements),
        _reproducible_check(measurements),
        _coverage_check(measurements.code_coverage_percent),
    )
    blockers = tuple(
        check.check_id for check in checks if check.status is not GateStatus.PASS
    )
    return S10GateEvaluation(
        checks=checks,
        verdict=GateStatus.FAIL if blockers else GateStatus.PASS,
        verdict_reason=(
            "all seven required G10 checks passed"
            if not blockers
            else "required checks are not PASS: " + ", ".join(blockers)
        ),
    )


def _traceability_check(measurements: S10GateMeasurements) -> S10GateCheck:
    measured = {
        "samples": measurements.sample_count,
        "fem_result_hash": measurements.fem_result_trace_count,
        "sampler_stream": measurements.sampler_stream_trace_count,
        "ambiguous": measurements.ambiguous_sample_trace_count,
        "cp_manifest_binding_verified": measurements.cp_manifest_binding_verified,
        "request_plan_binding_verified": measurements.request_plan_binding_verified,
        "fem_result_store_resolution_verified": (
            measurements.fem_result_store_resolution_verified
        ),
    }
    threshold = {
        "samples": ">= 1",
        "exactly_one_trace_per_sample": True,
        "ambiguous": 0,
        "cp_manifest_binding_verified": True,
        "request_plan_binding_verified": True,
        "fem_result_store_resolution_verified": True,
    }
    if measurements.sample_count == 0 or any(
        value is None
        for value in (
            measurements.cp_manifest_binding_verified,
            measurements.request_plan_binding_verified,
            measurements.fem_result_store_resolution_verified,
        )
    ):
        return S10GateCheck(
            "G10.traceability",
            GateStatus.NOT_RUN,
            measured,
            threshold,
            "all",
            "no samples were traced or checkpoint binding was not verified",
        )
    passed = (
        measurements.fem_result_trace_count + measurements.sampler_stream_trace_count
        == measurements.sample_count
        and measurements.ambiguous_sample_trace_count == 0
        and measurements.cp_manifest_binding_verified
        and measurements.request_plan_binding_verified
        and measurements.fem_result_store_resolution_verified
    )
    return S10GateCheck(
        "G10.traceability",
        GateStatus.PASS if passed else GateStatus.FAIL,
        measured,
        threshold,
        "all",
    )


def _projection_error_check(measurements: S10GateMeasurements) -> S10GateCheck:
    measured = {
        "projected_fields": measurements.projection_field_count,
        "error_artifacts": measurements.projection_error_artifact_count,
        "source_verified": measurements.source_verified_projection_count,
        "drop_projection_error_mutant_fired": (
            measurements.drop_projection_error_mutant_fired
        ),
    }
    threshold = {
        "projected_fields": ">= 1",
        "one_error_artifact_per_field": True,
        "source_verified_fields": "equals projected_fields",
        "drop_projection_error_mutant_fired": True,
    }
    if (
        measurements.projection_field_count == 0
        or measurements.drop_projection_error_mutant_fired is None
    ):
        return S10GateCheck(
            "G10.projection-error",
            GateStatus.NOT_RUN,
            measured,
            threshold,
            "all",
            "projection population or mutant evidence is missing",
        )
    passed = (
        measurements.projection_error_artifact_count
        == measurements.projection_field_count
        and measurements.source_verified_projection_count
        == measurements.projection_field_count
        and measurements.drop_projection_error_mutant_fired
    )
    return S10GateCheck(
        "G10.projection-error",
        GateStatus.PASS if passed else GateStatus.FAIL,
        measured,
        threshold,
        "all",
    )


def _cross_mesh_check(measurements: S10GateMeasurements) -> S10GateCheck:
    measured = {
        "physical_integral_comparisons": (
            measurements.physical_integral_comparison_count
        ),
        "source_verified": measurements.source_verified_integral_comparison_count,
        "stress_node_pairing_mutant_fired": (
            measurements.stress_node_pairing_mutant_fired
        ),
    }
    threshold = {
        "physical_integral_comparisons": ">= 1",
        "source_verified_comparisons": "equals physical_integral_comparisons",
        "stress_node_pairing_mutant_fired": True,
    }
    if (
        measurements.physical_integral_comparison_count == 0
        or measurements.stress_node_pairing_mutant_fired is None
    ):
        return S10GateCheck(
            "G10.cross-mesh",
            GateStatus.NOT_RUN,
            measured,
            threshold,
            "all",
            "physical-domain comparison or mutant evidence is missing",
        )
    return S10GateCheck(
        "G10.cross-mesh",
        (
            GateStatus.PASS
            if measurements.stress_node_pairing_mutant_fired
            and measurements.source_verified_integral_comparison_count
            == measurements.physical_integral_comparison_count
            else GateStatus.FAIL
        ),
        measured,
        threshold,
        "all",
    )


def _normalization_check(measurements: S10GateMeasurements) -> S10GateCheck:
    measured = {
        "statistics": measurements.normalization_statistic_count,
        "non_train_sources": measurements.normalization_non_train_source_count,
        "non_active_sources": measurements.normalization_non_active_source_count,
        "training_snapshot_binding_verified": (
            measurements.normalization_snapshot_binding_verified
        ),
        "normalize_on_all_splits_mutant_fired": (
            measurements.normalize_on_all_splits_mutant_fired
        ),
    }
    threshold = {
        "statistics": ">= 1",
        "non_train_sources": 0,
        "non_active_sources": 0,
        "training_snapshot_binding_verified": True,
        "normalize_on_all_splits_mutant_fired": True,
    }
    if (
        measurements.normalization_statistic_count == 0
        or measurements.normalization_snapshot_binding_verified is None
        or measurements.normalize_on_all_splits_mutant_fired is None
    ):
        return S10GateCheck(
            "G10.normalization",
            GateStatus.NOT_RUN,
            measured,
            threshold,
            "all",
            "normalization population or mutant evidence is missing",
        )
    passed = (
        measurements.normalization_non_train_source_count == 0
        and measurements.normalization_non_active_source_count == 0
        and measurements.normalization_snapshot_binding_verified
        and measurements.normalize_on_all_splits_mutant_fired
    )
    return S10GateCheck(
        "G10.normalization",
        GateStatus.PASS if passed else GateStatus.FAIL,
        measured,
        threshold,
        "all",
    )


def _failed_visible_check(measurements: S10GateMeasurements) -> S10GateCheck:
    measured = {
        "failed_fem_cases": measurements.failed_fem_case_count,
        "represented_failed_fem_cases": (
            measurements.represented_failed_fem_case_count
        ),
    }
    threshold = {
        "failed_fem_cases": ">= 1",
        "represented_failed_fem_cases": "equals failed_fem_cases",
    }
    if measurements.failed_fem_case_count == 0:
        return S10GateCheck(
            "G10.failed-visible",
            GateStatus.NOT_RUN,
            measured,
            threshold,
            "all",
            "no failed FEM case exercised manifest visibility",
        )
    return S10GateCheck(
        "G10.failed-visible",
        (
            GateStatus.PASS
            if measurements.represented_failed_fem_case_count
            == measurements.failed_fem_case_count
            else GateStatus.FAIL
        ),
        measured,
        threshold,
        "eq",
    )


def _reproducible_check(measurements: S10GateMeasurements) -> S10GateCheck:
    measured = {
        "replays": measurements.collocation_replay_count,
        "byte_identical": measurements.byte_identical_replay_count,
        "request_binding_verified": (measurements.collocation_request_binding_verified),
    }
    threshold = {
        "replays": ">= 1",
        "byte_identical": "equals replays",
        "request_binding_verified": True,
    }
    if (
        measurements.collocation_replay_count == 0
        or measurements.collocation_request_binding_verified is None
    ):
        return S10GateCheck(
            "G10.reproducible",
            GateStatus.NOT_RUN,
            measured,
            threshold,
            "all",
            "no collocation stream was replayed",
        )
    return S10GateCheck(
        "G10.reproducible",
        (
            GateStatus.PASS
            if measurements.byte_identical_replay_count
            == measurements.collocation_replay_count
            and measurements.collocation_request_binding_verified
            else GateStatus.FAIL
        ),
        measured,
        threshold,
        "eq",
    )


def _coverage_check(coverage: float | None) -> S10GateCheck:
    if coverage is None:
        return S10GateCheck(
            "G10.coverage",
            GateStatus.NOT_RUN,
            None,
            _MINIMUM_CODE_COVERAGE_PERCENT,
            "gte",
            "S10 source-package coverage was not measured",
        )
    return S10GateCheck(
        "G10.coverage",
        (
            GateStatus.PASS
            if coverage >= _MINIMUM_CODE_COVERAGE_PERCENT
            else GateStatus.FAIL
        ),
        coverage,
        _MINIMUM_CODE_COVERAGE_PERCENT,
        "gte",
    )


__all__ = (
    "GateStatus",
    "S10GateCheck",
    "S10GateEvaluation",
    "S10GateMeasurements",
    "evaluate_s10_gate",
)
