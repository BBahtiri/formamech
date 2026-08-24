"""Reducer semantics for the human-signoff S8 gate."""

from __future__ import annotations

from dataclasses import replace

import pytest

from piginot.fem.manufactured import MANUFACTURED_CASE_NAMES
from piginot.fem.s8_gate import (
    EstimatorGateEvidence,
    GateStatus,
    S8GateMeasurements,
    evaluate_s8_gate,
)
from piginot.fem.verification_meshes import VerificationMeshKind


def _estimator_evidence(
    *,
    safety_factor: float = 1.97,
    calibration_population_size: int = 16,
    validation_population_size: int = 16,
    bracketed_count: int = 16,
    conservative_count: int = 16,
    validation_mesh_kinds: tuple[VerificationMeshKind, ...] = (
        VerificationMeshKind.STRUCTURED,
        VerificationMeshKind.UNSTRUCTURED,
    ),
    validation_mesh_population_sizes: tuple[int, ...] = (8, 8),
) -> EstimatorGateEvidence:
    return EstimatorGateEvidence(
        safety_factor=safety_factor,
        calibration_population_size=calibration_population_size,
        validation_population_size=validation_population_size,
        bracketed_count=bracketed_count,
        conservative_count=conservative_count,
        calibration_mesh_kind=VerificationMeshKind.STRUCTURED,
        validation_mesh_kinds=validation_mesh_kinds,
        validation_mesh_population_sizes=validation_mesh_population_sizes,
    )


def _passing_measurements(
    *, coverage_percent: float | None = 90.0
) -> S8GateMeasurements:
    return S8GateMeasurements(
        displacement_l2_orders={
            f"{case_name}:p{order}": float(order + 1)
            for case_name in MANUFACTURED_CASE_NAMES
            for order in (1, 2)
        },
        first_piola_l2_orders={
            f"{case_name}:p{order}": float(order)
            for case_name in MANUFACTURED_CASE_NAMES
            for order in (1, 2)
        },
        patch_max_error=1.0e-13,
        objectivity_invariant_change=1.0e-13,
        angular_skew_ratio=1.0e-13,
        energy_relative_error=1.0e-10,
        reaction_relative_error=1.0e-11,
        small_strain_relative_error=1.0e-3,
        estimator_families={
            "richardson": _estimator_evidence(),
            "recovery": _estimator_evidence(
                safety_factor=2.61,
                calibration_population_size=24,
            ),
        },
        seeded_defects_detected=6,
        coverage_percent=coverage_percent,
    )


def test_unsigned_thresholds_are_not_run_and_block_the_gate() -> None:
    evaluation = evaluate_s8_gate(
        _passing_measurements(),
        thresholds_signed_off=False,
        exact_head_ci_complete=True,
    )

    assert evaluation.verdict is GateStatus.FAIL
    assert all(check.status is GateStatus.NOT_RUN for check in evaluation.checks)
    assert evaluation.checks[0].measured
    assert "human sign-off" in evaluation.verdict_reason


def test_missing_coverage_and_ci_each_block_an_otherwise_passing_gate() -> None:
    evaluation = evaluate_s8_gate(
        _passing_measurements(coverage_percent=None),
        thresholds_signed_off=True,
        exact_head_ci_complete=False,
    )

    checks = {check.check_id: check for check in evaluation.checks}
    assert checks["G8.coverage"].status is GateStatus.NOT_RUN
    assert all(
        check.status is GateStatus.PASS
        for check_id, check in checks.items()
        if check_id != "G8.coverage"
    )
    assert evaluation.verdict is GateStatus.FAIL
    assert "exact-head CI" in evaluation.verdict_reason


def test_signed_thresholds_complete_ci_and_all_measurements_pass() -> None:
    evaluation = evaluate_s8_gate(
        _passing_measurements(),
        thresholds_signed_off=True,
        exact_head_ci_complete=True,
    )

    assert tuple(check.check_id for check in evaluation.checks) == (
        "G8.order-p1",
        "G8.order-p2",
        "G8.order-stress",
        "G8.patch",
        "G8.objectivity",
        "G8.angular",
        "G8.energy",
        "G8.reaction",
        "G8.small-strain",
        "G8.estimator-brackets",
        "G8.estimator-conservative",
        "G8.seeded-defects",
        "G8.coverage",
    )
    assert all(check.status is GateStatus.PASS for check in evaluation.checks)
    assert evaluation.verdict is GateStatus.PASS


def test_estimator_checks_retain_counts_and_have_distinct_outcomes_at_15_of_16() -> (
    None
):
    measurements = replace(
        _passing_measurements(),
        estimator_families={
            "richardson": _estimator_evidence(
                bracketed_count=15,
                conservative_count=15,
            ),
            "recovery": _estimator_evidence(
                safety_factor=2.61,
                calibration_population_size=24,
                bracketed_count=15,
                conservative_count=15,
            ),
        },
    )

    evaluation = evaluate_s8_gate(
        measurements,
        thresholds_signed_off=True,
        exact_head_ci_complete=True,
    )

    checks = {check.check_id: check for check in evaluation.checks}
    assert checks["G8.estimator-brackets"].status is GateStatus.FAIL
    assert checks["G8.estimator-conservative"].status is GateStatus.PASS
    measured = checks["G8.estimator-brackets"].measured
    assert measured["richardson"]["count"] == 15
    assert measured["richardson"]["population_size"] == 16
    assert measured["richardson"]["safety_factor"] == pytest.approx(1.97)
    assert measured["richardson"]["validation_mesh_kinds"] == (
        "structured",
        "unstructured",
    )
    assert measured["richardson"]["validation_mesh_population_sizes"] == (8, 8)


def test_estimator_evidence_must_match_the_fixed_g8_population() -> None:
    passing = _passing_measurements()
    with pytest.raises(ValueError, match="fixed G8 estimator population"):
        replace(
            passing,
            estimator_families={
                "richardson": _estimator_evidence(
                    validation_population_size=1,
                    bracketed_count=1,
                    conservative_count=1,
                    validation_mesh_kinds=(VerificationMeshKind.STRUCTURED,),
                    validation_mesh_population_sizes=(1,),
                ),
                "recovery": passing.estimator_families["recovery"],
            },
        )


def test_one_failed_required_measurement_fails_its_named_check_and_gate() -> None:
    passing = _passing_measurements()
    failing = replace(passing, patch_max_error=2.0e-12)
    evaluation = evaluate_s8_gate(
        failing,
        thresholds_signed_off=True,
        exact_head_ci_complete=True,
    )

    checks = {check.check_id: check for check in evaluation.checks}
    assert checks["G8.patch"].status is GateStatus.FAIL
    assert evaluation.verdict is GateStatus.FAIL


def test_estimator_counts_outside_population_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        _estimator_evidence(bracketed_count=17)
