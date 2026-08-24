"""Reducer semantics for the human-signoff S8 gate."""

from __future__ import annotations

from dataclasses import replace

import pytest

from piginot.fem.manufactured import MANUFACTURED_CASE_NAMES
from piginot.fem.s8_gate import (
    GateStatus,
    S8GateMeasurements,
    evaluate_s8_gate,
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
        estimator_bracketing_rate=1.0,
        estimator_conservative_rate=1.0,
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


def test_one_failed_required_measurement_fails_its_named_check_and_gate() -> None:
    passing = _passing_measurements()
    failing = S8GateMeasurements(
        displacement_l2_orders=passing.displacement_l2_orders,
        first_piola_l2_orders=passing.first_piola_l2_orders,
        patch_max_error=2.0e-12,
        objectivity_invariant_change=passing.objectivity_invariant_change,
        angular_skew_ratio=passing.angular_skew_ratio,
        energy_relative_error=passing.energy_relative_error,
        reaction_relative_error=passing.reaction_relative_error,
        small_strain_relative_error=passing.small_strain_relative_error,
        estimator_bracketing_rate=passing.estimator_bracketing_rate,
        estimator_conservative_rate=passing.estimator_conservative_rate,
        seeded_defects_detected=passing.seeded_defects_detected,
        coverage_percent=passing.coverage_percent,
    )
    evaluation = evaluate_s8_gate(
        failing,
        thresholds_signed_off=True,
        exact_head_ci_complete=True,
    )

    checks = {check.check_id: check for check in evaluation.checks}
    assert checks["G8.patch"].status is GateStatus.FAIL
    assert evaluation.verdict is GateStatus.FAIL


def test_estimator_rates_outside_probability_range_are_rejected() -> None:
    with pytest.raises(ValueError, match="estimator rates"):
        replace(_passing_measurements(), estimator_bracketing_rate=1.01)
