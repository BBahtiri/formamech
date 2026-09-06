from __future__ import annotations

from dataclasses import replace

import pytest

from piginot.datasets.gate import (
    GateStatus,
    S10GateMeasurements,
    evaluate_s10_gate,
)


def _passing_measurements(**overrides: object) -> S10GateMeasurements:
    values: dict[str, object] = {
        "sample_count": 12,
        "fem_result_trace_count": 9,
        "sampler_stream_trace_count": 3,
        "ambiguous_sample_trace_count": 0,
        "cp_manifest_binding_verified": True,
        "request_plan_binding_verified": True,
        "fem_result_store_resolution_verified": True,
        "projection_field_count": 4,
        "projection_error_artifact_count": 4,
        "source_verified_projection_count": 4,
        "drop_projection_error_mutant_fired": True,
        "physical_integral_comparison_count": 2,
        "source_verified_integral_comparison_count": 2,
        "stress_node_pairing_mutant_fired": True,
        "normalization_statistic_count": 3,
        "normalization_non_train_source_count": 0,
        "normalization_non_active_source_count": 0,
        "normalization_snapshot_binding_verified": True,
        "normalize_on_all_splits_mutant_fired": True,
        "failed_fem_case_count": 2,
        "represented_failed_fem_case_count": 2,
        "failure_visibility_canary_count": 0,
        "represented_failure_visibility_canary_count": 0,
        "collocation_replay_count": 3,
        "byte_identical_replay_count": 3,
        "collocation_request_binding_verified": True,
        "code_coverage_percent": 85.0,
    }
    values.update(overrides)
    return S10GateMeasurements(**values)  # type: ignore[arg-type]


def test_all_seven_required_checks_pass_with_complete_evidence() -> None:
    evaluation = evaluate_s10_gate(_passing_measurements())

    assert evaluation.verdict is GateStatus.PASS
    assert tuple(check.check_id for check in evaluation.checks) == (
        "G10.traceability",
        "G10.projection-error",
        "G10.cross-mesh",
        "G10.normalization",
        "G10.failed-visible",
        "G10.reproducible",
        "G10.coverage",
    )


@pytest.mark.parametrize(
    ("overrides", "check_id", "status"),
    (
        (
            {
                "sample_count": 0,
                "fem_result_trace_count": 0,
                "sampler_stream_trace_count": 0,
            },
            "G10.traceability",
            GateStatus.NOT_RUN,
        ),
        ({"fem_result_trace_count": 8}, "G10.traceability", GateStatus.FAIL),
        ({"ambiguous_sample_trace_count": 1}, "G10.traceability", GateStatus.FAIL),
        ({"cp_manifest_binding_verified": False}, "G10.traceability", GateStatus.FAIL),
        (
            {"request_plan_binding_verified": False},
            "G10.traceability",
            GateStatus.FAIL,
        ),
        (
            {"fem_result_store_resolution_verified": False},
            "G10.traceability",
            GateStatus.FAIL,
        ),
        (
            {"cp_manifest_binding_verified": None},
            "G10.traceability",
            GateStatus.NOT_RUN,
        ),
        (
            {"projection_field_count": 0, "projection_error_artifact_count": 0},
            "G10.projection-error",
            GateStatus.NOT_RUN,
        ),
        (
            {"projection_error_artifact_count": 3},
            "G10.projection-error",
            GateStatus.FAIL,
        ),
        (
            {"source_verified_projection_count": 3},
            "G10.projection-error",
            GateStatus.FAIL,
        ),
        (
            {"drop_projection_error_mutant_fired": False},
            "G10.projection-error",
            GateStatus.FAIL,
        ),
        (
            {"drop_projection_error_mutant_fired": None},
            "G10.projection-error",
            GateStatus.NOT_RUN,
        ),
        (
            {"physical_integral_comparison_count": 0},
            "G10.cross-mesh",
            GateStatus.NOT_RUN,
        ),
        (
            {"stress_node_pairing_mutant_fired": False},
            "G10.cross-mesh",
            GateStatus.FAIL,
        ),
        (
            {"source_verified_integral_comparison_count": 1},
            "G10.cross-mesh",
            GateStatus.FAIL,
        ),
        ({"normalization_statistic_count": 0}, "G10.normalization", GateStatus.NOT_RUN),
        (
            {"normalization_non_train_source_count": 1},
            "G10.normalization",
            GateStatus.FAIL,
        ),
        (
            {"normalization_non_active_source_count": 1},
            "G10.normalization",
            GateStatus.FAIL,
        ),
        (
            {"normalize_on_all_splits_mutant_fired": False},
            "G10.normalization",
            GateStatus.FAIL,
        ),
        (
            {
                "failed_fem_case_count": 0,
                "represented_failed_fem_case_count": 0,
                "failure_visibility_canary_count": 0,
                "represented_failure_visibility_canary_count": 0,
            },
            "G10.failed-visible",
            GateStatus.NOT_RUN,
        ),
        (
            {"represented_failed_fem_case_count": 1},
            "G10.failed-visible",
            GateStatus.FAIL,
        ),
        (
            {"collocation_replay_count": 0, "byte_identical_replay_count": 0},
            "G10.reproducible",
            GateStatus.NOT_RUN,
        ),
        ({"byte_identical_replay_count": 2}, "G10.reproducible", GateStatus.FAIL),
        ({"code_coverage_percent": 84.999}, "G10.coverage", GateStatus.FAIL),
        ({"code_coverage_percent": None}, "G10.coverage", GateStatus.NOT_RUN),
    ),
)
def test_missing_or_bad_evidence_blocks_the_gate(
    overrides: dict[str, object], check_id: str, status: GateStatus
) -> None:
    evaluation = evaluate_s10_gate(_passing_measurements(**overrides))

    assert evaluation.verdict is GateStatus.FAIL
    assert evaluation.by_id(check_id).status is status


def test_coverage_threshold_is_inclusive() -> None:
    evaluation = evaluate_s10_gate(
        replace(_passing_measurements(), code_coverage_percent=85.0)
    )

    assert evaluation.by_id("G10.coverage").status is GateStatus.PASS


def test_failed_visible_passes_zero_failure_run_with_represented_canary() -> None:
    evaluation = evaluate_s10_gate(
        _passing_measurements(
            failed_fem_case_count=0,
            represented_failed_fem_case_count=0,
            failure_visibility_canary_count=1,
            represented_failure_visibility_canary_count=1,
        )
    )

    check = evaluation.by_id("G10.failed-visible")
    assert check.status is GateStatus.PASS
    assert check.measured == {
        "failed_fem_cases": 0,
        "represented_failed_fem_cases": 0,
        "failure_visibility_canaries": 1,
        "represented_failure_visibility_canaries": 1,
    }


def test_failed_visible_fails_zero_failure_run_when_canary_is_not_represented() -> None:
    evaluation = evaluate_s10_gate(
        _passing_measurements(
            failed_fem_case_count=0,
            represented_failed_fem_case_count=0,
            failure_visibility_canary_count=1,
            represented_failure_visibility_canary_count=0,
        )
    )

    assert evaluation.by_id("G10.failed-visible").status is GateStatus.FAIL


def test_failed_visible_fails_natural_failure_count_mismatch_despite_canary() -> None:
    evaluation = evaluate_s10_gate(
        _passing_measurements(
            failed_fem_case_count=2,
            represented_failed_fem_case_count=1,
            failure_visibility_canary_count=1,
            represented_failure_visibility_canary_count=1,
        )
    )

    assert evaluation.by_id("G10.failed-visible").status is GateStatus.FAIL


def test_failed_visible_fails_any_supplied_canary_mismatch() -> None:
    evaluation = evaluate_s10_gate(
        _passing_measurements(
            failed_fem_case_count=2,
            represented_failed_fem_case_count=2,
            failure_visibility_canary_count=1,
            represented_failure_visibility_canary_count=0,
        )
    )

    assert evaluation.by_id("G10.failed-visible").status is GateStatus.FAIL


def test_gate_requires_normalization_snapshot_and_collocation_request_bindings() -> (
    None
):
    normalization = evaluate_s10_gate(
        _passing_measurements(normalization_snapshot_binding_verified=False)
    )
    collocation = evaluate_s10_gate(
        _passing_measurements(collocation_request_binding_verified=False)
    )

    assert normalization.by_id("G10.normalization").status is GateStatus.FAIL
    assert collocation.by_id("G10.reproducible").status is GateStatus.FAIL


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"sample_count": -1}, "non-negative integer"),
        ({"sample_count": True}, "non-negative integer"),
        ({"failure_visibility_canary_count": -1}, "non-negative integer"),
        (
            {"represented_failure_visibility_canary_count": True},
            "non-negative integer",
        ),
        ({"code_coverage_percent": float("nan")}, "finite number"),
        ({"code_coverage_percent": 100.01}, r"\[0, 100\]"),
        ({"cp_manifest_binding_verified": "yes"}, "boolean or None"),
        ({"request_plan_binding_verified": "yes"}, "boolean or None"),
    ),
)
def test_measurements_reject_invalid_values(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _passing_measurements(**overrides)


def test_evaluator_rejects_untyped_measurements() -> None:
    with pytest.raises(TypeError, match="S10GateMeasurements"):
        evaluate_s10_gate(object())  # type: ignore[arg-type]
