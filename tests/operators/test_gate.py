from __future__ import annotations

from dataclasses import replace

import pytest

from piginot.operators.gate import (
    GateStatus,
    S11GateMeasurements,
    evaluate_s11_gate,
)


def _passing(**overrides: object) -> S11GateMeasurements:
    values: dict[str, object] = {
        "permutation_count": 100,
        "permutation_max_relative_change": 9.0e-6,
        "resampling_low_count": 320,
        "resampling_high_count": 480,
        "resampling_relative_change": 0.019,
        "node_order_relative_change": 9.0e-7,
        "query_batch_relative_change": 9.0e-7,
        "reachable_parameter_count": 7,
        "required_parameter_count": 7,
        "mask_mutant_fired": True,
        "pickle_refused_before_deserialization": True,
        "u_shape_verified": True,
        "p_shape_verified": True,
        "served_stress_is_predicted": True,
        "code_coverage_percent": 80.0,
    }
    values.update(overrides)
    return S11GateMeasurements(**values)  # type: ignore[arg-type]


def test_complete_measurements_pass_all_nine_checks() -> None:
    evaluation = evaluate_s11_gate(_passing())

    assert evaluation.verdict is GateStatus.PASS
    assert tuple(check.check_id for check in evaluation.checks) == (
        "G11.permutation",
        "G11.resampling",
        "G11.node-order",
        "G11.query-batch",
        "G11.reachability",
        "G11.mask",
        "G11.safetensors",
        "G11.heads",
        "G11.coverage",
    )


@pytest.mark.parametrize(
    ("overrides", "check_id", "status"),
    (
        (
            {"permutation_count": 99},
            "G11.permutation",
            GateStatus.FAIL,
        ),
        (
            {"permutation_max_relative_change": 1.0e-5},
            "G11.permutation",
            GateStatus.FAIL,
        ),
        (
            {"permutation_max_relative_change": None},
            "G11.permutation",
            GateStatus.NOT_RUN,
        ),
        (
            {"resampling_low_count": 319},
            "G11.resampling",
            GateStatus.FAIL,
        ),
        (
            {"resampling_relative_change": 0.02},
            "G11.resampling",
            GateStatus.FAIL,
        ),
        (
            {"resampling_relative_change": None},
            "G11.resampling",
            GateStatus.NOT_RUN,
        ),
        (
            {"node_order_relative_change": 1.0e-6},
            "G11.node-order",
            GateStatus.FAIL,
        ),
        (
            {"query_batch_relative_change": 1.0e-6},
            "G11.query-batch",
            GateStatus.FAIL,
        ),
        (
            {"reachable_parameter_count": 6},
            "G11.reachability",
            GateStatus.FAIL,
        ),
        (
            {"required_parameter_count": 0, "reachable_parameter_count": 0},
            "G11.reachability",
            GateStatus.NOT_RUN,
        ),
        ({"mask_mutant_fired": False}, "G11.mask", GateStatus.FAIL),
        ({"mask_mutant_fired": None}, "G11.mask", GateStatus.NOT_RUN),
        (
            {"pickle_refused_before_deserialization": False},
            "G11.safetensors",
            GateStatus.FAIL,
        ),
        ({"u_shape_verified": False}, "G11.heads", GateStatus.FAIL),
        (
            {"served_stress_is_predicted": None},
            "G11.heads",
            GateStatus.NOT_RUN,
        ),
        ({"code_coverage_percent": 79.999}, "G11.coverage", GateStatus.FAIL),
        ({"code_coverage_percent": None}, "G11.coverage", GateStatus.NOT_RUN),
    ),
)
def test_bad_or_missing_measurements_block_the_gate(
    overrides: dict[str, object],
    check_id: str,
    status: GateStatus,
) -> None:
    evaluation = evaluate_s11_gate(_passing(**overrides))

    assert evaluation.verdict is GateStatus.FAIL
    assert evaluation.by_id(check_id).status is status


@pytest.mark.parametrize(
    "changes",
    (
        {"permutation_count": -1},
        {"permutation_count": True},
        {"permutation_max_relative_change": float("nan")},
        {"resampling_relative_change": -1.0},
        {"mask_mutant_fired": 1},
        {"code_coverage_percent": 101.0},
    ),
)
def test_measurement_contract_rejects_invalid_values(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _passing(**changes)


def test_evaluation_lookup_is_exact_and_coverage_boundary_is_inclusive() -> None:
    evaluation = evaluate_s11_gate(replace(_passing(), code_coverage_percent=80.0))

    assert evaluation.by_id("G11.coverage").status is GateStatus.PASS
    with pytest.raises(KeyError):
        evaluation.by_id("G11.unknown")
