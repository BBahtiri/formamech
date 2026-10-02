from __future__ import annotations

from dataclasses import replace

import pytest

from piginot.operators import (
    ENERGY_G11_CHECK_IDS,
    ENERGY_G11_LIFTING_POINTS_PER_EDGE,
    EnergyG11Measurements,
    GateStatus,
    evaluate_energy_g11_gate,
)


def _passing() -> EnergyG11Measurements:
    return EnergyG11Measurements(
        node_order_relative_change=0.0,
        query_batch_relative_change=1.0e-16,
        reachable_parameter_count=5,
        required_parameter_count=5,
        pickle_refused_before_deserialization=True,
        u_shape_verified=True,
        p_shape_verified=True,
        displacement_only_output=True,
        served_stress_energy_gradient_relative_difference=1.0e-15,
        lifting_parameter_vector_count=64,
        lifting_points_per_edge=ENERGY_G11_LIFTING_POINTS_PER_EDGE,
        lifting_max_relative_error=0.0,
        lifting_mutant_control_relative_error=0.0,
        lifting_mutant_relative_error=0.5,
        code_coverage_percent=89.4,
    )


def test_all_eight_checks_pass_in_the_fixed_order() -> None:
    evaluation = evaluate_energy_g11_gate(_passing())

    assert tuple(check.check_id for check in evaluation.checks) == ENERGY_G11_CHECK_IDS
    assert evaluation.verdict is GateStatus.PASS
    assert all(check.status is GateStatus.PASS for check in evaluation.checks)
    assert evaluation.by_id("G11.lifting").measured == {
        "parameter_vectors": 64,
        "points_per_edge": [320, 480, 4800],
        "max_relative_error": 0.0,
    }


@pytest.mark.parametrize(
    ("change", "check_id"),
    [
        ({"node_order_relative_change": 1.0e-6}, "G11.node-order"),
        ({"query_batch_relative_change": 2.0e-6}, "G11.query-batch"),
        ({"reachable_parameter_count": 4}, "G11.reachability"),
        ({"pickle_refused_before_deserialization": False}, "G11.safetensors"),
        ({"u_shape_verified": False}, "G11.heads"),
        ({"displacement_only_output": False}, "G11.heads"),
        (
            {"served_stress_energy_gradient_relative_difference": 1.0e-10},
            "G11.heads",
        ),
        ({"lifting_max_relative_error": 1.0e-12}, "G11.lifting"),
        ({"lifting_parameter_vector_count": 63}, "G11.lifting"),
        ({"lifting_points_per_edge": (320, 480)}, "G11.lifting"),
        ({"lifting_mutant_relative_error": 1.0e-3}, "G11.lifting-mutant"),
        ({"lifting_mutant_control_relative_error": 1.0e-12}, "G11.lifting-mutant"),
        ({"code_coverage_percent": 79.99}, "G11.coverage"),
    ],
)
def test_each_threshold_fails_closed(change: dict[str, object], check_id: str) -> None:
    evaluation = evaluate_energy_g11_gate(replace(_passing(), **change))  # type: ignore[arg-type]

    assert evaluation.verdict is GateStatus.FAIL
    assert evaluation.by_id(check_id).status is GateStatus.FAIL
    assert [
        check.check_id
        for check in evaluation.checks
        if check.status is not GateStatus.PASS
    ] == [check_id]


@pytest.mark.parametrize(
    ("change", "check_id"),
    [
        ({"node_order_relative_change": None}, "G11.node-order"),
        (
            {"required_parameter_count": 0, "reachable_parameter_count": 0},
            "G11.reachability",
        ),
        ({"pickle_refused_before_deserialization": None}, "G11.safetensors"),
        ({"p_shape_verified": None}, "G11.heads"),
        ({"lifting_max_relative_error": None}, "G11.lifting"),
        ({"lifting_mutant_relative_error": None}, "G11.lifting-mutant"),
        ({"code_coverage_percent": None}, "G11.coverage"),
    ],
)
def test_missing_evidence_is_not_run_and_never_passes(
    change: dict[str, object], check_id: str
) -> None:
    evaluation = evaluate_energy_g11_gate(replace(_passing(), **change))  # type: ignore[arg-type]

    assert evaluation.verdict is GateStatus.FAIL
    assert evaluation.by_id(check_id).status is GateStatus.NOT_RUN
    assert evaluation.by_id(check_id).reason


@pytest.mark.parametrize(
    "change",
    [
        {"reachable_parameter_count": -1},
        {"lifting_parameter_vector_count": True},
        {"lifting_points_per_edge": [320, 480, 4800]},
        {"lifting_points_per_edge": (320, 0)},
        {"node_order_relative_change": float("nan")},
        {"lifting_max_relative_error": -1.0},
        {"u_shape_verified": 1},
        {"code_coverage_percent": 100.5},
        {"code_coverage_percent": True},
    ],
)
def test_measurements_reject_malformed_values(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(_passing(), **change)  # type: ignore[arg-type]


def test_reducer_requires_exact_measurements() -> None:
    with pytest.raises(TypeError):
        evaluate_energy_g11_gate(object())  # type: ignore[arg-type]
