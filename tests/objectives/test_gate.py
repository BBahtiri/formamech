from __future__ import annotations

from dataclasses import replace

import pytest

from piginot.objectives import (
    GateStatus,
    S12GateMeasurements,
    evaluate_s12_gate,
    run_s12_verification,
)


def _passing(**overrides: object) -> S12GateMeasurements:
    report = run_s12_verification()
    values: dict[str, object] = {
        "exact_max_residuals": report.exact_max_residuals,
        "perturbation_fired_terms": report.perturbation_fired_terms,
        "finite_difference_relative_errors": report.finite_difference_relative_errors,
        "detached_terms": report.detached_terms,
        "reachable_parameter_names": report.reachable_parameter_names,
        "required_parameter_names": report.required_parameter_names,
        "anisotropy_relative_change": report.anisotropy_relative_change,
        "coordinate_anisotropy_mutant_fired": report.coordinate_anisotropy_mutant_fired,
        "fem_solve_success": report.fem_solve_success,
        "weak_strong_difference": report.weak_strong_difference,
        "s8_discretization_error_bound": report.s8_discretization_error_bound,
        "constitutive_residual_shape": report.constitutive_residual_shape,
        "constitutive_rejects_out_of_plane": report.constitutive_rejects_out_of_plane,
        "holdout_refused": report.holdout_refused,
        "holdout_access_mutant_fired": report.holdout_access_mutant_fired,
        "code_coverage_percent": 90.0,
    }
    values.update(overrides)
    return S12GateMeasurements(**values)  # type: ignore[arg-type]


def test_complete_measurements_pass_exactly_the_nine_g12_checks() -> None:
    evaluation = evaluate_s12_gate(_passing())

    assert evaluation.verdict is GateStatus.PASS
    assert tuple(check.check_id for check in evaluation.checks) == (
        "G12.exact",
        "G12.perturbed",
        "G12.fd",
        "G12.reachability",
        "G12.anisotropy",
        "G12.weak-strong",
        "G12.constitutive-shape",
        "G12.role-isolation",
        "G12.coverage",
    )
    assert all(check.status is GateStatus.PASS for check in evaluation.checks)


@pytest.mark.parametrize(
    ("overrides", "check_id", "status"),
    (
        (
            {"exact_max_residuals": {"constitutive": 1.0e-8}},
            "G12.exact",
            GateStatus.FAIL,
        ),
        (
            {"perturbation_fired_terms": {"constitutive": ("constitutive", "angular")}},
            "G12.perturbed",
            GateStatus.FAIL,
        ),
        (
            {"finite_difference_relative_errors": {"data_u": 1.0e-5}},
            "G12.fd",
            GateStatus.FAIL,
        ),
        ({"detached_terms": ("data_u",)}, "G12.fd", GateStatus.FAIL),
        (
            {"reachable_parameter_names": ("L_total",)},
            "G12.reachability",
            GateStatus.FAIL,
        ),
        (
            {"anisotropy_relative_change": 1.0e-12},
            "G12.anisotropy",
            GateStatus.FAIL,
        ),
        (
            {"coordinate_anisotropy_mutant_fired": False},
            "G12.anisotropy",
            GateStatus.FAIL,
        ),
        (
            {"fem_solve_success": False},
            "G12.weak-strong",
            GateStatus.FAIL,
        ),
        (
            {"weak_strong_difference": 1.1e-12},
            "G12.weak-strong",
            GateStatus.FAIL,
        ),
        (
            {"constitutive_residual_shape": (3, 3)},
            "G12.constitutive-shape",
            GateStatus.FAIL,
        ),
        (
            {"holdout_refused": False},
            "G12.role-isolation",
            GateStatus.FAIL,
        ),
        (
            {"code_coverage_percent": 84.999},
            "G12.coverage",
            GateStatus.FAIL,
        ),
        (
            {"code_coverage_percent": None},
            "G12.coverage",
            GateStatus.NOT_RUN,
        ),
    ),
)
def test_bad_or_missing_evidence_fails_its_named_gate(
    overrides: dict[str, object], check_id: str, status: GateStatus
) -> None:
    evaluation = evaluate_s12_gate(_passing(**overrides))

    assert evaluation.verdict is GateStatus.FAIL
    assert evaluation.by_id(check_id).status is status


def test_gate_is_fail_closed_on_incomplete_populations() -> None:
    passing = _passing()
    incomplete = replace(
        passing,
        finite_difference_relative_errors={"data_u": 1.0e-12},
    )

    assert evaluate_s12_gate(incomplete).by_id("G12.fd").status is GateStatus.FAIL


@pytest.mark.parametrize(
    "changes",
    (
        {"anisotropy_relative_change": float("nan")},
        {"weak_strong_difference": -1.0},
        {"s8_discretization_error_bound": 0.0},
        {"constitutive_residual_shape": (2, True)},
        {"holdout_refused": 1},
        {"code_coverage_percent": 101.0},
    ),
)
def test_measurement_contract_rejects_malformed_values(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _passing(**changes)
