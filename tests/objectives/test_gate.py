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
        "perturbation_recomputed_term_counts": (
            report.perturbation_recomputed_term_counts
        ),
        "finite_difference_relative_errors": report.finite_difference_relative_errors,
        "detached_terms": report.detached_terms,
        "reachable_parameter_names": report.reachable_parameter_names,
        "required_parameter_names": report.required_parameter_names,
        "reachability_loss_paths": report.reachability_loss_paths,
        "anisotropy_relative_change": report.anisotropy_relative_change,
        "coordinate_anisotropy_mutant_fired": report.coordinate_anisotropy_mutant_fired,
        "fem_solve_success": report.fem_solve_success,
        "weak_strong_difference": report.weak_strong_difference,
        "weak_residual_max": report.weak_residual_max,
        "projected_strong_residual_max": report.projected_strong_residual_max,
        "reconstructed_stress_variation": report.reconstructed_stress_variation,
        "weak_strong_fem_request_id": report.weak_strong_fem_request_id,
        "weak_strong_quadrature_point_count": (
            report.weak_strong_quadrature_point_count
        ),
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
            {"perturbation_recomputed_term_counts": {"constitutive": 6}},
            "G12.perturbed",
            GateStatus.FAIL,
        ),
        (
            {
                "perturbation_recomputed_term_counts": {
                    term: 5
                    for term in (
                        "equilibrium_strong",
                        "equilibrium_weak",
                        "constitutive",
                        "angular",
                        "dirichlet",
                        "traction_free",
                    )
                }
            },
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
            {"reachability_loss_paths": {"geometry_axial_extent": "physics_total"}},
            "G12.reachability",
            GateStatus.FAIL,
        ),
        (
            {
                "reachability_loss_paths": {
                    **dict(run_s12_verification().reachability_loss_paths),
                    "geometry_axial_extent": "bogus_path",
                }
            },
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
            {"weak_residual_max": 0.0},
            "G12.weak-strong",
            GateStatus.FAIL,
        ),
        (
            {"projected_strong_residual_max": 0.0},
            "G12.weak-strong",
            GateStatus.FAIL,
        ),
        (
            {"reconstructed_stress_variation": 0.0},
            "G12.weak-strong",
            GateStatus.FAIL,
        ),
        (
            {"weak_strong_fem_request_id": "different-fem-fixture"},
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
        {"perturbation_recomputed_term_counts": {"constitutive": 0}},
        {"reachability_loss_paths": {"constitutive": ""}},
        {"weak_strong_fem_request_id": ""},
        {"weak_strong_quadrature_point_count": -1},
        {"s8_discretization_error_bound": 0.0},
        {"constitutive_residual_shape": (2, True)},
        {"holdout_refused": 1},
        {"code_coverage_percent": 101.0},
    ),
)
def test_measurement_contract_rejects_malformed_values(
    changes: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _passing(**changes)
