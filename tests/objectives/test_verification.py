from __future__ import annotations

from piginot.objectives import (
    PHYSICS_TERM_IDS,
    VERIFIED_GRADIENT_TERM_IDS,
    run_s12_verification,
)


def test_executable_verification_covers_every_required_physics_channel() -> None:
    report = run_s12_verification()

    assert tuple(report.exact_max_residuals) == PHYSICS_TERM_IDS
    assert max(report.exact_max_residuals.values()) < 1.0e-8
    assert tuple(report.perturbation_fired_terms) == PHYSICS_TERM_IDS
    for expected, fired in report.perturbation_fired_terms.items():
        assert fired == (expected,)


def test_all_objective_gradients_match_finite_differences_without_detach() -> None:
    report = run_s12_verification()

    assert tuple(report.finite_difference_relative_errors) == VERIFIED_GRADIENT_TERM_IDS
    assert max(report.finite_difference_relative_errors.values()) < 1.0e-5
    assert report.detached_terms == ()


def test_geometry_material_load_and_bc_parameters_reach_loss_paths() -> None:
    report = run_s12_verification()

    assert report.required_parameter_names == (
        "geometry_length",
        "geometry_width",
        "geometry_transition_radius",
        "geometry_shape",
        "elastic_modulus",
        "poisson_ratio",
        "load_amplitude",
        "prescribed_displacement",
    )
    assert report.reachable_parameter_names == report.required_parameter_names


def test_coordinate_and_role_mutants_are_detected() -> None:
    report = run_s12_verification()

    assert report.anisotropy_relative_change < 1.0e-12
    assert report.coordinate_anisotropy_mutant_fired
    assert report.holdout_refused
    assert report.holdout_access_mutant_fired


def test_weak_strong_comparison_uses_a_converged_fem_solution_and_s8_bound() -> None:
    report = run_s12_verification()

    assert report.fem_solve_success
    assert report.weak_strong_difference <= report.s8_discretization_error_bound
    assert report.s8_discretization_error_bound == 1.0e-12
    assert report.constitutive_residual_shape == (2, 2)
    assert report.constitutive_rejects_out_of_plane
