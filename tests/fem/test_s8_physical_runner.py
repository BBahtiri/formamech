"""Primary-backend physical checks required by the S8 gate."""

from __future__ import annotations

from piginot.fem.s8_physical import run_s8_physical_verification


def test_primary_solver_physical_metrics_meet_declared_g8_limits() -> None:
    report = run_s8_physical_verification()

    assert report.patch_max_error <= 1.0e-12
    assert report.objectivity_invariant_change <= 1.0e-10
    assert report.angular_skew_ratio < 1.0e-10
    assert report.energy_relative_error < 1.0e-8
    assert report.reaction_relative_error < 1.0e-9
    assert report.small_strain_relative_error < 5.0e-3
    assert set(report.patch_errors) == {
        "structured-p1",
        "structured-p2",
        "unstructured-p1",
        "unstructured-p2",
    }
    assert report.energy_path_quadrature_order == 16
    assert report.energy_path_solve_count == 17
