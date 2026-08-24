"""Held-out calibration of Richardson and recovered-stress estimators."""

from __future__ import annotations

import pytest

from piginot.fem.mms_convergence import run_mms_convergence_study
from piginot.fem.mms_estimators import (
    calibrate_mms_estimators,
    coarse_fine_displacement_l2_difference,
)


def test_cross_mesh_difference_uses_the_two_solved_fem_fields() -> None:
    study = run_mms_convergence_study("pure_dirichlet", 1)

    difference = coarse_fine_displacement_l2_difference(
        study.levels[0], study.levels[1]
    )

    assert difference > 0.0
    assert difference > study.levels[1].norms.displacement_l2_error


def test_held_out_estimator_populations_meet_g8_rates() -> None:
    report = calibrate_mms_estimators()

    assert report.richardson.calibration.population_size == 16
    assert report.richardson.validation.population_size == 8
    assert report.recovery.calibration.population_size == 24
    assert report.recovery.validation.population_size == 8
    assert report.richardson.safety_factor >= 1.0
    assert report.recovery.safety_factor >= 1.0
    assert report.richardson.validation.bracketing_rate >= 0.95
    assert report.richardson.validation.conservative_rate >= 0.90
    assert report.recovery.validation.bracketing_rate >= 0.95
    assert report.recovery.validation.conservative_rate >= 0.90
    assert report.minimum_validation_bracketing_rate == pytest.approx(
        min(
            report.richardson.validation.bracketing_rate,
            report.recovery.validation.bracketing_rate,
        )
    )
