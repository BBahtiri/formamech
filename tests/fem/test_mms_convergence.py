"""Four-level solved-FEM convergence evidence for every S8 MMS family."""

from __future__ import annotations

import math

import pytest

from piginot.fem.manufactured import MANUFACTURED_CASE_NAMES
from piginot.fem.mms_convergence import run_all_mms_convergence_studies
from piginot.fem.verification_meshes import VERIFICATION_LEVELS


def test_all_cases_and_orders_use_four_real_refinement_levels() -> None:
    studies = run_all_mms_convergence_studies()

    assert {(study.case_name, study.element_order) for study in studies} == {
        (case_name, order) for case_name in MANUFACTURED_CASE_NAMES for order in (1, 2)
    }
    for study in studies:
        assert tuple(level.level for level in study.levels) == VERIFICATION_LEVELS
        assert all(
            coarse.h_max_mm > fine.h_max_mm
            for coarse, fine in zip(study.levels, study.levels[1:], strict=False)
        )
        assert study.displacement_l2_order.sample_count == 2
        assert study.displacement_h1_order.sample_count == 2
        assert study.first_piola_l2_order.sample_count == 2
        assert study.all_level_displacement_l2_fit.sample_count == 4
        assert study.all_level_displacement_h1_fit.sample_count == 4
        assert study.all_level_first_piola_l2_fit.sample_count == 4
        for diagnostic_fit in (
            study.all_level_displacement_l2_fit,
            study.all_level_displacement_h1_fit,
            study.all_level_first_piola_l2_fit,
        ):
            assert math.isfinite(diagnostic_fit.order)
            assert diagnostic_fit.order > 0.0


@pytest.mark.parametrize("element_order", (1, 2))
def test_displacement_and_stress_orders_meet_g8_targets(element_order: int) -> None:
    studies = run_all_mms_convergence_studies()
    expected_l2 = element_order + 1.0
    expected_h1_and_stress = float(element_order)

    selected = [study for study in studies if study.element_order == element_order]
    assert {study.case_name for study in selected} == set(MANUFACTURED_CASE_NAMES)
    for study in selected:
        assert study.displacement_l2_order.order == pytest.approx(
            expected_l2, abs=0.15
        ), study.case_name
        assert study.first_piola_l2_order.order == pytest.approx(
            expected_h1_and_stress, abs=0.2
        ), study.case_name
        assert study.displacement_h1_order.order == pytest.approx(
            expected_h1_and_stress, abs=0.2
        ), study.case_name
