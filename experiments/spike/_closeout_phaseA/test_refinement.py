"""RED tests for Richardson-based convergence assessment (S1 closeout, Phase A)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.spike.refinement import (
    assess_refinement_study,
    richardson,
)

ITERATION8_CONFIRMATION = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "iteration8"
    / "iteration8_confirmation.json"
)


def _iteration8_sequences() -> list[tuple[str, list[float]]]:
    payload = json.loads(ITERATION8_CONFIRMATION.read_text())
    return [
        (
            sample["geometry_id"],
            [level["peak_stress_MPa"] for level in sample["levels"]],
        )
        for sample in payload["peak_regularity"]["samples"]
    ]


def test_richardson_recovers_a_known_second_order_sequence() -> None:
    limit, order = 100.0, 2.0
    values = [limit - 8.0 * (0.5 ** (order * level)) for level in range(3)]
    estimate = richardson(values)
    assert estimate.status == "OK"
    assert estimate.observed_order == pytest.approx(order, abs=1e-9)
    assert estimate.extrapolated_limit == pytest.approx(limit, rel=1e-9)


def test_richardson_reports_remaining_error_per_level() -> None:
    values = [92.0, 98.0, 99.5]
    estimate = richardson(values)
    assert estimate.status == "OK"
    assert len(estimate.relative_remaining_error) == len(values)
    assert estimate.relative_remaining_error[0] > estimate.relative_remaining_error[-1]


def test_richardson_refuses_a_non_monotone_sequence() -> None:
    estimate = richardson([50.0, 60.0, 55.0])
    assert estimate.status == "NOT_RUN"
    assert estimate.observed_order is None
    assert "monotone" in (estimate.reason or "")


def test_richardson_refuses_a_diverging_sequence() -> None:
    estimate = richardson([50.0, 60.0, 80.0])
    assert estimate.status == "NOT_RUN"
    assert "diverg" in (estimate.reason or "")


def test_richardson_refuses_a_non_finite_sequence() -> None:
    estimate = richardson([50.0, float("nan"), 55.0])
    assert estimate.status == "NOT_RUN"
    assert "finite" in (estimate.reason or "")


def test_richardson_refuses_fewer_than_three_levels() -> None:
    estimate = richardson([50.0, 60.0])
    assert estimate.status == "NOT_RUN"
    assert "three" in (estimate.reason or "")


# --- the load-bearing regression tests ----------------------------------------
# Iteration 8 passed its convergence check because that check tested only the last
# increment. The replacement must fail the study. Five of the six recorded
# sequences exceed the 2% remaining-error limit at the production mesh (nx=32,
# level index 1); the sixth is within it. One failure is enough to fail the gate.
def test_iteration8_production_mesh_fails_the_remaining_error_gate() -> None:
    sequences = dict(_iteration8_sequences())
    study = assess_refinement_study(sequences)
    assert study["verdict"] == "FAIL"
    failing = [
        geometry_id
        for geometry_id, record in study["geometries"].items()
        if not record["passes"]
    ]
    assert len(failing) == 5, f"expected 5 of 6 to exceed the limit, got {failing}"


def test_iteration8_remaining_errors_are_all_biased_low() -> None:
    """The production mesh under-predicts the peak in every recorded sequence."""
    for geometry_id, values in _iteration8_sequences():
        estimate = richardson(values)
        assert estimate.status == "OK", geometry_id
        assert estimate.extrapolated_limit is not None
        assert estimate.extrapolated_limit > values[1], (
            f"{geometry_id}: extrapolated peak must exceed the production-mesh peak"
        )


def test_iteration8_last_increment_check_would_have_passed_all_six() -> None:
    """Documents precisely what the superseded check missed."""
    for _geometry_id, values in _iteration8_sequences():
        last_increment = abs(values[2] - values[1]) / abs(values[2])
        assert last_increment < 0.10


def test_a_refused_sequence_fails_the_study_because_not_run_is_never_pass() -> None:
    study = assess_refinement_study({"oscillating": [50.0, 60.0, 55.0]})
    assert study["verdict"] == "FAIL"
    assert study["geometries"]["oscillating"]["status"] == "NOT_RUN"
    assert study["geometries"]["oscillating"]["passes"] is False


def test_a_converged_study_passes() -> None:
    study = assess_refinement_study({"fine": [99.0, 99.90, 99.975]})
    assert study["verdict"] == "PASS"
