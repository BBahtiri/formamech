from __future__ import annotations

from dataclasses import replace

import pytest

from piginot.doe.coverage import coverage_gate_checks, measure_design_coverage
from piginot.doe.design import (
    AxisSource,
    AxisSpec,
    CoupledDesignSpec,
    DifficultRegion,
    InteractionSpec,
    generate_coupled_design,
)
from piginot.doe.leakage import LeakagePolicy
from tests.doe.test_design import _CoupledPredicate, _spec


def test_coverage_reports_each_metric_without_an_aggregate_score() -> None:
    predicate = _CoupledPredicate()
    spec = _spec(validation_count=80)
    design = generate_coupled_design(predicate, spec)

    report = measure_design_coverage(predicate, spec, design)

    assert report.validation_count == 80
    assert report.empirical_fill_distance > 0.0
    assert 0.0 <= report.stratum_coverage <= 1.0
    assert 0.0 <= report.boundary_coverage <= 1.0
    assert 0.0 <= report.interaction_coverage <= 1.0
    assert 0.0 <= report.difficult_region_coverage <= 1.0
    assert report.minimum_stratum_count >= 0
    assert not hasattr(report, "coverage_percent")


def test_minimum_stratum_count_includes_zero_and_ids_uncovered_requirements() -> None:
    predicate = _CoupledPredicate()
    sparse_spec = CoupledDesignSpec(
        seed=b"sparse-v1",
        sample_count=1,
        reserve_count=0,
        candidate_count=40,
        validation_count=10,
        validation_candidate_count=40,
        guard_band_count=0,
        axes=(
            AxisSpec(
                "x",
                AxisSource.PRIMITIVE,
                strata=4,
                minimum_per_stratum=1,
                minimum_per_boundary=1,
            ),
            AxisSpec(
                "y",
                AxisSource.PRIMITIVE,
                strata=4,
                minimum_per_stratum=1,
                minimum_per_boundary=1,
            ),
            AxisSpec(
                "governing_ratio",
                AxisSource.DERIVED,
                strata=4,
                minimum_per_stratum=1,
                minimum_per_boundary=1,
            ),
        ),
        leakage_policy=LeakagePolicy(near_duplicate_tolerance=0.0),
        interactions=(InteractionSpec("xy", ("x", "y"), minimum_per_cell=1),),
        difficult_regions=(
            DifficultRegion(
                "upper-corner",
                {"x": (0.8, 1.0), "y": (0.8, 1.0)},
                minimum_count=2,
            ),
        ),
    )
    design = generate_coupled_design(predicate, sparse_spec)

    report = measure_design_coverage(predicate, sparse_spec, design)

    assert report.minimum_stratum_count == 0
    assert report.uncovered_ids
    assert set(report.uncovered_ids) <= set(report.unresolved_ids)
    assert "difficult:upper-corner" in report.unresolved_ids


def test_fill_distance_uses_independent_validation_points() -> None:
    predicate = _CoupledPredicate()
    spec = _spec(validation_count=25)
    design = generate_coupled_design(predicate, spec)
    report = measure_design_coverage(predicate, spec, design)

    selected = {
        tuple(point.all_coordinates.items()) for point in design.selected_points
    }
    validation = {
        tuple(point.all_coordinates.items()) for point in design.validation_points
    }

    assert selected.isdisjoint(validation)
    assert report.validation_stream_id == design.validation_stream_id


def test_coverage_gate_keeps_thresholds_separate_instead_of_averaging() -> None:
    predicate = _CoupledPredicate()
    spec = _spec(validation_count=60)
    design = generate_coupled_design(predicate, spec)
    report = measure_design_coverage(predicate, spec, design)

    checks = coverage_gate_checks(
        report,
        minimum_fraction=0.90,
        maximum_fill_distance=0.50,
        minimum_stratum_count=1,
    )

    assert tuple(checks) == (
        "empirical_fill_distance",
        "minimum_stratum_count",
        "stratum_coverage",
        "boundary_coverage",
        "interaction_coverage",
        "difficult_region_coverage",
        "unresolved_requirements",
    )
    assert all(isinstance(value, bool) for value in checks.values())
    assert "aggregate" not in checks


@pytest.mark.parametrize(
    "changes",
    [
        {"axes": ()},
        {"axes": ("x", "x")},
        {"empirical_fill_distance": float("nan")},
        {"minimum_stratum_count": -1},
        {"stratum_coverage": 1.1},
        {"uncovered_ids": ("missing",), "unresolved_ids": ()},
        {"validation_count": 0},
        {"validation_stream_id": ""},
        {"requirement_counts": {"bad": -1}},
    ],
)
def test_coverage_contract_rejects_incoherent_evidence(
    changes: dict[str, object],
) -> None:
    predicate = _CoupledPredicate()
    spec = _spec(validation_count=12)
    report = measure_design_coverage(
        predicate,
        spec,
        generate_coupled_design(predicate, spec),
    )

    with pytest.raises(ValueError):
        replace(report, **changes)


@pytest.mark.parametrize(
    "thresholds",
    [
        {
            "minimum_fraction": -0.1,
            "maximum_fill_distance": 0.5,
            "minimum_stratum_count": 1,
        },
        {
            "minimum_fraction": 0.9,
            "maximum_fill_distance": -0.1,
            "minimum_stratum_count": 1,
        },
        {
            "minimum_fraction": 0.9,
            "maximum_fill_distance": 0.5,
            "minimum_stratum_count": -1,
        },
    ],
)
def test_coverage_gate_rejects_invalid_thresholds(
    thresholds: dict[str, object],
) -> None:
    predicate = _CoupledPredicate()
    spec = _spec(validation_count=12)
    report = measure_design_coverage(
        predicate,
        spec,
        generate_coupled_design(predicate, spec),
    )

    with pytest.raises(ValueError):
        coverage_gate_checks(report, **thresholds)  # type: ignore[arg-type]
