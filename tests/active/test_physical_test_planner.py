"""Gate G-V5-2c tests: Active physical test-plan optimizer & retrospective test reduction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from piginot.active import (
    PhysicalTestCandidate,
    PhysicalTestPlanConfig,
    recommend_next_physical_tests,
    resolve_test_acquisition_backend,
    run_retrospective_test_reduction_benchmark,
)
from piginot.validation import (
    ComponentTestRecord,
    CouponRole,
    HybridGroundingConfig,
    SensorObservationKind,
)


def _make_record(
    sid: str, role: CouponRole, x0: float, x1: float
) -> ComponentTestRecord:
    """Synthetic multi-regime nonlinear discrepancy (like fast-charging / fatigue tests)."""
    y_sim = 0.5 * x0 + 0.3 * x1
    # Smooth yet spatially varied discrepancy across the 2D operating envelope
    delta = (
        0.65 * math.sin(1.4 * x0) * math.cos(1.1 * x1)
        + 0.35 * math.exp(-((x0 - 1.5) ** 2 + (x1 - 1.5) ** 2))
        + 0.15 * x0
    )
    return ComponentTestRecord(
        specimen_id=sid,
        role=role,
        sensor_kind=SensorObservationKind.DEGRADATION_METRIC,
        condition_vector=(x0, x1),
        observed_values=(float(y_sim + delta),),
        simulated_prior_values=(float(y_sim),),
        measurement_std=0.02,
        unit="cycles_k",
    )


def test_active_test_planner_deterministic_and_enforces_safety_and_holdout() -> None:
    """Plan hash is deterministic; VALIDATION_HOLDOUT & inadmissible tests are refused."""
    # Holdout candidate is refused at construction
    with pytest.raises(ValueError, match="VALIDATION_HOLDOUT"):
        PhysicalTestCandidate(
            candidate_id="bad-holdout",
            role=CouponRole.VALIDATION_HOLDOUT,
            condition_vector=(0.5, 0.5),
        )

    observed = [_make_record("obs-0", CouponRole.CALIBRATION, 0.0, 0.0)]
    candidates = [
        PhysicalTestCandidate(
            candidate_id="near-obs",
            role=CouponRole.CALIBRATION,
            condition_vector=(0.05, 0.05),
            sim_uncertainty_eta=0.05,
            test_cost=1.0,
        ),
        PhysicalTestCandidate(
            candidate_id="far-informative",
            role=CouponRole.CALIBRATION,
            condition_vector=(2.0, 2.0),
            sim_uncertainty_eta=0.50,
            test_cost=1.0,
        ),
        PhysicalTestCandidate(
            candidate_id="unsafe-buckling",
            role=CouponRole.CALIBRATION,
            condition_vector=(3.0, 3.0),
            sim_uncertainty_eta=10.0,
            test_cost=0.1,
            admissible=False,
        ),
    ]
    cfg = PhysicalTestPlanConfig(batch_size=2, sim_uncertainty_weight=0.25)
    p1 = recommend_next_physical_tests(observed, candidates, config=cfg)
    p2 = recommend_next_physical_tests(observed, candidates, config=cfg)

    assert p1.plan_hash == p2.plan_hash
    assert p1.refused_inadmissible_count == 1
    assert p1.eligible_candidate_count == 2
    assert p1.recommendations[0].candidate_id == "far-informative"
    assert all(r.candidate_id != "unsafe-buckling" for r in p1.recommendations)


def test_retrospective_test_reduction_beats_random_by_40_percent() -> None:
    """Gate G-V5-2c: Active physical test planner saves >= 40% physical tests vs. random."""
    # Clustered operating pool mimicking real test campaigns (many redundant baseline points
    # plus sparse high-information corners across 4 quadrants)
    pool: list[ComponentTestRecord] = []
    # Seed points clustered near (0, 0)
    pool.append(_make_record("seed-0", CouponRole.CALIBRATION, -1.8, -1.8))
    pool.append(_make_record("seed-1", CouponRole.CALIBRATION, -1.6, -1.7))

    # Redundant cluster around (-1.5, -1.5) (24 low-value repeats)
    rng = np.random.default_rng(7)
    for i in range(24):
        x0 = float(-1.6 + rng.uniform(-0.25, 0.25))
        x1 = float(-1.6 + rng.uniform(-0.25, 0.25))
        pool.append(_make_record(f"cluster-{i:02d}", CouponRole.CALIBRATION, x0, x1))

    # Sparse high-value exploratory regimes across the rest of the envelope
    informative_pts = [
        (1.6, 1.6),
        (1.6, -1.6),
        (-1.6, 1.6),
        (0.0, 0.0),
        (1.5, 0.0),
        (0.0, 1.5),
        (-1.5, 0.0),
        (0.0, -1.5),
    ]
    for i, (x0, x1) in enumerate(informative_pts):
        pool.append(_make_record(f"info-{i:02d}", CouponRole.CALIBRATION, x0, x1))

    # Dense blind physical holdout grid across the whole envelope
    holdout: list[ComponentTestRecord] = []
    idx = 0
    for x0 in (-1.4, -0.4, 0.6, 1.5):
        for x1 in (-1.4, -0.4, 0.6, 1.5):
            holdout.append(
                _make_record(
                    f"holdout-{idx:02d}", CouponRole.VALIDATION_HOLDOUT, x0, x1
                )
            )
            idx += 1

    report = run_retrospective_test_reduction_benchmark(
        pool,
        holdout,
        target_holdout_rmse=0.14,
        initial_seed_count=2,
        max_rounds=20,
        random_seeds=(0, 1, 2, 3, 4),
        grounding_config=HybridGroundingConfig(length_scale=1.2, u_num=0.01),
        plan_config=PhysicalTestPlanConfig(
            batch_size=1, length_scale=1.0, sim_uncertainty_weight=0.0
        ),
    )
    assert report.active_tests_needed < report.random_mean_tests_needed
    assert report.tests_saved_fraction >= 0.40, (
        f"Expected >= 40% test reduction, got {report.tests_saved_fraction:.1%} "
        f"(active={report.active_tests_needed}, random_mean={report.random_mean_tests_needed:.1f})"
    )
    assert len(report.benchmark_hash) == 64


def test_optional_bo_backends_resolve_or_fail_closed_with_hint() -> None:
    """Built-in rbf-gp-mi works always; unknown backend fails closed clearly."""
    backend = resolve_test_acquisition_backend("rbf-gp-mi")
    assert backend.backend_name == "rbf-gp-mi"
    with pytest.raises(LookupError, match="unknown"):
        resolve_test_acquisition_backend("nonexistent-bo")
