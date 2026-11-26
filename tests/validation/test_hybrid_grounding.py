"""Gates G-V5-2a & G-V5-2b tests: Hybrid sim+test grounding & ASME V&V 10 firewall."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from piginot.ui.workbench_3d.vv10_panel import run_hybrid_grounding_summary
from piginot.validation import (
    HYBRID_GROUNDING_MACHINE,
    ComponentTestRecord,
    CouponRole,
    ExperimentalValidationTier,
    HybridGroundingConfig,
    HybridGroundingState,
    SensorObservationKind,
    align_sensor_observations_to_mesh,
    ground_simulation_with_experiments,
)


def _synthetic_dic_records() -> list[ComponentTestRecord]:
    """Generate biased simulation priors + noisy physical DIC observations."""
    rng = np.random.default_rng(2026)
    sensor_pts = tuple(
        (float(x), float(y), 0.0)
        for x in np.linspace(0.0, 20.0, 4)
        for y in np.linspace(0.0, 10.0, 2)
    )
    records: list[ComponentTestRecord] = []

    def _make_record(
        sid: str, role: CouponRole, load_kn: float, temp_c: float
    ) -> ComponentTestRecord:
        sim_vals: list[float] = []
        obs_vals: list[float] = []
        for x, y, _ in sensor_pts:
            # Idealized simulation prior misses thermal softening & fixture compliance bias
            y_sim = 0.08 * load_kn * (x / 20.0) + 0.02 * (y / 10.0)
            true_bias = (
                0.15 * y_sim
                + 0.28 * math.sin(0.25 * load_kn)
                + 0.04 * (temp_c / 100.0) * (x / 20.0)
                + 0.12
            )
            noise = float(rng.normal(0.0, 0.008))
            sim_vals.append(float(y_sim))
            obs_vals.append(float(y_sim + true_bias + noise))
        return ComponentTestRecord(
            specimen_id=sid,
            role=role,
            sensor_kind=SensorObservationKind.DIC_DISPLACEMENT_FIELD,
            condition_vector=(load_kn, temp_c),
            observed_values=tuple(obs_vals),
            simulated_prior_values=tuple(sim_vals),
            measurement_std=0.03,
            unit="mm",
            sensor_locations=sensor_pts,
        )

    cal_conditions = [
        (2.0, 25.0),
        (4.0, 25.0),
        (6.0, 25.0),
        (8.0, 25.0),
        (3.0, 80.0),
        (5.0, 80.0),
        (7.0, 80.0),
        (9.0, 80.0),
        (4.0, 50.0),
        (8.0, 50.0),
    ]
    for idx, (load_kn, temp_c) in enumerate(cal_conditions):
        records.append(
            _make_record(f"dic-cal-{idx:02d}", CouponRole.CALIBRATION, load_kn, temp_c)
        )

    holdout_conditions = [(3.5, 40.0), (6.5, 60.0), (7.5, 35.0)]
    for idx, (load_kn, temp_c) in enumerate(holdout_conditions):
        records.append(
            _make_record(
                f"dic-holdout-{idx:02d}",
                CouponRole.VALIDATION_HOLDOUT,
                load_kn,
                temp_c,
            )
        )
    return records


def test_sensor_mesh_alignment_interpolates_3d_fields() -> None:
    """Deterministic KD-tree interpolation aligns arbitrary DIC coordinates onto mesh nodes."""
    gx, gy, gz = np.meshgrid(
        np.linspace(0.0, 10.0, 6),
        np.linspace(0.0, 10.0, 6),
        np.linspace(0.0, 2.0, 2),
        indexing="ij",
    )
    nodes = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    field = 2.0 * nodes[:, 0] - 0.5 * nodes[:, 1]
    sensors = np.array(
        [[2.0, 4.0, 0.0], [6.0, 8.0, 2.0], [5.0, 5.0, 1.0]], dtype=np.float64
    )
    report = align_sensor_observations_to_mesh(nodes, field, sensors, k_neighbors=4)
    assert report.sensor_count == 3
    assert report.max_registration_distance_mm <= 2.5
    # Exact grid points [2,4,0] and [6,8,2] match to machine precision
    assert math.isclose(
        float(report.matched_values[0]), 2.0 * 2.0 - 0.5 * 4.0, abs_tol=1.0e-9
    )
    assert math.isclose(
        float(report.matched_values[1]), 2.0 * 6.0 - 0.5 * 8.0, abs_tol=1.0e-9
    )


def test_hybrid_grounding_cuts_holdout_rmse_and_passes_vv10() -> None:
    """Gate G-V5-2a & G-V5-2b: >=50% RMSE drop, R^2>=0.94, ASME V&V 10 & ClaimLevel firewall."""
    records = _synthetic_dic_records()
    cfg = HybridGroundingConfig(
        length_scale=1.5,
        u_num=0.015,
        u_input=0.01,
        numerical_claim_level="HOLDOUT_VERIFIED",
    )
    model, dossier = ground_simulation_with_experiments(records, config=cfg)

    assert dossier.passed is True
    assert dossier.state is HybridGroundingState.VALIDATED
    assert dossier.experimental_validation_tier is (
        ExperimentalValidationTier.HOLDOUT_EXPERIMENTALLY_VALIDATED
    )
    # Epistemic firewall: numerical claim level stays orthogonal and unmodified
    assert dossier.numerical_claim_level == "HOLDOUT_VERIFIED"
    assert dossier.raw_sim_holdout_rmse is not None
    assert dossier.grounded_holdout_rmse is not None
    assert dossier.grounded_holdout_r2 is not None
    # Gate G-V5-2a thresholds
    assert dossier.grounded_holdout_rmse <= 0.50 * dossier.raw_sim_holdout_rmse
    assert dossier.grounded_holdout_r2 >= 0.94
    assert dossier.bias_reduction_fraction is not None
    assert dossier.bias_reduction_fraction >= 0.50
    # ASME V&V 10 quadratic uncertainty budget check
    expected_u_val = math.sqrt(
        dossier.u_num**2
        + dossier.u_input**2
        + dossier.u_discrepancy**2
        + dossier.u_exp**2
    )
    assert math.isclose(dossier.u_val, expected_u_val, rel_tol=1.0e-12)
    assert dossier.validation_ratio is not None and dossier.validation_ratio <= 1.0
    assert len(dossier.receipt_sha256) == 64
    assert dossier.snapshot().machine_id == HYBRID_GROUNDING_MACHINE.machine_id

    # Model prediction matches nonzero posterior std
    sample_x = np.array([[5.0, 50.0, 10.0, 5.0, 0.0]], dtype=np.float64)
    pred, std = model.predict(sample_x, np.array([0.21], dtype=np.float64))
    assert pred.shape == (1,) and std.shape == (1,) and float(std[0]) > 0.0


def test_hybrid_grounding_refuses_leakage_and_missing_holdout() -> None:
    """Gate G-V5-2a/2b: Fail-closed refusal on duplicate measurement leakage or missing holdout."""
    records = _synthetic_dic_records()
    # 1) Leakage via identical physical measurement cloned under a different specimen_id
    cal0 = records[0]
    leaked = ComponentTestRecord(
        specimen_id="cloned-holdout-leak",
        role=CouponRole.VALIDATION_HOLDOUT,
        sensor_kind=cal0.sensor_kind,
        condition_vector=cal0.condition_vector,
        observed_values=cal0.observed_values,
        simulated_prior_values=cal0.simulated_prior_values,
        measurement_std=cal0.measurement_std,
        unit=cal0.unit,
        sensor_locations=cal0.sensor_locations,
    )
    leaked_model, leaked_dossier = ground_simulation_with_experiments(
        [*records[:6], leaked]
    )
    assert leaked_model is None
    assert leaked_dossier.passed is False
    assert leaked_dossier.state is HybridGroundingState.REJECTED
    assert (
        leaked_dossier.experimental_validation_tier
        is ExperimentalValidationTier.UNVALIDATED
    )
    assert "leakage" in leaked_dossier.reason.lower()

    # 2) Missing holdout refuses HOLDOUT_EXPERIMENTALLY_VALIDATED promotion
    cal_only = [r for r in records if r.role is CouponRole.CALIBRATION]
    _, no_holdout_dossier = ground_simulation_with_experiments(cal_only)
    assert no_holdout_dossier.passed is False
    assert no_holdout_dossier.state is HybridGroundingState.REJECTED
    assert (
        no_holdout_dossier.experimental_validation_tier
        is ExperimentalValidationTier.COUPON_CALIBRATED
    )

    # 3) Gross unmodeled holdout shock fails E <= k * u_val gate
    shocked = list(records)
    h0 = [r for r in records if r.role is CouponRole.VALIDATION_HOLDOUT][0]
    shocked[-1] = ComponentTestRecord(
        specimen_id="shocked-holdout",
        role=CouponRole.VALIDATION_HOLDOUT,
        sensor_kind=h0.sensor_kind,
        condition_vector=h0.condition_vector,
        observed_values=tuple(v + 5.0 for v in h0.observed_values),
        simulated_prior_values=h0.simulated_prior_values,
        measurement_std=0.01,
        unit=h0.unit,
        sensor_locations=h0.sensor_locations,
    )
    _, fail_dossier = ground_simulation_with_experiments(
        shocked, config=HybridGroundingConfig(u_num=0.005, coverage_factor_k=1.5)
    )
    assert fail_dossier.passed is False
    assert fail_dossier.state is HybridGroundingState.REJECTED
    assert (
        fail_dossier.experimental_validation_tier
        is ExperimentalValidationTier.COUPON_CALIBRATED
    )


def test_workbench_vv10_panel_hybrid_summary_json(tmp_path: Path) -> None:
    """UI workbench helper executes hybrid grounding + test plan from canonical JSON."""
    records = _synthetic_dic_records()
    payload = {
        "records": [r.document() for r in records],
        "candidates": [
            {
                "candidate_id": "next-load-10kn",
                "role": CouponRole.CALIBRATION.value,
                "condition_vector": [10.0, 75.0],
                "sim_uncertainty_eta": 0.4,
                "test_cost": 1.0,
                "admissible": True,
            },
            {
                "candidate_id": "next-load-1kn",
                "role": CouponRole.CALIBRATION.value,
                "condition_vector": [1.0, 20.0],
                "sim_uncertainty_eta": 0.1,
                "test_cost": 2.0,
                "admissible": True,
            },
        ],
    }
    json_path = tmp_path / "component_tests.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    summary = run_hybrid_grounding_summary(json_path, u_num=0.015, batch_size=2)
    assert summary["grounding_dossier"]["passed"] is True
    assert len(summary["receipt_sha256"]) == 64
    assert (
        summary["physical_test_plan"]["recommendations"][0]["candidate_id"]
        == "next-load-10kn"
    )
    assert len(summary["plan_hash"]) == 64
