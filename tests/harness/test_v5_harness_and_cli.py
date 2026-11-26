"""End-to-end integration tests for FormaMech v0.5 Agent Harness capabilities & CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from piginot.cli import EXIT_OK, EXIT_REFUSED, main
from piginot.harness import FormaMechAgentHarness, HarnessBudget
from piginot.validation.contracts import CouponRole


def test_v5_agent_harness_progressive_discovery_and_invocation(tmp_path: Path) -> None:
    """All v0.5 capabilities are discoverable, budget-governed, and ledger-verified."""
    harness = FormaMechAgentHarness(
        tmp_path, budget=HarnessBudget(max_fem_solves=8, max_dofs=150_000)
    )
    caps_env = harness.invoke("discover.capabilities", {"search": "v5"})
    assert caps_env.ok and caps_env.result is not None
    names = {c["name"] for c in caps_env.result["capabilities"]}
    for expected in (
        "surrogate.check_regime_guards_3d",
        "validate.ground_surrogate_with_tests",
        "active.recommend_physical_test",
        "surrogate.estimate_archive_breakeven",
        "governance.run_industry_benchmarks",
        "cad.synthesize_brep_3d",
        "governance.record_hitl_signoff",
        "multiphysics.solve_electro_thermal_3d",
    ):
        assert expected in names

    # 1) V5-1 physics regime guard
    g_env = harness.invoke(
        "surrogate.check_regime_guards_3d",
        {"yield_stress_mpa": 250.0, "peak_von_mises_mpa": 300.0},
    )
    assert g_env.ok and g_env.result is not None
    assert g_env.result["passed"] is False

    # 2) V5-2 hybrid grounding + active physical test planner via workspace JSON
    comp_json = tmp_path / "component_tests.json"
    comp_json.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "specimen_id": f"c{i}",
                        "role": CouponRole.CALIBRATION.value,
                        "sensor_kind": "load_cell_reaction",
                        "condition_vector": [float(i + 1)],
                        "observed_values": [float(1.2 * (i + 1) + 0.5)],
                        "simulated_prior_values": [float(i + 1)],
                        "measurement_std": 0.05,
                        "unit": "kN",
                    }
                    for i in range(6)
                ]
                + [
                    {
                        "specimen_id": "h0",
                        "role": CouponRole.VALIDATION_HOLDOUT.value,
                        "sensor_kind": "load_cell_reaction",
                        "condition_vector": [3.5],
                        "observed_values": [1.2 * 3.5 + 0.5],
                        "simulated_prior_values": [3.5],
                        "measurement_std": 0.05,
                        "unit": "kN",
                    }
                ],
                "candidates": [
                    {
                        "candidate_id": "cand-high",
                        "role": CouponRole.CALIBRATION.value,
                        "condition_vector": [9.0],
                        "sim_uncertainty_eta": 0.8,
                        "test_cost": 1.0,
                    },
                    {
                        "candidate_id": "cand-low",
                        "role": CouponRole.CALIBRATION.value,
                        "condition_vector": [1.1],
                        "sim_uncertainty_eta": 0.1,
                        "test_cost": 1.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    hyb_env = harness.invoke(
        "validate.ground_surrogate_with_tests", {"dataset_json": "component_tests.json"}
    )
    assert hyb_env.ok and hyb_env.result is not None
    assert hyb_env.result["grounding_dossier"]["passed"] is True

    plan_env = harness.invoke(
        "active.recommend_physical_test",
        {"dataset_json": "component_tests.json", "batch_size": 1},
    )
    assert plan_env.ok and plan_env.result is not None
    assert plan_env.result["recommendations"][0]["candidate_id"] == "cand-high"

    # 3) V5-3 pre-flight break-even N*
    be_env = harness.invoke(
        "surrogate.estimate_archive_breakeven",
        {"expected_queries": 300, "fem_seconds_per_solve": 10.0},
    )
    assert be_env.ok and be_env.result is not None
    assert be_env.result["recommended_strategy"] == "archive_pretrained_lora"

    # 4) V5-4 HITL signoff + watertight B-Rep STEP synthesis
    hitl_env = harness.invoke(
        "governance.record_hitl_signoff",
        {
            "artifact_id": "bracket-rev-a",
            "artifact_sha256": "a" * 64,
            "reviewer_id": "lead-stress-01",
            "approved": True,
            "decision_note": "Verified hand-calc within 3%",
            "hand_calc_note": "M*c/I = 120 MPa",
            "hand_calc_value": 120.0,
            "model_predicted_value": 123.0,
        },
    )
    assert hitl_env.ok and hitl_env.result is not None
    assert hitl_env.result["state"] == "APPROVED"

    brep_env = harness.invoke(
        "cad.synthesize_brep_3d",
        {"bore_radius_mm": 2.0, "output_step": "part.step"},
    )
    assert brep_env.ok and brep_env.result is not None
    assert brep_env.result["watertight"] is True
    assert brep_env.result["remeshed_fem_verified"] is True

    # 5) V5-5 multi-material 3D ETM solve
    etm_env = harness.invoke(
        "multiphysics.solve_electro_thermal_3d",
        {"template": "parametric-chiplet-package-3d", "chiplet_power_w": 10.0},
    )
    assert etm_env.ok and etm_env.result is not None
    assert etm_env.result["converged"] is True

    # Hash-chained audit ledger verifies 100%
    ok, errs = harness.verify_ledger()
    assert ok is True and not errs


def test_v5_cli_bench_fetch_and_industry(tmp_path: Path) -> None:
    """CLI `bench fetch-dataset` enforces license tier & `bench industry` emits leaderboard."""
    # Tier C non-commercial dataset refused without --allow-noncommercial
    rc_refused = main(
        [
            "bench",
            "fetch-dataset",
            "--name",
            "shift-battery",
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert rc_refused == EXIT_REFUSED

    # Tier A commercial-clean dataset prepares synthetic fallback & verifies SHA-256
    out_fetch = tmp_path / "fetch.json"
    rc_ok = main(
        [
            "bench",
            "fetch-dataset",
            "--name",
            "deepjeb",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(out_fetch),
        ]
    )
    assert rc_ok == EXIT_OK
    assert out_fetch.is_file()

    # Neutral multi-scale + BC-shift industry benchmark CLI
    out_ind = tmp_path / "industry.json"
    rc_ind = main(
        [
            "bench",
            "industry",
            "--dataset",
            "deepjeb",
            "--scales",
            "3,4,5",
            "--epochs",
            "4",
            "--output-dir",
            str(tmp_path / "ind_work"),
            "--output",
            str(out_ind),
        ]
    )
    assert rc_ind == EXIT_OK
    assert (tmp_path / "ind_work" / "industry_benchmark.md").is_file()
