"""The completed S13b pilot remains falsifiable TRAIN evidence, not G13."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from campaigns.dogbone_v1.s13b_evaluate import compare_s13b_train_results
from campaigns.dogbone_v1.s13b_protocol import build_dogbone_s13b_pilot_plan
from piginot.contracts import hash_spec, strict_json_loads
from piginot.doe import EvidenceRole
from piginot.training import CheckpointLineage, SelectionMetrics, TrialResult

_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "campaigns"
    / "dogbone-v1"
    / "S13B-TRAIN-PILOT.json"
)
_DIAGNOSTIC = _ARTIFACT.with_name("S13B-TRAIN-INVALID-POINT.json")


def _report() -> dict[str, Any]:
    payload = strict_json_loads(_ARTIFACT.read_bytes())
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def _result(report: dict[str, Any], arm: dict[str, Any]) -> TrialResult:
    metrics = dict(arm["metrics"])
    metrics["hybrid_loss"] = metrics.pop("hybrid_loss_legacy_diagnostic")
    return TrialResult(
        lineage=CheckpointLineage(
            trial_id=arm["trial_id"],
            run_hash=arm["run_hash"],
            dataset_snapshot_hash=report["dataset_snapshot_hash"],
            execution_profile_hash=arm["execution_profile_hash"],
            weights_hash=arm["weights_hash"],
            model_state_hash=arm["model_state_hash"],
            checkpoint_hash=arm["checkpoint_manifest_hash"],
        ),
        role=EvidenceRole.TRAIN,
        metrics=SelectionMetrics(**metrics),
    )


def test_s13b_pilot_artifact_replays_exact_failed_signal() -> None:
    report = _report()
    plan = build_dogbone_s13b_pilot_plan()
    control = _result(report, report["control"])
    peak = _result(report, report["peak_arm"])

    comparison = compare_s13b_train_results(plan, control=control, peak=peak)

    assert report["pilot_plan_hash"] == plan.plan_hash
    assert report["objective_registry_hash"] == hash_spec(
        plan.objective_registry, "recipe"
    )
    assert report["pilot_id"] == plan.pilot_id
    assert report["dataset_snapshot_hash"] == plan.base_protocol.dataset_snapshot_hash
    assert report["source_bindings_hash"] == plan.base_protocol.source_bindings_hash
    assert report["training_revision"] == "65b091f136d72697698c1209427e96e0668a43e6"
    assert report["evaluator_revision"] == "f917d4f752d5ba6474b20167d0a607e6e01d9f55"
    assert report["control"]["checkpoint_manifest_hash"] == plan.control_manifest_hash
    assert report["control"]["run_hash"] == plan.control_run_hash
    assert report["control"]["checkpoint_step"] == 800
    assert report["peak_arm"]["checkpoint_step"] == 800
    assert report["criteria"] == dict(comparison.criteria)
    assert report["pilot_signal"] is comparison.pilot_signal is False
    assert [name for name, passed in comparison.criteria if not passed] == [
        "p_field_nonregression",
        "zero_invalid_collocation",
    ]
    frozen = plan.base_protocol.selection_policy.guardrails
    assert not frozen.admits(peak.metrics)
    assert frozen.admits(replace(peak.metrics, invalid_collocation_points=0))
    assert report["frozen_s11_guardrails_diagnostic_at_step_800"] == {
        "admitted": False,
        "failed_fields": ["invalid_collocation_points"],
        "not_g13_gate_evidence": True,
    }


def test_s13b_pilot_evidence_never_claims_gate_or_non_train_access() -> None:
    report = _report()

    assert report["schema_version"] == "1.0"
    assert report["step"] == "S13b"
    assert report["evidence_role"] == "TRAIN"
    assert report["screen_case_count"] == 64
    assert report["invalid_collocation_case_count"] == 256
    assert report["development_opened"] is False
    assert report["holdout_opened"] is False
    assert report["non_train_evidence_used"] is False
    assert report["g13_status"] == "NOT_PASSED"
    assert report["cp_freeze_status"] == "WITHHELD"
    assert report["s14_status"] == "BLOCKED"


def test_s13b_invalid_point_diagnostic_matches_failed_pilot() -> None:
    pilot = _report()
    diagnostic = strict_json_loads(_DIAGNOSTIC.read_bytes())

    assert diagnostic["schema_version"] == "1.0"
    assert diagnostic["step"] == "S13b"
    assert diagnostic["non_gate"] is True
    assert diagnostic["evidence_role"] == "TRAIN"
    assert diagnostic["pilot_artifact_sha256"] == (
        "6523895342ad68edf624991b65a4c1a6ad30bc295d5e944afb3c69d31077037b"
    )
    assert sha256(_ARTIFACT.read_bytes()).hexdigest() == (
        diagnostic["pilot_artifact_sha256"]
    )
    assert diagnostic["peak_checkpoint_manifest_hash"] == (
        pilot["peak_arm"]["checkpoint_manifest_hash"]
    )
    assert diagnostic["checked_case_count"] == pilot["invalid_collocation_case_count"]
    assert diagnostic["checked_interior_point_count"] == (
        diagnostic["checked_case_count"] * diagnostic["interior_points_per_case"]
    )
    assert diagnostic["invalid_case_count"] == 1
    assert diagnostic["invalid_point_count"] == (
        pilot["peak_arm"]["metrics"]["invalid_collocation_points"]
    )
    point = diagnostic["invalid_case"]
    assert point["case_id"] == "candidate-002485"
    assert point["population"] == "TRAIN_FIT"
    assert point["point_index"] == 25
    assert point["determinant"] == point["min_determinant"]
    assert point["determinant"] < 0 < diagnostic["determinant_margin"]
    assert diagnostic["development_opened"] is False
    assert diagnostic["holdout_opened"] is False
    assert diagnostic["g13_status"] == "NOT_PASSED"
