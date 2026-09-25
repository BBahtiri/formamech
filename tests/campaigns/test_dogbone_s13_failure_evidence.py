from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import pytest
import rfc8785

from campaigns.dogbone_v1.s13_protocol import build_dogbone_s13_protocol
from piginot.contracts import strict_json_loads
from piginot.doe import EvidenceRole
from piginot.training.selection import (
    CheckpointLineage,
    NoEligibleTrialError,
    SelectionMetrics,
    TrialResult,
    select_checkpoint,
)

_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "campaigns"
    / "dogbone-v1"
    / "S13-TRAIN-SCREEN-FAILED.json"
)
_STDOUT_SHA256 = "4093148a2418ca8c840b3ef0da7bd7f7033e46e4559805adf9bcee7b9501a46b"
_TRAINING_REVISION = "5473cdeebd06597d4ec0bdd50df7d230d00306cd"
_EVALUATOR_REVISION = "927bb1cd8d6528b7aa3c57184f8d1f92585d48ac"


def _report() -> dict[str, Any]:
    payload = strict_json_loads(_ARTIFACT.read_bytes())
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def _typed_result(payload: dict[str, Any]) -> TrialResult:
    assert payload["$type"] == "piginot.training.TrialResult"
    lineage_payload = dict(payload["lineage"])
    metrics_payload = dict(payload["metrics"])
    assert lineage_payload.pop("$type") == "piginot.training.CheckpointLineage"
    assert metrics_payload.pop("$type") == "piginot.training.SelectionMetrics"
    return TrialResult(
        lineage=CheckpointLineage(**lineage_payload),
        role=EvidenceRole(payload["role"]),
        metrics=SelectionMetrics(**metrics_payload),
    )


def _failed_guardrails(metrics: SelectionMetrics) -> list[str]:
    limits = build_dogbone_s13_protocol().selection_policy.guardrails
    failed = []
    if metrics.median_absolute_peak_error > limits.incumbent_median_absolute_peak_error:
        failed.append("median_absolute_peak_error")
    if metrics.p90_absolute_peak_error > limits.incumbent_p90_absolute_peak_error:
        failed.append("p90_absolute_peak_error")
    if max(0.0, -metrics.median_signed_peak_error) > (
        1.0 - limits.minimum_underprediction_reduction
    ) * max(0.0, -limits.incumbent_median_signed_peak_error):
        failed.append("median_signed_peak_error")
    if (
        metrics.u_relative_l2
        > (1.0 + limits.maximum_field_relative_regression)
        * limits.incumbent_u_relative_l2
    ):
        failed.append("u_relative_l2")
    if (
        metrics.p_relative_l2
        > (1.0 + limits.maximum_field_relative_regression)
        * limits.incumbent_p_relative_l2
    ):
        failed.append("p_relative_l2")
    if metrics.invalid_collocation_points != 0:
        failed.append("invalid_collocation_points")
    return failed


def test_failed_screen_artifact_preserves_the_exact_six_train_results() -> None:
    report = _report()
    protocol = build_dogbone_s13_protocol()
    source = report["source_log"]
    assert isinstance(source, dict)
    raw_results = report["trial_results"]
    assert isinstance(raw_results, list)

    assert report["schema_version"] == "1.0"
    assert report["step"] == "S13"
    assert report["evidence_role"] == "TRAIN"
    assert report["training_revision"] == _TRAINING_REVISION
    assert report["evaluator_revision"] == _EVALUATOR_REVISION
    assert report["protocol_hash"] == protocol.protocol_hash
    assert report["selection_policy_hash"] == protocol.selection_policy.policy_hash
    assert report["dataset_snapshot_hash"] == protocol.dataset_snapshot_hash
    assert report["fit_case_count"] == 192
    assert report["screen_case_count"] == 64
    assert report["optimizer_steps_per_trial"] == 1500
    assert source["path_from_training_worktree"] == (
        ".piginot-runs/s13-background-logs/batch-screen.stdout.log"
    )
    assert source["sha256"] == _STDOUT_SHA256
    assert source["byte_count"] == 6287
    assert source["line_ending"] == "CRLF"
    assert hashlib.sha256(rfc8785.dumps(raw_results) + b"\r\n").hexdigest() == (
        _STDOUT_SHA256
    )
    assert tuple(raw["lineage"]["trial_id"] for raw in raw_results) == (
        protocol.search_space.trial_ids
    )
    assert all(raw["role"] == "TRAIN" for raw in raw_results)


def test_failed_screen_artifact_replays_frozen_selection_and_rejections() -> None:
    report = _report()
    protocol = build_dogbone_s13_protocol()
    raw_results = report["trial_results"]
    assert isinstance(raw_results, list)
    results = tuple(_typed_result(raw) for raw in raw_results)
    guardrails = protocol.selection_policy.guardrails

    assert report["guardrail_limits"] == {
        "maximum_median_absolute_peak_error": guardrails.incumbent_median_absolute_peak_error,
        "maximum_p90_absolute_peak_error": guardrails.incumbent_p90_absolute_peak_error,
        "minimum_median_signed_peak_error": -(
            1.0 - guardrails.minimum_underprediction_reduction
        )
        * max(0.0, -guardrails.incumbent_median_signed_peak_error),
        "maximum_u_relative_l2": (1.0 + guardrails.maximum_field_relative_regression)
        * guardrails.incumbent_u_relative_l2,
        "maximum_p_relative_l2": (1.0 + guardrails.maximum_field_relative_regression)
        * guardrails.incumbent_p_relative_l2,
        "required_invalid_collocation_points": 0,
    }
    assert report["failed_guardrails"] == [
        {"trial_id": result.trial_id, "fields": _failed_guardrails(result.metrics)}
        for result in results
    ]
    assert all(not guardrails.admits(result.metrics) for result in results)
    with pytest.raises(NoEligibleTrialError, match="CP-FREEZE must be withheld"):
        select_checkpoint(results, policy=protocol.selection_policy)
    assert report["selected_checkpoint"] is None
    assert report["cp_freeze_status"] == "WITHHELD"
    assert report["g13_status"] == "NOT_PASSED"
    assert report["non_train_evidence_used"] is False
