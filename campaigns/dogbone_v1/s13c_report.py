"""Canonical S13c TRAIN-only evidence: screen report and G13 replay probe.

``write_screen_report`` records the registered checkpoint's TRAIN-screen
metrics, the frozen guardrail verdict, per-case diagnostics and the historical
comparison. ``compare_replay`` checks an independent re-execution of the
registered trial, interrupted and resumed at half of a segment, against the
registered checkpoint bit for bit (``G13.determinism`` and ``G13.resume``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from campaigns.dogbone_v1.s13_pilot import _git_provenance
from campaigns.dogbone_v1.s13c_evaluate import S13cEvaluation, evaluate_s13c_checkpoint
from campaigns.dogbone_v1.s13c_trial import TRAINING_LOG_NAME, latest_checkpoint
from piginot.contracts import canonical_json_bytes, strict_json_loads
from piginot.training import TrainingCheckpointManifest

_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACTS = _ROOT / "artifacts" / "campaigns" / "dogbone-v1"
SCREEN_REPORT_NAME = "S13C-TRAIN-SCREEN.json"
PROBE_REPORT_NAME = "S13C-G13-REPLAY-PROBE.json"


def _evaluator_revision() -> str:
    provenance = _git_provenance()
    if provenance.dirty_worktree:
        raise RuntimeError("S13c evidence must be written from a clean revision")
    return provenance.head


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _historical_metrics() -> dict[str, Any]:
    failed = strict_json_loads(
        (_ARTIFACTS / "S13-TRAIN-SCREEN-FAILED.json").read_bytes()
    )
    pilot = strict_json_loads((_ARTIFACTS / "S13B-TRAIN-PILOT.json").read_bytes())
    assert isinstance(failed, dict) and isinstance(pilot, dict)
    limits = failed["guardrail_limits"]
    trials = failed["trial_results"]
    assert isinstance(trials, list)
    first = trials[0]
    assert isinstance(first, dict)
    peak_arm = pilot["peak_arm"]
    assert isinstance(peak_arm, dict)
    return {
        "guardrail_limits": limits,
        "s13_trial_0": {
            key: value for key, value in first["metrics"].items() if key != "$type"
        },
        "s13b_peak_arm": peak_arm["metrics"],
    }


def _training_summary(run_root: Path) -> dict[str, Any]:
    raw = [
        json.loads(line)
        for line in (run_root / TRAINING_LOG_NAME).read_text("utf-8").splitlines()
        if line.strip()
    ]
    if not raw:
        raise ValueError("training log is empty")
    # A segment interrupted before its checkpoint is replayed bit for bit after
    # resume; the replayed record of a step supersedes the interrupted one.
    by_step = {record["step"]: record for record in raw}
    if sorted(by_step) != list(range(1, len(by_step) + 1)):
        raise ValueError("training log must cover every optimizer step")
    records = [by_step[step] for step in sorted(by_step)]

    def window(start: int, stop: int) -> float:
        return statistics.fmean(record["energy"] for record in records[start:stop])

    return {
        "logged_steps": len(records),
        "replayed_log_records": len(raw) - len(records),
        "mean_energy_first_100_steps": window(0, 100),
        "mean_energy_last_100_steps": window(len(records) - 100, len(records)),
        "maximum_barrier": max(record["barrier"] for record in records),
        "minimum_determinant": min(record["minimum_determinant"] for record in records),
        "inadmissible_points_total": sum(
            record["inadmissible_points"] for record in records
        ),
        "final_learning_rate": records[-1]["learning_rate"],
        "training_log_sha256": _sha256(run_root / TRAINING_LOG_NAME),
    }


def screen_report(evaluation: S13cEvaluation, run_root: Path) -> dict[str, Any]:
    """Build the canonical TRAIN-screen evidence document."""

    metrics = evaluation.result.metrics
    diagnostics = evaluation.diagnostics
    signed = [item.signed_peak_error for item in diagnostics]
    distances = [item.peak_vertex_distance_mm for item in diagnostics]
    selected = evaluation.decision is not None
    return {
        "schema_version": "1.0",
        "step": "S13c",
        "evidence_role": "TRAIN",
        "protocol_hash": evaluation.protocol_hash,
        "run_hash": evaluation.run_binding.run_hash,
        "training_revision": evaluation.run_binding.implementation_revision,
        "evaluator_revision": _evaluator_revision(),
        "execution_profile_hash": evaluation.run_binding.execution_profile_hash,
        "checkpoint_step": evaluation.manifest.step,
        "checkpoint_manifest_hash": evaluation.manifest.manifest_hash,
        "trial_result": evaluation.result,
        "failed_guardrails": list(evaluation.failed_guardrails),
        "selected_checkpoint": (evaluation.result.lineage if selected else None),
        "invalid_collocation_case_count": evaluation.invalid_case_count,
        "minimum_interior_collocation_determinant": (
            evaluation.minimum_collocation_determinant
        ),
        "screen_diagnostics": {
            "underpredicted_case_count": sum(value < 0.0 for value in signed),
            "case_count": len(diagnostics),
            "peak_at_reference_vertex_count": sum(
                item.predicted_peak_vertex == item.reference_peak_vertex
                for item in diagnostics
            ),
            "median_peak_vertex_distance_mm": statistics.median(distances),
            "median_root_p_relative_l2": statistics.median(
                item.root_p_relative_l2 for item in diagnostics
            ),
            "median_non_root_p_relative_l2": statistics.median(
                item.non_root_p_relative_l2 for item in diagnostics
            ),
            "minimum_vertex_determinant": min(
                item.minimum_vertex_determinant for item in diagnostics
            ),
            "median_relative_energy_gap_to_fem": statistics.median(
                item.relative_energy_gap for item in diagnostics
            ),
            "maximum_relative_energy_gap_to_fem": max(
                item.relative_energy_gap for item in diagnostics
            ),
            "cases": [
                {**asdict(item), "relative_energy_gap": item.relative_energy_gap}
                for item in diagnostics
            ],
        },
        "training_summary": _training_summary(run_root),
        "comparison": {
            **_historical_metrics(),
            "s13c": {
                key: getattr(metrics, key)
                for key in (
                    "median_absolute_peak_error",
                    "p90_absolute_peak_error",
                    "median_signed_peak_error",
                    "p_relative_l2",
                    "u_relative_l2",
                    "invalid_collocation_points",
                )
            },
        },
        "selection_status": "ELIGIBLE" if selected else "NO_ELIGIBLE_TRIAL",
        "cp_freeze_status": "WITHHELD_PENDING_USER_RATIFICATION",
        "g13_status": "NOT_CLOSED",
        "non_train_evidence_used": evaluation.development_opened,
    }


def write_screen_report(
    *,
    s10_run_root: Path,
    checkpoint_path: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    evaluation = evaluate_s13c_checkpoint(
        s10_run_root=s10_run_root,
        checkpoint_path=checkpoint_path,
    )
    report = screen_report(evaluation, checkpoint_path.parent)
    target = _ARTIFACTS / SCREEN_REPORT_NAME if output is None else output
    target.write_bytes(canonical_json_bytes(report))
    return report


def compare_replay(
    *,
    registered_run_root: Path,
    probe_run_root: Path,
    target_step: int,
    output: Path | None = None,
) -> dict[str, Any]:
    """Compare an independent, mid-way-resumed replay with the registered run.

    The replay must be produced by ``python -m campaigns.dogbone_v1.s13c_trial``
    executed from a clean checkout of the registered training revision, with
    ``--segment-steps`` set to half of ``target_step`` so that it is interrupted
    and resumed once before reaching the registered checkpoint step.
    """

    registered = registered_run_root / f"checkpoint-step-{target_step:04d}"
    reference = TrainingCheckpointManifest.from_bytes(
        (registered / "checkpoint.json").read_bytes()
    )
    half = target_step // 2
    if not (probe_run_root / f"checkpoint-step-{half:04d}").is_dir():
        raise RuntimeError("replay probe was not interrupted at half of the segment")
    latest, step = latest_checkpoint(
        probe_run_root, expected_run_hash=reference.run_hash
    )
    if latest is None or step != target_step:
        raise RuntimeError("replay probe did not reach the target step")
    replay = TrainingCheckpointManifest.from_bytes(
        (latest / "checkpoint.json").read_bytes()
    )
    files = ("model.safetensors", "state.safetensors", "checkpoint.json")
    identical = {
        name: (latest / name).read_bytes() == (registered / name).read_bytes()
        for name in files
    }
    report = {
        "schema_version": "1.0",
        "step": "S13c",
        "evidence_role": "TRAIN",
        "run_hash": reference.run_hash,
        "target_step": target_step,
        "replay_interrupted_and_resumed_at_step": half,
        "registered_model_state_hash": reference.model_state_hash,
        "replay_model_state_hash": replay.model_state_hash,
        "registered_weights_hash": reference.model_export_hash,
        "replay_weights_hash": replay.model_export_hash,
        "registered_manifest_hash": reference.manifest_hash,
        "replay_manifest_hash": replay.manifest_hash,
        "byte_identical_files": identical,
        "g13_determinism": (
            "PASS"
            if replay.model_export_hash == reference.model_export_hash
            else "FAIL"
        ),
        "g13_resume": (
            "PASS"
            if all(identical.values())
            and replay.model_state_hash == reference.model_state_hash
            else "FAIL"
        ),
    }
    target = _ARTIFACTS / PROBE_REPORT_NAME if output is None else output
    target.write_bytes(canonical_json_bytes(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write S13c TRAIN-only evidence.")
    sub = parser.add_subparsers(dest="command", required=True)
    screen = sub.add_parser("screen")
    screen.add_argument("--s10-run-root", required=True, type=Path)
    screen.add_argument("--checkpoint-path", required=True, type=Path)
    probe = sub.add_parser("probe")
    probe.add_argument("--registered-run-root", required=True, type=Path)
    probe.add_argument("--probe-run-root", required=True, type=Path)
    probe.add_argument("--target-step", type=int, default=400)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    if options.command == "screen":
        report = write_screen_report(
            s10_run_root=options.s10_run_root,
            checkpoint_path=options.checkpoint_path,
        )
        print(json.dumps(report["comparison"], indent=1), flush=True)
        print(report["selection_status"], report["failed_guardrails"], flush=True)
        return 0
    report = compare_replay(
        registered_run_root=options.registered_run_root,
        probe_run_root=options.probe_run_root,
        target_step=options.target_step,
    )
    print(json.dumps(report, indent=1), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = (
    "PROBE_REPORT_NAME",
    "SCREEN_REPORT_NAME",
    "build_parser",
    "compare_replay",
    "main",
    "screen_report",
    "write_screen_report",
)
