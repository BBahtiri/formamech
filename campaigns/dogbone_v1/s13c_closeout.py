"""G13 gate artifact and CP-FREEZE record for the accepted S13c candidate.

``g13_gate_payload`` measures every G13 check of STEPS-v1.0 on the registered
S13c run and the committed TRAIN-only evidence:

- determinism and resume from the independent replay probe, bound to the
  registered step-400 checkpoint files;
- search-space, holdout, identity, lineage and selection through the
  production S13 APIs, with the ``SELECT_ON_HOLDOUT`` and
  ``CHOOSE_NONRANKED_CHECKPOINT`` mutants;
- the training-package coverage supplied by the caller.

``cp_freeze_payload`` then freezes exactly that candidate once both the
amended G11 artifact and G13 pass (ADR 0018).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from campaigns.dogbone_v1.s11_closeout import (
    _clean_source_revision,
    _current_branch,
    _require_revision,
    _require_text,
)
from campaigns.dogbone_v1.s11_contingency_source import open_selective_s10_source
from campaigns.dogbone_v1.s13c_evaluate import (
    load_registered_checkpoint,
    load_run_binding,
)
from campaigns.dogbone_v1.s13c_g11 import write_gate_artifact
from campaigns.dogbone_v1.s13c_protocol import search_space
from campaigns.dogbone_v1.s13c_trial import (
    RUN_BINDING_NAME,
    load_committed_s13c_protocol,
)
from campaigns.history import UnknownRevisionError, resolve_revision
from piginot.contracts import strict_json_loads
from piginot.doe import EvidenceRole
from piginot.git_environment import isolated_git_environment
from piginot.operators import GateStatus
from piginot.training import (
    CheckpointLineage,
    GridSearchSpace,
    OutOfSearchSpaceError,
    SelectionMetrics,
    SelectionPolicy,
    TrainingCheckpointManifest,
    TrialResult,
    require_trial_in_space,
    select_checkpoint,
    select_training_trial,
)
from piginot.training.selection import (
    _choose_nonranked_checkpoint_mutant_for_test,
    _select_on_holdout_mutant_for_test,
)

_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACTS = _ROOT / "artifacts" / "campaigns" / "dogbone-v1"
_GATES = _ROOT / "artifacts" / "gates"
PROTOCOL_PATH = _ARTIFACTS / "S13C-PROTOCOL.json"
SCREEN_PATH = _ARTIFACTS / "S13C-TRAIN-SCREEN.json"
PROBE_PATH = _ARTIFACTS / "S13C-G13-REPLAY-PROBE.json"
MANIFEST_PATH = _ARTIFACTS / "public-case-manifest.json"
S10_GATE_PATH = _GATES / "S10.json"
G11_ENERGY_PATH = _GATES / "S11-ENERGY.json"
G13_PATH = _GATES / "S13.json"
CP_FREEZE_PATH = _ARTIFACTS / "CP-FREEZE.json"
_FINAL_STEP = 2_400
_PROBE_STEP = 400
_MINIMUM_COVERAGE_PERCENT = 80.0
_COVERAGE_COMMAND = (
    "pytest -q tests/training tests/test_s13_anti_scope.py "
    "--cov=src/piginot/training --cov-report=term-missing --cov-fail-under=80"
)
_RATIFICATION_DATE = "2026-10-02"


def _load_json(path: Path) -> dict[str, Any]:
    document = strict_json_loads(path.read_bytes())
    if not isinstance(document, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return document


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(run_root: Path, step: int) -> TrainingCheckpointManifest:
    path = run_root / f"checkpoint-step-{step:04d}" / "checkpoint.json"
    return TrainingCheckpointManifest.from_bytes(path.read_bytes())


def _check(
    check_id: str,
    passed: bool,
    measured: object,
    threshold: object,
    comparison: str = "all",
) -> dict[str, object]:
    return {
        "id": check_id,
        "status": (GateStatus.PASS if passed else GateStatus.FAIL).value,
        "measured": measured,
        "threshold": threshold,
        "comparison": comparison,
        "reason": None,
    }


def _strip_type(document: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "$type"}


def _screen_result(screen: dict[str, Any]) -> TrialResult:
    document = screen["trial_result"]
    return TrialResult(
        lineage=CheckpointLineage(**_strip_type(document["lineage"])),
        role=EvidenceRole(document["role"]),
        metrics=SelectionMetrics(**_strip_type(document["metrics"])),
    )


def _determinism_and_resume(
    run_root: Path, probe: dict[str, Any], run_hash: str
) -> tuple[dict[str, object], dict[str, object]]:
    registered = _manifest(run_root, _PROBE_STEP)
    determinism = {
        "probe_run_hash_matches": probe["run_hash"] == run_hash,
        "registered_weights_hash_matches_run_root": (
            probe["registered_weights_hash"] == registered.model_export_hash
        ),
        "replay_weights_hash_equals_registered": (
            probe["replay_weights_hash"] == probe["registered_weights_hash"]
        ),
        "probe_verdict": probe["g13_determinism"],
    }
    resume = {
        "interrupted_at_step": probe["replay_interrupted_and_resumed_at_step"],
        "target_step": probe["target_step"],
        "byte_identical_files": probe["byte_identical_files"],
        "registered_model_state_hash_matches_run_root": (
            probe["registered_model_state_hash"] == registered.model_state_hash
        ),
        "replay_model_state_hash_equals_registered": (
            probe["replay_model_state_hash"] == probe["registered_model_state_hash"]
        ),
        "probe_verdict": probe["g13_resume"],
    }
    determinism_passed = (
        all(
            value is True
            for key, value in determinism.items()
            if key != "probe_verdict"
        )
        and probe["g13_determinism"] == "PASS"
        and probe["target_step"] == _PROBE_STEP
    )
    resume_passed = (
        probe["target_step"] == _PROBE_STEP
        and probe["replay_interrupted_and_resumed_at_step"] * 2 == _PROBE_STEP
        and all(value is True for value in probe["byte_identical_files"].values())
        and len(probe["byte_identical_files"]) == 3
        and resume["registered_model_state_hash_matches_run_root"] is True
        and resume["replay_model_state_hash_equals_registered"] is True
        and probe["g13_resume"] == "PASS"
    )
    return (
        _check(
            "G13.determinism",
            determinism_passed,
            determinism,
            "independent replay from the clean training revision has the "
            "registered weights hash",
        ),
        _check(
            "G13.resume",
            resume_passed,
            resume,
            "interrupt at 50% of the 400-step segment and resume: model export, "
            "training state and manifest bitwise identical",
        ),
    )


def _search_space_check(screen_trial_id: str) -> dict[str, object]:
    space = search_space()
    (registered,) = space.trials
    authorized = require_trial_in_space(registered, space) == registered
    outside = (
        replace(registered, learning_rate=registered.learning_rate * 10.0),
        replace(registered, optimizer_steps=registered.optimizer_steps + 1),
        replace(registered, case_batch_size=registered.case_batch_size + 1),
    )
    refused = 0
    for trial in outside:
        try:
            require_trial_in_space(trial, space)
        except OutOfSearchSpaceError:
            refused += 1
    measured = {
        "registered_trials": len(space.trials),
        "maximum_trials": space.maximum_trials,
        "registered_trial_authorized": authorized,
        "registered_trial_matches_screen": registered.trial_id == screen_trial_id,
        "out_of_space_trials_refused": refused,
        "out_of_space_trials_tested": len(outside),
    }
    passed = (
        authorized
        and registered.trial_id == screen_trial_id
        and refused == len(outside)
        and len(space.trials) <= space.maximum_trials
    )
    return _check(
        "G13.search-space",
        passed,
        measured,
        "every out-of-space trial refuses; the registered trial is authorized",
    )


def _holdout_check(
    s10_run_root: Path, screen: dict[str, Any], result: TrialResult
) -> dict[str, object]:
    space = search_space()
    holdout_result = replace(result, role=EvidenceRole.HOLDOUT)
    production_refused = 0
    for attempt in (
        lambda: select_training_trial(
            space, (holdout_result,), role=EvidenceRole.HOLDOUT
        ),
        lambda: select_checkpoint((holdout_result,), policy=SelectionPolicy(space)),
    ):
        try:
            attempt()
        except PermissionError:
            production_refused += 1
    mutant = _select_on_holdout_mutant_for_test(
        space, (holdout_result,), role=EvidenceRole.HOLDOUT
    )
    mutant_selected = mutant.role is EvidenceRole.HOLDOUT

    manifest = _load_json(MANIFEST_PATH)
    development_id = next(
        case["case_id"]
        for case in manifest["cases"]
        if case.get("role") == "DEVELOPMENT"
    )
    development_refused = False
    with tempfile.TemporaryDirectory(prefix="piginot-s13c-g13-") as directory:
        record = Path(directory) / "development-access.json"
        source = open_selective_s10_source(
            s10_run_root, development_access_record=record
        )
        try:
            tuple(source.iter_train_geometry((development_id,)))
        except PermissionError:
            development_refused = True
        development_recorded = record.exists() or source.development_opened
    s10_gate = _load_json(S10_GATE_PATH)
    holdout_public = None
    for value in s10_gate.values():
        if isinstance(value, dict) and "holdout_public_count" in value:
            holdout_public = value["holdout_public_count"]
    measured = {
        "production_selection_refusals": production_refused,
        "select_on_holdout_mutant_selected_holdout": mutant_selected,
        "development_case_refused_by_training_source": development_refused,
        "development_access_recorded": development_recorded,
        "holdout_public_case_count": holdout_public,
        "screen_used_non_train_evidence": screen["non_train_evidence_used"],
    }
    passed = (
        production_refused == 2
        and mutant_selected
        and development_refused
        and not development_recorded
        and holdout_public == 0
        and screen["non_train_evidence_used"] is False
    )
    return _check(
        "G13.holdout",
        passed,
        measured,
        "HOLDOUT is unreachable from training and selection; the "
        "SELECT_ON_HOLDOUT mutant (role guard removed) fires",
    )


def _identity_check(run_root: Path) -> dict[str, object]:
    binding = load_run_binding(run_root / RUN_BINDING_NAME)
    profile = binding.execution_profile
    variants = {
        "data": replace(binding, dataset_snapshot_hash="1" * 64),
        "config": replace(binding, trainer_config_hash="2" * 64),
        "code": replace(binding, implementation_revision="3" * 40),
        "profile": replace(
            binding,
            execution_profile=replace(profile, blas_threads=profile.blas_threads + 1),
        ),
    }
    hashes = {name: variant.run_hash for name, variant in variants.items()}
    measured: dict[str, object] = {
        f"{name}_change_creates_new_run_hash": value != binding.run_hash
        for name, value in hashes.items()
    }
    distinct = len({binding.run_hash, *hashes.values()})
    measured["distinct_run_hashes"] = distinct
    passed = distinct == len(variants) + 1
    return _check(
        "G13.identity",
        passed,
        measured,
        "data, config, code and profile changes each create a new run_hash",
    )


def _is_ancestor(revision: str) -> bool:
    try:
        revision = resolve_revision(revision, _ROOT)
    except UnknownRevisionError:
        return False
    completed = subprocess.run(
        ("git", "merge-base", "--is-ancestor", revision, "HEAD"),
        cwd=_ROOT,
        env=isolated_git_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _lineage_check(
    run_root: Path, screen: dict[str, Any], result: TrialResult
) -> dict[str, object]:
    protocol = load_committed_s13c_protocol()
    final_path = run_root / f"checkpoint-step-{_FINAL_STEP:04d}"
    _, binding, manifest = load_registered_checkpoint(protocol, final_path)
    lineage = result.lineage
    measured = {
        "run_hash": manifest.run_hash
        == binding.run_hash
        == screen["run_hash"]
        == lineage.run_hash,
        "checkpoint_manifest_hash": manifest.manifest_hash
        == screen["checkpoint_manifest_hash"]
        == lineage.checkpoint_hash,
        "dataset_snapshot_hash": manifest.dataset_snapshot_hash
        == binding.dataset_snapshot_hash
        == protocol.dataset_snapshot_hash
        == lineage.dataset_snapshot_hash,
        "execution_profile_hash": manifest.execution_profile_hash
        == binding.execution_profile_hash
        == screen["execution_profile_hash"]
        == lineage.execution_profile_hash,
        "weights_hash": manifest.model_export_hash == lineage.weights_hash,
        "model_state_hash": manifest.model_state_hash == lineage.model_state_hash,
        "protocol_hash": protocol.protocol_hash == screen["protocol_hash"],
        "training_revision_recorded": binding.implementation_revision
        == screen["training_revision"],
        "training_revision_in_history": _is_ancestor(binding.implementation_revision),
        "evaluator_revision_in_history": _is_ancestor(screen["evaluator_revision"]),
        "checkpoint_files_verified": True,
    }
    return _check(
        "G13.lineage",
        all(value is True for value in measured.values()),
        measured,
        "the selected checkpoint resolves to its dataset, run, profile, "
        "protocol and source revisions",
    )


def _synthetic_two_trial_decision(result: TrialResult) -> tuple[Any, TrialResult]:
    space = GridSearchSpace(
        learning_rates=(1.0e-4, 2.0e-4),
        physics_weight_scales=(1.0,),
        optimizer_steps=1,
        case_batch_size=1,
        maximum_trials=2,
    )
    better, worse = space.trials
    results = (
        TrialResult(
            lineage=replace(result.lineage, trial_id=better.trial_id),
            role=EvidenceRole.TRAIN,
            metrics=result.metrics,
        ),
        TrialResult(
            lineage=replace(result.lineage, trial_id=worse.trial_id),
            role=EvidenceRole.TRAIN,
            metrics=replace(
                result.metrics,
                median_absolute_peak_error=2.0
                * result.metrics.median_absolute_peak_error,
            ),
        ),
    )
    decision = select_checkpoint(results, policy=SelectionPolicy(space))
    return decision, _choose_nonranked_checkpoint_mutant_for_test(decision)


def _selection_check(screen: dict[str, Any], result: TrialResult) -> dict[str, object]:
    decision = select_checkpoint((result,), policy=SelectionPolicy(search_space()))
    selected = decision.selected_lineage
    recorded = CheckpointLineage(**_strip_type(screen["selected_checkpoint"]))
    synthetic, mutant = _synthetic_two_trial_decision(result)
    measured = {
        "selected_trial_id": selected.trial_id,
        "selected_matches_screen_report": selected == recorded,
        "ranked_eligible_trials": len(decision.ranked_eligible_trial_ids),
        "screen_selection_status": screen["selection_status"],
        "failed_guardrails": screen["failed_guardrails"],
        "nonranked_mutant_detected": (
            mutant.trial_id != synthetic.selected_result.trial_id
            and synthetic.ranked_eligible_trial_ids[0]
            == synthetic.selected_result.trial_id
        ),
    }
    passed = (
        selected == recorded
        and screen["selection_status"] == "ELIGIBLE"
        and screen["failed_guardrails"] == []
        and measured["nonranked_mutant_detected"] is True
    )
    return _check(
        "G13.selection",
        passed,
        measured,
        "the chosen checkpoint is the first ranked eligible trial under the "
        "frozen policy; the CHOOSE_NONRANKED_CHECKPOINT mutant is detected",
    )


def _coverage_check(value: float) -> dict[str, object]:
    return _check(
        "G13.coverage",
        value >= _MINIMUM_COVERAGE_PERCENT,
        value,
        ">= 80.0",
        "gte",
    )


def g13_checks(
    *,
    s10_run_root: Path,
    run_root: Path,
    coverage_percent: float,
) -> list[dict[str, object]]:
    """Measure every G13 check in the STEPS-v1.0 order."""

    screen = _load_json(SCREEN_PATH)
    probe = _load_json(PROBE_PATH)
    result = _screen_result(screen)
    binding = load_run_binding(run_root / RUN_BINDING_NAME)
    determinism, resume = _determinism_and_resume(run_root, probe, binding.run_hash)
    return [
        determinism,
        resume,
        _search_space_check(result.trial_id),
        _holdout_check(s10_run_root, screen, result),
        _identity_check(run_root),
        _lineage_check(run_root, screen, result),
        _selection_check(screen, result),
        _coverage_check(coverage_percent),
    ]


def g13_gate_payload(
    checks: list[dict[str, object]],
    *,
    implementation_source_revision: str,
    branch: str,
    timestamp_utc: str,
    implementation_evidence: dict[str, object],
) -> dict[str, object]:
    """Assemble the public G13 artifact; every required check must PASS."""

    _require_revision(implementation_source_revision, "implementation_source_revision")
    _require_text(branch, "branch")
    _require_text(timestamp_utc, "timestamp_utc")
    expected = (
        "G13.determinism",
        "G13.resume",
        "G13.search-space",
        "G13.holdout",
        "G13.identity",
        "G13.lineage",
        "G13.selection",
        "G13.coverage",
    )
    if tuple(check["id"] for check in checks) != expected:
        raise ValueError("G13 checks must be complete and in STEPS order")
    passed = all(check["status"] == GateStatus.PASS.value for check in checks)
    screen = _load_json(SCREEN_PATH)
    return {
        "schema_version": "1.0",
        "step": "S13",
        "gate_id": "G13",
        "decision_record": "docs/adr/0018-s13c-energy-form-operator.md",
        "timestamp_utc": timestamp_utc,
        "branch": branch,
        "implementation_source_revision": implementation_source_revision,
        "scope": {
            "claim": (
                "The registered S13c trial satisfies every G13 check on "
                "TRAIN-only evidence. This is not an accuracy claim beyond the "
                "TRAIN screen and makes no release claim."
            ),
            "candidate_run_hash": screen["run_hash"],
            "candidate_checkpoint_manifest_hash": screen["checkpoint_manifest_hash"],
            "evidence_role": "TRAIN",
            "development_or_holdout_read": False,
            "adaptive_weighting": (
                "not used: one energy objective with a fixed barrier penalty "
                "(ADR 0018); the S13 EMA controller remains for hybrid objectives"
            ),
            "search_space": "one preregistered trial (maximum_trials = 1)",
        },
        "evidence": {
            "protocol_sha256": _sha256(PROTOCOL_PATH),
            "train_screen_sha256": _sha256(SCREEN_PATH),
            "replay_probe_sha256": _sha256(PROBE_PATH),
            "amended_g11_sha256": _sha256(G11_ENERGY_PATH),
        },
        "implementation_evidence": {
            "focused_coverage_command": _COVERAGE_COMMAND,
            **implementation_evidence,
        },
        "checks": checks,
        "verdict": (GateStatus.PASS if passed else GateStatus.FAIL).value,
        "verdict_reason": (
            "all eight required G13 checks passed"
            if passed
            else "at least one required G13 check did not pass"
        ),
        "limitations": [
            "All selection evidence is TRAIN-role; DEVELOPMENT and HOLDOUT were "
            "never opened.",
            "Determinism and resume are bitwise on the recorded Linux CPU "
            "profile; another profile produces another run_hash.",
            "The search space holds one registered trial; G13.selection proves "
            "rule conformance, not that a better configuration does not exist.",
        ],
    }


def cp_freeze_payload(
    *,
    run_root: Path,
    implementation_source_revision: str,
    timestamp_utc: str,
) -> dict[str, object]:
    """Freeze exactly the selected S13c checkpoint once G11 and G13 pass."""

    _require_revision(implementation_source_revision, "implementation_source_revision")
    _require_text(timestamp_utc, "timestamp_utc")
    g11 = _load_json(G11_ENERGY_PATH)
    g13 = _load_json(G13_PATH)
    if g11.get("verdict") != "PASS" or g13.get("verdict") != "PASS":
        raise RuntimeError("CP-FREEZE requires passing amended G11 and G13 gates")
    protocol = load_committed_s13c_protocol()
    screen = _load_json(SCREEN_PATH)
    final_path = run_root / f"checkpoint-step-{_FINAL_STEP:04d}"
    model, binding, manifest = load_registered_checkpoint(protocol, final_path)
    if (
        manifest.manifest_hash != screen["checkpoint_manifest_hash"]
        or binding.run_hash != g13["scope"]["candidate_run_hash"]
    ):
        raise ValueError("the checkpoint differs from the gated candidate")
    lineage = screen["selected_checkpoint"]
    return {
        "schema_version": "1.0",
        "checkpoint_id": "CP-FREEZE",
        "step": "S13",
        "created_utc": timestamp_utc,
        "implementation_source_revision": implementation_source_revision,
        "decision_record": "docs/adr/0018-s13c-energy-form-operator.md",
        "ratification": {
            "date": _RATIFICATION_DATE,
            "decisions": [
                "D1: adopt the S13c energy-form operator (ADR 0018)",
                "D2: park the ADR 0017 joint-hybrid contrast unexecuted",
                "D3: repair the shared repository and publish the S13 branches",
            ],
        },
        "frozen_candidate": {
            "architecture_id": protocol.architecture_id,
            "architecture_module_digest": protocol.architecture_module_digest,
            "trial_id": lineage["trial_id"],
            "run_hash": binding.run_hash,
            "checkpoint_step": manifest.step,
            "checkpoint_manifest_hash": manifest.manifest_hash,
            "model_state_hash": manifest.model_state_hash,
            "weights_hash": manifest.model_export_hash,
            "model_safetensors_sha256": _sha256(final_path / "model.safetensors"),
            "protocol_hash": protocol.protocol_hash,
            "objective_hash": protocol.objective_hash,
            "dataset_snapshot_hash": binding.dataset_snapshot_hash,
            "normalization_hash": binding.normalization_hash,
            "execution_profile_hash": binding.execution_profile_hash,
            "initial_weights_hash": binding.initial_weights_hash,
            "training_revision": binding.implementation_revision,
            "evaluator_revision": screen["evaluator_revision"],
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
        "selection": {
            "policy": "frozen S13 guardrails and ranking (piginot.training.selection)",
            "search_space_hash": search_space().search_space_hash,
            "status": screen["selection_status"],
            "failed_guardrails": screen["failed_guardrails"],
            "train_screen_metrics": _strip_type(screen["trial_result"]["metrics"]),
        },
        "gates": {
            "G11_amended": {
                "artifact": "artifacts/gates/S11-ENERGY.json",
                "sha256": _sha256(G11_ENERGY_PATH),
                "verdict": g11["verdict"],
            },
            "G13": {
                "artifact": "artifacts/gates/S13.json",
                "sha256": _sha256(G13_PATH),
                "verdict": g13["verdict"],
            },
        },
        "evidence": {
            "S13C-PROTOCOL.json": _sha256(PROTOCOL_PATH),
            "S13C-TRAIN-SCREEN.json": _sha256(SCREEN_PATH),
            "S13C-G13-REPLAY-PROBE.json": _sha256(PROBE_PATH),
        },
        "authorization": {
            "development": (
                "S14 may open DEVELOPMENT to calibrate per-query confidence and "
                "evaluate the error budget of exactly this frozen candidate"
            ),
            "holdout": (
                "S14 may consume HOLDOUT once, for exactly this frozen candidate, "
                "after DEVELOPMENT calibration is fixed; consumption must be "
                "durably recorded before reveal"
            ),
            "change_policy": (
                "any change to the weights, architecture, objective, data or "
                "conditioning voids this record and requires a new S13 gate"
            ),
        },
        "adr_0001_revisit": "docs/adr/0001-s1-go-with-contingency.md (Revisit 2)",
        "limitations": [
            "Selection evidence is TRAIN-role only.",
            "The candidate is conditioned on the DogBone family parameters.",
            "Bitwise reproduction requires the recorded Linux CPU profile.",
        ],
    }


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    g13 = sub.add_parser("g13")
    g13.add_argument("--s10-run-root", required=True, type=Path)
    g13.add_argument("--run-root", required=True, type=Path)
    g13.add_argument("--coverage-percent", required=True, type=float)
    g13.add_argument("--implementation-evidence", required=True, type=Path)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--run-root", required=True, type=Path)
    return parser


def main(argv: tuple[str, ...] | None = None) -> int:
    args = build_parser().parse_args(argv)
    revision = _clean_source_revision()
    if args.command == "g13":
        evidence = json.loads(args.implementation_evidence.read_text("utf-8"))
        if not isinstance(evidence, dict):
            raise ValueError("implementation evidence must be a JSON object")
        checks = g13_checks(
            s10_run_root=args.s10_run_root,
            run_root=args.run_root,
            coverage_percent=args.coverage_percent,
        )
        payload = g13_gate_payload(
            checks,
            implementation_source_revision=revision,
            branch=_current_branch(),
            timestamp_utc=_timestamp(),
            implementation_evidence=evidence,
        )
        write_gate_artifact(payload, G13_PATH)
    else:
        payload = cp_freeze_payload(
            run_root=args.run_root,
            implementation_source_revision=revision,
            timestamp_utc=_timestamp(),
        )
        write_gate_artifact(payload, CP_FREEZE_PATH)
    print(json.dumps(payload.get("checks", payload.get("frozen_candidate")), indent=1))
    print(f"verdict: {payload.get('verdict', 'FROZEN')}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = (
    "CP_FREEZE_PATH",
    "G11_ENERGY_PATH",
    "G13_PATH",
    "build_parser",
    "cp_freeze_payload",
    "g13_checks",
    "g13_gate_payload",
    "main",
)
