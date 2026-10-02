"""Approval-bound, single-attempt TRAIN-only joint U/P diagnostic coordinator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar, cast

import rfc8785
import torch

from campaigns.dogbone_v1.s11_contingency_models import model_state_hash
from campaigns.dogbone_v1.s11_contingency_run import qoi_vertex_indices
from campaigns.dogbone_v1.s11_contingency_source import open_selective_s10_source
from campaigns.dogbone_v1.s13_data import materialize_dogbone_s13_training_case
from campaigns.dogbone_v1.s13_initialization import load_selected_s11_initialization
from campaigns.dogbone_v1.s13_joint_hybrid_metrics import (
    JointHybridEndpointComparison,
    JointHybridSnapshot,
    evaluate_joint_snapshot,
    summarize_joint_endpoint,
)
from campaigns.dogbone_v1.s13_joint_hybrid_protocol import (
    JointHybridPlan,
    build_joint_hybrid_proposal,
)
from campaigns.dogbone_v1.s13_joint_hybrid_training import (
    JointArmTrainingResult,
    JointTrainingFailure,
    SampledJointObjective,
    build_joint_vertex_sampler,
    clone_joint_arms,
    train_joint_arm,
)
from campaigns.dogbone_v1.s13_objective import DogboneS13TrainingCase
from campaigns.dogbone_v1.s13_overfit_diagnostic import (
    _exclusive_write,
    _measure_case,
    _preflight_paths,
    _train_one_case,
    build_s13_overfit_plan,
)
from campaigns.dogbone_v1.s13_quadrature_audit import (
    QuadratureAuditPlan,
    audit_implementation_hashes,
    load_pinned_report,
    rebuild_fem_quadrature,
    require_bindings,
    validate_replay_runtime,
    verify_replay,
)
from piginot.contracts import canonical_json_bytes, hash_spec, strict_json_loads
from piginot.doe import EvidenceRole
from piginot.operators import OperatorConfig, PiGinotOperator, module_digest

_ROOT = Path(__file__).resolve().parents[2]
_PUBLIC = _ROOT / "artifacts/campaigns/dogbone-v1"


@dataclass(frozen=True, slots=True)
class JointHybridApproval:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"
    proposal_sha256: str = (
        "b613b4dc884e0b34d5a253fd81154abd11406e3b5d67ef1d89ab1e88f6cf7e0a"
    )
    plan_hash: str = "c7e698556c23b4163d5ff6a22a1a30cd600904312e7f7e2a445bafdf58b718e2"
    status: str = "APPROVED"
    human_message: str = "i approve and continue"
    ratification_date: str = "2026-10-01"
    scope: str = "ADR-0017-one-reconstruction-two-joint-arms-no-repeat-no-promotion"
    approval_is_cryptographic_attestation: bool = False

    def __post_init__(self) -> None:
        if any(
            type(getattr(self, item.name)) is not type(item.default)
            or getattr(self, item.name) != item.default
            for item in fields(self)
        ):
            raise ValueError("approval differs from this chat's exact ratification")


def _public_record(path: Path) -> dict[str, object]:
    if not path.is_absolute():
        raise ValueError("public record path must be absolute")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("public record must not traverse symlinks")
    encoded = path.read_bytes()
    value = strict_json_loads(encoded)
    if type(value) is not dict or rfc8785.dumps(value) + b"\n" != encoded:
        raise ValueError("public record must be canonical JSON plus newline")
    return cast(dict[str, object], value)


def load_approval(path: Path) -> JointHybridApproval:
    approval = JointHybridApproval()
    if _public_record(path) != strict_json_loads(canonical_json_bytes(approval)):
        raise ValueError("approval record differs from the exact ratification")
    return approval


def implementation_hashes() -> tuple[tuple[str, str], ...]:
    """Pin all project runtime Python sources; never inspect experiments or data."""
    paths = sorted(
        (
            *(_ROOT / "src/piginot").rglob("*.py"),
            *(_ROOT / "campaigns/dogbone_v1").rglob("*.py"),
        ),
        key=lambda path: path.relative_to(_ROOT).as_posix(),
    )
    requirement = _ROOT / "campaigns/dogbone_v1/s13_joint_tooling_requirements.txt"
    if requirement.exists():
        paths.append(requirement)
    if any(path.is_symlink() for path in paths):
        raise ValueError("runtime source must not be a symlink")
    return tuple(
        (
            path.relative_to(_ROOT).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in paths
    )


def dependency_versions() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (item.metadata["Name"].lower().replace("_", "-"), item.version)
            for item in importlib.metadata.distributions()
        )
    )


@dataclass(frozen=True, slots=True)
class JointHybridExecutionPin:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"
    role: EvidenceRole
    non_gate: bool
    plan_hash: str
    approval_sha256: str
    runtime_proof_sha256: str
    implementation_sha256: tuple[tuple[str, str], ...]
    dependency_versions: tuple[tuple[str, str], ...]


def build_execution_pin() -> JointHybridExecutionPin:
    proof_path = _PUBLIC / "S13-JOINT-RUNTIME-PROOF.json"
    proof = _public_record(proof_path)
    if (
        proof.get("$type") != "piginot.campaigns.JointRuntimeProof"
        or proof.get("audit_status") != "PASS"
        or proof.get("dependency_check_status") != "PASS"
        or proof.get("shared_runtime_unchanged") is not True
        or proof.get("scientific_packages_unchanged") is not True
    ):
        raise ValueError("isolated runtime proof must pass before execution pinning")
    return JointHybridExecutionPin(
        EvidenceRole.TRAIN,
        True,
        JointHybridPlan().content_hash,
        hashlib.sha256(canonical_json_bytes(JointHybridApproval()) + b"\n").hexdigest(),
        hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        implementation_hashes(),
        dependency_versions(),
    )


def validate_execution_pin() -> JointHybridExecutionPin:
    from campaigns.dogbone_v1.s13_joint_hybrid_runtime import (
        assert_project_origins,
        scientific_package_hashes,
        scientific_runtime,
    )

    approval = load_approval(_PUBLIC / "S13-JOINT-HYBRID-APPROVAL.json")
    proposal = build_joint_hybrid_proposal(_ROOT)
    if proposal.plan.content_hash != approval.plan_hash:
        raise ValueError("approved plan identity changed")
    assert_project_origins(_ROOT)
    scientific_runtime()
    validate_replay_runtime(QuadratureAuditPlan())
    pin = _public_record(_PUBLIC / "S13-JOINT-HYBRID-EXECUTION-PIN.json")
    expected = build_execution_pin()
    if pin != strict_json_loads(canonical_json_bytes(expected)):
        raise ValueError("execution code/dependencies binding mismatch")
    proof = _public_record(_PUBLIC / "S13-JOINT-RUNTIME-PROOF.json")
    require_bindings(
        proof,
        {"scientific_package_hashes": scientific_package_hashes()},
        "scientific binaries",
    )
    return expected


@dataclass(frozen=True, slots=True)
class JointHybridRunReport:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"
    role: EvidenceRole
    non_gate: bool
    status: str
    plan: JointHybridPlan
    plan_hash: str
    approval: JointHybridApproval
    git_revision: str
    execution_pin_hash: str
    source_bindings_hash: str | None
    original_s11_state_hash: str | None
    operator_config: OperatorConfig | None
    operator_module_digest: str | None
    warmstart: JointHybridSnapshot | None
    warmstart_optimizer_updates: int | None
    arms: tuple[JointArmTrainingResult, ...]
    comparison: JointHybridEndpointComparison | None
    failure_stage: str | None
    failure_reason: str | None
    failure_details: tuple[tuple[str, object], ...]
    development_access_absent: bool
    postflight_verified: bool
    g13_status: str = "NOT_PASSED"
    cp_freeze_status: str = "WITHHELD"
    s14_status: str = "BLOCKED"


def _clean_revision() -> str:
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty.strip():
        raise ValueError("reviewed runner must be committed with a clean worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_registered_population(
    case: DogboneS13TrainingCase, qoi: tuple[int, ...], plan: JointHybridPlan
) -> None:
    if (len(case.request.mesh.nodes), len(case.request.mesh.cells), len(qoi)) != (
        plan.expected_vertices,
        plan.quadrature_layout[0],
        20,
    ) or tuple(case.tensors.qoi_indices.tolist()) != qoi:
        raise ValueError("registered mesh/QoI population differs")


def run_joint_hybrid(
    *,
    s10_run_root: Path,
    s11_checkpoint_root: Path,
    development_access_record: Path,
    output_path: Path,
) -> JointHybridRunReport:
    # All authority, runtime, source-code and path checks precede private reads.
    pin = validate_execution_pin()
    revision = _clean_revision()
    s10, s11, access, output = _preflight_paths(
        s10_run_root, s11_checkpoint_root, development_access_record, output_path
    )
    assert output is not None
    plan = JointHybridPlan()
    v5_path = _PUBLIC / "S13-OVERFIT-BUDGET-V5.json"
    quad_path = _PUBLIC / "S13-QUADRATURE-GRADIENT-COMPARISON.json"
    v5 = load_pinned_report(
        v5_path, plan.v5_sha256, "piginot.campaigns.S13OverfitDiagnosticReport"
    )
    quad = load_pinned_report(
        quad_path, plan.quadrature_sha256, "piginot.campaigns.QuadratureAuditReport"
    )
    require_bindings(
        quad,
        {"implementation_sha256": audit_implementation_hashes()},
        "historical code",
    )
    stage = "initialization"
    source_hash = initial_hash = None
    operator_config = None
    operator_digest = None
    warmstart = None
    warm_updates: int | None = 0
    arms: list[JointArmTrainingResult] = []
    comparison = None
    failure_stage = failure_reason = None
    failure_details: tuple[tuple[str, object], ...] = ()
    postflight = False
    try:
        initialization = load_selected_s11_initialization(s11)
        operator_config = initialization.model.config
        operator_digest = module_digest(initialization.model)
        source = open_selective_s10_source(s10, development_access_record=access)

        def closed() -> None:
            if source.development_opened or access.exists():
                raise RuntimeError("DEVELOPMENT/HOLDOUT must remain closed")

        closed()
        source_hash = source.bindings.content_hash
        initial_hash = model_state_hash(initialization.model)
        require_bindings(
            v5,
            {
                "source_bindings_hash": source_hash,
                "s11_evidence_file_hash": initialization.evidence_file_blake2b256,
                "initial_checkpoint_file_hash": initialization.checkpoint_file_blake2b256,
                "initial_model_state_hash": initial_hash,
            },
            "original S11 inputs",
        )
        stage = "TRAIN-case-verification"
        case = materialize_dogbone_s13_training_case(
            source, initialization, plan.case_id
        )
        verified, _ = source.load_fit_with_collocation(plan.case_id)
        if (
            case.tensors.role is not EvidenceRole.TRAIN
            or verified.case.role is not EvidenceRole.TRAIN
            or verified.case.case_id != plan.case_id
            or verified.request != case.request
        ):
            raise PermissionError("only the registered TRAIN-fit case is permitted")
        require_bindings(
            quad,
            {
                "source_bindings_hash": source_hash,
                "request_content_hash": hash_spec(case.request, "content"),
                "result_content_hash": hash_spec(verified.result, "content"),
            },
            "TRAIN source",
        )
        qoi = qoi_vertex_indices(
            case.request.mesh, semantic_tag="right_arc", excluded_tags=("right_grip",)
        )
        _validate_registered_population(case, qoi, plan)
        reference = rebuild_fem_quadrature(case.request, verified.result, qoi)
        sampler = build_joint_vertex_sampler(plan)
        original = _measure_case(initialization.model, case)
        require_bindings(
            v5,
            {"before": original, "vertex_sampler_plan_hash": sampler.plan_hash},
            "original S11 metrics/sampler",
        )
        require_bindings(v5, {"plan_hash": plan.v5_plan_hash}, "V5 replay plan")
        stage = "V5-exact-reconstruction"
        closed()
        print("Reconstructing exact V5 once: 4,000 historical updates", flush=True)
        # The immutable historical loop cannot expose partial progress on error.
        warm_updates = None
        trace = _train_one_case(
            initialization.model, case, build_s13_overfit_plan(version="v5"), sampler
        )
        warm_updates = len(trace)
        verify_replay(
            initialization.model,
            trace,
            v5.get("optimizer_loss_trace"),
            plan.v5_state_hash,
            "V5",
        )
        baseline = initialization.model
        baseline_metrics = _measure_case(baseline, case)
        require_bindings(v5, {"after": baseline_metrics}, "V5 endpoint metrics")
        warmstart = evaluate_joint_snapshot(
            baseline,
            case,
            reference,
            arm_id="v5",
            step=0,
            original_s11_metrics=original,
            pinned_v5_metrics=baseline_metrics,
            plan=plan,
        )
        records = quad.get("arms")
        if (
            not isinstance(records, list)
            or not records
            or not isinstance(records[0], dict)
        ):
            raise ValueError("pinned quadrature warmstart record malformed")
        require_bindings(
            records[0],
            {
                "arm_id": "v5",
                "state_hash": plan.v5_state_hash,
                "metrics": warmstart.quadrature_metrics,
                "module_digest": operator_digest,
            },
            "V5 quadrature replay",
        )
        states = clone_joint_arms(baseline, plan)
        objective = SampledJointObjective(case)
        for state in states:
            stage = f"{state.arm_id}-joint-training"
            closed()
            if model_state_hash(state.model) != plan.v5_state_hash:
                raise ValueError("arm clone differs from exact warmstart")
            if (
                state.model.config != operator_config
                or module_digest(state.model) != operator_digest
            ):
                raise ValueError("arm config/module differs from baseline")

            def evaluate(
                model: PiGinotOperator, step: int, *, arm_id: str = state.arm_id
            ) -> JointHybridSnapshot:
                closed()
                measured = evaluate_joint_snapshot(
                    model,
                    case,
                    reference,
                    arm_id=arm_id,
                    step=step,
                    original_s11_metrics=original,
                    pinned_v5_metrics=baseline_metrics,
                    plan=plan,
                )
                print(
                    f"{arm_id} step {step}: U={measured.vertex_metrics.u_relative_l2:.4%}, "
                    f"P={measured.vertex_metrics.p_relative_l2:.4%}, "
                    f"grad H1={measured.quadrature_metrics.all_points.gradient_relative_h1:.4%}",
                    flush=True,
                )
                return measured

            arms.append(
                train_joint_arm(state, objective, sampler, plan, evaluate=evaluate)
            )
            if (
                arms[-1].config_hash != hash_spec(operator_config, "content")
                or arms[-1].module_digest != operator_digest
            ):
                raise RuntimeError("arm config/module binding changed during training")
        comparison = summarize_joint_endpoint(
            cast(JointHybridSnapshot, arms[1].evaluations[-1][1]),
            cast(JointHybridSnapshot, arms[0].evaluations[-1][1]),
            warmstart,
            plan=plan,
        )
        stage = "postflight"
        closed()
        latest, _ = source.load_fit_with_collocation(plan.case_id)
        if latest.request != verified.request or latest.result != verified.result:
            raise RuntimeError("TRAIN request/result changed during the contrast")
        if model_state_hash(baseline) != plan.v5_state_hash:
            raise RuntimeError("immutable baseline changed during the contrast")
        if (
            baseline.config != operator_config
            or module_digest(baseline) != operator_digest
        ):
            raise RuntimeError("immutable baseline config/module changed")
        if validate_execution_pin() != pin or _clean_revision() != revision:
            raise RuntimeError("execution pin or revision changed during the contrast")
        postflight = True
    except Exception as error:
        failure_stage, failure_reason = stage, f"{type(error).__name__}: {error}"
        if isinstance(error, JointTrainingFailure):
            failure_details = (
                ("arm_id", error.arm_id),
                ("attempted_step", error.attempted_step),
                ("optimizer_updates", error.optimizer_updates),
                ("completed_step_trace", error.step_trace),
                ("completed_evaluations", error.evaluations),
                ("cause_type", error.cause_type),
            )
        print(f"STOPPED without retry at {failure_stage}: {failure_reason}", flush=True)
    report = JointHybridRunReport(
        EvidenceRole.TRAIN,
        True,
        "COMPLETED" if postflight else "FAILED",
        plan,
        plan.content_hash,
        JointHybridApproval(),
        revision,
        hash_spec(pin, "content"),
        source_hash,
        initial_hash,
        operator_config,
        operator_digest,
        warmstart,
        warm_updates,
        tuple(arms),
        comparison,
        failure_stage,
        failure_reason,
        failure_details,
        not access.exists(),
        postflight,
    )
    _exclusive_write(output, canonical_json_bytes(report) + b"\n")
    return report


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "s10-run-root",
        "s11-checkpoint-root",
        "development-access-record",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(arguments)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    report = run_joint_hybrid(
        s10_run_root=args.s10_run_root,
        s11_checkpoint_root=args.s11_checkpoint_root,
        development_access_record=args.development_access_record,
        output_path=args.output,
    )
    return 0 if report.status == "COMPLETED" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
