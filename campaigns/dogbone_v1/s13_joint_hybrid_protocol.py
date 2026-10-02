"""Public, approval-required S13 proposal. This module cannot execute training."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar, cast

import rfc8785

from piginot.contracts import canonical_json_bytes, hash_spec, strict_json_loads
from piginot.doe import EvidenceRole


def _same_fixed_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, tuple):
        actual_tuple = cast(tuple[object, ...], actual)
        return len(actual_tuple) == len(expected) and all(
            _same_fixed_value(left, right)
            for left, right in zip(actual_tuple, expected, strict=True)
        )
    return bool(actual == expected)


def _require_fixed_defaults(contract: JointHybridPlan | JointHybridProposal) -> None:
    if any(
        not _same_fixed_value(getattr(contract, item.name), item.default)
        for item in fields(contract)
    ):
        raise ValueError("contract differs from the fixed draft proposal")


@dataclass(frozen=True, slots=True)
class JointHybridPlan:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    diagnostic_id: str = "dogbone-s13-joint-hybrid-contrast-v1"
    role: EvidenceRole = EvidenceRole.TRAIN
    non_gate: bool = True
    case_id: str = "candidate-002485"
    v5_sha256: str = "d443f6b51ab1883fc1cc704e3094a3a408c410de0cb14979359ec5b5848b1326"
    quadrature_sha256: str = (
        "874303154e156d93949f1fe1b8f1d1a408ea61ed5fe5a58b0425803cabf92a57"
    )
    v5_plan_hash: str = (
        "c144f6f3b3a7025d779e6cfe564679aa57601bad97e4668c91b0c17075f38840"
    )
    v5_state_hash: str = (
        "1db8494d95792f99d8566c2c078ba1ceabd1df416fe2df85cadfb6b76d91adab"
    )
    vertex_sampler_plan_hash: str = (
        "f894eeab7ea354a2d2ee0085ba8fef17b521df59329071e380e02749495df861"
    )
    source_bindings_hash: str = (
        "e32148e4efde6c54fe02ca25b86124d1a43a620eecddb235122919f1649461e3"
    )
    request_content_hash: str = (
        "330d6424b1aa5d506e7e89f0baf5f2b560ff4d8ce141646530701bb6255d11e7"
    )
    result_content_hash: str = (
        "956c21386b01a451dc0a85cce9baa1fc6a17df04e30393ea093b961620ae5956"
    )
    architecture_policy: str = "exact-v5-baseline-both-arms"
    trainable_policy: str = "all-shared-u-head-p-head-parameters"
    warmstart_policy: str = "exact-v5-replay-all-losses-and-state-no-new-clipping"
    warmstart_optimizer_steps: int = 4000
    arm_optimizer_steps: int = 4000
    optimizer_policy: str = "fresh-adam-each-arm-float64-cpu-foreach-false-fused-false"
    learning_rate: float = 2e-5
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    seed: int = 13001
    vertex_batch_size: int = 57
    expected_vertices: int = 969
    physics_collocation_layout: tuple[int, int, int] = (128, 64, 64)
    collocation_policy: str = "same-recorded-interior-background-boundary-right-arc"
    residual_reduction: str = "existing-nondimensional-unweighted-per-term-mean-square"
    control_term_weights: tuple[tuple[str, float], ...] = (
        ("data_u", 1.0),
        ("data_P", 1.0),
    )
    candidate_term_weights: tuple[tuple[str, float], ...] = control_term_weights + (
        ("equilibrium_strong", 1e-3),
        ("constitutive", 1e-3),
        ("angular", 1e-3),
        ("dirichlet", 1e-3),
        ("traction_free", 1e-3),
    )
    adaptive_weighting: bool = False
    scheduler: bool = False
    gradient_clip_norm: float = 1.0
    orientation_policy: str = "existing-v2-candidate-only-no-material-extension"
    determinant_margin: float = 1e-12
    invalid_row_buffer: float = 0.1
    invalid_row_gain: float = 100.0
    common_abort_policy: str = (
        "nonfinite-loss-gradient-parameter-or-valid-F-kernel-failure"
    )
    cpu_num_threads: int = 8
    cpu_interop_threads: int = 8
    torch_version: str = "2.13.0+cpu"
    evaluation_steps: tuple[int, ...] = (0, 1000, 2000, 3000, 4000)
    comparison_endpoint: int = 4000
    evaluation_policy: str = "no-rng-mode-optimizer-or-parameter-mutation"
    quadrature_layout: tuple[int, int] = (1792, 7)
    query_chunk_size: int = 512
    quadrature_policy: str = "existing-physical-area-all-root-incident-nonroot"
    conditional_ratio_policy: str = "NA-unless-admissible-point-support-identical"
    metrics_policy: str = (
        "U-H1-served-P-material-from-grad-P-constitutive-A-B-C-coverage"
    )
    comparison_policy: str = "descriptive-endpoint-ratios-vs-control-and-pinned-v5"
    numeric_repair_threshold: None = None
    vertex_classification_policy: str = "unchanged-original-S11-MSE-denominator"
    checkpoint_policy: str = "none-no-promotion-no-best-step"
    repeat_policy: str = (
        "no-repeat-extension-third-arm-or-retuning-without-new-approval"
    )
    non_train_access: bool = False
    new_fem_solves: int = 0

    def __post_init__(self) -> None:
        _require_fixed_defaults(self)

    @property
    def total_optimizer_updates(self) -> int:
        return self.warmstart_optimizer_steps + 2 * self.arm_optimizer_steps

    @property
    def content_hash(self) -> str:
        return hash_spec(self, "content")


@dataclass(frozen=True, slots=True)
class JointHybridProposal:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    plan: JointHybridPlan = JointHybridPlan()
    role: EvidenceRole = EvidenceRole.TRAIN
    non_gate: bool = True
    status: str = "DRAFT_REQUIRES_EXACT_APPROVAL"
    execution_authorized: bool = False
    training_performed: bool = False
    g13_status: str = "NOT_PASSED"
    cp_freeze_status: str = "WITHHELD"
    s14_status: str = "BLOCKED"

    def __post_init__(self) -> None:
        _require_fixed_defaults(self)

    @property
    def content_hash(self) -> str:
        return hash_spec(self, "content")


def _read_public_evidence(
    path: Path, expected_sha256: str, expected_type: str
) -> dict[str, object]:
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("public evidence must not traverse symlinks")
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ValueError("public evidence SHA-256 differs from the pinned report")
    record = strict_json_loads(encoded)
    if type(record) is not dict or rfc8785.dumps(record) + b"\n" != encoded:
        raise ValueError("public evidence must be canonical JSON with one newline")
    if (
        record.get("$type") != expected_type
        or record.get("role") != "TRAIN"
        or record.get("non_gate") is not True
        or record.get("case_id") != "candidate-002485"
    ):
        raise ValueError("public evidence type, role or case binding differs")
    return cast(dict[str, object], record)


def build_joint_hybrid_proposal(repository_root: Path) -> JointHybridProposal:
    """Verify two committed public reports, without opening any source payload."""
    if not repository_root.is_absolute():
        raise ValueError("repository_root must be absolute")
    proposal = JointHybridProposal()
    plan = proposal.plan
    public = repository_root / "artifacts/campaigns/dogbone-v1"
    v5 = _read_public_evidence(
        public / "S13-OVERFIT-BUDGET-V5.json",
        plan.v5_sha256,
        "piginot.campaigns.S13OverfitDiagnosticReport",
    )
    quadrature = _read_public_evidence(
        public / "S13-QUADRATURE-GRADIENT-COMPARISON.json",
        plan.quadrature_sha256,
        "piginot.campaigns.QuadratureAuditReport",
    )
    expected = (
        (v5, "plan_hash", plan.v5_plan_hash),
        (v5, "final_model_state_hash", plan.v5_state_hash),
        (v5, "vertex_sampler_plan_hash", plan.vertex_sampler_plan_hash),
        (v5, "source_bindings_hash", plan.source_bindings_hash),
        (quadrature, "source_bindings_hash", plan.source_bindings_hash),
        (quadrature, "request_content_hash", plan.request_content_hash),
        (quadrature, "result_content_hash", plan.result_content_hash),
    )
    if any(record.get(key) != value for record, key, value in expected):
        raise ValueError("public evidence lineage differs from the fixed draft")
    return proposal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output: Path = args.output
    if not output.is_absolute():
        raise ValueError("output must be absolute")
    if any(parent.is_symlink() for parent in (output, *output.parents)):
        raise ValueError("output must not traverse symlinks")
    root = Path(__file__).resolve().parents[2]
    proposal = build_joint_hybrid_proposal(root)
    encoded = canonical_json_bytes(proposal) + b"\n"
    with output.open("xb") as handle:
        handle.write(encoded)
    print(f"Draft proposal SHA-256: {hashlib.sha256(encoded).hexdigest()}")
    print(f"Plan content hash: {proposal.plan.content_hash}")
    print("No execution authorized; no training performed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
