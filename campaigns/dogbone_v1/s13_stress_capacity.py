"""Matched TRAIN-only stress-capacity contrast with frozen displacement."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import ClassVar

import torch

from campaigns.dogbone_v1.s11_contingency_models import model_state_hash
from campaigns.dogbone_v1.s11_contingency_source import open_selective_s10_source
from campaigns.dogbone_v1.s13_data import materialize_dogbone_s13_training_case
from campaigns.dogbone_v1.s13_initialization import load_selected_s11_initialization
from campaigns.dogbone_v1.s13_objective import (
    _COORDINATE_CONVERSION,
    DogboneS13TrainingCase,
    _scales,
    _validated_operator,
)
from campaigns.dogbone_v1.s13_overfit_diagnostic import (
    OverfitCaseMetrics,
    OverfitClassification,
    S13OverfitPlan,
    _build_vertex_sampler_plan,
    _exclusive_write,
    _measure_case,
    _preflight_paths,
    _train_one_case,
    build_s13_overfit_plan,
    classify_s13_overfit,
)
from campaigns.dogbone_v1.s13_spatial_audit import (
    SpatialCaseMetrics,
    _load_pinned_report,
    _vertex_prediction_and_physics,
    summarize_spatial_case,
)
from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.datasets import SemanticBCClass
from piginot.doe import EvidenceRole
from piginot.objectives import displacement_gradient
from piginot.operators import OperatorConfig, PiGinotOperator, module_digest

STRESS_PREFIXES = (
    "p_head.",
    "local_boundary_projections.",
    "stress_enrichment_blocks.",
)
_V5_SHA256 = "d443f6b51ab1883fc1cc704e3094a3a408c410de0cb14979359ec5b5848b1326"


@dataclass(frozen=True, slots=True)
class StressCapacityPlan:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    diagnostic_id: str = "dogbone-s13-stress-capacity-contrast-v1"
    role: EvidenceRole = EvidenceRole.TRAIN
    non_gate: bool = True
    fine_tune_plan: S13OverfitPlan = field(
        default_factory=lambda: build_s13_overfit_plan(version="v5")
    )
    warmstart_report_sha256: str = _V5_SHA256
    kernel_scales: tuple[float, ...] = (0.08, 0.25)
    stress_layers: int = 2
    initialization_seed: int = 13_002
    migration_policy_id: str = "strict-shared-copy-zero-stress-residual-v1"
    sampler_policy: str = "restart-v5-identical-schedule-both-arms"
    mse_denominator: str = "original-pinned-s11-initial-state"
    control_trainable_prefixes: tuple[str, ...] = ("p_head.",)
    candidate_trainable_prefixes: tuple[str, ...] = STRESS_PREFIXES
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1.0e-8
    weight_decay: float = 0.0
    data_u_loss_weight: float = 1.0
    data_p_loss_weight: float = 1.0
    peak_loss_weight: float = 0.0
    minimum_root_improvement: float = 0.25
    maximum_nonroot_regression: float = 0.05

    def __post_init__(self) -> None:
        frozen = {
            "diagnostic_id": "dogbone-s13-stress-capacity-contrast-v1",
            "role": EvidenceRole.TRAIN,
            "non_gate": True,
            "fine_tune_plan": build_s13_overfit_plan(version="v5"),
            "warmstart_report_sha256": _V5_SHA256,
            "kernel_scales": (0.08, 0.25),
            "stress_layers": 2,
            "initialization_seed": 13_002,
            "migration_policy_id": "strict-shared-copy-zero-stress-residual-v1",
            "sampler_policy": "restart-v5-identical-schedule-both-arms",
            "mse_denominator": "original-pinned-s11-initial-state",
            "control_trainable_prefixes": ("p_head.",),
            "candidate_trainable_prefixes": STRESS_PREFIXES,
            "adam_betas": (0.9, 0.999),
            "adam_epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "data_u_loss_weight": 1.0,
            "data_p_loss_weight": 1.0,
            "peak_loss_weight": 0.0,
            "minimum_root_improvement": 0.25,
            "maximum_nonroot_regression": 0.05,
        }
        if any(
            type(getattr(self, name)) is not type(value) or getattr(self, name) != value
            for name, value in frozen.items()
        ):
            raise ValueError("stress-capacity plan differs from the frozen protocol")

    @property
    def content_hash(self) -> str:
        return hash_spec(self, "content")


def trainable_names(model: PiGinotOperator) -> tuple[str, ...]:
    return tuple(
        name for name, value in model.named_parameters() if value.requires_grad
    )


def assert_trainable_scope(model: PiGinotOperator) -> None:
    expected = tuple(
        name for name, _ in model.named_parameters() if name.startswith(STRESS_PREFIXES)
    )
    if trainable_names(model) != expected:
        raise ValueError("trainable parameter scope must contain only the stress path")


def frozen_state_hash(model: PiGinotOperator) -> str:
    return hash_spec(
        tuple(
            (
                name,
                str(value.dtype),
                tuple(value.shape),
                hashlib.sha256(
                    value.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
            )
            for name, value in sorted(model.state_dict().items())
            if not name.startswith(STRESS_PREFIXES)
        ),
        "content",
    )


def build_stress_arm(
    source: PiGinotOperator, plan: StressCapacityPlan, *, enriched: bool
) -> PiGinotOperator:
    """Copy a baseline exactly and make any added stress path initially neutral."""
    _validated_operator(source)
    if type(plan) is not StressCapacityPlan or type(enriched) is not bool:
        raise TypeError("migration requires the exact plan and a boolean arm")
    if (
        source.config.local_boundary_kernel_scales
        or source.config.stress_enrichment_layers
    ):
        raise ValueError("warmstart must use the baseline stress architecture")
    config = replace(
        source.config,
        local_boundary_kernel_scales=plan.kernel_scales if enriched else (),
        stress_enrichment_layers=plan.stress_layers if enriched else 0,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(plan.initialization_seed)
        target = PiGinotOperator(config).to(dtype=torch.float64)
    original, expanded = source.state_dict(), target.state_dict()
    new_names = set(expanded) - set(original)
    expected_shared = {
        name
        for name in expanded
        if not name.startswith(
            ("local_boundary_projections.", "stress_enrichment_blocks.")
        )
    }
    if set(original) != expected_shared:
        raise ValueError("migration state schema differs from the baseline")
    copied: dict[str, torch.Tensor] = {}
    for name, value in expanded.items():
        if name not in new_names:
            old = original[name]
            if old.shape != value.shape or old.dtype is not value.dtype:
                raise ValueError("migration state schema has shape/dtype mismatch")
            copied[name] = old.detach().clone()
        elif name.startswith("local_boundary_projections.") or ".3." in name:
            copied[name] = torch.zeros_like(value)
        else:
            copied[name] = value.detach().clone()
    target.load_state_dict(copied, strict=True)
    for name, value in target.named_parameters():
        value.requires_grad_(name.startswith(STRESS_PREFIXES))
    assert_trainable_scope(target)
    return target


def displacement_signature(
    model: PiGinotOperator, case: DogboneS13TrainingCase
) -> tuple[torch.Tensor, ...]:
    """Retain full u, physical Jacobian and det F at vertices and interior points."""
    tensors = case.tensors
    interior = torch.tensor(
        tuple(
            point.coordinates
            for point in case.collocation.points
            if point.semantic_class is SemanticBCClass.INTERIOR
        ),
        dtype=torch.float64,
    )
    values: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        for points in (tensors.query_xy, interior):
            coordinates = points.detach().clone().unsqueeze(0).requires_grad_(True)
            with (
                torch.enable_grad(),
                torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
            ):
                output = model(
                    boundary_xy=tensors.boundary_xy.unsqueeze(0),
                    boundary_tags=tensors.boundary_tags.unsqueeze(0),
                    boundary_normals=tensors.boundary_normals.unsqueeze(0),
                    boundary_weights=tensors.boundary_weights.unsqueeze(0),
                    boundary_mask=tensors.boundary_mask.unsqueeze(0),
                    query_xy=coordinates,
                    parameters=tensors.parameters_for(
                        model.config.parameter_names
                    ).unsqueeze(0),
                )
                gradient = displacement_gradient(
                    output.u * _scales(case.request).displacement_mm,
                    coordinates,
                    _COORDINATE_CONVERSION,
                )
                determinant = torch.linalg.det(
                    torch.eye(2, dtype=gradient.dtype) + gradient
                )
            values.extend((output.u.detach(), gradient.detach(), determinant.detach()))
    finally:
        model.train(was_training)
    return tuple(values)


def assert_fixed_displacement(
    before: tuple[torch.Tensor, ...], after: tuple[torch.Tensor, ...]
) -> None:
    if (
        len(before) != 6
        or len(after) != 6
        or any(
            not torch.equal(old, new) for old, new in zip(before, after, strict=True)
        )
    ):
        raise ValueError("frozen displacement, Jacobian or determinant changed")


@dataclass(frozen=True, slots=True)
class CapacityDecision:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"
    success: bool
    root_improvement_fraction: float
    nonroot_regression_fraction: float
    failed_criteria: tuple[str, ...]


def classify_capacity_contrast(
    plan: StressCapacityPlan,
    control: SpatialCaseMetrics,
    candidate: SpatialCaseMetrics,
    *,
    candidate_primary_success: bool,
) -> CapacityDecision:
    if control.root_p_relative_l2 <= 0 or control.nonroot_p_relative_l2 <= 0:
        raise ValueError("control slice errors need positive denominators")
    root = 1.0 - candidate.root_p_relative_l2 / control.root_p_relative_l2
    nonroot = candidate.nonroot_p_relative_l2 / control.nonroot_p_relative_l2 - 1.0
    failed = []
    if not candidate_primary_success:
        failed.append("original_one_case_criteria")
    if (
        candidate.root_p_relative_l2
        > (1 - plan.minimum_root_improvement) * control.root_p_relative_l2
    ):
        failed.append("root_tensor_improvement")
    if (
        candidate.nonroot_p_relative_l2
        > (1 + plan.maximum_nonroot_regression) * control.nonroot_p_relative_l2
    ):
        failed.append("nonroot_p_nonregression")
    return CapacityDecision(not failed, root, nonroot, tuple(failed))


@dataclass(frozen=True, slots=True)
class StressCapacityArm:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"
    arm_id: str
    config: OperatorConfig
    operator_module_digest: str
    initial_state_hash: str
    final_state_hash: str
    frozen_state_hash: str
    copied_state_names: tuple[str, ...]
    new_state_names: tuple[str, ...]
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_count: int
    after: OverfitCaseMetrics
    spatial: SpatialCaseMetrics
    classification: OverfitClassification
    additional_data_mse_drop_fraction: float
    optimizer_loss_trace: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class StressCapacityReport:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"
    role: EvidenceRole
    non_gate: bool
    plan: StressCapacityPlan
    plan_hash: str
    source_bindings_hash: str
    s11_evidence_file_hash: str
    s11_checkpoint_file_hash: str
    warmstart_state_hash: str
    warmstart_frozen_state_hash: str
    fixed_displacement_verified: bool
    sampler_hash: str
    implementation_sha256: tuple[tuple[str, str], ...]
    original_before: OverfitCaseMetrics
    warmstart_before: OverfitCaseMetrics
    arms: tuple[StressCapacityArm, ...]
    decision: CapacityDecision


def _implementation_hashes() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "campaigns/dogbone_v1/s13_stress_capacity.py",
        "campaigns/dogbone_v1/s13_overfit_diagnostic.py",
        "campaigns/dogbone_v1/s13_spatial_audit.py",
        "campaigns/dogbone_v1/s13_objective.py",
        "src/piginot/operators/architecture.py",
        "src/piginot/objectives/kinematics.py",
    )
    return tuple(
        (name, hashlib.sha256((root / name).read_bytes()).hexdigest()) for name in names
    )


def run_stress_capacity_contrast(
    *,
    s10_run_root: Path,
    s11_checkpoint_root: Path,
    development_access_record: Path,
    v5_report_path: Path,
    output_path: Path,
) -> StressCapacityReport:
    plan = StressCapacityPlan()
    implementation = _implementation_hashes()
    s10, s11, access, _ = _preflight_paths(
        s10_run_root, s11_checkpoint_root, development_access_record, output_path
    )
    prior = _load_pinned_report(v5_report_path, plan.warmstart_report_sha256)
    initialization = load_selected_s11_initialization(s11)
    source = open_selective_s10_source(s10, development_access_record=access)
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT access forbidden in stress-capacity contrast")
    case = materialize_dogbone_s13_training_case(
        source, initialization, plan.fine_tune_plan.case_id
    )
    if case.tensors.role is not EvidenceRole.TRAIN or source.development_opened:
        raise PermissionError("stress-capacity contrast is TRAIN only")
    if len(case.tensors.query_xy) != 969 or len(case.tensors.qoi_indices) != 20:
        raise ValueError("stress-capacity population differs from the frozen case")
    sampler = _build_vertex_sampler_plan(plan.fine_tune_plan, 969)
    bindings = {
        "plan_hash": plan.fine_tune_plan.content_hash,
        "vertex_sampler_plan_hash": sampler.plan_hash,
        "source_bindings_hash": source.bindings.content_hash,
        "s11_evidence_file_hash": initialization.evidence_file_blake2b256,
        "initial_checkpoint_file_hash": initialization.checkpoint_file_blake2b256,
        "initial_model_state_hash": model_state_hash(initialization.model),
    }
    if any(prior.get(name) != value for name, value in bindings.items()):
        raise ValueError("V5 warmstart input binding mismatch")
    original_before = _measure_case(initialization.model, case)
    print("Replaying pinned V5 warmstart: 4,000 steps", flush=True)
    replay = _train_one_case(initialization.model, case, plan.fine_tune_plan, sampler)
    if list(replay) != prior.get("optimizer_loss_trace") or model_state_hash(
        initialization.model
    ) != prior.get("final_model_state_hash"):
        raise ValueError("V5 warmstart replay differs from frozen evidence")
    warmstart = initialization.model
    warm_before = _measure_case(warmstart, case)
    fixed_signature = displacement_signature(warmstart, case)
    warm_state = model_state_hash(warmstart)
    warm_fixed_hash = frozen_state_hash(warmstart)
    arms: list[StressCapacityArm] = []
    for enriched in (False, True):
        if source.development_opened:
            raise RuntimeError("DEVELOPMENT access forbidden before stress training")
        model = build_stress_arm(warmstart, plan, enriched=enriched)
        fixed_hash = frozen_state_hash(model)
        if fixed_hash != warm_fixed_hash:
            raise ValueError("migration changed the frozen shared state")
        initial_state = model_state_hash(model)
        assert_fixed_displacement(fixed_signature, displacement_signature(model, case))
        # A neutral migration must preserve all initial P values, not just the peak.
        expected_p, _, _ = _vertex_prediction_and_physics(warmstart, case)
        actual_p, _, _ = _vertex_prediction_and_physics(model, case)
        if not torch.equal(expected_p, actual_p):
            raise ValueError("neutral stress migration changed initial predictions")
        assert_trainable_scope(model)
        print(
            f"Training matched {'enriched' if enriched else 'control'} arm: 4,000 steps",
            flush=True,
        )
        trace = _train_one_case(model, case, plan.fine_tune_plan, sampler)
        assert_trainable_scope(model)
        if frozen_state_hash(model) != fixed_hash:
            raise ValueError("frozen shared state changed during stress training")
        assert_fixed_displacement(fixed_signature, displacement_signature(model, case))
        after = _measure_case(model, case)
        predicted, residual, determinants = _vertex_prediction_and_physics(model, case)
        spatial = summarize_spatial_case(
            predicted,
            case.tensors.target_P,
            case.tensors.qoi_indices,
            case.tensors.query_xy,
            residual,
            determinants,
        )
        print(
            f"{'Enriched' if enriched else 'Control'} complete: "
            f"P={after.p_relative_l2:.4%}, root P={spatial.root_p_relative_l2:.4%}, "
            f"peak={after.absolute_peak_error:.4%}; displacement unchanged",
            flush=True,
        )
        arms.append(
            StressCapacityArm(
                "enriched" if enriched else "control",
                model.config,
                module_digest(model),
                initial_state,
                model_state_hash(model),
                fixed_hash,
                tuple(sorted(warmstart.state_dict())),
                tuple(sorted(set(model.state_dict()) - set(warmstart.state_dict()))),
                trainable_names(model),
                sum(p.numel() for p in model.parameters() if p.requires_grad),
                after,
                spatial,
                classify_s13_overfit(plan.fine_tune_plan, original_before, after),
                1.0 - after.normalized_data_mse / warm_before.normalized_data_mse,
                trace,
            )
        )
    if source.development_opened or model_state_hash(warmstart) != warm_state:
        raise RuntimeError("TRAIN scope or immutable warmstart changed during contrast")
    if _implementation_hashes() != implementation:
        raise RuntimeError("diagnostic source files changed during the contrast")
    return StressCapacityReport(
        EvidenceRole.TRAIN,
        True,
        plan,
        plan.content_hash,
        source.bindings.content_hash,
        initialization.evidence_file_blake2b256,
        initialization.checkpoint_file_blake2b256,
        warm_state,
        warm_fixed_hash,
        True,
        sampler.plan_hash,
        implementation,
        original_before,
        warm_before,
        tuple(arms),
        classify_capacity_contrast(
            plan,
            arms[0].spatial,
            arms[1].spatial,
            candidate_primary_success=arms[1].classification.success,
        ),
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--s11-checkpoint-root", required=True, type=Path)
    parser.add_argument("--development-access-record", required=True, type=Path)
    parser.add_argument("--v5-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args(arguments)
    report = run_stress_capacity_contrast(
        s10_run_root=options.s10_run_root,
        s11_checkpoint_root=options.s11_checkpoint_root,
        development_access_record=options.development_access_record,
        v5_report_path=options.v5_report,
        output_path=options.output,
    )
    _exclusive_write(options.output, canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
