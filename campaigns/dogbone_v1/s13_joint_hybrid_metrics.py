"""Read-only snapshots and descriptive comparisons of the joint S13 contrast."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import ClassVar

import torch

from campaigns.dogbone_v1.s11_contingency_models import model_state_hash
from campaigns.dogbone_v1.s13_joint_hybrid_protocol import JointHybridPlan
from campaigns.dogbone_v1.s13_objective import (
    DogboneS13TrainingCase,
    _physics_material,
    _scales,
    _validated_operator,
)
from campaigns.dogbone_v1.s13_overfit_diagnostic import (
    OverfitCaseMetrics,
    OverfitClassification,
    _measure_case,
    build_s13_overfit_plan,
    classify_s13_overfit,
)
from campaigns.dogbone_v1.s13_quadrature_audit import (
    FemQuadratureFields,
    predict_quadrature,
)
from campaigns.dogbone_v1.s13_quadrature_metrics import (
    ConstitutiveDecomposition,
    QuadratureCaseMetrics,
    QuadraturePopulationMetrics,
    summarize_quadrature_case,
)
from campaigns.dogbone_v1.s13_spatial_audit import (
    SpatialCaseMetrics,
    _vertex_prediction_and_physics,
    summarize_spatial_case,
)
from piginot.doe import EvidenceRole
from piginot.operators import PiGinotOperator

_PLAN = JointHybridPlan()


@dataclass(frozen=True, slots=True)
class JointHybridSnapshot:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    arm_id: str
    step: int
    state_hash: str
    quadrature_reference_hash: str
    vertex_metrics: OverfitCaseMetrics
    vertex_classification: OverfitClassification
    additional_data_mse_drop_from_v5: float | None
    spatial_metrics: SpatialCaseMetrics | None
    spatial_unavailable_reason: str | None
    quadrature_metrics: QuadratureCaseMetrics


def _reference_hash(reference: FemQuadratureFields) -> str:
    digest = hashlib.sha256()
    for item in fields(reference):
        value = getattr(reference, item.name)
        digest.update(item.name.encode("ascii"))
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode("ascii"))
            digest.update(value.detach().contiguous().numpy().tobytes())
        else:
            digest.update(float(value).hex().encode("ascii"))
    return digest.hexdigest()


def _snapshot_inputs(
    case: DogboneS13TrainingCase,
    reference: FemQuadratureFields,
    arm_id: str,
    step: int,
    original_s11_metrics: OverfitCaseMetrics,
    pinned_v5_metrics: OverfitCaseMetrics,
    plan: JointHybridPlan,
) -> None:
    if type(plan) is not JointHybridPlan:
        raise TypeError("plan must be the exact JointHybridPlan")
    if (
        type(case) is not DogboneS13TrainingCase
        or case.tensors.role is not EvidenceRole.TRAIN
        or case.tensors.case_id != plan.case_id
    ):
        raise PermissionError("snapshot requires the registered exact TRAIN case")
    if type(arm_id) is not str or arm_id not in ("candidate", "control", "v5"):
        raise ValueError("snapshot arm_id must identify candidate, control or v5")
    if type(step) is not int or step not in plan.evaluation_steps:
        raise ValueError("snapshot step must be a declared evaluation step")
    if type(reference) is not FemQuadratureFields or tuple(reference.xy_mm.shape) != (
        *plan.quadrature_layout,
        2,
    ):
        raise ValueError("snapshot must retain the complete fixed quadrature layout")
    if (
        type(original_s11_metrics) is not OverfitCaseMetrics
        or type(pinned_v5_metrics) is not OverfitCaseMetrics
    ):
        raise TypeError("original S11 and pinned v5 metrics must be exact metrics")
    if original_s11_metrics.normalized_data_mse <= 0:
        raise ValueError("original S11 data MSE denominator must be positive")


def evaluate_joint_snapshot(
    model: PiGinotOperator,
    case: DogboneS13TrainingCase,
    reference: FemQuadratureFields,
    *,
    arm_id: str,
    step: int,
    original_s11_metrics: OverfitCaseMetrics,
    pinned_v5_metrics: OverfitCaseMetrics,
    plan: JointHybridPlan = _PLAN,
) -> JointHybridSnapshot:
    """Measure supplied TRAIN fields without training, loading or writing evidence.

    The historical recovered-P classification retains its original S11 denominator.
    Every arm predicts its own current U, grad U and served P; no frozen-kinematics
    assumption is made. The unchanged spatial audit is unavailable on invalid
    vertices, but the unconditional vertex and quadrature measurements remain.
    """
    _snapshot_inputs(
        case, reference, arm_id, step, original_s11_metrics, pinned_v5_metrics, plan
    )
    _validated_operator(model)
    state_hash = model_state_hash(model)
    modes = tuple((module, module.training) for module in model.modules())
    try:
        with torch.random.fork_rng(devices=[]):
            vertex = _measure_case(model, case)
            spatial, spatial_reason = None, None
            if vertex.invalid_vertex_count:
                spatial_reason = "inadmissible_vertex_deformation"
            else:
                predicted_p, constitutive, determinants = (
                    _vertex_prediction_and_physics(model, case)
                )
                spatial = summarize_spatial_case(
                    predicted_p,
                    case.tensors.target_P,
                    case.tensors.qoi_indices,
                    case.tensors.query_xy,
                    constitutive,
                    determinants,
                )
            predicted = predict_quadrature(
                model, case, reference.xy_mm, chunk_size=plan.query_chunk_size
            )
            quadrature = summarize_quadrature_case(
                grad_model=predicted.gradient,
                grad_fem=reference.gradient,
                served_p=predicted.p_mpa,
                raw_fem_p=reference.raw_p_mpa,
                area_weights=reference.area_weights,
                root_cell_mask=reference.root_cell_mask,
                xy_mm=reference.xy_mm,
                material=_physics_material(case.request.material),
                scales=_scales(case.request),
                u_model=predicted.u_mm,
                u_fem=reference.u_mm,
            )
    finally:
        # Restore individual flags, including intentionally mixed submodule modes.
        for module, training in modes:
            module.training = training
        if model_state_hash(model) != state_hash:
            raise RuntimeError("model state changed during read-only evaluation")
    v5_mse = pinned_v5_metrics.normalized_data_mse
    return JointHybridSnapshot(
        arm_id,
        step,
        state_hash,
        _reference_hash(reference),
        vertex,
        classify_s13_overfit(
            build_s13_overfit_plan(version="v5"), original_s11_metrics, vertex
        ),
        (v5_mse - vertex.normalized_data_mse) / v5_mse if v5_mse else None,
        spatial,
        spatial_reason,
        quadrature,
    )


@dataclass(frozen=True, slots=True)
class MetricContrast:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    metric: str
    candidate: float | None
    reference: float | None
    signed_change: float | None
    ratio: float | None
    ratio_unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class PopulationContrast:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    admissible_point_support_identical: bool
    unconditional: tuple[MetricContrast, ...]
    conditional: tuple[MetricContrast, ...]


@dataclass(frozen=True, slots=True)
class QuadratureContrast:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    all_points: PopulationContrast
    root_incident: PopulationContrast
    nonroot: PopulationContrast


@dataclass(frozen=True, slots=True)
class JointHybridEndpointComparison:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    comparison_step: int
    candidate_state_hash: str
    control_state_hash: str
    pinned_v5_state_hash: str
    candidate_vs_control: QuadratureContrast
    candidate_vs_pinned_v5: QuadratureContrast
    vertex_candidate_vs_control: tuple[MetricContrast, ...]
    vertex_candidate_vs_pinned_v5: tuple[MetricContrast, ...]


def _metric_contrast(
    name: str,
    candidate: float | None,
    reference: float | None,
    *,
    same_support: bool = True,
) -> MetricContrast:
    if candidate is None or reference is None:
        return MetricContrast(name, candidate, reference, None, None, "unavailable")
    reason = (
        "different_admissible_support"
        if not same_support
        else "zero_reference"
        if reference == 0
        else None
    )
    return MetricContrast(
        name,
        candidate,
        reference,
        candidate - reference,
        candidate / reference if reason is None else None,
        reason,
    )


def _numeric_fields(
    value: QuadraturePopulationMetrics | ConstitutiveDecomposition | OverfitCaseMetrics,
) -> tuple[tuple[str, float | None], ...]:
    return tuple(
        (item.name, None if entry is None else float(entry))
        for item in fields(value)
        if isinstance(entry := getattr(value, item.name), (int, float))
        or (entry is None and item.name in ("u_relative_l2", "u_component_rms_mm"))
    )


def _population_contrast(
    candidate: QuadraturePopulationMetrics, reference: QuadraturePopulationMetrics
) -> PopulationContrast:
    candidate_ids = tuple(
        (point.cell_index, point.quadrature_index) for point in candidate.invalid_points
    )
    reference_ids = tuple(
        (point.cell_index, point.quadrature_index) for point in reference.invalid_points
    )
    same_support = candidate_ids == reference_ids
    unconditional = tuple(
        _metric_contrast(name, left, right)
        for (name, left), (_, right) in zip(
            _numeric_fields(candidate), _numeric_fields(reference), strict=True
        )
    ) + tuple(
        _metric_contrast(
            f"gradient_bias.{row}.{column}",
            candidate.gradient_bias[row][column],
            reference.gradient_bias[row][column],
        )
        for row in range(2)
        for column in range(2)
    )
    left, right = candidate.decomposition, reference.decomposition
    conditional = tuple(
        _metric_contrast(
            item.name,
            None if left is None else float(getattr(left, item.name)),
            None if right is None else float(getattr(right, item.name)),
            same_support=same_support,
        )
        for item in fields(ConstitutiveDecomposition)
    )
    return PopulationContrast(same_support, unconditional, conditional)


def _quadrature_contrast(
    candidate: QuadratureCaseMetrics, reference: QuadratureCaseMetrics
) -> QuadratureContrast:
    return QuadratureContrast(
        *(
            _population_contrast(getattr(candidate, name), getattr(reference, name))
            for name in ("all_points", "root_incident", "nonroot")
        )
    )


def _vertex_contrast(
    candidate: OverfitCaseMetrics, reference: OverfitCaseMetrics
) -> tuple[MetricContrast, ...]:
    return tuple(
        _metric_contrast(name, left, right)
        for (name, left), (_, right) in zip(
            _numeric_fields(candidate), _numeric_fields(reference), strict=True
        )
    )


def summarize_joint_endpoint(
    candidate: JointHybridSnapshot,
    control: JointHybridSnapshot,
    pinned_v5: JointHybridSnapshot,
    *,
    plan: JointHybridPlan = _PLAN,
) -> JointHybridEndpointComparison:
    """Compare only the declared endpoint; conditional ratios need identical IDs.

    Signed changes are candidate minus reference, and ratios are candidate over
    reference. A support change is recorded explicitly, with its conditional
    ratios unavailable. No accuracy-repair criterion or pass label is added.
    """
    if type(plan) is not JointHybridPlan or any(
        type(snapshot) is not JointHybridSnapshot
        for snapshot in (candidate, control, pinned_v5)
    ):
        raise TypeError("endpoint requires exact joint snapshots and plan")
    if (candidate.arm_id, control.arm_id, pinned_v5.arm_id) != (
        "candidate",
        "control",
        "v5",
    ) or (candidate.step, control.step, pinned_v5.step) != (
        plan.comparison_endpoint,
        plan.comparison_endpoint,
        0,
    ):
        raise ValueError("endpoint arm IDs or fixed comparison steps differ")
    if (
        len(
            {
                snapshot.quadrature_reference_hash
                for snapshot in (candidate, control, pinned_v5)
            }
        )
        != 1
    ):
        raise ValueError("paired snapshots have different FEM quadrature references")
    return JointHybridEndpointComparison(
        plan.comparison_endpoint,
        candidate.state_hash,
        control.state_hash,
        pinned_v5.state_hash,
        _quadrature_contrast(candidate.quadrature_metrics, control.quadrature_metrics),
        _quadrature_contrast(
            candidate.quadrature_metrics, pinned_v5.quadrature_metrics
        ),
        _vertex_contrast(candidate.vertex_metrics, control.vertex_metrics),
        _vertex_contrast(candidate.vertex_metrics, pinned_v5.vertex_metrics),
    )


__all__ = (
    "JointHybridEndpointComparison",
    "JointHybridSnapshot",
    "MetricContrast",
    "PopulationContrast",
    "QuadratureContrast",
    "evaluate_joint_snapshot",
    "summarize_joint_endpoint",
)
