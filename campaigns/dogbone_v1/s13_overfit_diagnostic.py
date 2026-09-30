"""Non-gate, TRAIN-only one-case supervised S13 overfit diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypedDict, cast

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
from campaigns.dogbone_v1.s13_screen import _determinants_for_case
from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.datasets import SemanticBCClass
from piginot.doe import EvidenceRole
from piginot.objectives import displacement_gradient
from piginot.operators import OperatorOutput, PiGinotOperator
from piginot.physics._validation import DETERMINANT_MARGIN
from piginot.training import DeterministicSamplerPlan

_CASE_ID = "candidate-002485"
_MAXIMUM_OPTIMIZER_STEPS = 800
_EXPECTED_VERTEX_COUNT = 969
_VERTEX_BATCH_SIZE = 57
_VERTEX_SAMPLER_POLICY_ID = "dogbone-s13-overfit-blake2b-epoch-permutation-v1"
_VERTEX_SAMPLER_PERSON = b"piginot-vtx-v1"


class _PlanArguments(TypedDict):
    diagnostic_id: str
    role: EvidenceRole
    non_gate: bool
    case_id: str
    seed: int
    learning_rate: float
    optimizer_steps: int
    maximum_optimizer_steps: int
    expected_vertex_count: int
    vertex_batch_size: int
    vertex_sampler_policy_id: str
    minimum_data_mse_drop_fraction: float
    maximum_final_u_relative_l2: float
    maximum_final_p_relative_l2: float
    maximum_final_absolute_peak_error: float
    maximum_final_invalid_collocation_count: int


_FROZEN_DIAGNOSTICS = {
    "dogbone-s13-one-case-overfit-v3": (5.0e-6, _MAXIMUM_OPTIMIZER_STEPS, 0.0),
    "dogbone-s13-one-case-overfit-v4": (2.0e-5, _MAXIMUM_OPTIMIZER_STEPS, 0.0),
    "dogbone-s13-one-case-overfit-v5": (2.0e-5, 4_000, 0.0),
    "dogbone-s13-one-case-overfit-v6": (2.0e-5, 4_000, 0.1),
    "dogbone-s13-one-case-overfit-v7": (2.0e-5, 4_000, 0.0),
}


@dataclass(frozen=True, slots=True)
class S13OverfitPlan:
    """Predeclared diagnostic budget and conjunctive success criteria."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    diagnostic_id: str
    role: EvidenceRole
    non_gate: bool
    case_id: str
    seed: int
    learning_rate: float
    optimizer_steps: int
    maximum_optimizer_steps: int
    expected_vertex_count: int
    vertex_batch_size: int
    vertex_sampler_policy_id: str
    minimum_data_mse_drop_fraction: float
    maximum_final_u_relative_l2: float
    maximum_final_p_relative_l2: float
    maximum_final_absolute_peak_error: float
    maximum_final_invalid_collocation_count: int

    def __post_init__(self) -> None:
        if self.diagnostic_id not in _FROZEN_DIAGNOSTICS:
            raise ValueError("diagnostic_id must identify the frozen diagnostic")
        if self.diagnostic_id.endswith("-v7") != (type(self) is S13OverfitPlanV7):
            raise ValueError("v7 requires its exact weighted plan type")
        if self.role is not EvidenceRole.TRAIN:
            raise PermissionError("overfit diagnostic is restricted to TRAIN evidence")
        if self.non_gate is not True:
            raise ValueError("overfit diagnostic must remain non-gate")
        if self.case_id != _CASE_ID:
            raise ValueError(f"case_id must be {_CASE_ID}")
        if type(self.seed) is not int or self.seed != 13_001:
            raise ValueError("seed must remain 13001")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        learning_rate, maximum_steps, _ = _FROZEN_DIAGNOSTICS[self.diagnostic_id]
        if self.learning_rate != learning_rate:
            raise ValueError("learning_rate differs from the frozen diagnostic")
        if self.maximum_optimizer_steps != maximum_steps:
            raise ValueError(
                "maximum_optimizer_steps differs from the frozen diagnostic"
            )
        if (
            type(self.optimizer_steps) is not int
            or self.optimizer_steps <= 0
            or self.optimizer_steps > self.maximum_optimizer_steps
        ):
            raise ValueError("optimizer_steps must be within the maximum budget")
        if self.expected_vertex_count != _EXPECTED_VERTEX_COUNT:
            raise ValueError(
                f"expected_vertex_count must remain {_EXPECTED_VERTEX_COUNT}"
            )
        if self.vertex_batch_size != _VERTEX_BATCH_SIZE:
            raise ValueError(f"vertex_batch_size must remain {_VERTEX_BATCH_SIZE}")
        if self.expected_vertex_count % self.vertex_batch_size != 0:
            raise ValueError("vertex_batch_size must divide expected_vertex_count")
        if self.vertex_sampler_policy_id != _VERTEX_SAMPLER_POLICY_ID:
            raise ValueError("vertex_sampler_policy_id differs from the frozen policy")
        for name in (
            "minimum_data_mse_drop_fraction",
            "maximum_final_u_relative_l2",
            "maximum_final_p_relative_l2",
            "maximum_final_absolute_peak_error",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
        if not 0.0 <= self.minimum_data_mse_drop_fraction <= 1.0:
            raise ValueError("minimum_data_mse_drop_fraction must lie in [0, 1]")
        if (
            self.minimum_data_mse_drop_fraction != 0.80
            or self.maximum_final_u_relative_l2 != 0.02
            or self.maximum_final_p_relative_l2 != 0.05
            or self.maximum_final_absolute_peak_error != 0.05
        ):
            raise ValueError("overfit success thresholds differ from the frozen policy")
        if type(self.maximum_final_invalid_collocation_count) is not int or (
            self.maximum_final_invalid_collocation_count != 0
        ):
            raise ValueError("maximum_final_invalid_collocation_count must remain zero")

    @property
    def content_hash(self) -> str:
        return hash_spec(self, "content")

    @property
    def peak_loss_weight(self) -> float:
        return _FROZEN_DIAGNOSTICS[self.diagnostic_id][2]

    @property
    def field_p_loss_weight(self) -> float:
        return 1.0


@dataclass(frozen=True, slots=True)
class S13OverfitPlanV7(S13OverfitPlan):
    """Versioned plan that canonically binds the changed P-field weight."""

    p_field_weight: float

    def __post_init__(self) -> None:
        S13OverfitPlan.__post_init__(self)
        if self.p_field_weight != 2.0:
            raise ValueError("p_field_weight must remain 2.0 in v7")

    @property
    def field_p_loss_weight(self) -> float:
        return self.p_field_weight


def _is_exact_overfit_plan(value: object) -> bool:
    return type(value) in (S13OverfitPlan, S13OverfitPlanV7)


@dataclass(frozen=True, slots=True)
class OverfitCaseMetrics:
    """Full-vertex supervised and orientation measurements for one case."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    normalized_data_mse: float
    normalized_u_mse: float
    normalized_p_mse: float
    u_relative_l2: float
    p_relative_l2: float
    absolute_peak_error: float
    signed_peak_error: float
    min_vertex_determinant: float
    invalid_vertex_count: int
    min_collocation_determinant: float
    invalid_collocation_count: int

    def __post_init__(self) -> None:
        for name in (
            "normalized_data_mse",
            "normalized_u_mse",
            "normalized_p_mse",
            "u_relative_l2",
            "p_relative_l2",
            "absolute_peak_error",
            "signed_peak_error",
            "min_vertex_determinant",
            "min_collocation_determinant",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in (
            "normalized_data_mse",
            "normalized_u_mse",
            "normalized_p_mse",
            "u_relative_l2",
            "p_relative_l2",
            "absolute_peak_error",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not math.isclose(
            self.normalized_data_mse,
            self.normalized_u_mse + self.normalized_p_mse,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError("normalized_data_mse must equal u_mse + p_mse")
        if not math.isclose(
            self.absolute_peak_error,
            abs(self.signed_peak_error),
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError("absolute_peak_error must match signed_peak_error")
        for name in ("invalid_vertex_count", "invalid_collocation_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class OverfitClassification:
    """Result of the frozen, conjunctive diagnostic decision rule."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    data_mse_drop_fraction: float
    success: bool
    failed_criteria: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class S13OverfitDiagnosticReport:
    """Canonical evidence for the TRAIN-only one-case fit test."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    role: EvidenceRole
    non_gate: bool
    plan: S13OverfitPlan
    plan_hash: str
    case_id: str
    source_bindings_hash: str
    s11_evidence_file_hash: str
    initial_checkpoint_file_hash: str
    initial_model_state_hash: str
    final_model_state_hash: str
    vertex_count: int
    vertex_sampler_plan_hash: str
    before: OverfitCaseMetrics
    after: OverfitCaseMetrics
    classification: OverfitClassification
    optimizer_loss_trace: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.role is not EvidenceRole.TRAIN or self.non_gate is not True:
            raise PermissionError("report must remain TRAIN-only and non-gate")
        if not _is_exact_overfit_plan(self.plan):
            raise TypeError("plan must be an exact frozen S13OverfitPlan")
        if self.plan_hash != self.plan.content_hash:
            raise ValueError("plan_hash differs from the frozen plan")
        if self.case_id != self.plan.case_id:
            raise ValueError("report case differs from the plan")
        for name in (
            "source_bindings_hash",
            "s11_evidence_file_hash",
            "initial_checkpoint_file_hash",
            "initial_model_state_hash",
            "final_model_state_hash",
            "vertex_sampler_plan_hash",
        ):
            _digest(getattr(self, name), name)
        if self.vertex_count != self.plan.expected_vertex_count:
            raise ValueError("vertex_count differs from the frozen plan")
        if (
            self.vertex_sampler_plan_hash
            != _build_vertex_sampler_plan(self.plan, self.vertex_count).plan_hash
        ):
            raise ValueError("vertex sampler hash differs from the frozen schedule")
        if (
            type(self.before) is not OverfitCaseMetrics
            or type(self.after) is not OverfitCaseMetrics
        ):
            raise TypeError("before and after must be exact OverfitCaseMetrics")
        if type(self.classification) is not OverfitClassification:
            raise TypeError("classification must be exact OverfitClassification")
        if self.classification != classify_s13_overfit(
            self.plan, self.before, self.after
        ):
            raise ValueError("classification differs from the frozen decision rule")
        if (
            type(self.optimizer_loss_trace) is not tuple
            or len(self.optimizer_loss_trace) != self.plan.optimizer_steps
            or any(
                not math.isfinite(value) or value < 0.0
                for value in self.optimizer_loss_trace
            )
        ):
            raise ValueError("optimizer_loss_trace must cover the finite step budget")


def build_s13_overfit_plan(
    *,
    optimizer_steps: int | None = None,
    version: str = "v3",
) -> S13OverfitPlan:
    """Build the frozen default plan, allowing only a strict diagnostic prefix."""

    if type(version) is not str or version not in ("v3", "v4", "v5", "v6", "v7"):
        raise ValueError("version must identify v3, v4, v5, v6 or v7")
    diagnostic_id = f"dogbone-s13-one-case-overfit-{version}"
    learning_rate, maximum_steps, _ = _FROZEN_DIAGNOSTICS[diagnostic_id]
    plan_arguments: _PlanArguments = {
        "diagnostic_id": diagnostic_id,
        "role": EvidenceRole.TRAIN,
        "non_gate": True,
        "case_id": _CASE_ID,
        "seed": 13_001,
        "learning_rate": learning_rate,
        "optimizer_steps": maximum_steps
        if optimizer_steps is None
        else optimizer_steps,
        "maximum_optimizer_steps": maximum_steps,
        "expected_vertex_count": _EXPECTED_VERTEX_COUNT,
        "vertex_batch_size": _VERTEX_BATCH_SIZE,
        "vertex_sampler_policy_id": _VERTEX_SAMPLER_POLICY_ID,
        "minimum_data_mse_drop_fraction": 0.80,
        "maximum_final_u_relative_l2": 0.02,
        "maximum_final_p_relative_l2": 0.05,
        "maximum_final_absolute_peak_error": 0.05,
        "maximum_final_invalid_collocation_count": 0,
    }
    if version == "v7":
        return S13OverfitPlanV7(**plan_arguments, p_field_weight=2.0)
    return S13OverfitPlan(**plan_arguments)


def classify_s13_overfit(
    plan: S13OverfitPlan,
    before: OverfitCaseMetrics,
    after: OverfitCaseMetrics,
) -> OverfitClassification:
    """Apply every preregistered criterion without early exit or softening."""

    if not _is_exact_overfit_plan(plan):
        raise TypeError("plan must be an exact frozen S13OverfitPlan")
    if type(before) is not OverfitCaseMetrics or type(after) is not OverfitCaseMetrics:
        raise TypeError("before and after must be exact OverfitCaseMetrics")
    if before.normalized_data_mse <= 0.0:
        raise ValueError("initial normalized_data_mse must be positive")
    drop = (before.normalized_data_mse - after.normalized_data_mse) / (
        before.normalized_data_mse
    )
    failed: list[str] = []
    if drop < plan.minimum_data_mse_drop_fraction:
        failed.append("combined_data_mse_drop")
    if after.u_relative_l2 > plan.maximum_final_u_relative_l2:
        failed.append("final_u_relative_l2")
    if after.p_relative_l2 > plan.maximum_final_p_relative_l2:
        failed.append("final_p_relative_l2")
    if after.absolute_peak_error > plan.maximum_final_absolute_peak_error:
        failed.append("final_absolute_peak_error")
    if after.invalid_collocation_count > plan.maximum_final_invalid_collocation_count:
        failed.append("final_invalid_collocation_count")
    return OverfitClassification(drop, not failed, tuple(failed))


def _predict_case(
    model: PiGinotOperator,
    case: DogboneS13TrainingCase,
) -> OperatorOutput:
    operator = _validated_operator(model)
    tensors = case.tensors
    with torch.no_grad():
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            return cast(
                OperatorOutput,
                operator(
                    boundary_xy=tensors.boundary_xy.unsqueeze(0),
                    boundary_tags=tensors.boundary_tags.unsqueeze(0),
                    boundary_normals=tensors.boundary_normals.unsqueeze(0),
                    boundary_weights=tensors.boundary_weights.unsqueeze(0),
                    boundary_mask=tensors.boundary_mask.unsqueeze(0),
                    query_xy=tensors.query_xy.unsqueeze(0),
                    parameters=tensors.parameters_for(
                        operator.config.parameter_names
                    ).unsqueeze(0),
                ),
            )


def _predict_case_vertices(
    model: PiGinotOperator,
    case: DogboneS13TrainingCase,
    vertex_indices: torch.Tensor,
) -> OperatorOutput:
    operator = _validated_operator(model)
    tensors = case.tensors
    if vertex_indices.dtype is not torch.long or vertex_indices.device.type != "cpu":
        raise ValueError("vertex_indices must be a CPU int64 tensor")
    if vertex_indices.ndim != 1 or vertex_indices.numel() == 0:
        raise ValueError("vertex_indices must be a non-empty vector")
    if int(vertex_indices.min()) < 0 or int(vertex_indices.max()) >= int(
        tensors.query_xy.shape[0]
    ):
        raise ValueError("vertex_indices must refer to query vertices")
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return cast(
            OperatorOutput,
            operator(
                boundary_xy=tensors.boundary_xy.unsqueeze(0),
                boundary_tags=tensors.boundary_tags.unsqueeze(0),
                boundary_normals=tensors.boundary_normals.unsqueeze(0),
                boundary_weights=tensors.boundary_weights.unsqueeze(0),
                boundary_mask=tensors.boundary_mask.unsqueeze(0),
                query_xy=tensors.query_xy.index_select(0, vertex_indices).unsqueeze(0),
                parameters=tensors.parameters_for(
                    operator.config.parameter_names
                ).unsqueeze(0),
            ),
        )


def _vertex_determinants(
    model: PiGinotOperator,
    case: DogboneS13TrainingCase,
) -> torch.Tensor:
    operator = _validated_operator(model)
    tensors = case.tensors
    coordinates = tensors.query_xy.detach().clone().unsqueeze(0)
    coordinates.requires_grad_(True)
    with torch.enable_grad():
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            output = operator(
                boundary_xy=tensors.boundary_xy.unsqueeze(0),
                boundary_tags=tensors.boundary_tags.unsqueeze(0),
                boundary_normals=tensors.boundary_normals.unsqueeze(0),
                boundary_weights=tensors.boundary_weights.unsqueeze(0),
                boundary_mask=tensors.boundary_mask.unsqueeze(0),
                query_xy=coordinates,
                parameters=tensors.parameters_for(
                    operator.config.parameter_names
                ).unsqueeze(0),
            )
            gradient = displacement_gradient(
                output.u * _scales(case.request).displacement_mm,
                coordinates,
                _COORDINATE_CONVERSION,
            )
    deformation = torch.eye(2, dtype=gradient.dtype, device=gradient.device) + gradient
    determinant = (
        deformation[..., 0, 0] * deformation[..., 1, 1]
        - deformation[..., 0, 1] * deformation[..., 1, 0]
    )
    return determinant.detach().reshape(-1)


def _collocation_determinants(
    model: PiGinotOperator,
    case: DogboneS13TrainingCase,
) -> torch.Tensor:
    """Reuse the S13 screen's physical-displacement determinant semantics."""

    return _determinants_for_case(model, case)


def _measure_case(
    model: PiGinotOperator,
    case: DogboneS13TrainingCase,
) -> OverfitCaseMetrics:
    operator = _validated_operator(model)
    if type(case) is not DogboneS13TrainingCase:
        raise TypeError("case must be an exact DogboneS13TrainingCase")
    if case.tensors.role is not EvidenceRole.TRAIN:
        raise PermissionError("only TRAIN tensors may enter the diagnostic")
    was_training = operator.training
    operator.eval()
    try:
        output = _predict_case(operator, case)
        determinant = _vertex_determinants(operator, case)
        collocation_determinant = _collocation_determinants(operator, case)
    finally:
        operator.train(was_training)
    tensors = case.tensors
    if output.u.shape != (1, *tensors.target_u.shape) or output.P.shape != (
        1,
        *tensors.target_P.shape,
    ):
        raise ValueError("full-vertex predictions do not align with FEM targets")
    if determinant.shape != (tensors.query_xy.shape[0],):
        raise ValueError("vertex determinants do not cover every query vertex")
    interior_count = sum(
        point.semantic_class is SemanticBCClass.INTERIOR
        for point in case.collocation.points
    )
    if collocation_determinant.shape != (interior_count,):
        raise ValueError("collocation determinants do not cover every interior point")
    for value in (output.u, output.P, determinant, collocation_determinant):
        if value.dtype is not torch.float64 or value.device.type != "cpu":
            raise ValueError("diagnostic values must be CPU float64")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("diagnostic values must be finite")
    u_delta = output.u[0] - tensors.target_u
    p_delta = output.P[0] - tensors.target_P
    u_target_norm = torch.linalg.vector_norm(tensors.target_u)
    p_target_norm = torch.linalg.vector_norm(tensors.target_P)
    if float(u_target_norm) <= 0.0 or float(p_target_norm) <= 0.0:
        raise ValueError("relative-error FEM targets must have positive norms")
    true_peak = torch.linalg.vector_norm(
        tensors.target_P[tensors.qoi_indices], dim=(-2, -1)
    ).amax()
    predicted_peak = torch.linalg.vector_norm(
        output.P[0, tensors.qoi_indices], dim=(-2, -1)
    ).amax()
    if float(true_peak) <= 0.0:
        raise ValueError("QoI FEM target must have a positive stress peak")
    u_mse = float(torch.mean(u_delta.square()))
    p_mse = float(torch.mean(p_delta.square()))
    signed_peak = float((predicted_peak - true_peak) / true_peak)
    return OverfitCaseMetrics(
        normalized_data_mse=u_mse + p_mse,
        normalized_u_mse=u_mse,
        normalized_p_mse=p_mse,
        u_relative_l2=float(torch.linalg.vector_norm(u_delta) / u_target_norm),
        p_relative_l2=float(torch.linalg.vector_norm(p_delta) / p_target_norm),
        absolute_peak_error=abs(signed_peak),
        signed_peak_error=signed_peak,
        min_vertex_determinant=float(determinant.min()),
        invalid_vertex_count=int((determinant <= DETERMINANT_MARGIN).sum()),
        min_collocation_determinant=float(collocation_determinant.min()),
        invalid_collocation_count=int(
            (collocation_determinant <= DETERMINANT_MARGIN).sum()
        ),
    )


def _train_one_case(
    model: PiGinotOperator,
    case: DogboneS13TrainingCase,
    plan: S13OverfitPlan,
    sampler_plan: DeterministicSamplerPlan | None = None,
) -> tuple[float, ...]:
    """Fit vertex fields and the optional root peak with deterministic CPU Adam."""

    operator = _validated_operator(model)
    if type(case) is not DogboneS13TrainingCase:
        raise TypeError("case must be an exact DogboneS13TrainingCase")
    if not _is_exact_overfit_plan(plan):
        raise TypeError("plan must be an exact frozen S13OverfitPlan")
    if (
        case.tensors.case_id != plan.case_id
        or case.tensors.role is not EvidenceRole.TRAIN
    ):
        raise PermissionError("training case must be the preregistered TRAIN case")
    if any(
        parameter.dtype is not torch.float64 or parameter.device.type != "cpu"
        for parameter in operator.parameters()
    ):
        raise ValueError("model parameters must be deterministic CPU float64")
    tensors = case.tensors
    vertex_count = int(tensors.query_xy.shape[0])
    expected_sampler = _build_vertex_sampler_plan(plan, vertex_count)
    sampler = expected_sampler if sampler_plan is None else sampler_plan
    if type(sampler) is not DeterministicSamplerPlan or sampler != expected_sampler:
        raise ValueError("sampler_plan differs from the deterministic vertex schedule")
    optimizer = torch.optim.Adam(
        operator.parameters(),
        lr=plan.learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    peak_target = (
        torch.linalg.vector_norm(
            tensors.target_P[tensors.qoi_indices], dim=(-2, -1)
        ).amax()
        if plan.peak_loss_weight > 0.0
        else None
    )
    was_training = operator.training
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    trace: list[float] = []
    operator.train()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(plan.seed)
            for batch in sampler.batches:
                indices = torch.tensor(batch, dtype=torch.long)
                query_indices = (
                    torch.cat((indices, tensors.qoi_indices))
                    if peak_target is not None
                    else indices
                )
                optimizer.zero_grad(set_to_none=True)
                output = _predict_case_vertices(operator, case, query_indices)
                target_u = tensors.target_u.index_select(0, indices)
                target_p = tensors.target_P.index_select(0, indices)
                u_mse = torch.mean((output.u[0, : len(batch)] - target_u).square())
                p_mse = torch.mean((output.P[0, : len(batch)] - target_p).square())
                loss = u_mse + plan.field_p_loss_weight * p_mse
                if peak_target is not None:
                    predicted_peak = torch.linalg.vector_norm(
                        output.P[0, len(batch) :], dim=(-2, -1)
                    ).amax()
                    loss = (
                        loss
                        + plan.peak_loss_weight
                        * (predicted_peak - peak_target).square()
                    )
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(
                        "supervised overfit loss became non-finite"
                    )
                loss.backward()
                if any(
                    parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all().item())
                    for parameter in operator.parameters()
                ):
                    raise FloatingPointError(
                        "supervised overfit gradient became non-finite"
                    )
                optimizer.step()
                trace.append(float(loss.detach()))
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        operator.train(was_training)
    return tuple(trace)


def _vertex_epoch_order(vertex_count: int, seed: int, epoch: int) -> tuple[int, ...]:
    prefix = seed.to_bytes(8, "big") + epoch.to_bytes(8, "big")
    return tuple(
        sorted(
            range(vertex_count),
            key=lambda index: (
                hashlib.blake2b(
                    prefix + index.to_bytes(8, "big"),
                    digest_size=32,
                    person=_VERTEX_SAMPLER_PERSON,
                ).digest(),
                index,
            ),
        )
    )


def _build_vertex_sampler_plan(
    plan: S13OverfitPlan,
    vertex_count: int,
) -> DeterministicSamplerPlan:
    """Build the finite, cross-platform deterministic vertex schedule."""

    if type(vertex_count) is not int or vertex_count <= 0:
        raise ValueError("vertex_count must be a positive integer")
    if not _is_exact_overfit_plan(plan):
        raise TypeError("plan must be an exact frozen S13OverfitPlan")
    batch_size = min(plan.vertex_batch_size, vertex_count)
    if vertex_count % batch_size != 0:
        raise ValueError("vertex_count must divide exactly into vertex batches")
    batches: list[tuple[int, ...]] = []
    epoch = 0
    while len(batches) < plan.optimizer_steps:
        order = _vertex_epoch_order(vertex_count, plan.seed, epoch)
        batches.extend(
            order[start : start + batch_size]
            for start in range(0, vertex_count, batch_size)
        )
        epoch += 1
    return DeterministicSamplerPlan(
        sampler_id=(
            f"{plan.vertex_sampler_policy_id}:{plan.seed}:batch-size-{batch_size}"
        ),
        population_size=vertex_count,
        batches=tuple(batches[: plan.optimizer_steps]),
    )


def _absolute_directory(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{label} must be a pathlib.Path")
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be an existing non-symlink directory")
    return path.resolve(strict=True)


def _new_external_path(
    path: Path,
    label: str,
    *,
    source_roots: tuple[Path, Path],
) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{label} must be a pathlib.Path")
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} must not already exist")
    parent = _absolute_directory(path.parent, f"{label} parent")
    resolved = parent / path.name
    if any(resolved.is_relative_to(root) for root in source_roots):
        raise ValueError(f"{label} must remain outside source roots")
    return resolved


def _preflight_paths(
    s10_run_root: Path,
    s11_checkpoint_root: Path,
    development_access_record: Path,
    output_path: Path | None,
) -> tuple[Path, Path, Path, Path | None]:
    s10 = _absolute_directory(s10_run_root, "s10_run_root")
    s11 = _absolute_directory(s11_checkpoint_root, "s11_checkpoint_root")
    if s10 == s11 or s10.is_relative_to(s11) or s11.is_relative_to(s10):
        raise ValueError("S10 and S11 source roots must be separate")
    roots = (s10, s11)
    access = _new_external_path(
        development_access_record,
        "development_access_record",
        source_roots=roots,
    )
    output = (
        None
        if output_path is None
        else _new_external_path(output_path, "output_path", source_roots=roots)
    )
    if output is not None and output == access:
        raise ValueError("output_path and development_access_record must differ")
    return s10, s11, access, output


def run_s13_overfit_diagnostic(
    *,
    s10_run_root: Path,
    s11_checkpoint_root: Path,
    development_access_record: Path,
    plan: S13OverfitPlan | None = None,
    output_path: Path | None = None,
) -> S13OverfitDiagnosticReport:
    """Run the bounded diagnostic after freezing S11 lineage and before S10 access."""

    frozen_plan = build_s13_overfit_plan() if plan is None else plan
    if not _is_exact_overfit_plan(frozen_plan):
        raise TypeError("plan must be an exact frozen S13OverfitPlan")
    s10, s11, access, _ = _preflight_paths(
        s10_run_root,
        s11_checkpoint_root,
        development_access_record,
        output_path,
    )
    vertex_sampler = _build_vertex_sampler_plan(
        frozen_plan, frozen_plan.expected_vertex_count
    )

    # Freeze the exact selected S11 state before resolving any FEM result bytes.
    initialization = load_selected_s11_initialization(s11)
    initial_state_hash = model_state_hash(initialization.model)

    source = open_selective_s10_source(
        s10,
        development_access_record=access,
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened before TRAIN diagnostic")
    case = materialize_dogbone_s13_training_case(
        source,
        initialization,
        frozen_plan.case_id,
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during TRAIN materialization")
    vertex_count = int(case.tensors.query_xy.shape[0])
    if vertex_count != frozen_plan.expected_vertex_count:
        raise ValueError("TRAIN case vertex count differs from the frozen plan")
    before = _measure_case(initialization.model, case)
    trace = _train_one_case(
        initialization.model,
        case,
        frozen_plan,
        vertex_sampler,
    )
    after = _measure_case(initialization.model, case)
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during TRAIN diagnostic")
    final_state_hash = model_state_hash(initialization.model)
    return S13OverfitDiagnosticReport(
        role=EvidenceRole.TRAIN,
        non_gate=True,
        plan=frozen_plan,
        plan_hash=frozen_plan.content_hash,
        case_id=frozen_plan.case_id,
        source_bindings_hash=source.bindings.content_hash,
        s11_evidence_file_hash=initialization.evidence_file_blake2b256,
        initial_checkpoint_file_hash=initialization.checkpoint_file_blake2b256,
        initial_model_state_hash=initial_state_hash,
        final_model_state_hash=final_state_hash,
        vertex_count=vertex_count,
        vertex_sampler_plan_hash=vertex_sampler.plan_hash,
        before=before,
        after=after,
        classification=classify_s13_overfit(frozen_plan, before, after),
        optimizer_loss_trace=trace,
    )


def _exclusive_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the non-gate, TRAIN-only S13 one-case overfit diagnostic."
    )
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--s11-checkpoint-root", required=True, type=Path)
    parser.add_argument("--development-access-record", required=True, type=Path)
    parser.add_argument(
        "--optimizer-steps",
        type=int,
        default=None,
        help="positive diagnostic prefix, bounded by the selected version",
    )
    parser.add_argument(
        "--version", choices=("v3", "v4", "v5", "v6", "v7"), default="v3"
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    plan = build_s13_overfit_plan(
        optimizer_steps=options.optimizer_steps,
        version=options.version,
    )
    report = run_s13_overfit_diagnostic(
        s10_run_root=options.s10_run_root,
        s11_checkpoint_root=options.s11_checkpoint_root,
        development_access_record=options.development_access_record,
        plan=plan,
        output_path=options.output,
    )
    payload = canonical_json_bytes(report) + b"\n"
    if options.output is not None:
        _exclusive_write(options.output, payload)
    print(payload[:-1].decode("utf-8"), flush=True)
    return 0


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a lowercase 256-bit digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a lowercase 256-bit digest") from exc
    if value.lower() != value:
        raise ValueError(f"{label} must be a lowercase 256-bit digest")
    return value


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = (
    "OverfitCaseMetrics",
    "OverfitClassification",
    "S13OverfitDiagnosticReport",
    "S13OverfitPlan",
    "S13OverfitPlanV7",
    "build_parser",
    "build_s13_overfit_plan",
    "classify_s13_overfit",
    "main",
    "run_s13_overfit_diagnostic",
)
