"""Matched joint-arm mechanics; no source access, replay, or artifact loading."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import ClassVar

import numpy as np
import torch

from campaigns.dogbone_v1.s13_joint_hybrid_protocol import JointHybridPlan
from campaigns.dogbone_v1.s13_objective import (
    DogboneS13TrainingCase,
    _physics_residuals,
    _validate_model_inputs,
    _validated_operator,
)
from campaigns.dogbone_v1.s13_overfit_diagnostic import (
    _build_vertex_sampler_plan,
    build_s13_overfit_plan,
)
from piginot.contracts import hash_spec
from piginot.datasets import CollocationKind
from piginot.doe import EvidenceRole
from piginot.objectives import ObjectiveResiduals, ObjectiveTerm
from piginot.operators import PiGinotOperator, module_digest
from piginot.training import DeterministicSamplerPlan

EvaluationCallback = Callable[[PiGinotOperator, int], object]
ResidualCallback = Callable[..., ObjectiveResiduals]
ParameterManifest = tuple[tuple[str, tuple[int, ...], int, bool], ...]


def _require_plan(plan: JointHybridPlan) -> None:
    if type(plan) is not JointHybridPlan or plan != JointHybridPlan():
        raise ValueError("plan must be the exact fixed JointHybridPlan")


def build_joint_vertex_sampler(plan: JointHybridPlan) -> DeterministicSamplerPlan:
    """Reuse the historical v5 schedule without modifying its algorithm."""
    _require_plan(plan)
    sampler = _build_vertex_sampler_plan(
        build_s13_overfit_plan(version="v5"), plan.expected_vertices
    )
    if sampler.plan_hash != plan.vertex_sampler_plan_hash:
        raise ValueError("vertex schedule differs from the pinned v5 sampler")
    return sampler


class SampledJointObjective:
    """Only data queries are minibatched; candidate physics remains fixed."""

    __slots__ = ("_case",)

    def __init__(self, case: DogboneS13TrainingCase) -> None:
        if type(case) is not DogboneS13TrainingCase:
            raise TypeError("case must be an exact DogboneS13TrainingCase")
        if case.tensors.role is not EvidenceRole.TRAIN:
            raise PermissionError("joint objective accepts TRAIN cases only")
        self._case = case

    @property
    def case(self) -> DogboneS13TrainingCase:
        return self._case

    def __call__(
        self,
        model: torch.nn.Module,
        vertex_indices: torch.Tensor,
        *,
        candidate: bool,
    ) -> ObjectiveResiduals:
        operator = _validated_operator(model)
        _validate_model_inputs(operator, (self._case,))
        if type(candidate) is not bool:
            raise TypeError("candidate must be a bool")
        tensors = self._case.tensors
        if (
            vertex_indices.dtype is not torch.long
            or vertex_indices.device.type != "cpu"
            or vertex_indices.ndim != 1
            or vertex_indices.numel() == 0
            or int(vertex_indices.min()) < 0
            or int(vertex_indices.max()) >= len(tensors.query_xy)
            or len(set(vertex_indices.tolist())) != vertex_indices.numel()
        ):
            raise ValueError("vertex_indices must be unique in-range CPU int64 indices")
        parameters = tensors.parameters_for(operator.config.parameter_names).unsqueeze(
            0
        )
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            encoding = operator.encode_boundary(
                boundary_xy=tensors.boundary_xy.unsqueeze(0),
                boundary_tags=tensors.boundary_tags.unsqueeze(0),
                boundary_normals=tensors.boundary_normals.unsqueeze(0),
                boundary_weights=tensors.boundary_weights.unsqueeze(0),
                boundary_mask=tensors.boundary_mask.unsqueeze(0),
            )
            data = operator.decode_queries(
                encoding=encoding,
                query_xy=tensors.query_xy.index_select(0, vertex_indices).unsqueeze(0),
                parameters=parameters,
            )
            values = {
                ObjectiveTerm.DATA_U: data.u
                - tensors.target_u.index_select(0, vertex_indices).unsqueeze(0),
                ObjectiveTerm.DATA_P: data.P
                - tensors.target_P.index_select(0, vertex_indices).unsqueeze(0),
            }
            if candidate:
                coordinates = (
                    torch.tensor(
                        [point.coordinates for point in self._case.collocation.points],
                        dtype=parameters.dtype,
                        device=parameters.device,
                    )
                    .unsqueeze(0)
                    .requires_grad_(True)
                )
                physics = operator.decode_queries(
                    encoding=encoding, query_xy=coordinates, parameters=parameters
                )
                values.update(
                    _physics_residuals(
                        (self._case,), (coordinates,), physics.u, physics.P
                    )
                )
        return ObjectiveResiduals(values)


def joint_parameter_manifest(model: PiGinotOperator) -> ParameterManifest:
    """Record every named parameter, not a P-head-only subset."""
    return tuple(
        (name, tuple(parameter.shape), parameter.numel(), parameter.requires_grad)
        for name, parameter in _validated_operator(model).named_parameters()
    )


@dataclass(frozen=True, slots=True)
class JointArmState:
    arm_id: str
    model: PiGinotOperator
    optimizer: torch.optim.Adam
    expected_config_hash: str
    expected_module_digest: str

    def __post_init__(self) -> None:
        if self.arm_id not in ("control", "candidate"):
            raise ValueError("arm_id must be control or candidate")


def clone_joint_arms(
    baseline: PiGinotOperator, plan: JointHybridPlan
) -> tuple[JointArmState, JointArmState]:
    """Clone only model state; baseline optimizer moments are never reused."""
    _require_plan(plan)
    _validated_operator(baseline)
    _finite_parameters(baseline)
    config_hash = hash_spec(baseline.config, "content")
    baseline_digest = module_digest(baseline)
    states: list[JointArmState] = []
    for arm_id in ("control", "candidate"):
        model = copy.deepcopy(baseline)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=plan.learning_rate,
            betas=plan.adam_betas,
            eps=plan.adam_epsilon,
            weight_decay=plan.weight_decay,
            foreach=False,
            fused=False,
        )
        if not _same_value(model.state_dict(), baseline.state_dict()):
            raise RuntimeError("joint clone differs from the baseline state")
        state = JointArmState(arm_id, model, optimizer, config_hash, baseline_digest)
        _validate_architecture_binding(state)
        states.append(state)
    return states[0], states[1]


@dataclass(frozen=True, slots=True)
class JointStepTrace:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    step: int
    vertex_indices: tuple[int, ...]
    unweighted_terms: tuple[tuple[str, float], ...]
    weighted_terms: tuple[tuple[str, float], ...]
    total_loss: float
    preclip_gradient_norm: float
    clip_coefficient: float
    optimizer_updates: int
    finite_checks: tuple[tuple[str, bool], ...]


@dataclass(frozen=True, slots=True)
class JointArmTrainingResult:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    arm_id: str
    step_trace: tuple[JointStepTrace, ...]
    evaluations: tuple[tuple[int, object], ...]
    sampler_plan_hash: str
    optimizer_updates: int
    parameter_manifest: ParameterManifest
    config_hash: str
    module_digest: str


class JointTrainingFailure(RuntimeError):
    """A single failed attempt with completed evidence, never an implicit retry."""

    def __init__(
        self,
        cause: Exception,
        *,
        arm_id: str,
        attempted_step: int,
        step_trace: tuple[JointStepTrace, ...],
        evaluations: tuple[tuple[int, object], ...],
        optimizer_updates: int,
    ) -> None:
        super().__init__(str(cause))
        self.reason = str(cause)
        self.cause_type = type(cause).__name__
        self.arm_id = arm_id
        self.attempted_step = attempted_step
        self.step_trace = step_trace
        self.evaluations = evaluations
        self.optimizer_updates = optimizer_updates


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return tuple(left) == tuple(right) and all(
            _same_value(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and bool(left == right)


def _finite_parameters(model: PiGinotOperator) -> None:
    if any(
        not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    ):
        raise FloatingPointError("joint parameter became non-finite")


def _validate_architecture_binding(state: JointArmState) -> None:
    if hash_spec(state.model.config, "content") != state.expected_config_hash:
        raise ValueError("joint config differs from its immutable baseline binding")
    if module_digest(state.model) != state.expected_module_digest:
        raise ValueError(
            "joint module digest differs from its immutable baseline binding"
        )


def _validate_arm(state: JointArmState, plan: JointHybridPlan) -> None:
    if type(state) is not JointArmState:
        raise TypeError("state must be an exact JointArmState")
    _validate_architecture_binding(state)
    _validated_operator(state.model)
    if not all(parameter.requires_grad for parameter in state.model.parameters()):
        raise ValueError("every joint parameter must be trainable")
    if type(state.optimizer) is not torch.optim.Adam or state.optimizer.state:
        raise ValueError("each arm must start with a fresh Adam optimizer")
    if len(state.optimizer.param_groups) != 1:
        raise ValueError("joint Adam must have exactly one all-parameter group")
    group = state.optimizer.param_groups[0]
    if tuple(id(parameter) for parameter in group["params"]) != tuple(
        id(parameter) for parameter in state.model.parameters()
    ):
        raise ValueError("Adam must cover every joint parameter in named order")
    expected = {
        "lr": plan.learning_rate,
        "betas": plan.adam_betas,
        "eps": plan.adam_epsilon,
        "weight_decay": plan.weight_decay,
        "foreach": False,
        "fused": False,
        "amsgrad": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
    }
    if any(not _same_value(group[name], value) for name, value in expected.items()):
        raise ValueError("fresh Adam settings differ from the fixed plan")
    _finite_parameters(state.model)


def _evaluate_preserving_state(
    state: JointArmState,
    sampler: DeterministicSamplerPlan,
    step: int,
    evaluate: EvaluationCallback,
) -> object:
    model = state.model
    _validate_architecture_binding(state)
    config_before = copy.deepcopy(model.config)
    lifting_before = copy.deepcopy(model.lifting_verification)
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(state.optimizer.state_dict())
    sampler_before = copy.deepcopy(sampler)
    parameters_before = tuple(model.named_parameters())
    structure = tuple(
        (module, dict(module._modules), dict(module._parameters), dict(module._buffers))
        for module in model.modules()
    )
    trainability = tuple(parameter.requires_grad for _, parameter in parameters_before)
    gradients = tuple(
        None if parameter.grad is None else parameter.grad.clone()
        for _, parameter in parameters_before
    )
    modes = tuple((module, module.training) for module in model.modules())
    rng = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
    deterministic = torch.are_deterministic_algorithms_enabled()
    model.eval()
    mutated = False
    evaluation_error: Exception | None = None
    try:
        with torch.enable_grad():
            result = evaluate(model, step)
    except Exception as cause:
        evaluation_error = cause
        raise
    finally:
        try:
            _validate_architecture_binding(state)
            architecture_mutated = False
        except (TypeError, ValueError):
            architecture_mutated = True
        same_parameters = tuple(
            (name, id(parameter)) for name, parameter in model.named_parameters()
        ) == tuple((name, id(parameter)) for name, parameter in parameters_before)
        mutated = (
            architecture_mutated
            or not same_parameters
            or not _same_value(model.state_dict(), model_before)
            or not _same_value(state.optimizer.state_dict(), optimizer_before)
            or sampler != sampler_before
            or tuple(parameter.requires_grad for _, parameter in parameters_before)
            != trainability
            or not _same_value(
                tuple(parameter.grad for _, parameter in parameters_before), gradients
            )
        )
        for module, modules, parameters, buffers in structure:
            module._modules = modules
            module._parameters = parameters
            module._buffers = buffers
        model.config = config_before
        model.lifting_verification = lifting_before
        model.load_state_dict(model_before, strict=True)
        state.optimizer.load_state_dict(optimizer_before)
        for (_, parameter), requires_grad, gradient in zip(
            parameters_before, trainability, gradients, strict=True
        ):
            parameter.requires_grad_(requires_grad)
            parameter.grad = gradient
        for module, mode in modes:
            module.training = mode
        torch.set_rng_state(rng[0])
        random.setstate(rng[1])
        np.random.set_state(rng[2])
        torch.use_deterministic_algorithms(deterministic)
        if sampler != sampler_before:
            for field in fields(sampler):
                object.__setattr__(
                    sampler, field.name, getattr(sampler_before, field.name)
                )
        if mutated:
            reason = "evaluation mutated config, module, optimizer, gradients or sampler progress"
            if evaluation_error is not None:
                reason += f"; evaluator also failed: {type(evaluation_error).__name__}: {evaluation_error}"
            raise RuntimeError(reason) from evaluation_error
    return result


def train_joint_arm(
    state: JointArmState,
    objective: SampledJointObjective,
    sampler: DeterministicSamplerPlan,
    plan: JointHybridPlan,
    evaluate: EvaluationCallback | None = None,
) -> JointArmTrainingResult:
    """Execute only the complete approved arm budget and fixed evaluation steps."""
    _require_plan(plan)
    if type(objective) is not SampledJointObjective:
        raise TypeError("objective must be an exact SampledJointObjective")
    if sampler != build_joint_vertex_sampler(plan):
        raise ValueError("sampler differs from the fixed 57-of-969 vertex schedule")
    case = objective.case
    if (
        case.tensors.case_id != plan.case_id
        or len(case.tensors.query_xy) != plan.expected_vertices
    ):
        raise ValueError("objective differs from the fixed 969-vertex TRAIN case")
    stream = case.collocation.stream
    if (
        (
            stream.interior_count,
            stream.boundary_count,
            sum(quota.count for quota in stream.qoi_quotas),
        )
        != plan.physics_collocation_layout
        or len(stream.qoi_quotas) != 1
        or stream.qoi_quotas[0].kind is not CollocationKind.BOUNDARY
        or stream.qoi_quotas[0].semantic_tag != "right_arc"
        or len(case.collocation.points) != sum(plan.physics_collocation_layout)
    ):
        raise ValueError("physics collocations differ from the fixed 128/64/64 layout")
    if evaluate is None:
        raise ValueError("the fixed arm run requires an evaluation callback")
    return _train_arm_steps(
        state,
        objective,
        sampler,
        plan,
        optimizer_steps=plan.arm_optimizer_steps,
        evaluation_steps=plan.evaluation_steps,
        evaluate=evaluate,
    )


def _train_arm_steps(
    state: JointArmState,
    objective: ResidualCallback,
    sampler: DeterministicSamplerPlan,
    plan: JointHybridPlan,
    *,
    optimizer_steps: int,
    evaluation_steps: tuple[int, ...],
    evaluate: EvaluationCallback | None = None,
) -> JointArmTrainingResult:
    """Small synthetic-loop seam; the public entrypoint never changes its budget."""
    _require_plan(plan)
    _validate_arm(state, plan)
    if type(sampler) is not DeterministicSamplerPlan:
        raise TypeError("sampler must be an exact DeterministicSamplerPlan")
    if type(optimizer_steps) is not int or not 0 < optimizer_steps <= min(
        len(sampler.batches), plan.arm_optimizer_steps
    ):
        raise ValueError("optimizer_steps must be a finite in-budget prefix")
    if (
        type(evaluation_steps) is not tuple
        or evaluation_steps != tuple(sorted(set(evaluation_steps)))
        or any(
            type(step) is not int or not 0 <= step <= optimizer_steps
            for step in evaluation_steps
        )
    ):
        raise ValueError("evaluation_steps must be sorted unique in-budget steps")
    model = state.model
    modes = tuple((module, module.training) for module in model.modules())
    deterministic = torch.are_deterministic_algorithms_enabled()
    weights = (
        plan.candidate_term_weights
        if state.arm_id == "candidate"
        else plan.control_term_weights
    )
    trace: list[JointStepTrace] = []
    evaluations: list[tuple[int, object]] = []
    updates = 0
    attempted_step = 0
    model.train()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(plan.seed)
            if evaluate is not None and 0 in evaluation_steps:
                evaluations.append(
                    (0, _evaluate_preserving_state(state, sampler, 0, evaluate))
                )
            for step, batch in enumerate(sampler.batches[:optimizer_steps], start=1):
                attempted_step = step
                state.optimizer.zero_grad(set_to_none=True)
                residuals = objective(
                    model,
                    torch.tensor(batch, dtype=torch.long),
                    candidate=state.arm_id == "candidate",
                )
                if tuple(term.value for term in residuals.values) != tuple(
                    name for name, _ in weights
                ):
                    raise ValueError(
                        "objective terms differ from the fixed arm weights"
                    )
                losses = tuple(
                    (name, torch.mean(residuals.values[ObjectiveTerm(name)].square()))
                    for name, _ in weights
                )
                weighted = tuple(
                    (name, loss * weight)
                    for (name, loss), (_, weight) in zip(losses, weights, strict=True)
                )
                total = sum(
                    (value for _, value in weighted),
                    torch.zeros((), dtype=torch.float64),
                )
                if not all(
                    bool(torch.isfinite(value)) for _, value in (*losses, *weighted)
                ) or not bool(torch.isfinite(total)):
                    raise FloatingPointError("joint objective loss became non-finite")
                torch.autograd.backward(total)
                if any(
                    parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError("joint gradient became non-finite")
                norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        plan.gradient_clip_norm,
                        error_if_nonfinite=True,
                        foreach=False,
                    )
                )
                if not math.isfinite(norm):
                    raise FloatingPointError(
                        "joint preclip gradient norm became non-finite"
                    )
                state.optimizer.step()
                updates += 1
                parameters_finite = all(
                    bool(torch.isfinite(parameter).all())
                    for parameter in model.parameters()
                )
                trace.append(
                    JointStepTrace(
                        step,
                        batch,
                        tuple((name, float(value.detach())) for name, value in losses),
                        tuple(
                            (name, float(value.detach())) for name, value in weighted
                        ),
                        float(total.detach()),
                        norm,
                        min(1.0, plan.gradient_clip_norm / (norm + 1e-6)),
                        updates,
                        (
                            ("loss", True),
                            ("gradient", True),
                            ("parameters", parameters_finite),
                        ),
                    )
                )
                _finite_parameters(model)
                if evaluate is not None and step in evaluation_steps:
                    evaluations.append(
                        (
                            step,
                            _evaluate_preserving_state(state, sampler, step, evaluate),
                        )
                    )
        _validate_architecture_binding(state)
    except Exception as cause:
        raise JointTrainingFailure(
            cause,
            arm_id=state.arm_id,
            attempted_step=attempted_step,
            step_trace=tuple(trace),
            evaluations=tuple(evaluations),
            optimizer_updates=updates,
        ) from cause
    finally:
        torch.use_deterministic_algorithms(deterministic)
        for module, mode in modes:
            module.training = mode
    return JointArmTrainingResult(
        state.arm_id,
        tuple(trace),
        tuple(evaluations),
        sampler.plan_hash,
        updates,
        joint_parameter_manifest(model),
        state.expected_config_hash,
        state.expected_module_digest,
    )
