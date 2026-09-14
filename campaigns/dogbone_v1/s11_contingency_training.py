"""Pure deterministic training core for the preregistered S11 contingency.

This module consumes already materialized, normalized tensors and never opens the
S10 object store. Optimizer updates accept TRAIN-fit tensors only; evaluation may
consume TRAIN-screen or one-shot DEVELOPMENT tensors. HOLDOUT, reserve, PILOT, and
legacy evidence have no path into this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from campaigns.dogbone_v1.s11_contingency import (
    ArchitectureCandidate,
    ArchitectureScreenMetrics,
    ArchitectureScreenProtocol,
)
from campaigns.dogbone_v1.s11_contingency_models import (
    build_candidate_models,
    model_state_hash,
)
from campaigns.dogbone_v1.s11_contingency_tensors import (
    ArchitectureCaseTensors,
    CandidateTrainingSummary,
    TrainingSchedule,
    _owned_float64,
)
from piginot.doe import EvidenceRole
from piginot.operators import OperatorOutput, PiGinotOperator, module_digest

TrainingProgress = Callable[[int, float], None]


@dataclass(frozen=True, slots=True)
class ArchitectureScreenLoss:
    """Differentiable components of the fixed supervised screen objective."""

    total: torch.Tensor
    u_mse: torch.Tensor
    P_mse: torch.Tensor
    peak_P_mse: torch.Tensor


def build_training_schedule(
    protocol: ArchitectureScreenProtocol,
    fit_cases: tuple[ArchitectureCaseTensors, ...],
) -> TrainingSchedule:
    """Build the exact 1,500-step schedule independently of any candidate."""

    _require_protocol(protocol)
    ordered = _ordered_population(
        fit_cases,
        protocol.fit_case_ids,
        "fit",
        expected_role=EvidenceRole.TRAIN,
    )
    return _build_schedule(protocol, ordered)


def architecture_screen_loss(
    protocol: ArchitectureScreenProtocol,
    output: OperatorOutput,
    *,
    target_u: torch.Tensor,
    target_P: torch.Tensor,
    qoi_query_count: int,
) -> ArchitectureScreenLoss:
    """Compute u MSE + P MSE + registered-weight root-peak P MSE."""

    _require_protocol(protocol)
    if type(output) is not OperatorOutput:
        raise TypeError("output must be an exact OperatorOutput")
    expected_u = _checked_batch_target(target_u, "target_u", trailing=(2,))
    expected_P = _checked_batch_target(target_P, "target_P", trailing=(2, 2))
    if output.u.dtype is not torch.float64 or output.P.dtype is not torch.float64:
        raise ValueError("operator outputs must use torch.float64")
    if output.u.device.type != "cpu" or output.P.device.type != "cpu":
        raise ValueError("architecture screen is fixed to deterministic CPU tensors")
    if not bool(torch.isfinite(output.u).all().item()) or not bool(
        torch.isfinite(output.P).all().item()
    ):
        raise ValueError("operator outputs must be finite")
    if output.u.shape != expected_u.shape or output.P.shape != expected_P.shape:
        raise ValueError("operator outputs and supervised targets must align")
    query_count = output.u.shape[1]
    if (
        type(qoi_query_count) is not int
        or qoi_query_count <= 0
        or qoi_query_count >= query_count
    ):
        raise ValueError("qoi_query_count must split generic and QoI queries")

    u_mse = torch.mean((output.u - expected_u).square())
    P_mse = torch.mean((output.P - expected_P).square())
    predicted_qoi = torch.linalg.vector_norm(
        output.P[:, -qoi_query_count:], dim=(-2, -1)
    )
    expected_qoi = torch.linalg.vector_norm(
        expected_P[:, -qoi_query_count:], dim=(-2, -1)
    )
    peak_P_mse = torch.mean(
        (predicted_qoi.amax(dim=1) - expected_qoi.amax(dim=1)).square()
    )
    total = u_mse + P_mse + protocol.peak_supervised_p_weight * peak_P_mse
    return ArchitectureScreenLoss(total, u_mse, P_mse, peak_P_mse)


def train_candidate_model(
    protocol: ArchitectureScreenProtocol,
    candidate: ArchitectureCandidate,
    model: PiGinotOperator,
    fit_cases: tuple[ArchitectureCaseTensors, ...],
    schedule: TrainingSchedule,
    *,
    progress: TrainingProgress | None = None,
) -> CandidateTrainingSummary:
    """Execute the complete registered budget; production has no step override."""

    if schedule.optimizer_steps != protocol.optimizer_steps:
        raise ValueError("production training requires the fixed 1,500 optimizer steps")
    _validate_registered_fresh_initialization(
        protocol,
        candidate,
        model,
        fit_cases,
    )
    return _train_candidate(
        protocol,
        candidate,
        model,
        fit_cases,
        schedule,
        optimizer_steps=protocol.optimizer_steps,
        progress=progress,
    )


def _train_candidate_prefix_for_test(
    protocol: ArchitectureScreenProtocol,
    candidate: ArchitectureCandidate,
    model: PiGinotOperator,
    fit_cases: tuple[ArchitectureCaseTensors, ...],
    schedule: TrainingSchedule,
    *,
    optimizer_steps: int,
    progress: TrainingProgress | None = None,
) -> CandidateTrainingSummary:
    """Exercise a strict prefix in tests without changing the protocol or schedule."""

    if (
        type(optimizer_steps) is not int
        or optimizer_steps <= 0
        or optimizer_steps >= protocol.optimizer_steps
    ):
        raise ValueError("test optimizer_steps must be a strict positive prefix")
    return _train_candidate(
        protocol,
        candidate,
        model,
        fit_cases,
        schedule,
        optimizer_steps=optimizer_steps,
        progress=progress,
    )


def evaluate_train_screen_candidate(
    protocol: ArchitectureScreenProtocol,
    candidate: ArchitectureCandidate,
    model: PiGinotOperator,
    screen_cases: tuple[ArchitectureCaseTensors, ...],
    *,
    elapsed_seconds: float,
) -> ArchitectureScreenMetrics:
    """Evaluate all vertices of the preregistered held-aside TRAIN population."""

    return _evaluate_candidate_population(
        protocol,
        candidate,
        model,
        screen_cases,
        expected_case_ids=protocol.screen_case_ids,
        expected_role=EvidenceRole.TRAIN,
        label="screen",
        elapsed_seconds=elapsed_seconds,
    )


def evaluate_development_candidate(
    protocol: ArchitectureScreenProtocol,
    candidate: ArchitectureCandidate,
    model: PiGinotOperator,
    development_cases: tuple[ArchitectureCaseTensors, ...],
    *,
    elapsed_seconds: float,
) -> ArchitectureScreenMetrics:
    """Evaluate all vertices of the once-opened DEVELOPMENT population."""

    return _evaluate_candidate_population(
        protocol,
        candidate,
        model,
        development_cases,
        expected_case_ids=protocol.development_case_ids,
        expected_role=EvidenceRole.DEVELOPMENT,
        label="DEVELOPMENT",
        elapsed_seconds=elapsed_seconds,
    )


def _evaluate_candidate_population(
    protocol: ArchitectureScreenProtocol,
    candidate: ArchitectureCandidate,
    model: PiGinotOperator,
    cases: tuple[ArchitectureCaseTensors, ...],
    *,
    expected_case_ids: tuple[str, ...],
    expected_role: EvidenceRole,
    label: str,
    elapsed_seconds: float,
) -> ArchitectureScreenMetrics:
    """Evaluate a role-bound public population without changing model state."""

    _require_protocol(protocol)
    ordered = _ordered_population(
        cases,
        expected_case_ids,
        label,
        expected_role=expected_role,
    )
    _validate_model(candidate, protocol, model, ordered)
    outputs: list[OperatorOutput] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(ordered), protocol.case_batch_size):
                batch = ordered[start : start + protocol.case_batch_size]
                _require_uniform_query_shape(batch)
                output = model(
                    boundary_xy=torch.stack([case.boundary_xy for case in batch]),
                    boundary_tags=torch.stack([case.boundary_tags for case in batch]),
                    boundary_normals=torch.stack(
                        [case.boundary_normals for case in batch]
                    ),
                    boundary_weights=torch.stack(
                        [case.boundary_weights for case in batch]
                    ),
                    boundary_mask=torch.stack([case.boundary_mask for case in batch]),
                    query_xy=torch.stack([case.query_xy for case in batch]),
                    parameters=torch.stack(
                        [
                            case.parameters_for(candidate.parameter_names)
                            for case in batch
                        ]
                    ),
                )
                for index in range(len(batch)):
                    outputs.append(
                        OperatorOutput(
                            output.u[index : index + 1].detach().clone(),
                            output.P[index : index + 1].detach().clone(),
                        )
                    )
    finally:
        model.train(was_training)
    return calculate_architecture_screen_metrics(
        candidate.candidate_id,
        ordered,
        tuple(outputs),
        trainable_parameter_count=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        elapsed_seconds=elapsed_seconds,
    )


def calculate_architecture_screen_metrics(
    candidate_id: str,
    cases: tuple[ArchitectureCaseTensors, ...],
    predictions: tuple[OperatorOutput, ...],
    *,
    trainable_parameter_count: int,
    elapsed_seconds: float,
) -> ArchitectureScreenMetrics:
    """Reduce per-case relative field and smooth-root peak errors."""

    if type(cases) is not tuple or not cases:
        raise ValueError("cases must be a non-empty tuple")
    if any(type(case) is not ArchitectureCaseTensors for case in cases):
        raise TypeError("cases must contain exact ArchitectureCaseTensors")
    if type(predictions) is not tuple or len(predictions) != len(cases):
        raise ValueError("predictions must align one-to-one with cases")
    if any(type(output) is not OperatorOutput for output in predictions):
        raise TypeError("predictions must contain exact OperatorOutput values")

    signed_peak_errors: list[float] = []
    u_relative_errors: list[float] = []
    P_relative_errors: list[float] = []
    for case, output in zip(cases, predictions, strict=True):
        if output.u.shape != (1, *case.target_u.shape) or output.P.shape != (
            1,
            *case.target_P.shape,
        ):
            raise ValueError("prediction tensors must align with every case")
        if output.u.dtype is not torch.float64 or output.P.dtype is not torch.float64:
            raise ValueError("prediction tensors must use torch.float64")
        if output.u.device.type != "cpu" or output.P.device.type != "cpu":
            raise ValueError("prediction tensors must reside on CPU")
        if not bool(torch.isfinite(output.u).all().item()) or not bool(
            torch.isfinite(output.P).all().item()
        ):
            raise ValueError("prediction tensors must be finite")
        target_u_norm = torch.linalg.vector_norm(case.target_u)
        target_P_norm = torch.linalg.vector_norm(case.target_P)
        if float(target_u_norm) <= 0.0 or float(target_P_norm) <= 0.0:
            raise ValueError("relative L2 targets must have positive norms")
        u_relative_errors.append(
            float(torch.linalg.vector_norm(output.u[0] - case.target_u) / target_u_norm)
        )
        P_relative_errors.append(
            float(torch.linalg.vector_norm(output.P[0] - case.target_P) / target_P_norm)
        )
        true_peak = torch.linalg.vector_norm(
            case.target_P[case.qoi_indices], dim=(-2, -1)
        ).amax()
        predicted_peak = torch.linalg.vector_norm(
            output.P[0, case.qoi_indices], dim=(-2, -1)
        ).amax()
        if float(true_peak) <= 0.0:
            raise ValueError("every root QoI target must have a positive stress peak")
        signed_peak_errors.append(float((predicted_peak - true_peak) / true_peak))

    absolute_peak_errors = [abs(value) for value in signed_peak_errors]
    return ArchitectureScreenMetrics(
        candidate_id=candidate_id,
        median_signed_peak_error=_linear_quantile(signed_peak_errors, 0.5),
        median_absolute_peak_error=_linear_quantile(absolute_peak_errors, 0.5),
        p90_absolute_peak_error=_linear_quantile(absolute_peak_errors, 0.9),
        median_u_relative_l2=_linear_quantile(u_relative_errors, 0.5),
        median_p_relative_l2=_linear_quantile(P_relative_errors, 0.5),
        trainable_parameter_count=trainable_parameter_count,
        elapsed_seconds=elapsed_seconds,
    )


def _train_candidate(
    protocol: ArchitectureScreenProtocol,
    candidate: ArchitectureCandidate,
    model: PiGinotOperator,
    fit_cases: tuple[ArchitectureCaseTensors, ...],
    schedule: TrainingSchedule,
    *,
    optimizer_steps: int,
    progress: TrainingProgress | None,
) -> CandidateTrainingSummary:
    _require_protocol(protocol)
    ordered = _ordered_population(
        fit_cases,
        protocol.fit_case_ids,
        "fit",
        expected_role=EvidenceRole.TRAIN,
    )
    _validate_model(candidate, protocol, model, ordered)
    expected_schedule = _build_schedule(protocol, ordered)
    if schedule.content_hash != expected_schedule.content_hash:
        raise ValueError("schedule does not match the registered sampler exactly")
    if optimizer_steps > schedule.optimizer_steps:
        raise ValueError("optimizer_steps exceed the registered schedule")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=protocol.learning_rate,
        foreach=False,
        fused=False,
    )
    model.train()
    final_loss: ArchitectureScreenLoss | None = None
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for step in range(optimizer_steps):
            output, target_u, target_P = _training_batch(
                model,
                candidate,
                ordered,
                schedule.case_indices[step],
                schedule.query_indices[step],
            )
            optimizer.zero_grad(set_to_none=True)
            final_loss = architecture_screen_loss(
                protocol,
                output,
                target_u=target_u,
                target_P=target_P,
                qoi_query_count=schedule.qoi_query_count,
            )
            final_loss.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), protocol.gradient_clip_norm
            )
            optimizer.step()
            completed = step + 1
            if progress is not None and (
                completed == 1 or completed % 50 == 0 or completed == optimizer_steps
            ):
                progress(completed, float(final_loss.total.detach()))
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
    if final_loss is None:
        raise RuntimeError("training executed no optimizer steps")
    return CandidateTrainingSummary(
        candidate_id=candidate.candidate_id,
        optimizer_steps=optimizer_steps,
        schedule_hash=schedule.content_hash,
        final_total_loss=float(final_loss.total.detach()),
        final_u_mse=float(final_loss.u_mse.detach()),
        final_P_mse=float(final_loss.P_mse.detach()),
        final_peak_P_mse=float(final_loss.peak_P_mse.detach()),
    )


def _training_batch(
    model: PiGinotOperator,
    candidate: ArchitectureCandidate,
    cases: tuple[ArchitectureCaseTensors, ...],
    case_indices: torch.Tensor,
    query_indices: torch.Tensor,
) -> tuple[OperatorOutput, torch.Tensor, torch.Tensor]:
    selected = tuple(cases[int(index)] for index in case_indices)
    query_rows = tuple(query_indices[index] for index in range(len(selected)))
    query_xy = torch.stack(
        [
            case.query_xy[indices]
            for case, indices in zip(selected, query_rows, strict=True)
        ]
    )
    target_u = torch.stack(
        [
            case.target_u[indices]
            for case, indices in zip(selected, query_rows, strict=True)
        ]
    )
    target_P = torch.stack(
        [
            case.target_P[indices]
            for case, indices in zip(selected, query_rows, strict=True)
        ]
    )
    output = model(
        boundary_xy=torch.stack([case.boundary_xy for case in selected]),
        boundary_tags=torch.stack([case.boundary_tags for case in selected]),
        boundary_normals=torch.stack([case.boundary_normals for case in selected]),
        boundary_weights=torch.stack([case.boundary_weights for case in selected]),
        boundary_mask=torch.stack([case.boundary_mask for case in selected]),
        query_xy=query_xy,
        parameters=torch.stack(
            [case.parameters_for(candidate.parameter_names) for case in selected]
        ),
    )
    return output, target_u, target_P


def _build_schedule(
    protocol: ArchitectureScreenProtocol,
    ordered: tuple[ArchitectureCaseTensors, ...],
) -> TrainingSchedule:
    qoi_count_float = protocol.query_points_per_case * protocol.qoi_fraction
    qoi_count = int(qoi_count_float)
    if float(qoi_count) != qoi_count_float:
        raise ValueError("qoi_fraction must yield an integral query count")
    generic_count = protocol.query_points_per_case - qoi_count
    generator = torch.Generator(device="cpu")
    generator.manual_seed(protocol.sampler_seed)
    case_indices = torch.empty(
        (protocol.optimizer_steps, protocol.case_batch_size), dtype=torch.int64
    )
    query_indices = torch.empty(
        (
            protocol.optimizer_steps,
            protocol.case_batch_size,
            protocol.query_points_per_case,
        ),
        dtype=torch.int64,
    )
    for step in range(protocol.optimizer_steps):
        selected = torch.randperm(len(ordered), generator=generator)[
            : protocol.case_batch_size
        ]
        case_indices[step] = selected
        for batch_index, case_index in enumerate(selected.tolist()):
            case = ordered[case_index]
            qoi_pool = case.qoi_indices
            is_qoi = torch.zeros(case.query_xy.shape[0], dtype=torch.bool)
            is_qoi[qoi_pool] = True
            generic_pool = torch.arange(case.query_xy.shape[0], dtype=torch.int64)[
                ~is_qoi
            ]
            if generic_pool.numel() == 0:
                raise ValueError(f"case {case.case_id} has no non-QoI query pool")
            query_indices[step, batch_index, :generic_count] = _cycle_sample(
                generic_pool, generic_count, generator
            )
            query_indices[step, batch_index, generic_count:] = _cycle_sample(
                qoi_pool, qoi_count, generator
            )
    return TrainingSchedule(
        protocol_id=protocol.protocol_id,
        sampler_seed=protocol.sampler_seed,
        case_ids=tuple(case.case_id for case in ordered),
        case_indices=case_indices,
        query_indices=query_indices,
        qoi_query_count=qoi_count,
    )


def _cycle_sample(
    pool: torch.Tensor, count: int, generator: torch.Generator
) -> torch.Tensor:
    permutation = pool[torch.randperm(pool.numel(), generator=generator)]
    repeats = math.ceil(count / pool.numel())
    return permutation.repeat(repeats)[:count]


def _ordered_population(
    cases: tuple[ArchitectureCaseTensors, ...],
    expected_case_ids: tuple[str, ...],
    label: str,
    *,
    expected_role: EvidenceRole,
) -> tuple[ArchitectureCaseTensors, ...]:
    if type(cases) is not tuple or any(
        type(case) is not ArchitectureCaseTensors for case in cases
    ):
        raise TypeError(f"{label} cases must contain exact ArchitectureCaseTensors")
    by_id = {case.case_id: case for case in cases}
    if len(by_id) != len(cases) or set(by_id) != set(expected_case_ids):
        raise ValueError(f"{label} cases must match the registered {label}_case_ids")
    ordered = tuple(by_id[case_id] for case_id in expected_case_ids)
    if any(case.role is not expected_role for case in ordered):
        raise ValueError(f"{label} cases must have role {expected_role.value}")
    if len({case.leakage_group_id for case in ordered}) != len(ordered):
        raise ValueError(f"{label} cases must have unique leakage groups")
    if len({case.normalization_hash for case in ordered}) != 1:
        raise ValueError(f"{label} cases must share one normalization_hash")
    first = ordered[0]
    for case in ordered[1:]:
        if case.parameter_names != first.parameter_names:
            raise ValueError(f"{label} cases must share parameter_names")
        if case.boundary_tag_names != first.boundary_tag_names:
            raise ValueError(f"{label} cases must share boundary_tag_names")
        for name in (
            "boundary_xy",
            "boundary_tags",
            "boundary_normals",
            "boundary_weights",
            "boundary_mask",
        ):
            if getattr(case, name).shape != getattr(first, name).shape:
                raise ValueError(f"{label} cases must share boundary cloud shapes")
    return ordered


def _validate_model(
    candidate: ArchitectureCandidate,
    protocol: ArchitectureScreenProtocol,
    model: PiGinotOperator,
    cases: tuple[ArchitectureCaseTensors, ...],
) -> None:
    if (
        type(candidate) is not ArchitectureCandidate
        or candidate not in protocol.candidates
    ):
        raise ValueError("candidate must be an exact registered architecture")
    if type(model) is not PiGinotOperator:
        raise TypeError("model must be an exact PiGinotOperator")
    if model.config.parameter_names != candidate.parameter_names:
        raise ValueError("model parameter names do not match the candidate")
    if (
        model.config.local_boundary_kernel_scales
        != candidate.local_boundary_kernel_scales
        or model.config.stress_enrichment_layers != candidate.stress_enrichment_layers
    ):
        raise ValueError("model enrichment configuration does not match the candidate")
    if module_digest(model) != candidate.operator_module_digest:
        raise ValueError("model operator schema does not match the preregistration")
    if any(
        parameter.dtype is not torch.float64 or parameter.device.type != "cpu"
        for parameter in model.parameters()
    ):
        raise ValueError("model parameters must use deterministic CPU torch.float64")
    if model.config.boundary_tag_names != cases[0].boundary_tag_names:
        raise ValueError("model boundary tag order does not match the case tensors")
    for case in cases:
        case.parameters_for(candidate.parameter_names)


def _validate_registered_fresh_initialization(
    protocol: ArchitectureScreenProtocol,
    candidate: ArchitectureCandidate,
    model: PiGinotOperator,
    fit_cases: tuple[ArchitectureCaseTensors, ...],
) -> None:
    """Reject resumed or caller-perturbed state before the production budget."""

    _require_protocol(protocol)
    ordered = _ordered_population(
        fit_cases,
        protocol.fit_case_ids,
        "fit",
        expected_role=EvidenceRole.TRAIN,
    )
    _validate_model(candidate, protocol, model, ordered)
    expected_models = build_candidate_models(
        protocol,
        boundary_tag_names=ordered[0].boundary_tag_names,
    )
    candidate_index = protocol.candidates.index(candidate)
    if model_state_hash(model) != model_state_hash(expected_models[candidate_index]):
        raise ValueError("model differs from its registered fresh initialization")


def _require_uniform_query_shape(cases: tuple[ArchitectureCaseTensors, ...]) -> None:
    first_shape = cases[0].query_xy.shape
    if any(case.query_xy.shape != first_shape for case in cases[1:]):
        raise ValueError("evaluation batches require one shared vertex-query shape")


def _require_protocol(protocol: ArchitectureScreenProtocol) -> None:
    if type(protocol) is not ArchitectureScreenProtocol:
        raise TypeError("protocol must be an exact ArchitectureScreenProtocol")


def _checked_batch_target(
    value: torch.Tensor, name: str, *, trailing: tuple[int, ...]
) -> torch.Tensor:
    tensor = _owned_float64(value, name)
    if (
        tensor.ndim != 2 + len(trailing)
        or tuple(tensor.shape[-len(trailing) :]) != trailing
    ):
        raise ValueError(f"{name} has an invalid batch shape")
    return tensor


def _linear_quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile population must be non-empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


__all__ = (
    "ArchitectureCaseTensors",
    "ArchitectureScreenLoss",
    "CandidateTrainingSummary",
    "TrainingSchedule",
    "TrainingProgress",
    "architecture_screen_loss",
    "build_training_schedule",
    "calculate_architecture_screen_metrics",
    "evaluate_development_candidate",
    "evaluate_train_screen_candidate",
    "train_candidate_model",
)
