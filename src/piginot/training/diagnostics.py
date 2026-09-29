"""Read-only TRAIN diagnostics for composite S13 objective gradients."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from piginot.doe import EvidenceRole
from piginot.objectives import (
    EquilibriumForm,
    ObjectiveMode,
    ObjectiveRegistry,
    ObjectiveResiduals,
    ObjectiveTerm,
)
from piginot.training.adaptive import (
    ADAPTIVE_PHYSICS_TERMS,
    REFERENCE_PHYSICS_TERM,
    AdaptiveWeightConfig,
    AdaptiveWeightState,
)
from piginot.training.engine import TrainingPopulation

_DATA_TERMS = frozenset(
    (ObjectiveTerm.DATA_U, ObjectiveTerm.DATA_P, ObjectiveTerm.DATA_PEAK)
)


@dataclass(frozen=True, slots=True)
class TermGradientDiagnostic:
    """Loss and full-parameter gradient magnitudes for one objective term."""

    term: ObjectiveTerm
    unweighted_loss: float
    effective_weight: float
    weighted_loss: float
    unweighted_gradient_l2: float
    weighted_gradient_l2: float


@dataclass(frozen=True, slots=True)
class PairwiseGradientDiagnostic:
    """Pairwise relationship between two effective weighted gradients."""

    left_term: ObjectiveTerm
    right_term: ObjectiveTerm
    weighted_dot_product: float
    cosine: float | None
    conflict: bool


@dataclass(frozen=True, slots=True)
class AdaptiveTargetDiagnostic:
    """One exact gradient-ratio target before and after configured bounds."""

    term: ObjectiveTerm
    reference_statistic: float
    term_statistic: float
    raw_target: float
    bounded_target: float
    hit_minimum: bool
    hit_maximum: bool
    current_multiplier: float
    projected_multiplier: float


@dataclass(frozen=True, slots=True)
class ObjectiveGradientDiagnostic:
    """Immutable result of one read-only TRAIN objective replay."""

    role: EvidenceRole
    dataset_snapshot_hash: str
    public_manifest_hash: str
    sample_ids: tuple[str, ...]
    step: int
    physics_weight_scale: float
    terms: tuple[TermGradientDiagnostic, ...]
    pairs: tuple[PairwiseGradientDiagnostic, ...]
    adaptive_targets: tuple[AdaptiveTargetDiagnostic, ...]


def _validate_inputs(
    model: torch.nn.Module,
    *,
    population: TrainingPopulation,
    sample_ids: tuple[str, ...],
    step: int,
    base_registry: ObjectiveRegistry,
    physics_weight_scale: float,
    adaptive_config: AdaptiveWeightConfig,
    adaptive_state: AdaptiveWeightState,
    objective: Callable[[torch.nn.Module, tuple[str, ...], int], ObjectiveResiduals],
    role: EvidenceRole,
) -> tuple[torch.nn.Parameter, ...]:
    if type(role) is not EvidenceRole:
        raise TypeError("role must be an exact EvidenceRole")
    if role is not EvidenceRole.TRAIN:
        raise PermissionError(f"{role.value} evidence cannot enter TRAIN diagnostics")
    if type(population) is not TrainingPopulation:
        raise TypeError("population must be an exact TrainingPopulation")
    if population.role is not EvidenceRole.TRAIN:
        raise PermissionError("population must contain TRAIN evidence")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if type(sample_ids) is not tuple or not sample_ids:
        raise ValueError("sample_ids must be a non-empty tuple")
    if any(
        not isinstance(sample_id, str)
        or not sample_id
        or sample_id.strip() != sample_id
        for sample_id in sample_ids
    ) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be canonical and unique")
    if any(sample_id not in population.sample_ids for sample_id in sample_ids):
        raise ValueError("sample_ids must be contained in the TRAIN population")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer")
    if type(base_registry) is not ObjectiveRegistry:
        raise TypeError("base_registry must be an exact ObjectiveRegistry")
    if (
        base_registry.mode is not ObjectiveMode.HYBRID
        or base_registry.equilibrium_form is not EquilibriumForm.STRONG
    ):
        raise ValueError("gradient diagnostics require a strong HYBRID registry")
    if type(adaptive_config) is not AdaptiveWeightConfig:
        raise TypeError("adaptive_config must be exact AdaptiveWeightConfig")
    if type(adaptive_state) is not AdaptiveWeightState:
        raise TypeError("adaptive_state must be exact AdaptiveWeightState")
    if (
        isinstance(physics_weight_scale, bool)
        or not isinstance(physics_weight_scale, (int, float))
        or not math.isfinite(float(physics_weight_scale))
        or float(physics_weight_scale) <= 0.0
    ):
        raise ValueError("physics_weight_scale must be finite and positive")
    if not callable(objective):
        raise TypeError("objective must be callable")
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("model must contain trainable parameters")
    return parameters


def _effective_weights(
    registry: ObjectiveRegistry,
    physics_weight_scale: float,
    adaptive_state: AdaptiveWeightState,
) -> dict[ObjectiveTerm, float]:
    weights: dict[ObjectiveTerm, float] = {}
    for term in registry.required_terms:
        weight = registry.term_weights[term]
        if term not in _DATA_TERMS:
            weight *= physics_weight_scale
        if term in ADAPTIVE_PHYSICS_TERMS:
            weight *= adaptive_state.multipliers[term]
        weights[term] = weight
    return weights


def _term_gradients(
    losses: dict[ObjectiveTerm, torch.Tensor],
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[dict[ObjectiveTerm, torch.Tensor], dict[ObjectiveTerm, float]]:
    flattened: dict[ObjectiveTerm, torch.Tensor] = {}
    adaptive_statistics: dict[ObjectiveTerm, float] = {}
    for term, loss in losses.items():
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        connected = tuple(gradient for gradient in gradients if gradient is not None)
        if not connected:
            raise ValueError(f"{term.value} loss must use a trainable parameter")
        pieces = tuple(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
            for parameter, gradient in zip(parameters, gradients, strict=True)
        )
        vector = torch.cat(pieces)
        if not bool(torch.isfinite(vector).all().item()):
            raise ValueError(f"{term.value} gradient must be finite")
        flattened[term] = vector
        if term is REFERENCE_PHYSICS_TERM:
            adaptive_statistics[term] = max(
                float(gradient.detach().abs().max().item())
                for gradient in connected
            )
        elif term in ADAPTIVE_PHYSICS_TERMS:
            absolute_sum = sum(
                float(gradient.detach().abs().sum().item())
                for gradient in connected
            )
            element_count = sum(gradient.numel() for gradient in connected)
            adaptive_statistics[term] = absolute_sum / element_count
    return flattened, adaptive_statistics


def measure_objective_gradient_diagnostic(
    model: torch.nn.Module,
    *,
    population: TrainingPopulation,
    sample_ids: tuple[str, ...],
    step: int,
    base_registry: ObjectiveRegistry,
    physics_weight_scale: float,
    adaptive_config: AdaptiveWeightConfig,
    adaptive_state: AdaptiveWeightState,
    objective: Callable[[torch.nn.Module, tuple[str, ...], int], ObjectiveResiduals],
    role: EvidenceRole,
) -> ObjectiveGradientDiagnostic:
    """Replay one TRAIN batch without changing model or runtime state."""

    parameters = _validate_inputs(
        model,
        population=population,
        sample_ids=sample_ids,
        step=step,
        base_registry=base_registry,
        physics_weight_scale=physics_weight_scale,
        adaptive_config=adaptive_config,
        adaptive_state=adaptive_state,
        objective=objective,
        role=role,
    )
    checked_scale = float(physics_weight_scale)
    weights = _effective_weights(base_registry, checked_scale, adaptive_state)
    was_training = model.training
    saved_rng = torch.random.get_rng_state().clone()
    saved_gradients = tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    )
    saved_buffers = tuple(buffer.detach().clone() for buffer in model.buffers())
    try:
        model.train()
        residuals = objective(model, sample_ids, step)
        if type(residuals) is not ObjectiveResiduals:
            raise TypeError("objective must return exact ObjectiveResiduals")
        missing = tuple(
            term for term in base_registry.required_terms if term not in residuals.values
        )
        if missing:
            raise ValueError(
                "missing objective residual: "
                + ", ".join(term.value for term in missing)
            )
        losses = {
            term: residuals.values[term].square().mean()
            for term in base_registry.required_terms
        }
        if any(not bool(torch.isfinite(loss).item()) for loss in losses.values()):
            raise ValueError("objective losses must be finite")
        gradients, adaptive_statistics = _term_gradients(losses, parameters)

        terms = tuple(
            TermGradientDiagnostic(
                term=term,
                unweighted_loss=float(losses[term].detach().item()),
                effective_weight=weights[term],
                weighted_loss=float(losses[term].detach().item()) * weights[term],
                unweighted_gradient_l2=float(torch.linalg.vector_norm(gradients[term]).item()),
                weighted_gradient_l2=float(torch.linalg.vector_norm(gradients[term] * weights[term]).item()),
            )
            for term in base_registry.required_terms
        )
        weighted = {
            term: gradient * weights[term] for term, gradient in gradients.items()
        }
        ordered_terms = base_registry.required_terms
        pairs: list[PairwiseGradientDiagnostic] = []
        for left_index, left_term in enumerate(ordered_terms):
            left = weighted[left_term]
            left_norm = float(torch.linalg.vector_norm(left).item())
            for right_term in ordered_terms[left_index + 1 :]:
                right = weighted[right_term]
                right_norm = float(torch.linalg.vector_norm(right).item())
                dot = float(torch.dot(left, right).item())
                cosine = (
                    None
                    if left_norm == 0.0 or right_norm == 0.0
                    else dot / (left_norm * right_norm)
                )
                pairs.append(
                    PairwiseGradientDiagnostic(
                        left_term=left_term,
                        right_term=right_term,
                        weighted_dot_product=dot,
                        cosine=cosine,
                        conflict=cosine is not None and cosine < 0.0,
                    )
                )

        reference_statistic = adaptive_statistics[REFERENCE_PHYSICS_TERM]
        targets: list[AdaptiveTargetDiagnostic] = []
        for term in ADAPTIVE_PHYSICS_TERMS:
            term_statistic = adaptive_statistics[term]
            if term_statistic <= 0.0:
                raise ValueError(f"{term.value} adaptive gradient must be non-zero")
            raw_target = (
                base_registry.term_weights[REFERENCE_PHYSICS_TERM]
                * reference_statistic
                / (base_registry.term_weights[term] * term_statistic)
            )
            bounded_target = min(
                adaptive_config.maximum_multiplier,
                max(adaptive_config.minimum_multiplier, raw_target),
            )
            current = adaptive_state.multipliers[term]
            projected = (
                (1.0 - adaptive_config.beta) * current
                + adaptive_config.beta * bounded_target
            )
            targets.append(
                AdaptiveTargetDiagnostic(
                    term=term,
                    reference_statistic=reference_statistic,
                    term_statistic=term_statistic,
                    raw_target=raw_target,
                    bounded_target=bounded_target,
                    hit_minimum=raw_target < adaptive_config.minimum_multiplier,
                    hit_maximum=raw_target > adaptive_config.maximum_multiplier,
                    current_multiplier=current,
                    projected_multiplier=projected,
                )
            )
        return ObjectiveGradientDiagnostic(
            role=role,
            dataset_snapshot_hash=population.dataset_snapshot_hash,
            public_manifest_hash=population.public_manifest_hash,
            sample_ids=sample_ids,
            step=step,
            physics_weight_scale=checked_scale,
            terms=terms,
            pairs=tuple(pairs),
            adaptive_targets=tuple(targets),
        )
    finally:
        for parameter, gradient in zip(parameters, saved_gradients, strict=True):
            parameter.grad = gradient
        for buffer, saved in zip(model.buffers(), saved_buffers, strict=True):
            buffer.copy_(saved)
        model.train(was_training)
        torch.random.set_rng_state(saved_rng)


__all__ = (
    "AdaptiveTargetDiagnostic",
    "ObjectiveGradientDiagnostic",
    "PairwiseGradientDiagnostic",
    "TermGradientDiagnostic",
    "measure_objective_gradient_diagnostic",
)
