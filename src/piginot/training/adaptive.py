"""Deterministic gradient-ratio EMA weights for S13 physics objectives."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar

import torch

from piginot.objectives.models import (
    EquilibriumForm,
    ObjectiveRegistry,
    ObjectiveTerm,
)

REFERENCE_PHYSICS_TERM = ObjectiveTerm.EQUILIBRIUM_STRONG
ADAPTIVE_PHYSICS_TERMS = (
    ObjectiveTerm.CONSTITUTIVE,
    ObjectiveTerm.ANGULAR,
    ObjectiveTerm.DIRICHLET,
    ObjectiveTerm.TRACTION_FREE,
)
_REQUIRED_GRADIENT_TERMS = (REFERENCE_PHYSICS_TERM, *ADAPTIVE_PHYSICS_TERMS)
_CANONICAL_NAMESPACE = "piginot.training"


def _finite_number(value: object, *, label: str, positive: bool) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        condition = "finite and strictly positive" if positive else "finite"
        raise ValueError(f"{label} must be {condition}")
    return float(value)


def _nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class AdaptiveWeightConfig:
    """Pre-registered schedule and bounds for gradient-ratio adaptation."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    beta: float = 0.1
    update_period: int = 100
    warmup_steps: int = 200
    minimum_multiplier: float = 0.1
    maximum_multiplier: float = 10.0

    def __post_init__(self) -> None:
        beta = _finite_number(self.beta, label="beta", positive=True)
        if beta > 1.0:
            raise ValueError("beta must be at most 1")
        update_period = _nonnegative_integer(self.update_period, label="update_period")
        if update_period == 0:
            raise ValueError("update_period must be strictly positive")
        warmup_steps = _nonnegative_integer(self.warmup_steps, label="warmup_steps")
        minimum = _finite_number(
            self.minimum_multiplier,
            label="minimum_multiplier",
            positive=True,
        )
        maximum = _finite_number(
            self.maximum_multiplier,
            label="maximum_multiplier",
            positive=True,
        )
        if minimum > maximum:
            raise ValueError("multiplier bounds must be ordered")
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "update_period", update_period)
        object.__setattr__(self, "warmup_steps", warmup_steps)
        object.__setattr__(self, "minimum_multiplier", minimum)
        object.__setattr__(self, "maximum_multiplier", maximum)

    def is_update_step(self, step: int) -> bool:
        """Return whether ``step`` is on the pre-registered update schedule."""

        checked_step = _nonnegative_integer(step, label="step")
        return (
            checked_step >= self.warmup_steps
            and (checked_step - self.warmup_steps) % self.update_period == 0
        )


@dataclass(frozen=True, slots=True)
class GradientStatistic:
    """One finite gradient magnitude computed from an unweighted term loss."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    magnitude: float

    def __post_init__(self) -> None:
        magnitude = _finite_number(
            self.magnitude,
            label="gradient magnitude",
            positive=False,
        )
        if magnitude < 0.0:
            raise ValueError("gradient magnitude must be non-negative")
        object.__setattr__(self, "magnitude", magnitude)


def _initial_multipliers() -> dict[ObjectiveTerm, float]:
    return {term: 1.0 for term in ADAPTIVE_PHYSICS_TERMS}


@dataclass(frozen=True, slots=True)
class AdaptiveWeightState:
    """Exact resumable EMA multipliers; mappings are defensively frozen."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    multipliers: Mapping[ObjectiveTerm, float] = field(
        default_factory=_initial_multipliers
    )
    update_count: int = 0
    last_update_step: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.multipliers, Mapping):
            raise TypeError("multipliers must be a mapping")
        if any(type(term) is not ObjectiveTerm for term in self.multipliers):
            raise TypeError("multiplier keys must be ObjectiveTerm values")
        if set(self.multipliers) != set(ADAPTIVE_PHYSICS_TERMS):
            raise ValueError(
                "multipliers must contain exactly the adaptive physics terms"
            )
        copied: dict[ObjectiveTerm, float] = {}
        for term in ADAPTIVE_PHYSICS_TERMS:
            copied[term] = _finite_number(
                self.multipliers[term],
                label=f"{term.value} multiplier",
                positive=True,
            )
        update_count = _nonnegative_integer(self.update_count, label="update_count")
        if self.last_update_step is None:
            if update_count != 0:
                raise ValueError(
                    "last_update_step is required when update_count is non-zero"
                )
            last_update_step = None
        else:
            last_update_step = _nonnegative_integer(
                self.last_update_step, label="last_update_step"
            )
            if update_count == 0:
                raise ValueError(
                    "update_count must be non-zero when last_update_step is set"
                )
        object.__setattr__(self, "multipliers", MappingProxyType(copied))
        object.__setattr__(self, "update_count", update_count)
        object.__setattr__(self, "last_update_step", last_update_step)

    @classmethod
    def initial(
        cls,
        *,
        multipliers: Mapping[ObjectiveTerm, float] | None = None,
    ) -> AdaptiveWeightState:
        """Create an initial state with optional explicit multiplier overrides."""

        merged = _initial_multipliers()
        if multipliers is not None:
            if not isinstance(multipliers, Mapping):
                raise TypeError("multipliers must be a mapping")
            for term, multiplier in multipliers.items():
                if (
                    type(term) is not ObjectiveTerm
                    or term not in ADAPTIVE_PHYSICS_TERMS
                ):
                    raise ValueError(
                        "initial overrides must name adaptive physics terms"
                    )
                merged[term] = multiplier
        return cls(multipliers=merged)


@dataclass(frozen=True, slots=True)
class AdaptiveWeightUpdate:
    """The new resumable state and immutable weighted objective registry."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    state: AdaptiveWeightState
    registry: ObjectiveRegistry

    def __post_init__(self) -> None:
        if type(self.state) is not AdaptiveWeightState:
            raise TypeError("state must be an exact AdaptiveWeightState")
        if type(self.registry) is not ObjectiveRegistry:
            raise TypeError("registry must be an exact ObjectiveRegistry")


def _validate_strong_registry(registry: ObjectiveRegistry) -> None:
    if type(registry) is not ObjectiveRegistry:
        raise TypeError("registry must be an exact ObjectiveRegistry")
    if registry.equilibrium_form is not EquilibriumForm.STRONG:
        raise ValueError("adaptive weighting requires strong equilibrium")
    required = set(registry.required_terms)
    if any(term not in required for term in _REQUIRED_GRADIENT_TERMS):
        raise ValueError(
            "adaptive weighting requires every registered strong-form physics term"
        )


def _registry_with_state(
    registry: ObjectiveRegistry,
    state: AdaptiveWeightState,
) -> ObjectiveRegistry:
    _validate_strong_registry(registry)
    if type(state) is not AdaptiveWeightState:
        raise TypeError("state must be an exact AdaptiveWeightState")
    weights = dict(registry.term_weights)
    for term in ADAPTIVE_PHYSICS_TERMS:
        weights[term] = registry.term_weights[term] * state.multipliers[term]
    return ObjectiveRegistry(
        mode=registry.mode,
        equilibrium_form=registry.equilibrium_form,
        term_weights=weights,
    )


def measure_unweighted_gradient_statistics(
    unweighted_losses: Mapping[ObjectiveTerm, torch.Tensor],
    parameters: Iterable[torch.Tensor],
) -> Mapping[ObjectiveTerm, GradientStatistic]:
    """Measure max-reference/mean-term gradients without changing ``.grad``.

    The equilibrium reference follows the DogBone implementation's maximum
    absolute gradient. Each adapted term uses its mean absolute gradient. The
    graph is retained so the caller can subsequently backpropagate the weighted
    total loss.
    """

    if not isinstance(unweighted_losses, Mapping):
        raise TypeError("unweighted_losses must be a mapping")
    for term in unweighted_losses:
        if type(term) is not ObjectiveTerm:
            raise TypeError("unweighted loss keys must be ObjectiveTerm values")
    missing = tuple(
        term for term in _REQUIRED_GRADIENT_TERMS if term not in unweighted_losses
    )
    if missing:
        raise ValueError(
            "missing unweighted loss: " + ", ".join(term.value for term in missing)
        )

    parameter_tuple = tuple(parameters)
    if not parameter_tuple:
        raise ValueError("parameters must be non-empty")
    if len({id(parameter) for parameter in parameter_tuple}) != len(parameter_tuple):
        raise ValueError("parameters must not contain duplicate tensors")
    for parameter in parameter_tuple:
        if not isinstance(parameter, torch.Tensor):
            raise TypeError("parameters must contain torch.Tensor values")
        if not parameter.requires_grad:
            raise ValueError("every parameter must have requires_grad=True")
        if not parameter.is_floating_point() or parameter.layout is not torch.strided:
            raise ValueError("parameters must be dense floating tensors")

    measured: dict[ObjectiveTerm, GradientStatistic] = {}
    for term in _REQUIRED_GRADIENT_TERMS:
        loss = unweighted_losses[term]
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"{term.value} loss must be a torch.Tensor")
        if loss.layout is not torch.strided or not loss.is_floating_point():
            raise ValueError(f"{term.value} loss must be a dense floating tensor")
        if loss.ndim != 0:
            raise ValueError(f"{term.value} loss must be scalar")
        if not bool(torch.isfinite(loss).item()):
            raise ValueError(f"{term.value} loss must be finite")
        if not loss.requires_grad:
            raise ValueError(f"{term.value} loss must be attached to the graph")
        try:
            gradients = torch.autograd.grad(
                loss,
                parameter_tuple,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
        except RuntimeError as exc:
            raise ValueError(f"{term.value} gradient measurement failed") from exc
        connected = tuple(gradient for gradient in gradients if gradient is not None)
        if not connected:
            raise ValueError(
                f"{term.value} loss must use at least one trainable parameter"
            )
        if any(
            gradient.layout is not torch.strided
            or not gradient.is_floating_point()
            or not bool(torch.isfinite(gradient).all().item())
            for gradient in connected
        ):
            raise ValueError(f"{term.value} must have a finite gradient")

        if term is REFERENCE_PHYSICS_TERM:
            magnitude = max(
                float(gradient.detach().abs().max().item()) for gradient in connected
            )
        else:
            absolute_sum = sum(
                float(gradient.detach().abs().sum().item()) for gradient in connected
            )
            element_count = sum(gradient.numel() for gradient in connected)
            magnitude = absolute_sum / element_count
        if magnitude <= 0.0:
            raise ValueError(f"{term.value} must have a non-zero gradient")
        measured[term] = GradientStatistic(magnitude)
    return MappingProxyType(measured)


def update_adaptive_weights(
    base_registry: ObjectiveRegistry,
    state: AdaptiveWeightState,
    gradients: Mapping[ObjectiveTerm, GradientStatistic],
    *,
    config: AdaptiveWeightConfig,
    step: int,
) -> AdaptiveWeightUpdate:
    """Apply EMA against the immutable, pre-registered base registry.

    The weighted registry returned from this function is an output. It must not
    be passed back as ``base_registry``; the separate state carries adaptation.
    Gradient statistics are consumed and validated only on scheduled update
    steps, so callers may pass an empty mapping during warmup and between them.
    """

    _validate_strong_registry(base_registry)
    if type(state) is not AdaptiveWeightState:
        raise TypeError("state must be an exact AdaptiveWeightState")
    if type(config) is not AdaptiveWeightConfig:
        raise TypeError("config must be an exact AdaptiveWeightConfig")
    checked_step = _nonnegative_integer(step, label="step")
    if state.last_update_step is not None and checked_step <= state.last_update_step:
        raise ValueError("step must be after last_update_step")
    if not config.is_update_step(checked_step):
        return AdaptiveWeightUpdate(
            state,
            _registry_with_state(base_registry, state),
        )

    if not isinstance(gradients, Mapping):
        raise TypeError("gradients must be a mapping")
    if any(type(term) is not ObjectiveTerm for term in gradients):
        raise TypeError("gradient keys must be ObjectiveTerm values")
    if set(gradients) != set(_REQUIRED_GRADIENT_TERMS):
        raise ValueError("gradients must contain exactly the adaptive gradient terms")
    magnitudes: dict[ObjectiveTerm, float] = {}
    for term in _REQUIRED_GRADIENT_TERMS:
        statistic = gradients[term]
        if type(statistic) is not GradientStatistic:
            raise TypeError("gradient values must be exact GradientStatistic values")
        if statistic.magnitude <= 0.0:
            raise ValueError(f"{term.value} gradient must be non-zero")
        magnitudes[term] = statistic.magnitude

    weighted_reference = (
        base_registry.term_weights[REFERENCE_PHYSICS_TERM]
        * magnitudes[REFERENCE_PHYSICS_TERM]
    )
    multipliers: dict[ObjectiveTerm, float] = {}
    for term in ADAPTIVE_PHYSICS_TERMS:
        target = weighted_reference / (
            base_registry.term_weights[term] * magnitudes[term]
        )
        bounded_target = min(
            config.maximum_multiplier,
            max(config.minimum_multiplier, target),
        )
        previous = state.multipliers[term]
        multipliers[term] = (
            1.0 - config.beta
        ) * previous + config.beta * bounded_target

    updated_state = AdaptiveWeightState(
        multipliers=multipliers,
        update_count=state.update_count + 1,
        last_update_step=checked_step,
    )
    return AdaptiveWeightUpdate(
        updated_state,
        _registry_with_state(base_registry, updated_state),
    )


__all__ = (
    "ADAPTIVE_PHYSICS_TERMS",
    "REFERENCE_PHYSICS_TERM",
    "AdaptiveWeightConfig",
    "AdaptiveWeightState",
    "AdaptiveWeightUpdate",
    "GradientStatistic",
    "measure_unweighted_gradient_statistics",
    "update_adaptive_weights",
)
