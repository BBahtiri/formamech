"""Immutable sampler, scheduler, and Adam resume state for S13."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import ClassVar

import torch

from piginot.contracts import hash_spec
from piginot.contracts.errors import ContractValidationError
from piginot.contracts.specs import SpecContract

_CANONICAL_NAMESPACE = "piginot.training"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_SAFE_INTEGER = 2**53 - 1


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ContractValidationError(f"{label} must be non-empty canonical text")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ContractValidationError(f"{label} must be a lowercase 256-bit digest")
    return value


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (1 if positive else 0)
        or value > _MAX_SAFE_INTEGER
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ContractValidationError(
            f"{label} must be a {qualifier} RFC 8785-safe integer"
        )
    return value


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "finite and strictly positive" if positive else "finite"
        raise ContractValidationError(f"{label} must be {qualifier}")
    return float(value)


@dataclass(frozen=True, slots=True)
class DeterministicSamplerPlan(SpecContract):
    """The complete finite batch schedule bound into a training run."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    sampler_id: str
    population_size: int
    batches: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _text(self.sampler_id, "sampler_id")
        population_size = _integer(
            self.population_size,
            "population_size",
            positive=True,
        )
        if type(self.batches) is not tuple or not self.batches:
            raise ContractValidationError("batches must be a non-empty tuple")
        batches: list[tuple[int, ...]] = []
        batch_size: int | None = None
        for batch in self.batches:
            if type(batch) is not tuple or not batch:
                raise ContractValidationError(
                    "batches must have the same positive batch size"
                )
            if batch_size is None:
                batch_size = len(batch)
            elif len(batch) != batch_size:
                raise ContractValidationError(
                    "batches must have the same positive batch size"
                )
            checked = tuple(_integer(index, "batch index") for index in batch)
            if any(index >= population_size for index in checked):
                raise ContractValidationError(
                    "batch index must be smaller than population_size"
                )
            if len(set(checked)) != len(checked):
                raise ContractValidationError(
                    "one batch must not repeat a population index"
                )
            batches.append(checked)
        object.__setattr__(self, "population_size", population_size)
        object.__setattr__(self, "batches", tuple(batches))

    @property
    def plan_hash(self) -> str:
        return hash_spec(self, "recipe")

    @property
    def initial_state(self) -> DeterministicSamplerState:
        return DeterministicSamplerState(self.plan_hash, 0)


@dataclass(frozen=True, slots=True)
class DeterministicSamplerState(SpecContract):
    """Exact cursor into one immutable deterministic sampler plan."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    plan_hash: str
    next_step: int

    def __post_init__(self) -> None:
        _digest(self.plan_hash, "plan_hash")
        _integer(self.next_step, "next_step")


def next_sampler_batch(
    plan: DeterministicSamplerPlan,
    state: DeterministicSamplerState,
) -> tuple[tuple[int, ...], DeterministicSamplerState]:
    """Return one registered batch and an advanced immutable cursor."""

    if type(plan) is not DeterministicSamplerPlan:
        raise TypeError("plan must be an exact DeterministicSamplerPlan")
    if type(state) is not DeterministicSamplerState:
        raise TypeError("state must be an exact DeterministicSamplerState")
    if state.plan_hash != plan.plan_hash:
        raise ContractValidationError("sampler state plan_hash does not match plan")
    if state.next_step >= len(plan.batches):
        raise ContractValidationError("deterministic sampler plan is exhausted")
    batch = plan.batches[state.next_step]
    return batch, DeterministicSamplerState(plan.plan_hash, state.next_step + 1)


@dataclass(frozen=True, slots=True)
class TrainingSchedulerState(SpecContract):
    """Exact finite state for the registered fit-loss scheduler."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    scheduler_id: str
    step: int
    learning_rate: float
    best_metric: float | None
    bad_observations: int
    cooldown_remaining: int
    observation_count: int

    def __post_init__(self) -> None:
        _text(self.scheduler_id, "scheduler_id")
        _integer(self.step, "step")
        object.__setattr__(
            self,
            "learning_rate",
            _finite(self.learning_rate, "learning_rate", positive=True),
        )
        if self.best_metric is not None:
            object.__setattr__(
                self,
                "best_metric",
                _finite(self.best_metric, "best_metric"),
            )
        _integer(self.bad_observations, "bad_observations")
        _integer(self.cooldown_remaining, "cooldown_remaining")
        _integer(self.observation_count, "observation_count")


def _names(values: object, label: str, *, allow_empty: bool) -> tuple[str, ...]:
    if type(values) is not tuple or (not allow_empty and not values):
        qualifier = "an immutable tuple" if allow_empty else "a non-empty tuple"
        raise ContractValidationError(f"{label} must be {qualifier}")
    names = tuple(_text(value, label) for value in values)
    if any("::" in name for name in names):
        raise ContractValidationError(f"{label} must not contain '::'")
    if len(set(names)) != len(names):
        raise ContractValidationError(f"{label} must not contain duplicates")
    return names


@dataclass(frozen=True, slots=True)
class AdamOptimizerState(SpecContract):
    """Named-parameter Adam structure and exact resumable group settings."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    parameter_names: tuple[str, ...]
    state_parameter_names: tuple[str, ...]
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool
    maximize: bool
    optimizer_id: str = "torch-adam-cpu-v1"

    def __post_init__(self) -> None:
        if self.optimizer_id != "torch-adam-cpu-v1":
            raise ContractValidationError("unsupported optimizer_id")
        parameter_names = _names(
            self.parameter_names,
            "parameter_names",
            allow_empty=False,
        )
        state_names = _names(
            self.state_parameter_names,
            "state_parameter_names",
            allow_empty=True,
        )
        if any(name not in parameter_names for name in state_names):
            raise ContractValidationError(
                "state_parameter_names must be a subset of parameter_names"
            )
        if (
            tuple(name for name in parameter_names if name in state_names)
            != state_names
        ):
            raise ContractValidationError(
                "state_parameter_names must preserve parameter_names order"
            )
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "state_parameter_names", state_names)
        object.__setattr__(
            self,
            "learning_rate",
            _finite(self.learning_rate, "learning_rate", positive=True),
        )
        if type(self.betas) is not tuple or len(self.betas) != 2:
            raise ContractValidationError("betas must be a two-value tuple")
        beta_values = tuple(_finite(value, "betas") for value in self.betas)
        if any(value < 0.0 or value >= 1.0 for value in beta_values):
            raise ContractValidationError("betas must lie in [0, 1)")
        object.__setattr__(self, "betas", beta_values)
        object.__setattr__(self, "eps", _finite(self.eps, "eps", positive=True))
        weight_decay = _finite(self.weight_decay, "weight_decay")
        if weight_decay < 0.0:
            raise ContractValidationError("weight_decay must be non-negative")
        object.__setattr__(self, "weight_decay", weight_decay)
        if type(self.amsgrad) is not bool:
            raise ContractValidationError("amsgrad must be boolean")
        if type(self.maximize) is not bool:
            raise ContractValidationError("maximize must be boolean")


def capture_adam_optimizer_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
) -> AdamOptimizerState:
    """Capture exact deterministic CPU Adam structure by parameter name."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if type(optimizer) is not torch.optim.Adam:
        raise TypeError("optimizer must be an exact torch.optim.Adam")
    named_parameters = tuple(model.named_parameters())
    if not named_parameters:
        raise ContractValidationError("model must contain trainable parameters")
    names = tuple(name for name, _ in named_parameters)
    parameters = tuple(parameter for _, parameter in named_parameters)
    if len(optimizer.param_groups) != 1:
        raise ContractValidationError("S13 Adam requires one parameter group")
    group = optimizer.param_groups[0]
    group_parameters = tuple(group["params"])
    if len(group_parameters) != len(parameters) or any(
        actual is not expected
        for actual, expected in zip(group_parameters, parameters, strict=True)
    ):
        raise ContractValidationError(
            "optimizer must contain the exact model parameters in named order"
        )
    if any(
        parameter.device.type != "cpu"
        or not parameter.is_floating_point()
        or parameter.layout is not torch.strided
        for parameter in parameters
    ):
        raise ContractValidationError(
            "S13 deterministic Adam parameters must be dense floating CPU tensors"
        )
    for flag in ("foreach", "capturable", "differentiable", "fused"):
        if group.get(flag) is not False:
            raise ContractValidationError(f"Adam {flag} must be explicitly false")
    if group.get("decoupled_weight_decay", False) is not False:
        raise ContractValidationError("Adam decoupled_weight_decay must be false")
    if group.get("maximize") is not False:
        raise ContractValidationError("Adam maximize must be false")
    state_names = tuple(
        name
        for name, parameter in named_parameters
        if bool(optimizer.state.get(parameter))
    )
    return AdamOptimizerState(
        parameter_names=names,
        state_parameter_names=state_names,
        learning_rate=group["lr"],
        betas=group["betas"],
        eps=group["eps"],
        weight_decay=group["weight_decay"],
        amsgrad=group["amsgrad"],
        maximize=group["maximize"],
    )


__all__ = (
    "AdamOptimizerState",
    "DeterministicSamplerPlan",
    "DeterministicSamplerState",
    "TrainingSchedulerState",
    "capture_adam_optimizer_state",
    "next_sampler_batch",
)
