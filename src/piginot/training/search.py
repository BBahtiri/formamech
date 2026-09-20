"""Bounded, preregistered S13 hyperparameter search contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from piginot.contracts import hash_spec
from piginot.contracts.errors import ContractValidationError

_CANONICAL_NAMESPACE = "piginot.training"
_MAX_SAFE_INTEGER = 2**53 - 1


class OutOfSearchSpaceError(ContractValidationError):
    """A requested trial is outside the preregistered S13 search space."""


def _positive_finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ContractValidationError(f"{label} must be finite and strictly positive")
    return float(value)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{label} must be a strictly positive integer")
    if value <= 0 or value > _MAX_SAFE_INTEGER:
        raise ContractValidationError(
            f"{label} must be a strictly positive RFC 8785-safe integer"
        )
    return value


def _strictly_increasing_positive(values: object, label: str) -> tuple[float, ...]:
    if type(values) is not tuple:
        raise ContractValidationError(f"{label} must be an immutable tuple")
    converted = tuple(_positive_finite(value, label) for value in values)
    if not converted:
        raise ContractValidationError(f"{label} must be non-empty")
    if tuple(sorted(converted)) != converted or len(set(converted)) != len(converted):
        raise ContractValidationError(f"{label} must be strictly increasing and unique")
    return converted


@dataclass(frozen=True, slots=True)
class TrialConfig:
    """One exact candidate configuration inside the S13 search grid."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    learning_rate: float
    physics_weight_scale: float
    optimizer_steps: int
    case_batch_size: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "learning_rate", _positive_finite(self.learning_rate, "learning_rate")
        )
        object.__setattr__(
            self,
            "physics_weight_scale",
            _positive_finite(self.physics_weight_scale, "physics_weight_scale"),
        )
        object.__setattr__(
            self,
            "optimizer_steps",
            _positive_int(self.optimizer_steps, "optimizer_steps"),
        )
        object.__setattr__(
            self,
            "case_batch_size",
            _positive_int(self.case_batch_size, "case_batch_size"),
        )

    @property
    def trial_hash(self) -> str:
        """Return the canonical content identity of this exact trial."""

        return hash_spec(self, "content")

    @property
    def trial_id(self) -> str:
        """Return the deterministic identifier derived from trial content."""

        return f"trial-{self.trial_hash}"


@dataclass(frozen=True, slots=True)
class GridSearchSpace:
    """Finite Cartesian search space that refuses non-preregistered trials."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    learning_rates: tuple[float, ...]
    physics_weight_scales: tuple[float, ...]
    optimizer_steps: int
    case_batch_size: int
    maximum_trials: int

    def __post_init__(self) -> None:
        learning_rates = _strictly_increasing_positive(
            self.learning_rates, "learning_rates"
        )
        physics_weight_scales = _strictly_increasing_positive(
            self.physics_weight_scales, "physics_weight_scales"
        )
        optimizer_steps = _positive_int(self.optimizer_steps, "optimizer_steps")
        case_batch_size = _positive_int(self.case_batch_size, "case_batch_size")
        maximum_trials = _positive_int(self.maximum_trials, "maximum_trials")
        trial_count = len(learning_rates) * len(physics_weight_scales)
        if trial_count > maximum_trials:
            raise OutOfSearchSpaceError(
                f"search space contains {trial_count} trials, above cap {maximum_trials}"
            )
        object.__setattr__(self, "learning_rates", learning_rates)
        object.__setattr__(self, "physics_weight_scales", physics_weight_scales)
        object.__setattr__(self, "optimizer_steps", optimizer_steps)
        object.__setattr__(self, "case_batch_size", case_batch_size)
        object.__setattr__(self, "maximum_trials", maximum_trials)

    @property
    def trials(self) -> tuple[TrialConfig, ...]:
        """Enumerate candidates in stable Cartesian order."""

        return tuple(
            TrialConfig(
                learning_rate=learning_rate,
                physics_weight_scale=physics_weight_scale,
                optimizer_steps=self.optimizer_steps,
                case_batch_size=self.case_batch_size,
            )
            for learning_rate in self.learning_rates
            for physics_weight_scale in self.physics_weight_scales
        )

    @property
    def trial_ids(self) -> tuple[str, ...]:
        """Return every registered trial identifier in enumeration order."""

        return tuple(trial.trial_id for trial in self.trials)

    @property
    def search_space_hash(self) -> str:
        """Bind axes, fixed settings, and cap into one recipe identity."""

        return hash_spec(self, "recipe")

    def authorize(self, config: TrialConfig) -> TrialConfig:
        """Return the exact registered trial or fail closed if it is outside."""

        if type(config) is not TrialConfig:
            raise TypeError("config must be an exact TrialConfig")
        for trial in self.trials:
            if trial == config:
                return trial
        raise OutOfSearchSpaceError(
            f"{config.trial_id} is outside the preregistered S13 search space"
        )


def enumerate_trials(space: GridSearchSpace) -> tuple[TrialConfig, ...]:
    """Functional form of the stable Cartesian search-space enumeration."""

    if type(space) is not GridSearchSpace:
        raise TypeError("space must be an exact GridSearchSpace")
    return space.trials


def require_trial_in_space(
    requested: TrialConfig,
    space: GridSearchSpace,
) -> TrialConfig:
    """Functional form of the preregistered-trial authorization boundary."""

    if type(space) is not GridSearchSpace:
        raise TypeError("space must be an exact GridSearchSpace")
    return space.authorize(requested)


__all__ = (
    "GridSearchSpace",
    "OutOfSearchSpaceError",
    "TrialConfig",
    "enumerate_trials",
    "require_trial_in_space",
)
