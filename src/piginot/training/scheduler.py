"""Pure deterministic fit-loss scheduler transitions for S13."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from piginot.training.state import TrainingSchedulerState

_SCHEDULER_ID = "fit-loss-plateau-v1"


def _integer(value: object, label: str, *, positive: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _finite(value: object, label: str, *, positive: bool) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return float(value)


@dataclass(frozen=True, slots=True)
class PlateauSchedulerConfig:
    """Pre-registered absolute-improvement plateau policy."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    patience: int = 100
    reduction_factor: float = 0.5
    absolute_threshold: float = 0.0
    cooldown_steps: int = 0
    minimum_learning_rate: float = 1.0e-8
    scheduler_id: str = _SCHEDULER_ID

    def __post_init__(self) -> None:
        if self.scheduler_id != _SCHEDULER_ID:
            raise ValueError(f"scheduler_id must be {_SCHEDULER_ID!r}")
        patience = _integer(self.patience, "patience", positive=True)
        factor = _finite(
            self.reduction_factor,
            "reduction_factor",
            positive=True,
        )
        if factor >= 1.0:
            raise ValueError("reduction_factor must be smaller than one")
        threshold = _finite(
            self.absolute_threshold,
            "absolute_threshold",
            positive=False,
        )
        if threshold < 0.0:
            raise ValueError("absolute_threshold must be non-negative")
        cooldown = _integer(
            self.cooldown_steps,
            "cooldown_steps",
            positive=False,
        )
        minimum = _finite(
            self.minimum_learning_rate,
            "minimum_learning_rate",
            positive=True,
        )
        object.__setattr__(self, "patience", patience)
        object.__setattr__(self, "reduction_factor", factor)
        object.__setattr__(self, "absolute_threshold", threshold)
        object.__setattr__(self, "cooldown_steps", cooldown)
        object.__setattr__(self, "minimum_learning_rate", minimum)


def initial_scheduler_state(
    config: PlateauSchedulerConfig,
    *,
    learning_rate: float,
) -> TrainingSchedulerState:
    """Create the exact zero-observation state for a registered policy."""

    if type(config) is not PlateauSchedulerConfig:
        raise TypeError("config must be an exact PlateauSchedulerConfig")
    rate = _finite(learning_rate, "learning_rate", positive=True)
    if rate < config.minimum_learning_rate:
        raise ValueError("learning_rate must not be below minimum_learning_rate")
    return TrainingSchedulerState(
        scheduler_id=config.scheduler_id,
        step=0,
        learning_rate=rate,
        best_metric=None,
        bad_observations=0,
        cooldown_remaining=0,
        observation_count=0,
    )


def advance_plateau_scheduler(
    config: PlateauSchedulerConfig,
    state: TrainingSchedulerState,
    *,
    metric: float,
) -> TrainingSchedulerState:
    """Observe one finite fit loss and return the next immutable state."""

    if type(config) is not PlateauSchedulerConfig:
        raise TypeError("config must be an exact PlateauSchedulerConfig")
    if type(state) is not TrainingSchedulerState:
        raise TypeError("state must be an exact TrainingSchedulerState")
    if state.scheduler_id != config.scheduler_id:
        raise ValueError("scheduler_id does not match the registered policy")
    if state.step != state.observation_count:
        raise ValueError("scheduler step and observation_count must match")
    if state.learning_rate < config.minimum_learning_rate:
        raise ValueError("scheduler learning_rate is below the registered minimum")
    if state.cooldown_remaining > config.cooldown_steps:
        raise ValueError("scheduler cooldown exceeds the registered policy")
    observed = _finite(metric, "metric", positive=False)

    best = state.best_metric
    improved = best is None or observed < best - config.absolute_threshold
    bad_observations = state.bad_observations
    cooldown = state.cooldown_remaining
    learning_rate = state.learning_rate
    if improved:
        best = observed
        bad_observations = 0
    elif cooldown > 0:
        cooldown -= 1
        bad_observations = 0
    else:
        bad_observations += 1
        if bad_observations >= config.patience:
            learning_rate = max(
                config.minimum_learning_rate,
                learning_rate * config.reduction_factor,
            )
            bad_observations = 0
            cooldown = config.cooldown_steps

    return TrainingSchedulerState(
        scheduler_id=config.scheduler_id,
        step=state.step + 1,
        learning_rate=learning_rate,
        best_metric=best,
        bad_observations=bad_observations,
        cooldown_remaining=cooldown,
        observation_count=state.observation_count + 1,
    )


__all__ = (
    "PlateauSchedulerConfig",
    "advance_plateau_scheduler",
    "initial_scheduler_state",
)
