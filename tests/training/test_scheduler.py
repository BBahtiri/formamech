from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from piginot.training.scheduler import (
    PlateauSchedulerConfig,
    advance_plateau_scheduler,
    initial_scheduler_state,
)


def test_plateau_scheduler_has_exact_deterministic_resume_state() -> None:
    config = PlateauSchedulerConfig(
        patience=2,
        reduction_factor=0.5,
        absolute_threshold=0.01,
        cooldown_steps=1,
        minimum_learning_rate=0.01,
    )
    state = initial_scheduler_state(config, learning_rate=0.08)

    state = advance_plateau_scheduler(config, state, metric=1.0)
    state = advance_plateau_scheduler(config, state, metric=0.995)
    state = advance_plateau_scheduler(config, state, metric=0.994)

    assert state.step == 3
    assert state.observation_count == 3
    assert state.best_metric == 1.0
    assert state.learning_rate == pytest.approx(0.04)
    assert state.bad_observations == 0
    assert state.cooldown_remaining == 1

    cooled = advance_plateau_scheduler(config, state, metric=0.993)
    assert cooled.step == 4
    assert cooled.learning_rate == pytest.approx(0.04)
    assert cooled.bad_observations == 0
    assert cooled.cooldown_remaining == 0

    improved = advance_plateau_scheduler(config, cooled, metric=0.97)
    assert improved.best_metric == pytest.approx(0.97)
    assert improved.bad_observations == 0


def test_scheduler_refuses_policy_or_state_drift() -> None:
    config = PlateauSchedulerConfig()
    state = initial_scheduler_state(config, learning_rate=1.0e-3)

    with pytest.raises(ValueError, match="scheduler_id"):
        advance_plateau_scheduler(
            config,
            replace(state, scheduler_id="foreign-scheduler"),
            metric=1.0,
        )
    with pytest.raises(ValueError, match="finite"):
        advance_plateau_scheduler(config, state, metric=float("nan"))
    with pytest.raises(FrozenInstanceError):
        config.patience = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    (
        {"patience": 0},
        {"reduction_factor": 0.0},
        {"reduction_factor": 1.0},
        {"absolute_threshold": -1.0},
        {"cooldown_steps": -1},
        {"minimum_learning_rate": 0.0},
    ),
)
def test_scheduler_policy_is_fail_closed(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        PlateauSchedulerConfig(**changes)  # type: ignore[arg-type]
