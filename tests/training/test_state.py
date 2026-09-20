from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest
import torch

from piginot.contracts import hash_spec
from piginot.contracts.errors import ContractValidationError
from piginot.training.state import (
    AdamOptimizerState,
    DeterministicSamplerPlan,
    DeterministicSamplerState,
    TrainingSchedulerState,
    capture_adam_optimizer_state,
    next_sampler_batch,
)


def _plan() -> DeterministicSamplerPlan:
    return DeterministicSamplerPlan(
        sampler_id="blake2b-batch-plan-v1",
        population_size=6,
        batches=((4, 1), (0, 5), (2, 3), (1, 4)),
    )


def test_sampler_plan_is_canonical_finite_and_exactly_resumable() -> None:
    plan = _plan()
    state = plan.initial_state

    assert plan.plan_hash == hash_spec(plan, "recipe")
    first, state = next_sampler_batch(plan, state)
    second, state = next_sampler_batch(plan, state)

    assert first == (4, 1)
    assert second == (0, 5)
    assert state == DeterministicSamplerState(plan.plan_hash, 2)
    with pytest.raises(FrozenInstanceError):
        state.next_step = 0  # type: ignore[misc]


def test_sampler_refuses_plan_drift_exhaustion_and_invalid_indices() -> None:
    plan = _plan()
    with pytest.raises(ContractValidationError, match="plan_hash"):
        next_sampler_batch(plan, DeterministicSamplerState("0" * 64, 0))
    with pytest.raises(ContractValidationError, match="exhausted"):
        next_sampler_batch(
            plan,
            DeterministicSamplerState(plan.plan_hash, len(plan.batches)),
        )
    with pytest.raises(ContractValidationError, match="population_size"):
        replace(plan, batches=((0, 6),))
    with pytest.raises(ContractValidationError, match="same positive batch size"):
        replace(plan, batches=((0,), (1, 2)))


def test_scheduler_state_carries_plateau_resume_counters() -> None:
    state = TrainingSchedulerState(
        scheduler_id="fit-loss-plateau-v1",
        step=25,
        learning_rate=1.0e-4,
        best_metric=0.25,
        bad_observations=3,
        cooldown_remaining=1,
        observation_count=7,
    )

    assert state.step == 25
    assert state.best_metric == 0.25
    with pytest.raises(ContractValidationError, match="best_metric"):
        replace(state, best_metric=float("nan"))
    with pytest.raises(ContractValidationError, match="learning_rate"):
        replace(state, learning_rate=0.0)


def test_adam_metadata_binds_named_parameters_and_exact_group_state() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.Tanh(),
        torch.nn.Linear(3, 1),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2.0e-4,
        betas=(0.8, 0.95),
        eps=1.0e-7,
        weight_decay=1.0e-5,
        foreach=False,
        fused=False,
    )
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    metadata = capture_adam_optimizer_state(model, optimizer)

    assert metadata.optimizer_id == "torch-adam-cpu-v1"
    assert metadata.parameter_names == tuple(
        name for name, _ in model.named_parameters()
    )
    assert metadata.state_parameter_names == metadata.parameter_names
    assert metadata.learning_rate == 2.0e-4
    assert metadata.betas == (0.8, 0.95)
    assert metadata.eps == 1.0e-7
    assert metadata.weight_decay == 1.0e-5
    assert metadata.amsgrad is False
    assert metadata.maximize is False


def test_adam_metadata_refuses_foreign_groups_and_nondeterministic_modes() -> None:
    model = torch.nn.Linear(2, 1)
    wrong_model = torch.nn.Linear(2, 1)
    foreign = torch.optim.Adam(wrong_model.parameters(), lr=1.0e-3)
    with pytest.raises(ContractValidationError, match="exact model parameters"):
        capture_adam_optimizer_state(model, foreign)

    groups: list[dict[str, Any]] = [
        {"params": [model.weight]},
        {"params": [model.bias]},
    ]
    grouped = torch.optim.Adam(
        groups,
        lr=1.0e-3,
    )
    with pytest.raises(ContractValidationError, match="one parameter group"):
        capture_adam_optimizer_state(model, grouped)

    maximizing = torch.optim.Adam(
        model.parameters(),
        lr=1.0e-3,
        maximize=True,
        foreach=False,
        fused=False,
    )
    with pytest.raises(ContractValidationError, match="maximize"):
        capture_adam_optimizer_state(model, maximizing)


@pytest.mark.parametrize(
    "changes",
    (
        {"parameter_names": ()},
        {"parameter_names": ("weight", "weight")},
        {"state_parameter_names": ("outside",)},
        {"learning_rate": float("inf")},
        {"betas": (0.9, 1.0)},
        {"eps": 0.0},
        {"weight_decay": -1.0},
    ),
)
def test_adam_metadata_is_fail_closed(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "parameter_names": ("weight", "bias"),
        "state_parameter_names": ("weight",),
        "learning_rate": 1.0e-3,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
    }
    values.update(changes)

    with pytest.raises(ContractValidationError):
        AdamOptimizerState(**values)  # type: ignore[arg-type]
