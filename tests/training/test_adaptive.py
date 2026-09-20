from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import cast

import pytest
import torch

from piginot.contracts import canonical_json_bytes
from piginot.objectives.models import (
    EquilibriumForm,
    ObjectiveMode,
    ObjectiveRegistry,
    ObjectiveTerm,
)
from piginot.training.adaptive import (
    ADAPTIVE_PHYSICS_TERMS,
    AdaptiveWeightConfig,
    AdaptiveWeightState,
    GradientStatistic,
    measure_unweighted_gradient_statistics,
    update_adaptive_weights,
)


def _registry() -> ObjectiveRegistry:
    return ObjectiveRegistry(
        mode=ObjectiveMode.HYBRID,
        equilibrium_form=EquilibriumForm.STRONG,
        term_weights={
            ObjectiveTerm.DATA_U: 1.0,
            ObjectiveTerm.DATA_P: 2.0,
            ObjectiveTerm.EQUILIBRIUM_STRONG: 100.0,
            ObjectiveTerm.CONSTITUTIVE: 20.0,
            ObjectiveTerm.ANGULAR: 1.0,
            ObjectiveTerm.DIRICHLET: 20.0,
            ObjectiveTerm.TRACTION_FREE: 20.0,
        },
    )


def test_adaptive_weights_follow_dogbone_gradient_ratio_ema_with_base_weights() -> None:
    registry = _registry()
    state = AdaptiveWeightState.initial()
    gradients = {
        ObjectiveTerm.EQUILIBRIUM_STRONG: GradientStatistic(10.0),
        ObjectiveTerm.CONSTITUTIVE: GradientStatistic(5.0),
        ObjectiveTerm.ANGULAR: GradientStatistic(100.0),
        ObjectiveTerm.DIRICHLET: GradientStatistic(1.0),
        ObjectiveTerm.TRACTION_FREE: GradientStatistic(0.5),
    }

    update = update_adaptive_weights(
        registry,
        state,
        gradients,
        config=AdaptiveWeightConfig(beta=0.1, update_period=100, warmup_steps=200),
        step=200,
    )

    assert update.state.update_count == 1
    assert update.state.multipliers[ObjectiveTerm.CONSTITUTIVE] == pytest.approx(1.9)
    assert update.state.multipliers[ObjectiveTerm.ANGULAR] == pytest.approx(1.9)
    assert update.state.multipliers[ObjectiveTerm.DIRICHLET] == pytest.approx(1.9)
    assert update.state.multipliers[ObjectiveTerm.TRACTION_FREE] == pytest.approx(1.9)
    assert update.registry.term_weights[ObjectiveTerm.DATA_U] == pytest.approx(1.0)
    assert update.registry.term_weights[ObjectiveTerm.DATA_P] == pytest.approx(2.0)
    assert update.registry.term_weights[ObjectiveTerm.CONSTITUTIVE] == pytest.approx(
        38.0
    )


def test_nonclamped_target_uses_each_terms_base_weight_without_compounding() -> None:
    base_registry = _registry()
    gradients = {
        ObjectiveTerm.EQUILIBRIUM_STRONG: GradientStatistic(10.0),
        ObjectiveTerm.CONSTITUTIVE: GradientStatistic(10.0),
        ObjectiveTerm.ANGULAR: GradientStatistic(100.0),
        ObjectiveTerm.DIRICHLET: GradientStatistic(10.0),
        ObjectiveTerm.TRACTION_FREE: GradientStatistic(10.0),
    }

    first = update_adaptive_weights(
        base_registry,
        AdaptiveWeightState.initial(),
        gradients,
        config=AdaptiveWeightConfig(),
        step=200,
    )
    second = update_adaptive_weights(
        base_registry,
        first.state,
        gradients,
        config=AdaptiveWeightConfig(),
        step=300,
    )

    # (100 * 10) / (20 * 10) = 5, safely below the 10x upper clamp.
    assert first.state.multipliers[ObjectiveTerm.CONSTITUTIVE] == pytest.approx(1.4)
    assert first.registry.term_weights[ObjectiveTerm.CONSTITUTIVE] == pytest.approx(
        28.0
    )
    assert second.state.multipliers[ObjectiveTerm.CONSTITUTIVE] == pytest.approx(1.76)
    assert second.registry.term_weights[ObjectiveTerm.CONSTITUTIVE] == pytest.approx(
        35.2
    )
    assert base_registry.term_weights[ObjectiveTerm.CONSTITUTIVE] == 20.0


def test_target_is_lower_clamped_before_the_ema() -> None:
    gradients = {
        ObjectiveTerm.EQUILIBRIUM_STRONG: GradientStatistic(1.0),
        ObjectiveTerm.CONSTITUTIVE: GradientStatistic(1.0),
        ObjectiveTerm.ANGULAR: GradientStatistic(2000.0),
        ObjectiveTerm.DIRICHLET: GradientStatistic(1.0),
        ObjectiveTerm.TRACTION_FREE: GradientStatistic(1.0),
    }

    update = update_adaptive_weights(
        _registry(),
        AdaptiveWeightState.initial(),
        gradients,
        config=AdaptiveWeightConfig(),
        step=200,
    )

    # Raw angular target is 0.05, so beta=.1 moves 1.0 toward clamped 0.1.
    assert update.state.multipliers[ObjectiveTerm.ANGULAR] == pytest.approx(0.91)


def test_adaptive_weights_do_not_update_before_warmup_or_off_period() -> None:
    registry = _registry()
    state = AdaptiveWeightState.initial(multipliers={ObjectiveTerm.CONSTITUTIVE: 2.0})

    skipped = update_adaptive_weights(
        registry,
        state,
        {},
        config=AdaptiveWeightConfig(),
        step=199,
    )
    off_period = update_adaptive_weights(
        registry,
        state,
        {},
        config=AdaptiveWeightConfig(),
        step=250,
    )

    assert skipped.state == state
    assert off_period.state == state
    assert skipped.registry.term_weights[ObjectiveTerm.CONSTITUTIVE] == pytest.approx(
        40.0
    )


def test_schedule_is_relative_to_warmup_not_to_step_zero() -> None:
    gradients = {
        term: GradientStatistic(1.0)
        for term in (ObjectiveTerm.EQUILIBRIUM_STRONG, *ADAPTIVE_PHYSICS_TERMS)
    }
    update = update_adaptive_weights(
        _registry(),
        AdaptiveWeightState.initial(),
        gradients,
        config=AdaptiveWeightConfig(warmup_steps=250, update_period=100),
        step=250,
    )

    assert update.state.update_count == 1
    assert update.state.last_update_step == 250


def test_adaptation_rejects_weak_or_data_only_registries() -> None:
    weak = ObjectiveRegistry(
        mode=ObjectiveMode.HYBRID,
        equilibrium_form=EquilibriumForm.WEAK,
    )
    data_only = ObjectiveRegistry(
        mode=ObjectiveMode.DATA_ONLY,
        equilibrium_form=EquilibriumForm.STRONG,
    )

    with pytest.raises(ValueError, match="strong equilibrium"):
        update_adaptive_weights(
            weak,
            AdaptiveWeightState.initial(),
            {},
            config=AdaptiveWeightConfig(),
            step=0,
        )
    with pytest.raises(ValueError, match="every registered"):
        update_adaptive_weights(
            data_only,
            AdaptiveWeightState.initial(),
            {},
            config=AdaptiveWeightConfig(),
            step=0,
        )


@pytest.mark.parametrize(
    "bad_gradients",
    (
        {},
        {ObjectiveTerm.EQUILIBRIUM_STRONG: GradientStatistic(1.0)},
        {
            ObjectiveTerm.EQUILIBRIUM_STRONG: GradientStatistic(0.0),
            ObjectiveTerm.CONSTITUTIVE: GradientStatistic(1.0),
            ObjectiveTerm.ANGULAR: GradientStatistic(1.0),
            ObjectiveTerm.DIRICHLET: GradientStatistic(1.0),
            ObjectiveTerm.TRACTION_FREE: GradientStatistic(1.0),
        },
    ),
)
def test_adaptive_weights_fail_closed_on_missing_or_zero_gradients(
    bad_gradients: dict[ObjectiveTerm, GradientStatistic],
) -> None:
    with pytest.raises(ValueError):
        update_adaptive_weights(
            _registry(),
            AdaptiveWeightState.initial(),
            bad_gradients,
            config=AdaptiveWeightConfig(),
            step=200,
        )


def test_adaptive_contracts_are_frozen() -> None:
    state = AdaptiveWeightState.initial()

    with pytest.raises(FrozenInstanceError):
        state.update_count = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        state.multipliers[ObjectiveTerm.ANGULAR] = 2.0  # type: ignore[index]

    assert isinstance(state.multipliers, MappingProxyType)
    assert canonical_json_bytes(state).startswith(
        b'{"$type":"piginot.training.AdaptiveWeightState"'
    )
    with pytest.raises(TypeError, match="ObjectiveTerm"):
        AdaptiveWeightState(
            multipliers=cast(
                Mapping[ObjectiveTerm, float],
                {term.value: 1.0 for term in ADAPTIVE_PHYSICS_TERMS},
            ),
        )


def test_gradient_statistics_use_reference_max_and_adapted_term_mean() -> None:
    parameter = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    losses = {
        ObjectiveTerm.EQUILIBRIUM_STRONG: parameter.square().sum(),
        ObjectiveTerm.CONSTITUTIVE: parameter.sum(),
        ObjectiveTerm.ANGULAR: (2.0 * parameter).sum(),
        ObjectiveTerm.DIRICHLET: (3.0 * parameter).sum(),
        ObjectiveTerm.TRACTION_FREE: (4.0 * parameter).sum(),
    }

    statistics = measure_unweighted_gradient_statistics(losses, (parameter,))

    assert statistics[ObjectiveTerm.EQUILIBRIUM_STRONG].magnitude == 4.0
    assert statistics[ObjectiveTerm.CONSTITUTIVE].magnitude == 1.0
    assert statistics[ObjectiveTerm.ANGULAR].magnitude == 2.0
    assert statistics[ObjectiveTerm.DIRICHLET].magnitude == 3.0
    assert statistics[ObjectiveTerm.TRACTION_FREE].magnitude == 4.0
    assert parameter.grad is None


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        (lambda parameter: parameter.detach().sum(), "attached"),
        (lambda parameter: parameter.square(), "scalar"),
        (lambda parameter: parameter.sum() * float("nan"), "finite"),
        (lambda parameter: (parameter * 0.0).sum(), "non-zero"),
    ),
)
def test_gradient_measurement_fails_closed_on_invalid_losses(
    replacement: Callable[[torch.Tensor], torch.Tensor],
    message: str,
) -> None:
    parameter = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    losses = {
        term: parameter.square().sum()
        for term in (ObjectiveTerm.EQUILIBRIUM_STRONG, *ADAPTIVE_PHYSICS_TERMS)
    }
    losses[ObjectiveTerm.ANGULAR] = replacement(parameter)

    with pytest.raises(ValueError, match=message):
        measure_unweighted_gradient_statistics(losses, (parameter,))


def test_gradient_measurement_rejects_missing_and_disconnected_terms() -> None:
    parameter = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    losses = {
        term: parameter.square().sum()
        for term in (ObjectiveTerm.EQUILIBRIUM_STRONG, *ADAPTIVE_PHYSICS_TERMS)
    }
    del losses[ObjectiveTerm.DIRICHLET]
    with pytest.raises(ValueError, match="missing unweighted loss"):
        measure_unweighted_gradient_statistics(losses, (parameter,))

    losses[ObjectiveTerm.DIRICHLET] = parameter.square().sum()
    losses[ObjectiveTerm.ANGULAR] = torch.tensor(
        1.0, dtype=torch.float64, requires_grad=True
    )
    with pytest.raises(ValueError, match="at least one trainable parameter"):
        measure_unweighted_gradient_statistics(losses, (parameter,))


def test_gradient_measurement_ignores_validly_disconnected_output_head() -> None:
    used = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    unused = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    losses = {
        term: used.square().sum()
        for term in (ObjectiveTerm.EQUILIBRIUM_STRONG, *ADAPTIVE_PHYSICS_TERMS)
    }

    statistics = measure_unweighted_gradient_statistics(losses, (used, unused))

    assert statistics[ObjectiveTerm.EQUILIBRIUM_STRONG].magnitude == 2.0
    assert statistics[ObjectiveTerm.CONSTITUTIVE].magnitude == 2.0


def test_gradient_measurement_rejects_nonfinite_gradient() -> None:
    parameter = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    losses = {
        term: parameter.square().sum()
        for term in (ObjectiveTerm.EQUILIBRIUM_STRONG, *ADAPTIVE_PHYSICS_TERMS)
    }
    losses[ObjectiveTerm.EQUILIBRIUM_STRONG] = parameter.sqrt().sum()

    with pytest.raises(ValueError, match="finite gradient"):
        measure_unweighted_gradient_statistics(losses, (parameter,))


@pytest.mark.parametrize(
    "changes",
    (
        {"beta": 0.0},
        {"beta": 1.1},
        {"update_period": 0},
        {"warmup_steps": -1},
        {"minimum_multiplier": 0.0},
        {"minimum_multiplier": 2.0, "maximum_multiplier": 1.0},
    ),
)
def test_adaptive_config_rejects_invalid_policy(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        AdaptiveWeightConfig(**changes)  # type: ignore[arg-type]


def test_resume_state_tracks_exact_updates_and_rejects_repeated_step() -> None:
    gradients = {
        term: GradientStatistic(1.0)
        for term in (ObjectiveTerm.EQUILIBRIUM_STRONG, *ADAPTIVE_PHYSICS_TERMS)
    }
    first = update_adaptive_weights(
        _registry(),
        AdaptiveWeightState.initial(),
        gradients,
        config=AdaptiveWeightConfig(),
        step=200,
    )

    assert first.state.update_count == 1
    assert first.state.last_update_step == 200
    with pytest.raises(ValueError, match="after last_update_step"):
        update_adaptive_weights(
            _registry(),
            first.state,
            gradients,
            config=AdaptiveWeightConfig(),
            step=200,
        )
