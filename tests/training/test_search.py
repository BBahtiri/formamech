from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from piginot.contracts import hash_spec
from piginot.contracts.errors import ContractValidationError
from piginot.training.search import (
    GridSearchSpace,
    OutOfSearchSpaceError,
    TrialConfig,
)


def _space() -> GridSearchSpace:
    return GridSearchSpace(
        learning_rates=(5e-5, 1e-4),
        physics_weight_scales=(0.01, 0.1, 1.0),
        optimizer_steps=400,
        case_batch_size=4,
        maximum_trials=6,
    )


def test_grid_search_enumerates_a_stable_bounded_cartesian_product() -> None:
    trials = _space().trials

    assert tuple(
        (trial.learning_rate, trial.physics_weight_scale) for trial in trials
    ) == (
        (5e-5, 0.01),
        (5e-5, 0.1),
        (5e-5, 1.0),
        (1e-4, 0.01),
        (1e-4, 0.1),
        (1e-4, 1.0),
    )
    assert len({trial.trial_id for trial in trials}) == 6
    assert len({trial.trial_hash for trial in trials}) == 6
    assert all(trial.trial_id == f"trial-{trial.trial_hash}" for trial in trials)
    assert trials[0].trial_hash == hash_spec(trials[0], "content")
    assert _space().trials == trials
    assert _space().search_space_hash == hash_spec(_space(), "recipe")


def test_trial_identity_changes_with_every_trial_field() -> None:
    baseline = _space().trials[0]

    for change in (
        {"learning_rate": 7.5e-5},
        {"physics_weight_scale": 0.2},
        {"optimizer_steps": 401},
        {"case_batch_size": 8},
    ):
        assert replace(baseline, **change).trial_hash != baseline.trial_hash


def test_grid_search_authorizes_only_exact_registered_trials() -> None:
    space = _space()
    requested = TrialConfig(
        learning_rate=1e-4,
        physics_weight_scale=0.1,
        optimizer_steps=400,
        case_batch_size=4,
    )

    authorized = space.authorize(requested)

    assert authorized == space.trials[4]
    for outside in (
        replace(requested, learning_rate=2e-4),
        replace(requested, physics_weight_scale=0.5),
        replace(requested, optimizer_steps=401),
        replace(requested, case_batch_size=8),
    ):
        with pytest.raises(OutOfSearchSpaceError, match="outside the preregistered"):
            space.authorize(outside)
    with pytest.raises(TypeError, match="exact TrialConfig"):
        space.authorize(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("learning_rate", 0.0),
        ("learning_rate", -1.0),
        ("learning_rate", float("nan")),
        ("learning_rate", float("inf")),
        ("learning_rate", True),
        ("learning_rate", "0.001"),
        ("physics_weight_scale", 0.0),
        ("physics_weight_scale", float("nan")),
        ("physics_weight_scale", False),
        ("optimizer_steps", 0),
        ("optimizer_steps", 4.0),
        ("optimizer_steps", True),
        ("optimizer_steps", 2**53),
        ("case_batch_size", 0),
        ("case_batch_size", 4.0),
        ("case_batch_size", False),
        ("case_batch_size", 2**53),
    ),
)
def test_trial_config_rejects_noncanonical_values(field: str, value: object) -> None:
    values: dict[str, object] = {
        "learning_rate": 1e-4,
        "physics_weight_scale": 0.1,
        "optimizer_steps": 400,
        "case_batch_size": 4,
    }
    values[field] = value

    with pytest.raises(ContractValidationError, match=field):
        TrialConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    (
        {"learning_rates": [5e-5, 1e-4]},
        {"learning_rates": (1e-4, 5e-5)},
        {"learning_rates": (1e-4, 1e-4)},
        {"learning_rates": ()},
        {"learning_rates": (True,)},
        {"learning_rates": (float("nan"),)},
        {"physics_weight_scales": [0.1, 1.0]},
        {"physics_weight_scales": (1.0, 0.1)},
        {"physics_weight_scales": (0.1, 0.1)},
        {"physics_weight_scales": ()},
        {"physics_weight_scales": (False,)},
        {"optimizer_steps": True},
        {"case_batch_size": 0},
        {"maximum_trials": 0},
        {"maximum_trials": 6.0},
        {"maximum_trials": 2**53},
    ),
)
def test_grid_rejects_mutable_unsorted_duplicate_or_invalid_axes(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "learning_rates": (5e-5, 1e-4),
        "physics_weight_scales": (0.1, 1.0),
        "optimizer_steps": 400,
        "case_batch_size": 4,
        "maximum_trials": 4,
    }
    values.update(changes)

    with pytest.raises(ContractValidationError):
        GridSearchSpace(**values)  # type: ignore[arg-type]


def test_grid_refuses_a_cartesian_product_above_the_preregistered_cap() -> None:
    with pytest.raises(OutOfSearchSpaceError, match="above cap"):
        GridSearchSpace(
            learning_rates=(5e-5, 1e-4),
            physics_weight_scales=(0.01, 0.1, 1.0),
            optimizer_steps=400,
            case_batch_size=4,
            maximum_trials=5,
        )


def test_search_contracts_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        _space().maximum_trials = 12  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        _space().trials[0].learning_rate = 0.2  # type: ignore[misc]
