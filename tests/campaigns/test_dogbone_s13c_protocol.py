from __future__ import annotations

from collections import Counter
from dataclasses import fields, replace

import pytest
import torch

from campaigns.dogbone_v1.s13c_protocol import (
    DOGBONE_S13C_CONDITIONING_PARAMETERS,
    DOGBONE_S13C_PROTOCOL_ID,
    DogboneS13cProtocol,
    build_dogbone_s13c_protocol,
    build_dogbone_s13c_sampler_plan,
    build_dogbone_s13c_trainer_config,
    build_initial_operator,
    energy_config,
    load_s13c_normalization,
    operator_config,
    schedule,
    search_space,
)
from campaigns.dogbone_v1.s13c_trial import load_committed_s13c_protocol
from piginot.objectives import PotentialEnergyConfig
from piginot.operators import ENERGY_ARCHITECTURE_ID, energy_module_digest
from piginot.training import (
    GridSearchSpace,
    SelectionGuardrails,
    SelectionPolicy,
    model_state_hash,
)


@pytest.fixture(scope="module")
def protocol() -> DogboneS13cProtocol:
    return build_dogbone_s13c_protocol()


def test_protocol_matches_the_committed_preregistration(
    protocol: DogboneS13cProtocol,
) -> None:
    committed = load_committed_s13c_protocol()

    assert committed == protocol
    assert committed.to_bytes() == protocol.to_bytes()
    assert protocol.protocol_id == DOGBONE_S13C_PROTOCOL_ID
    assert protocol.architecture_id == ENERGY_ARCHITECTURE_ID
    assert len(protocol.protocol_hash) == 64


def test_protocol_reuses_the_ratified_s10_s11_population(
    protocol: DogboneS13cProtocol,
) -> None:
    normalization = load_s13c_normalization()

    assert len(protocol.fit_case_ids) == 192
    assert len(protocol.screen_case_ids) == 64
    assert not set(protocol.fit_case_ids) & set(protocol.screen_case_ids)
    assert protocol.normalization_hash == normalization.content_hash
    assert protocol.objective.conditioning_parameter_names == (
        DOGBONE_S13C_CONDITIONING_PARAMETERS
    )
    assert all(
        name in normalization.parameter_names
        for name in DOGBONE_S13C_CONDITIONING_PARAMETERS
    )


def test_protocol_keeps_the_frozen_s13_guardrails(
    protocol: DogboneS13cProtocol,
) -> None:
    guardrails = protocol.selection_policy.guardrails
    metrics = protocol.s11_baseline_metrics

    assert guardrails == SelectionGuardrails()
    assert guardrails.incumbent_median_absolute_peak_error == (
        metrics.median_absolute_peak_error
    )
    assert guardrails.incumbent_p_relative_l2 == metrics.median_p_relative_l2
    assert protocol.selection_policy.registered_trial_ids == (
        protocol.search_space.trial_ids
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"protocol_id": "other"},
        {"normalization_hash": "0" * 64},
        {"architecture_id": "pi-ginot-s11-boundary-cross-attention-v1"},
        {"architecture_config": replace(operator_config(), decoder_layers=5)},
        {
            "search_space": GridSearchSpace(
                learning_rates=(3.0e-4, 1.0e-3),
                physics_weight_scales=(1.0,),
                optimizer_steps=2400,
                case_batch_size=4,
                maximum_trials=2,
            )
        },
        {"schedule": replace(schedule(), constant_steps=1500)},
        {"gradient_clip_norm": 2.0},
        {"training_seed": 1},
        {"execution_dtype": "float64"},
        {"execution_blas_threads": 8},
    ],
)
def test_protocol_rejects_any_unregistered_change(
    protocol: DogboneS13cProtocol, changes: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        replace(protocol, **changes)


def test_protocol_rejects_relaxed_selection(protocol: DogboneS13cProtocol) -> None:
    relaxed = object.__new__(SelectionGuardrails)
    for field in fields(SelectionGuardrails):
        object.__setattr__(
            relaxed, field.name, getattr(SelectionGuardrails(), field.name)
        )
    object.__setattr__(relaxed, "maximum_field_relative_regression", 0.5)
    policy = object.__new__(SelectionPolicy)
    object.__setattr__(policy, "search_space", protocol.search_space)
    object.__setattr__(policy, "guardrails", relaxed)
    with pytest.raises(ValueError, match="frozen S13 guardrails"):
        replace(protocol, selection_policy=policy)


def test_single_trial_matches_the_enhanced_recipe(
    protocol: DogboneS13cProtocol,
) -> None:
    config = build_dogbone_s13c_trainer_config(protocol)

    assert protocol.search_space == search_space()
    assert config.trial.learning_rate == 3.0e-4
    assert config.trial.optimizer_steps == 2400
    assert config.trial.case_batch_size == 4
    assert config.schedule.learning_rate(1599) == 3.0e-4
    assert config.schedule.learning_rate(2400) == pytest.approx(3.0e-6)
    assert config.gradient_clip_norm == 1.0
    assert energy_config() == PotentialEnergyConfig()


def test_sampler_plan_visits_every_fit_case_once_per_epoch(
    protocol: DogboneS13cProtocol,
) -> None:
    plan = build_dogbone_s13c_sampler_plan(protocol)

    assert len(plan.batches) == 2400
    assert all(len(batch) == 4 for batch in plan.batches)
    first_epoch = [index for batch in plan.batches[:48] for index in batch]
    assert sorted(first_epoch) == list(range(192))
    counts = Counter(index for batch in plan.batches for index in batch)
    assert set(counts.values()) == {50}
    assert plan == build_dogbone_s13c_sampler_plan(protocol)


def test_initial_operator_is_deterministic_and_preserves_global_rng(
    protocol: DogboneS13cProtocol,
) -> None:
    before = torch.random.get_rng_state()
    first = build_initial_operator()
    after = torch.random.get_rng_state()
    second = build_initial_operator()

    assert torch.equal(before, after)
    assert model_state_hash(first) == model_state_hash(second)
    assert model_state_hash(first) == protocol.initial_weights_hash
    assert energy_module_digest(first) == protocol.architecture_module_digest
    assert all(parameter.dtype is torch.float32 for parameter in first.parameters())


def test_builders_require_the_exact_protocol() -> None:
    with pytest.raises(TypeError):
        build_dogbone_s13c_sampler_plan(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_dogbone_s13c_trainer_config(object())  # type: ignore[arg-type]


def test_protocol_pins_initial_weights_independently_of_local_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import campaigns.dogbone_v1.s13c_protocol as protocol_module

    original = protocol_module.build_initial_operator

    def perturbed() -> torch.nn.Module:
        model = original()
        with torch.no_grad():
            model.output.weight.add_(1.0e-7)
        return model

    monkeypatch.setattr(protocol_module, "build_initial_operator", perturbed)
    rebuilt = protocol_module.build_dogbone_s13c_protocol()

    assert rebuilt.initial_weights_hash == protocol_module.INITIAL_WEIGHTS_HASH
    assert rebuilt == load_committed_s13c_protocol()
    with pytest.raises(ValueError, match="initial_weights_hash"):
        replace(rebuilt, initial_weights_hash="0" * 64)
