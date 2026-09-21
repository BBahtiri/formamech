from __future__ import annotations

from dataclasses import fields, replace

import pytest

from campaigns.dogbone_v1.s11_contingency import BASELINE_CANDIDATE_ID
from campaigns.dogbone_v1.s11_contingency_run import load_committed_protocol
from campaigns.dogbone_v1.s13_initialization import (
    load_selected_s11_checkpoint_evidence,
)
from campaigns.dogbone_v1.s13_protocol import (
    DOGBONE_S13_PROTOCOL_ID,
    DOGBONE_S13_SAMPLER_POLICY_ID,
    DogboneS13Protocol,
    _build_dogbone_s13_protocol,
    build_dogbone_s13_protocol,
    build_dogbone_s13_sampler_plan,
    build_dogbone_s13_trainer_configs,
)
from piginot.contracts import canonical_json_bytes
from piginot.training import SelectionGuardrails


def test_protocol_freezes_the_bounded_train_only_search() -> None:
    protocol = build_dogbone_s13_protocol()

    assert type(protocol) is DogboneS13Protocol
    assert protocol.protocol_id == DOGBONE_S13_PROTOCOL_ID
    assert len(protocol.fit_case_ids) == 192
    assert len(protocol.screen_case_ids) == 64
    assert set(protocol.fit_case_ids).isdisjoint(protocol.screen_case_ids)
    assert protocol.selected_candidate_id == BASELINE_CANDIDATE_ID
    assert protocol.search_space.learning_rates == (1.0e-5, 3.0e-5)
    assert protocol.search_space.physics_weight_scales == (1.0e-4, 1.0e-3, 1.0e-2)
    assert protocol.search_space.optimizer_steps == 1_500
    assert protocol.search_space.case_batch_size == 4
    assert protocol.search_space.maximum_trials == 6
    assert len(protocol.search_space.trials) == 6
    assert protocol.selection_policy.search_space == protocol.search_space
    assert protocol.training_seed == 13_001
    assert protocol.sampler_seed == 13_002
    assert protocol.sampler_policy_id == DOGBONE_S13_SAMPLER_POLICY_ID
    assert protocol.initial_model_state_hash == (
        "51b83fd2468d66d5556b75286feb08a13b2e9ca16d0b7f4cdcb83ff26eaba906"
    )
    assert protocol.initial_trainer_weights_hash == (
        "da58cb840fa5401a75474b7dca7f4a46f47c42032920d1331987b9fff48dbd7a"
    )
    assert protocol.initial_trainer_weights_hash != protocol.initial_model_state_hash
    assert len(protocol.protocol_hash) == 64
    assert canonical_json_bytes(protocol) == protocol.to_bytes()


def test_protocol_incumbent_guardrails_are_provenanced_to_s11_evidence() -> None:
    protocol = build_dogbone_s13_protocol()
    incumbent = protocol.s11_baseline_metrics
    guardrails = protocol.selection_policy.guardrails

    assert type(guardrails) is SelectionGuardrails
    assert guardrails.incumbent_median_absolute_peak_error == (
        incumbent.median_absolute_peak_error
    )
    assert guardrails.incumbent_p90_absolute_peak_error == (
        incumbent.p90_absolute_peak_error
    )
    assert guardrails.incumbent_median_signed_peak_error == (
        incumbent.median_signed_peak_error
    )
    assert guardrails.incumbent_u_relative_l2 == incumbent.median_u_relative_l2
    assert guardrails.incumbent_p_relative_l2 == incumbent.median_p_relative_l2


def test_sampler_plan_is_deterministic_and_epoch_balanced() -> None:
    protocol = build_dogbone_s13_protocol()

    first = build_dogbone_s13_sampler_plan(protocol)
    second = build_dogbone_s13_sampler_plan(protocol)

    assert first == second
    assert first.plan_hash == second.plan_hash
    assert first.sampler_id == f"{DOGBONE_S13_SAMPLER_POLICY_ID}:{13_002}"
    assert first.population_size == 192
    assert len(first.batches) == 1_500
    assert {len(batch) for batch in first.batches} == {4}
    assert sorted(index for batch in first.batches[:48] for index in batch) == list(
        range(192)
    )
    assert sorted(index for batch in first.batches[48:96] for index in batch) == list(
        range(192)
    )
    assert first.batches[:48] != first.batches[48:96]


def test_every_trainer_config_is_registered_and_uses_shared_policies() -> None:
    protocol = build_dogbone_s13_protocol()

    configs = build_dogbone_s13_trainer_configs(protocol)

    assert tuple(config.trial for config in configs) == protocol.search_space.trials
    assert len({config.config_hash for config in configs}) == 6
    assert all(config.search_space == protocol.search_space for config in configs)
    assert all(config.adaptive == protocol.adaptive for config in configs)
    assert all(config.scheduler == protocol.scheduler for config in configs)
    assert all(config.gradient_clip_norm == 1.0 for config in configs)


def test_protocol_refuses_upstream_identity_drift() -> None:
    s11_protocol = load_committed_protocol()
    evidence = load_selected_s11_checkpoint_evidence()
    changed = replace(
        evidence,
        source_bindings_hash="0" * 64,
    )

    with pytest.raises(ValueError, match="source bindings"):
        _build_dogbone_s13_protocol(
            s11_protocol,
            changed,
        )


def test_protocol_rejects_subclasses_at_execution_boundaries() -> None:
    class ProtocolSubclass(DogboneS13Protocol):
        pass

    protocol = build_dogbone_s13_protocol()
    subclass = ProtocolSubclass(
        **{
            field.name: getattr(protocol, field.name)
            for field in fields(DogboneS13Protocol)
        }
    )

    with pytest.raises(TypeError, match="exact DogboneS13Protocol"):
        build_dogbone_s13_sampler_plan(subclass)
