"""The S13b pilot is separate from, but matched to, frozen S13."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from campaigns.dogbone_v1.s13_protocol import (
    build_dogbone_s13_protocol,
    build_dogbone_s13_sampler_plan,
)
from campaigns.dogbone_v1.s13b_protocol import (
    DOGBONE_S13B_PILOT_ID,
    DogboneS13bPilotPlan,
    build_dogbone_s13b_pilot_plan,
    build_dogbone_s13b_sampler_plan,
    build_dogbone_s13b_trainer_config,
)
from piginot.contracts import hash_spec
from piginot.objectives import (
    EquilibriumForm,
    ObjectiveMode,
    ObjectiveRegistry,
    ObjectiveTerm,
)


def test_pilot_preserves_s13_data_split_guardrails_and_initialization() -> None:
    base = build_dogbone_s13_protocol()
    plan = build_dogbone_s13b_pilot_plan()

    assert type(plan) is DogboneS13bPilotPlan
    assert plan.pilot_id == DOGBONE_S13B_PILOT_ID
    assert base.protocol_hash == (
        "256cf8748489ea84ed3277a380d599eb7dc4227efb693204c2c99cd83d47cced"
    )
    assert plan.base_protocol == base
    assert plan.base_protocol.fit_case_ids == base.fit_case_ids
    assert plan.base_protocol.screen_case_ids == base.screen_case_ids
    assert plan.base_protocol.selection_policy == base.selection_policy
    assert plan.base_protocol.initial_trainer_weights_hash == (
        base.initial_trainer_weights_hash
    )
    assert plan.plan_hash == hash_spec(plan, "recipe")
    assert plan.plan_hash != base.protocol_hash


def test_pilot_has_one_opt_in_peak_arm_and_original_s13_control() -> None:
    plan = build_dogbone_s13b_pilot_plan()
    base = plan.base_protocol
    trial = plan.search_space.trials[0]
    config = build_dogbone_s13b_trainer_config(plan)

    assert plan.objective_registry.mode is ObjectiveMode.HYBRID
    assert plan.objective_registry.equilibrium_form is EquilibriumForm.STRONG
    assert ObjectiveTerm.DATA_PEAK in plan.objective_registry.required_terms
    assert plan.objective_registry.term_weights[ObjectiveTerm.DATA_PEAK] == 1.0
    assert hash_spec(plan.objective_registry, "recipe") != hash_spec(
        base.objective_registry, "recipe"
    )
    assert len(plan.search_space.trials) == 1
    assert (trial.learning_rate, trial.physics_weight_scale) == (1.0e-5, 1.0e-4)
    assert (trial.optimizer_steps, trial.case_batch_size) == (800, 4)
    assert config.trial == trial
    assert config.search_space == plan.search_space
    assert config.adaptive == base.adaptive
    assert config.scheduler == base.scheduler
    assert config.gradient_clip_norm == base.gradient_clip_norm
    assert plan.trainer_config == config
    assert plan.control_trial_id == base.search_space.trials[0].trial_id
    assert plan.control_checkpoint_step == 800
    assert plan.control_run_hash == (
        "8be4870d6323bb2880953d98195c6076d65c71760203faab0c59fd86c1f476e8"
    )
    assert plan.control_manifest_hash == (
        "dedb463a0a8eb490c4b413a911c0d02b9ac84625a506acd2054cdca9d493fcbd"
    )


def test_pilot_sampler_is_exact_s13_800_batch_prefix() -> None:
    plan = build_dogbone_s13b_pilot_plan()
    base_sampler = build_dogbone_s13_sampler_plan(plan.base_protocol)

    first = build_dogbone_s13b_sampler_plan(plan)
    second = build_dogbone_s13b_sampler_plan(plan)

    assert first == second
    assert first.plan_hash == second.plan_hash
    assert first.sampler_id == base_sampler.sampler_id
    assert first.population_size == 192
    assert first.batches == base_sampler.batches[:800]
    assert first.plan_hash != base_sampler.plan_hash
    assert plan.sampler_plan == first


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"pilot_id": "changed"}, "pilot_id"),
        ({"control_checkpoint_step": 799}, "control_checkpoint_step"),
        ({"control_trial_id": "trial-changed"}, "control_trial_id"),
        ({"control_run_hash": "0" * 64}, "control_run_hash"),
        ({"control_manifest_hash": "0" * 64}, "control_manifest_hash"),
        (
            {
                "objective_registry": ObjectiveRegistry(
                    mode=ObjectiveMode.HYBRID,
                    equilibrium_form=EquilibriumForm.STRONG,
                )
            },
            "objective_registry",
        ),
    ),
)
def test_pilot_rejects_identity_and_recipe_drift(
    changes: dict[str, object], message: str
) -> None:
    plan = build_dogbone_s13b_pilot_plan()

    with pytest.raises(ValueError, match=message):
        replace(plan, **changes)  # type: ignore[arg-type]


def test_pilot_rejects_upstream_or_search_space_drift() -> None:
    plan = build_dogbone_s13b_pilot_plan()

    with pytest.raises(ValueError, match="base_protocol"):
        replace(plan, base_protocol=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="search_space"):
        replace(
            plan,
            search_space=replace(plan.search_space, optimizer_steps=801),
        )


def test_pilot_builders_reject_subclass_at_execution_boundary() -> None:
    class PilotSubclass(DogboneS13bPilotPlan):
        pass

    pilot = build_dogbone_s13b_pilot_plan()
    subclass = PilotSubclass(
        **{field.name: getattr(pilot, field.name) for field in fields(pilot)}
    )

    with pytest.raises(TypeError, match="exact DogboneS13bPilotPlan"):
        build_dogbone_s13b_sampler_plan(subclass)
    with pytest.raises(TypeError, match="exact DogboneS13bPilotPlan"):
        build_dogbone_s13b_trainer_config(subclass)
