"""Separate TRAIN-only matched pilot for S13 root-peak preservation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from campaigns.dogbone_v1.s13_protocol import (
    DogboneS13Protocol,
    build_dogbone_s13_protocol,
    build_dogbone_s13_sampler_plan,
)
from piginot.contracts import hash_spec
from piginot.objectives import (
    EquilibriumForm,
    ObjectiveMode,
    ObjectiveRegistry,
    ObjectiveTerm,
)
from piginot.training import (
    DeterministicSamplerPlan,
    DeterministicTrainerConfig,
    GridSearchSpace,
)

DOGBONE_S13B_PILOT_ID = "dogbone-s13b-peak-preservation-pilot-v1"
_BASE_PROTOCOL_HASH = "256cf8748489ea84ed3277a380d599eb7dc4227efb693204c2c99cd83d47cced"
_CONTROL_RUN_HASH = "8be4870d6323bb2880953d98195c6076d65c71760203faab0c59fd86c1f476e8"
_CONTROL_MANIFEST_HASH = (
    "dedb463a0a8eb490c4b413a911c0d02b9ac84625a506acd2054cdca9d493fcbd"
)
_PILOT_STEPS = 800


def _search_space() -> GridSearchSpace:
    return GridSearchSpace(
        learning_rates=(1.0e-5,),
        physics_weight_scales=(1.0e-4,),
        optimizer_steps=_PILOT_STEPS,
        case_batch_size=4,
        maximum_trials=1,
    )


def _objective_registry() -> ObjectiveRegistry:
    return ObjectiveRegistry(
        mode=ObjectiveMode.HYBRID,
        equilibrium_form=EquilibriumForm.STRONG,
        term_weights={ObjectiveTerm.DATA_PEAK: 1.0},
    )


@dataclass(frozen=True, slots=True)
class DogboneS13bPilotPlan:
    """One peak-loss arm matched to the original S13 trial-0 step-800 control."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    pilot_id: str
    base_protocol: DogboneS13Protocol
    search_space: GridSearchSpace
    objective_registry: ObjectiveRegistry
    control_trial_id: str
    control_checkpoint_step: int
    control_run_hash: str
    control_manifest_hash: str

    def __post_init__(self) -> None:
        if self.pilot_id != DOGBONE_S13B_PILOT_ID:
            raise ValueError("pilot_id differs from the registered S13b pilot")
        if type(self.base_protocol) is not DogboneS13Protocol or (
            self.base_protocol.protocol_hash != _BASE_PROTOCOL_HASH
        ):
            raise ValueError("base_protocol differs from the frozen S13 protocol")
        if type(self.search_space) is not GridSearchSpace or (
            self.search_space != _search_space()
        ):
            raise ValueError("search_space differs from the one-arm 800-step pilot")
        if type(self.objective_registry) is not ObjectiveRegistry or (
            self.objective_registry != _objective_registry()
        ):
            raise ValueError("objective_registry differs from the peak-loss recipe")
        if self.control_trial_id != self.base_protocol.search_space.trials[0].trial_id:
            raise ValueError("control_trial_id must name original S13 trial 0")
        if type(self.control_checkpoint_step) is not int or (
            self.control_checkpoint_step != _PILOT_STEPS
        ):
            raise ValueError("control_checkpoint_step must be 800")
        if self.control_run_hash != _CONTROL_RUN_HASH:
            raise ValueError("control_run_hash differs from the pinned S13 run")
        if self.control_manifest_hash != _CONTROL_MANIFEST_HASH:
            raise ValueError(
                "control_manifest_hash differs from the pinned S13 checkpoint"
            )

    @property
    def plan_hash(self) -> str:
        return hash_spec(self, "recipe")

    @property
    def sampler_plan(self) -> DeterministicSamplerPlan:
        return build_dogbone_s13b_sampler_plan(self)

    @property
    def trainer_config(self) -> DeterministicTrainerConfig:
        return build_dogbone_s13b_trainer_config(self)


def build_dogbone_s13b_pilot_plan() -> DogboneS13bPilotPlan:
    """Bind one opt-in peak arm to frozen S13 evidence without altering it."""

    base = build_dogbone_s13_protocol()
    return DogboneS13bPilotPlan(
        pilot_id=DOGBONE_S13B_PILOT_ID,
        base_protocol=base,
        search_space=_search_space(),
        objective_registry=_objective_registry(),
        control_trial_id=base.search_space.trials[0].trial_id,
        control_checkpoint_step=_PILOT_STEPS,
        control_run_hash=_CONTROL_RUN_HASH,
        control_manifest_hash=_CONTROL_MANIFEST_HASH,
    )


def build_dogbone_s13b_sampler_plan(
    pilot: DogboneS13bPilotPlan,
) -> DeterministicSamplerPlan:
    """Use exactly the first 800 batches of S13's frozen 1,500-step plan."""

    if type(pilot) is not DogboneS13bPilotPlan:
        raise TypeError("pilot must be exact DogboneS13bPilotPlan")
    base = build_dogbone_s13_sampler_plan(pilot.base_protocol)
    return DeterministicSamplerPlan(
        sampler_id=base.sampler_id,
        population_size=base.population_size,
        batches=base.batches[: pilot.search_space.optimizer_steps],
    )


def build_dogbone_s13b_trainer_config(
    pilot: DogboneS13bPilotPlan,
) -> DeterministicTrainerConfig:
    """Use S13's optimizer schedule for the single 800-step pilot arm."""

    if type(pilot) is not DogboneS13bPilotPlan:
        raise TypeError("pilot must be exact DogboneS13bPilotPlan")
    return DeterministicTrainerConfig(
        search_space=pilot.search_space,
        trial=pilot.search_space.trials[0],
        adaptive=pilot.base_protocol.adaptive,
        scheduler=pilot.base_protocol.scheduler,
        gradient_clip_norm=pilot.base_protocol.gradient_clip_norm,
    )


__all__ = (
    "DOGBONE_S13B_PILOT_ID",
    "DogboneS13bPilotPlan",
    "build_dogbone_s13b_pilot_plan",
    "build_dogbone_s13b_sampler_plan",
    "build_dogbone_s13b_trainer_config",
)
