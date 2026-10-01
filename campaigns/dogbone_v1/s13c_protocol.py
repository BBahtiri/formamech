"""Frozen TRAIN-only protocol for the DogBone S13c energy-form operator.

S13c replaces the S11 supervised two-head operator with the energy-form
PI-GINOT (``pi-ginot-enhanced`` commit ``12c9c1c``): exact quarter-model
Dirichlet lifting, parameter-token FiLM cross-attention decoder, potential
energy objective, and constitutive served stress. The S10 TRAIN population,
the S11 parameter normalization and the frozen S13 selection guardrails are
reused unchanged; DEVELOPMENT and HOLDOUT remain outside every code path.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import ClassVar, Final

import torch

from campaigns.dogbone_v1.s11_contingency import (
    BASELINE_CANDIDATE_ID,
    ArchitectureScreenMetrics,
)
from campaigns.dogbone_v1.s11_contingency_data import CaseNormalizationStatistics
from campaigns.dogbone_v1.s11_contingency_run import load_committed_protocol
from campaigns.dogbone_v1.s13_initialization import (
    load_selected_s11_checkpoint_evidence,
)
from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.objectives import PotentialEnergyConfig
from piginot.operators import (
    ENERGY_ARCHITECTURE_ID,
    EnergyFormOperator,
    EnergyOperatorConfig,
    energy_module_digest,
)
from piginot.training import (
    CosineTailSchedule,
    DeterministicSamplerPlan,
    EnergyTrainerConfig,
    GridSearchSpace,
    SelectionGuardrails,
    SelectionPolicy,
    model_state_hash,
)

DOGBONE_S13C_PROTOCOL_ID: Final = "dogbone-s13c-energy-form-operator-v1"
DOGBONE_S13C_SAMPLER_POLICY_ID: Final = "dogbone-s13c-blake2b-epoch-permutation-v1"
DOGBONE_S13C_LIFTING_POLICY_ID: Final = (
    "dogbone-quarter-axis-aligned-tension-lifting-v1"
)
DOGBONE_S13C_CONDITIONING_POLICY_ID: Final = "s11-train-zscore-geometry-and-strain-v1"
DOGBONE_S13C_QUADRATURE_POLICY_ID: Final = (
    "s10-morph-mesh-triangles-torch-cpu-stratified-v1"
)
DOGBONE_S13C_SCREEN_METRIC_POLICY_ID: Final = (
    "s13c-screen-64-constitutive-vertex-stress-all-train-orientation-v1"
)
DOGBONE_S13C_CONDITIONING_PARAMETERS: Final = (
    "L_total",
    "W_grip",
    "W_gauge",
    "R_fillet",
    "nominal_axial_strain",
)

_S11_PROTOCOL_HASH = "f16a4f63c3a8cfc3ee321bc4e15162802cbc6f106ed807f5aa4bc59eceac81df"
_S11_SOURCE_BINDINGS_HASH = (
    "e32148e4efde6c54fe02ca25b86124d1a43a620eecddb235122919f1649461e3"
)
_TRAINING_SNAPSHOT_HASH = (
    "af04a059e0efb647a1ba768f722d2fde7500d06d11695d1132a0b26dded24845"
)
_PUBLIC_MANIFEST_HASH = (
    "ce95c03f5b22308a7aa9bf627966a6b21300a4e7b0272d3affa7dd503af564a9"
)
_NORMALIZATION_HASH = "b3f59978629e11a4f7535d4f7be17984ffa7323d3a164128a2b8cd4fbf7420df"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SAMPLER_PERSON = b"piginot-s13c-v1"
_TRAINING_SEED = 13_101
_SAMPLER_SEED = 13_102
_INITIALIZATION_SEED = 13_103
_LEARNING_RATE = 3.0e-4
_FINAL_LEARNING_RATE = 3.0e-6
_CONSTANT_STEPS = 1_600
_OPTIMIZER_STEPS = 2_400
_CASE_BATCH_SIZE = 4
_MAXIMUM_TRIALS = 1
_GRADIENT_CLIP_NORM = 1.0
EXECUTION_DTYPE: Final = "float32"
EXECUTION_BLAS_THREADS: Final = 2


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 256-bit digest")
    return value


def _identifiers(values: object, label: str, count: int) -> tuple[str, ...]:
    if type(values) is not tuple or len(values) != count:
        raise ValueError(f"{label} must contain exactly {count} identifiers")
    if any(
        not isinstance(value, str) or not value or value.strip() != value
        for value in values
    ):
        raise ValueError(f"{label} must contain canonical non-empty text")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{label} must be sorted and unique")
    return values


def operator_config() -> EnergyOperatorConfig:
    """Return the ported enhanced architecture, conditioned on five parameters."""

    return EnergyOperatorConfig(parameter_names=DOGBONE_S13C_CONDITIONING_PARAMETERS)


def energy_config() -> PotentialEnergyConfig:
    """Return the enhanced energy policy: 3 points per cell, barrier 1e3 at 0.05."""

    return PotentialEnergyConfig(
        points_per_cell=3,
        barrier_weight=1.0e3,
        barrier_threshold=0.05,
        admissibility_floor=1.0e-6,
    )


def search_space() -> GridSearchSpace:
    """Return the single registered trial (the enhanced paper learning rate)."""

    return GridSearchSpace(
        learning_rates=(_LEARNING_RATE,),
        physics_weight_scales=(1.0,),
        optimizer_steps=_OPTIMIZER_STEPS,
        case_batch_size=_CASE_BATCH_SIZE,
        maximum_trials=_MAXIMUM_TRIALS,
    )


def schedule() -> CosineTailSchedule:
    """Return 1,600 constant steps at 3e-4, then cosine to 3e-6 at 2,400."""

    return CosineTailSchedule(
        peak_learning_rate=_LEARNING_RATE,
        final_learning_rate=_FINAL_LEARNING_RATE,
        constant_steps=_CONSTANT_STEPS,
        total_steps=_OPTIMIZER_STEPS,
    )


def build_initial_operator(
    config: EnergyOperatorConfig | None = None,
    *,
    seed: int = _INITIALIZATION_SEED,
) -> EnergyFormOperator:
    """Construct the registered float32 initial operator deterministically."""

    resolved = operator_config() if config is None else config
    generator_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = EnergyFormOperator(resolved).to(torch.float32)
    finally:
        torch.random.set_rng_state(generator_state)
    return model


@dataclass(frozen=True, slots=True)
class DogboneS13cObjectivePolicy:
    """Everything that defines the training objective, bound into ``run_hash``."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    energy: PotentialEnergyConfig
    lifting_policy_id: str
    quadrature_policy_id: str
    conditioning_policy_id: str
    conditioning_parameter_names: tuple[str, ...]
    normalization_hash: str

    @property
    def policy_hash(self) -> str:
        return hash_spec(self, "recipe")


@dataclass(frozen=True, slots=True)
class DogboneS13cProtocol:
    """Complete preregistration surface before any S13c training or screening."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    protocol_id: str
    s11_protocol_hash: str
    s13_protocol_hash: str
    source_bindings_hash: str
    dataset_snapshot_hash: str
    public_manifest_hash: str
    normalization_hash: str
    architecture_id: str
    architecture_config: EnergyOperatorConfig
    architecture_module_digest: str
    initial_weights_hash: str
    objective: DogboneS13cObjectivePolicy
    fit_case_ids: tuple[str, ...]
    screen_case_ids: tuple[str, ...]
    search_space: GridSearchSpace
    schedule: CosineTailSchedule
    gradient_clip_norm: float
    training_seed: int
    sampler_seed: int
    initialization_seed: int
    sampler_policy_id: str
    execution_dtype: str
    execution_blas_threads: int
    screen_metric_policy_id: str
    selection_policy: SelectionPolicy
    s11_baseline_metrics: ArchitectureScreenMetrics

    def __post_init__(self) -> None:
        for value, expected, label in (
            (self.s11_protocol_hash, _S11_PROTOCOL_HASH, "s11_protocol_hash"),
            (
                self.source_bindings_hash,
                _S11_SOURCE_BINDINGS_HASH,
                "source_bindings_hash",
            ),
            (
                self.dataset_snapshot_hash,
                _TRAINING_SNAPSHOT_HASH,
                "dataset_snapshot_hash",
            ),
            (
                self.public_manifest_hash,
                _PUBLIC_MANIFEST_HASH,
                "public_manifest_hash",
            ),
            (self.normalization_hash, _NORMALIZATION_HASH, "normalization_hash"),
        ):
            if _digest(value, label) != expected:
                raise ValueError(f"{label} differs from the ratified upstream evidence")
        _digest(self.s13_protocol_hash, "s13_protocol_hash")
        _digest(self.initial_weights_hash, "initial_weights_hash")
        if self.protocol_id != DOGBONE_S13C_PROTOCOL_ID:
            raise ValueError("unexpected DogBone S13c protocol_id")
        if self.architecture_id != ENERGY_ARCHITECTURE_ID:
            raise ValueError("architecture_id must be the energy-form operator")
        if self.architecture_config != operator_config():
            raise ValueError("architecture_config differs from the ported operator")
        if self.objective != _objective_policy():
            raise ValueError("objective policy differs from the preregistration")
        fit = _identifiers(self.fit_case_ids, "fit_case_ids", 192)
        screen = _identifiers(self.screen_case_ids, "screen_case_ids", 64)
        if set(fit) & set(screen):
            raise ValueError("TRAIN-fit and TRAIN-screen case IDs must be disjoint")
        if self.search_space != search_space():
            raise ValueError("search_space differs from the registered single trial")
        if self.schedule != schedule():
            raise ValueError("schedule differs from the registered policy")
        if (
            isinstance(self.gradient_clip_norm, bool)
            or not isinstance(self.gradient_clip_norm, (int, float))
            or not math.isfinite(float(self.gradient_clip_norm))
            or float(self.gradient_clip_norm) != _GRADIENT_CLIP_NORM
        ):
            raise ValueError("gradient_clip_norm must remain fixed at 1.0")
        for value, expected, label in (
            (self.training_seed, _TRAINING_SEED, "training_seed"),
            (self.sampler_seed, _SAMPLER_SEED, "sampler_seed"),
            (self.initialization_seed, _INITIALIZATION_SEED, "initialization_seed"),
            (
                self.execution_blas_threads,
                EXECUTION_BLAS_THREADS,
                "execution_blas_threads",
            ),
        ):
            if value != expected:
                raise ValueError(f"{label} must remain {expected}")
        if self.sampler_policy_id != DOGBONE_S13C_SAMPLER_POLICY_ID:
            raise ValueError("sampler_policy_id changed")
        if self.execution_dtype != EXECUTION_DTYPE:
            raise ValueError("execution_dtype must remain float32")
        if self.screen_metric_policy_id != DOGBONE_S13C_SCREEN_METRIC_POLICY_ID:
            raise ValueError("screen_metric_policy_id changed")
        if self.selection_policy != SelectionPolicy(
            self.search_space,
            SelectionGuardrails(),
        ):
            raise ValueError("selection_policy must keep the frozen S13 guardrails")
        if type(self.s11_baseline_metrics) is not ArchitectureScreenMetrics:
            raise TypeError("s11_baseline_metrics must be ArchitectureScreenMetrics")
        if self.s11_baseline_metrics.candidate_id != BASELINE_CANDIDATE_ID:
            raise ValueError("incumbent metrics must describe the S11 baseline")
        object.__setattr__(self, "fit_case_ids", fit)
        object.__setattr__(self, "screen_case_ids", screen)
        object.__setattr__(self, "gradient_clip_norm", _GRADIENT_CLIP_NORM)

    @property
    def protocol_hash(self) -> str:
        return hash_spec(self, "recipe")

    @property
    def objective_hash(self) -> str:
        return self.objective.policy_hash

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self)


def _objective_policy() -> DogboneS13cObjectivePolicy:
    return DogboneS13cObjectivePolicy(
        energy=energy_config(),
        lifting_policy_id=DOGBONE_S13C_LIFTING_POLICY_ID,
        quadrature_policy_id=DOGBONE_S13C_QUADRATURE_POLICY_ID,
        conditioning_policy_id=DOGBONE_S13C_CONDITIONING_POLICY_ID,
        conditioning_parameter_names=DOGBONE_S13C_CONDITIONING_PARAMETERS,
        normalization_hash=_NORMALIZATION_HASH,
    )


def build_dogbone_s13c_protocol() -> DogboneS13cProtocol:
    """Build the exact S13c protocol from pinned, tracked evidence only."""

    from campaigns.dogbone_v1.s13_protocol import build_dogbone_s13_protocol

    s11_protocol = load_committed_protocol()
    evidence = load_selected_s11_checkpoint_evidence()
    if s11_protocol.content_hash != _S11_PROTOCOL_HASH:
        raise ValueError("S11 protocol identity changed")
    if evidence.source_bindings_hash != _S11_SOURCE_BINDINGS_HASH:
        raise ValueError("S11 source bindings differ from the ratified evidence")
    model = build_initial_operator()
    space = search_space()
    return DogboneS13cProtocol(
        protocol_id=DOGBONE_S13C_PROTOCOL_ID,
        s11_protocol_hash=s11_protocol.content_hash,
        s13_protocol_hash=build_dogbone_s13_protocol().protocol_hash,
        source_bindings_hash=evidence.source_bindings_hash,
        dataset_snapshot_hash=s11_protocol.training_snapshot_hash,
        public_manifest_hash=s11_protocol.public_case_manifest_hash,
        normalization_hash=evidence.normalization.content_hash,
        architecture_id=model.architecture_id,
        architecture_config=model.config,
        architecture_module_digest=energy_module_digest(model),
        initial_weights_hash=model_state_hash(model),
        objective=_objective_policy(),
        fit_case_ids=s11_protocol.fit_case_ids,
        screen_case_ids=s11_protocol.screen_case_ids,
        search_space=space,
        schedule=schedule(),
        gradient_clip_norm=_GRADIENT_CLIP_NORM,
        training_seed=_TRAINING_SEED,
        sampler_seed=_SAMPLER_SEED,
        initialization_seed=_INITIALIZATION_SEED,
        sampler_policy_id=DOGBONE_S13C_SAMPLER_POLICY_ID,
        execution_dtype=EXECUTION_DTYPE,
        execution_blas_threads=EXECUTION_BLAS_THREADS,
        screen_metric_policy_id=DOGBONE_S13C_SCREEN_METRIC_POLICY_ID,
        selection_policy=SelectionPolicy(space, SelectionGuardrails()),
        s11_baseline_metrics=evidence.baseline_metrics,
    )


def load_s13c_normalization() -> CaseNormalizationStatistics:
    """Return the frozen S11 TRAIN normalization used for conditioning."""

    normalization = load_selected_s11_checkpoint_evidence().normalization
    if normalization.content_hash != _NORMALIZATION_HASH:
        raise ValueError("S11 normalization differs from the S13c protocol")
    return normalization


def _epoch_order(population_size: int, seed: int, epoch: int) -> tuple[int, ...]:
    prefix = seed.to_bytes(8, "big") + epoch.to_bytes(8, "big")
    return tuple(
        sorted(
            range(population_size),
            key=lambda index: (
                hashlib.blake2b(
                    prefix + index.to_bytes(8, "big"),
                    digest_size=32,
                    person=_SAMPLER_PERSON,
                ).digest(),
                index,
            ),
        )
    )


def build_dogbone_s13c_sampler_plan(
    protocol: DogboneS13cProtocol,
) -> DeterministicSamplerPlan:
    """Build the complete epoch-permutation batch schedule of the single trial."""

    if type(protocol) is not DogboneS13cProtocol:
        raise TypeError("protocol must be exact DogboneS13cProtocol")
    population_size = len(protocol.fit_case_ids)
    batch_size = protocol.search_space.case_batch_size
    if population_size % batch_size != 0:
        raise ValueError("TRAIN-fit population must divide exactly into batches")
    batches: list[tuple[int, ...]] = []
    epoch = 0
    while len(batches) < protocol.search_space.optimizer_steps:
        order = _epoch_order(population_size, protocol.sampler_seed, epoch)
        batches.extend(
            order[start : start + batch_size]
            for start in range(0, population_size, batch_size)
        )
        epoch += 1
    return DeterministicSamplerPlan(
        sampler_id=f"{protocol.sampler_policy_id}:{protocol.sampler_seed}",
        population_size=population_size,
        batches=tuple(batches[: protocol.search_space.optimizer_steps]),
    )


def build_dogbone_s13c_trainer_config(
    protocol: DogboneS13cProtocol,
) -> EnergyTrainerConfig:
    """Materialize the single registered trial's optimizer policy."""

    if type(protocol) is not DogboneS13cProtocol:
        raise TypeError("protocol must be exact DogboneS13cProtocol")
    (trial,) = protocol.search_space.trials
    return EnergyTrainerConfig(
        search_space=protocol.search_space,
        trial=trial,
        schedule=protocol.schedule,
        gradient_clip_norm=protocol.gradient_clip_norm,
    )


__all__ = (
    "DOGBONE_S13C_CONDITIONING_PARAMETERS",
    "DOGBONE_S13C_CONDITIONING_POLICY_ID",
    "DOGBONE_S13C_LIFTING_POLICY_ID",
    "DOGBONE_S13C_PROTOCOL_ID",
    "DOGBONE_S13C_QUADRATURE_POLICY_ID",
    "DOGBONE_S13C_SAMPLER_POLICY_ID",
    "DOGBONE_S13C_SCREEN_METRIC_POLICY_ID",
    "EXECUTION_BLAS_THREADS",
    "EXECUTION_DTYPE",
    "DogboneS13cObjectivePolicy",
    "DogboneS13cProtocol",
    "build_dogbone_s13c_protocol",
    "build_dogbone_s13c_sampler_plan",
    "build_dogbone_s13c_trainer_config",
    "build_initial_operator",
    "energy_config",
    "load_s13c_normalization",
    "operator_config",
    "schedule",
    "search_space",
)
