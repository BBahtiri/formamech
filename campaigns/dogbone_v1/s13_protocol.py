"""Frozen TRAIN-only protocol for the DogBone S13 hybrid fine-tune."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import ClassVar

from campaigns.dogbone_v1.s11_contingency import (
    BASELINE_CANDIDATE_ID,
    ArchitectureScreenMetrics,
    ArchitectureScreenProtocol,
)
from campaigns.dogbone_v1.s11_contingency_run import load_committed_protocol
from campaigns.dogbone_v1.s13_initialization import (
    SelectedS11CheckpointEvidence,
    load_selected_s11_checkpoint_evidence,
)
from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.objectives import EquilibriumForm, ObjectiveMode, ObjectiveRegistry
from piginot.training import (
    AdaptiveWeightConfig,
    DeterministicSamplerPlan,
    DeterministicTrainerConfig,
    GridSearchSpace,
    PlateauSchedulerConfig,
    SelectionGuardrails,
    SelectionPolicy,
)

DOGBONE_S13_PROTOCOL_ID = "dogbone-s13-hybrid-finetune-v1"
DOGBONE_S13_SAMPLER_POLICY_ID = "dogbone-s13-blake2b-epoch-permutation-v1"

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
_S11_EVIDENCE_HASH = "260eca6c345ee1edea17b9098859baaa6de4a35e27de65c6ccdead0cad831f36"
_INITIAL_CHECKPOINT_FILE_HASH = (
    "c646f8fe58d2759012f2e1392d613395d5256bd6d8947d7728d3352a1bcf7c8e"
)
_INITIAL_MODEL_STATE_HASH = (
    "51b83fd2468d66d5556b75286feb08a13b2e9ca16d0b7f4cdcb83ff26eaba906"
)
_INITIAL_TRAINER_WEIGHTS_HASH = (
    "da58cb840fa5401a75474b7dca7f4a46f47c42032920d1331987b9fff48dbd7a"
)
_ARCHITECTURE_MODULE_DIGEST = (
    "blake2b256:ac064556f53e573b4f9de63a2610077dfd38141b6eef695731302cea3a560f47"
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MODULE_DIGEST = re.compile(r"blake2b256:[0-9a-f]{64}\Z")
_SAMPLER_PERSON = b"piginot-s13-v1"
_TRAINING_SEED = 13_001
_SAMPLER_SEED = 13_002
_LEARNING_RATES = (1.0e-5, 3.0e-5)
_PHYSICS_WEIGHT_SCALES = (1.0e-4, 1.0e-3, 1.0e-2)
_OPTIMIZER_STEPS = 1_500
_CASE_BATCH_SIZE = 4
_MAXIMUM_TRIALS = 6


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


@dataclass(frozen=True, slots=True)
class DogboneS13Protocol:
    """Complete preregistration surface before TRAIN-screen labels are read."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    protocol_id: str
    s11_protocol_hash: str
    s11_evidence_hash: str
    source_bindings_hash: str
    dataset_snapshot_hash: str
    public_manifest_hash: str
    normalization_hash: str
    selected_candidate_id: str
    architecture_module_digest: str
    initial_checkpoint_file_hash: str
    initial_model_state_hash: str
    initial_trainer_weights_hash: str
    fit_case_ids: tuple[str, ...]
    screen_case_ids: tuple[str, ...]
    search_space: GridSearchSpace
    adaptive: AdaptiveWeightConfig
    scheduler: PlateauSchedulerConfig
    gradient_clip_norm: float
    objective_registry: ObjectiveRegistry
    training_seed: int
    sampler_seed: int
    sampler_policy_id: str
    selection_policy: SelectionPolicy
    s11_baseline_metrics: ArchitectureScreenMetrics

    def __post_init__(self) -> None:
        expected_digests = (
            (self.s11_protocol_hash, _S11_PROTOCOL_HASH, "s11_protocol_hash"),
            (self.s11_evidence_hash, _S11_EVIDENCE_HASH, "s11_evidence_hash"),
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
            (
                self.normalization_hash,
                _NORMALIZATION_HASH,
                "normalization_hash",
            ),
            (
                self.initial_checkpoint_file_hash,
                _INITIAL_CHECKPOINT_FILE_HASH,
                "initial_checkpoint_file_hash",
            ),
            (
                self.initial_model_state_hash,
                _INITIAL_MODEL_STATE_HASH,
                "initial_model_state_hash",
            ),
            (
                self.initial_trainer_weights_hash,
                _INITIAL_TRAINER_WEIGHTS_HASH,
                "initial_trainer_weights_hash",
            ),
        )
        for value, expected, label in expected_digests:
            if _digest(value, label) != expected:
                raise ValueError(f"{label} differs from the ratified upstream evidence")
        if self.protocol_id != DOGBONE_S13_PROTOCOL_ID:
            raise ValueError("unexpected DogBone S13 protocol_id")
        if self.selected_candidate_id != BASELINE_CANDIDATE_ID:
            raise ValueError("selected_candidate_id must remain the S11 fallback")
        if (
            not isinstance(self.architecture_module_digest, str)
            or _MODULE_DIGEST.fullmatch(self.architecture_module_digest) is None
            or self.architecture_module_digest != _ARCHITECTURE_MODULE_DIGEST
        ):
            raise ValueError("architecture_module_digest differs from S11 evidence")
        fit = _identifiers(self.fit_case_ids, "fit_case_ids", 192)
        screen = _identifiers(self.screen_case_ids, "screen_case_ids", 64)
        if set(fit) & set(screen):
            raise ValueError("TRAIN-fit and TRAIN-screen case IDs must be disjoint")
        if type(self.search_space) is not GridSearchSpace or self.search_space != (
            _search_space()
        ):
            raise ValueError("search_space differs from the frozen bounded grid")
        if type(self.adaptive) is not AdaptiveWeightConfig or self.adaptive != (
            _adaptive_config()
        ):
            raise ValueError("adaptive policy differs from the frozen schedule")
        if type(self.scheduler) is not PlateauSchedulerConfig or self.scheduler != (
            _scheduler_config()
        ):
            raise ValueError("scheduler differs from the frozen policy")
        if (
            isinstance(self.gradient_clip_norm, bool)
            or not isinstance(self.gradient_clip_norm, (int, float))
            or not math.isfinite(float(self.gradient_clip_norm))
            or float(self.gradient_clip_norm) != 1.0
        ):
            raise ValueError("gradient_clip_norm must remain fixed at 1.0")
        if type(self.objective_registry) is not ObjectiveRegistry or (
            self.objective_registry != _objective_registry()
        ):
            raise ValueError(
                "objective_registry differs from the S12 strong HYBRID policy"
            )
        if self.training_seed != _TRAINING_SEED:
            raise ValueError(f"training_seed must remain {_TRAINING_SEED}")
        if self.sampler_seed != _SAMPLER_SEED:
            raise ValueError(f"sampler_seed must remain {_SAMPLER_SEED}")
        if self.sampler_policy_id != DOGBONE_S13_SAMPLER_POLICY_ID:
            raise ValueError("sampler_policy_id changed")
        if type(self.selection_policy) is not SelectionPolicy:
            raise TypeError("selection_policy must be exact SelectionPolicy")
        if self.selection_policy != SelectionPolicy(
            self.search_space,
            SelectionGuardrails(),
        ):
            raise ValueError("selection_policy differs from the preregistered rule")
        if type(self.s11_baseline_metrics) is not ArchitectureScreenMetrics:
            raise TypeError(
                "s11_baseline_metrics must be exact ArchitectureScreenMetrics"
            )
        _validate_incumbent_provenance(
            self.selection_policy.guardrails,
            self.s11_baseline_metrics,
        )
        object.__setattr__(self, "fit_case_ids", fit)
        object.__setattr__(self, "screen_case_ids", screen)
        object.__setattr__(self, "gradient_clip_norm", 1.0)

    @property
    def protocol_hash(self) -> str:
        return hash_spec(self, "recipe")

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self)


def _search_space() -> GridSearchSpace:
    return GridSearchSpace(
        learning_rates=_LEARNING_RATES,
        physics_weight_scales=_PHYSICS_WEIGHT_SCALES,
        optimizer_steps=_OPTIMIZER_STEPS,
        case_batch_size=_CASE_BATCH_SIZE,
        maximum_trials=_MAXIMUM_TRIALS,
    )


def _adaptive_config() -> AdaptiveWeightConfig:
    return AdaptiveWeightConfig(
        beta=0.1,
        update_period=100,
        warmup_steps=200,
        minimum_multiplier=0.1,
        maximum_multiplier=10.0,
    )


def _scheduler_config() -> PlateauSchedulerConfig:
    return PlateauSchedulerConfig(
        patience=250,
        reduction_factor=0.5,
        absolute_threshold=0.0,
        cooldown_steps=48,
        minimum_learning_rate=1.0e-7,
    )


def _objective_registry() -> ObjectiveRegistry:
    return ObjectiveRegistry(
        mode=ObjectiveMode.HYBRID,
        equilibrium_form=EquilibriumForm.STRONG,
    )


def _validate_incumbent_provenance(
    guardrails: SelectionGuardrails,
    metrics: ArchitectureScreenMetrics,
) -> None:
    if metrics.candidate_id != BASELINE_CANDIDATE_ID:
        raise ValueError("incumbent metrics must describe the S11 baseline")
    comparisons = (
        (
            guardrails.incumbent_median_absolute_peak_error,
            metrics.median_absolute_peak_error,
        ),
        (
            guardrails.incumbent_p90_absolute_peak_error,
            metrics.p90_absolute_peak_error,
        ),
        (
            guardrails.incumbent_median_signed_peak_error,
            metrics.median_signed_peak_error,
        ),
        (guardrails.incumbent_u_relative_l2, metrics.median_u_relative_l2),
        (guardrails.incumbent_p_relative_l2, metrics.median_p_relative_l2),
    )
    if any(frozen != evidenced for frozen, evidenced in comparisons):
        raise ValueError("selection guardrails differ from S11 baseline evidence")


def build_dogbone_s13_protocol() -> DogboneS13Protocol:
    """Build the exact S13 protocol from pinned, tracked S11 evidence only."""

    return _build_dogbone_s13_protocol(
        load_committed_protocol(),
        load_selected_s11_checkpoint_evidence(),
    )


def _build_dogbone_s13_protocol(
    s11_protocol: ArchitectureScreenProtocol,
    evidence: SelectedS11CheckpointEvidence,
) -> DogboneS13Protocol:
    if type(s11_protocol) is not ArchitectureScreenProtocol:
        raise TypeError("s11_protocol must be exact ArchitectureScreenProtocol")
    if type(evidence) is not SelectedS11CheckpointEvidence:
        raise TypeError("evidence must be exact SelectedS11CheckpointEvidence")
    if s11_protocol.content_hash != _S11_PROTOCOL_HASH:
        raise ValueError("S11 protocol identity changed")
    if evidence.protocol_content_hash != s11_protocol.content_hash:
        raise ValueError("S11 evidence differs from the committed protocol")
    if evidence.source_bindings_hash != _S11_SOURCE_BINDINGS_HASH:
        raise ValueError("S11 source bindings differ from the ratified evidence")
    if evidence.normalization.training_snapshot_hash != (
        s11_protocol.training_snapshot_hash
    ):
        raise ValueError("S11 normalization differs from the TRAIN snapshot")
    search_space = _search_space()
    selection_policy = SelectionPolicy(search_space, SelectionGuardrails())
    return DogboneS13Protocol(
        protocol_id=DOGBONE_S13_PROTOCOL_ID,
        s11_protocol_hash=s11_protocol.content_hash,
        s11_evidence_hash=evidence.evidence_file_blake2b256,
        source_bindings_hash=evidence.source_bindings_hash,
        dataset_snapshot_hash=s11_protocol.training_snapshot_hash,
        public_manifest_hash=s11_protocol.public_case_manifest_hash,
        normalization_hash=evidence.normalization.content_hash,
        selected_candidate_id=evidence.candidate_id,
        architecture_module_digest=evidence.operator_module_digest,
        initial_checkpoint_file_hash=evidence.file_blake2b256,
        initial_model_state_hash=evidence.final_state_hash,
        initial_trainer_weights_hash=_INITIAL_TRAINER_WEIGHTS_HASH,
        fit_case_ids=s11_protocol.fit_case_ids,
        screen_case_ids=s11_protocol.screen_case_ids,
        search_space=search_space,
        adaptive=_adaptive_config(),
        scheduler=_scheduler_config(),
        gradient_clip_norm=1.0,
        objective_registry=_objective_registry(),
        training_seed=_TRAINING_SEED,
        sampler_seed=_SAMPLER_SEED,
        sampler_policy_id=DOGBONE_S13_SAMPLER_POLICY_ID,
        selection_policy=selection_policy,
        s11_baseline_metrics=evidence.baseline_metrics,
    )


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


def build_dogbone_s13_sampler_plan(
    protocol: DogboneS13Protocol,
) -> DeterministicSamplerPlan:
    """Build the shared, complete batch schedule used by all six trials."""

    if type(protocol) is not DogboneS13Protocol:
        raise TypeError("protocol must be exact DogboneS13Protocol")
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


def build_dogbone_s13_trainer_configs(
    protocol: DogboneS13Protocol,
) -> tuple[DeterministicTrainerConfig, ...]:
    """Materialize every registered trial with one shared optimizer policy."""

    if type(protocol) is not DogboneS13Protocol:
        raise TypeError("protocol must be exact DogboneS13Protocol")
    return tuple(
        DeterministicTrainerConfig(
            search_space=protocol.search_space,
            trial=trial,
            adaptive=protocol.adaptive,
            scheduler=protocol.scheduler,
            gradient_clip_norm=protocol.gradient_clip_norm,
        )
        for trial in protocol.search_space.trials
    )


__all__ = (
    "DOGBONE_S13_PROTOCOL_ID",
    "DOGBONE_S13_SAMPLER_POLICY_ID",
    "DogboneS13Protocol",
    "build_dogbone_s13_protocol",
    "build_dogbone_s13_sampler_plan",
    "build_dogbone_s13_trainer_configs",
)
