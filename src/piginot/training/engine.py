"""TRAIN-only deterministic optimizer orchestration for S13."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Protocol

import torch

from piginot.contracts import hash_spec
from piginot.contracts.errors import ContractValidationError
from piginot.contracts.specs import SpecContract
from piginot.doe import EvidenceRole
from piginot.objectives import (
    EquilibriumForm,
    ObjectiveMode,
    ObjectiveRegistry,
    ObjectiveResiduals,
    ObjectiveTerm,
)
from piginot.training.adaptive import (
    AdaptiveWeightConfig,
    AdaptiveWeightState,
    measure_unweighted_gradient_statistics,
    update_adaptive_weights,
)
from piginot.training.checkpoint import (
    WrittenTrainingCheckpoint,
    load_training_checkpoint,
    model_state_hash,
    write_training_checkpoint,
)
from piginot.training.identity import TrainingRunBinding
from piginot.training.scheduler import (
    PlateauSchedulerConfig,
    advance_plateau_scheduler,
    initial_scheduler_state,
)
from piginot.training.search import GridSearchSpace, TrialConfig
from piginot.training.state import (
    DeterministicSamplerPlan,
    DeterministicSamplerState,
    TrainingSchedulerState,
    next_sampler_batch,
)

_DIGEST_LENGTH = 64
_DATA_TERMS = frozenset((ObjectiveTerm.DATA_U, ObjectiveTerm.DATA_P))


class ObjectiveBuilder(Protocol):
    """Build one graph-connected S12 residual set for registered sample IDs."""

    def __call__(
        self,
        model: torch.nn.Module,
        sample_ids: tuple[str, ...],
        step: int,
    ) -> ObjectiveResiduals: ...


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != _DIGEST_LENGTH:
        raise ContractValidationError(f"{label} must be a lowercase 256-bit digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ContractValidationError(
            f"{label} must be a lowercase 256-bit digest"
        ) from exc
    if value.lower() != value:
        raise ContractValidationError(f"{label} must be a lowercase 256-bit digest")
    return value


def _finite(value: object, label: str, *, positive: bool) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "finite and positive" if positive else "finite"
        raise ContractValidationError(f"{label} must be {qualifier}")
    return float(value)


@dataclass(frozen=True, slots=True)
class TrainingPopulation(SpecContract):
    """TRAIN-only identity surface presented to the optimizer."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    dataset_snapshot_hash: str
    public_manifest_hash: str
    role: EvidenceRole
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _digest(self.dataset_snapshot_hash, "dataset_snapshot_hash")
        _digest(self.public_manifest_hash, "public_manifest_hash")
        if type(self.role) is not EvidenceRole:
            raise TypeError("role must be an exact EvidenceRole")
        if self.role is not EvidenceRole.TRAIN:
            raise PermissionError(
                f"{self.role.value} evidence cannot enter the S13 trainer"
            )
        if type(self.sample_ids) is not tuple or not self.sample_ids:
            raise ContractValidationError("sample_ids must be a non-empty tuple")
        if any(
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id.strip() != sample_id
            for sample_id in self.sample_ids
        ):
            raise ContractValidationError("sample_ids must be canonical text")
        if tuple(sorted(set(self.sample_ids))) != self.sample_ids:
            raise ContractValidationError("sample_ids must be sorted and unique")


@dataclass(frozen=True, slots=True)
class DeterministicTrainerConfig(SpecContract):
    """Complete optimizer policy bound by ``trainer_config_hash``."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    search_space: GridSearchSpace
    trial: TrialConfig
    adaptive: AdaptiveWeightConfig
    scheduler: PlateauSchedulerConfig
    gradient_clip_norm: float
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1.0e-8
    adam_weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if type(self.search_space) is not GridSearchSpace:
            raise ContractValidationError(
                "search_space must be an exact GridSearchSpace"
            )
        if type(self.trial) is not TrialConfig:
            raise ContractValidationError("trial must be an exact TrialConfig")
        if self.trial.optimizer_steps != self.search_space.optimizer_steps:
            raise ContractValidationError(
                "trial optimizer_steps differ from search space"
            )
        if self.trial.case_batch_size != self.search_space.case_batch_size:
            raise ContractValidationError(
                "trial case_batch_size differs from search space"
            )
        self.search_space.authorize(self.trial)
        if type(self.adaptive) is not AdaptiveWeightConfig:
            raise ContractValidationError(
                "adaptive must be an exact AdaptiveWeightConfig"
            )
        if type(self.scheduler) is not PlateauSchedulerConfig:
            raise ContractValidationError(
                "scheduler must be an exact PlateauSchedulerConfig"
            )
        object.__setattr__(
            self,
            "gradient_clip_norm",
            _finite(self.gradient_clip_norm, "gradient_clip_norm", positive=True),
        )
        if type(self.adam_betas) is not tuple or len(self.adam_betas) != 2:
            raise ContractValidationError("adam_betas must contain two values")
        betas = tuple(
            _finite(value, "adam beta", positive=False) for value in self.adam_betas
        )
        if any(value < 0.0 or value >= 1.0 for value in betas):
            raise ContractValidationError("adam betas must lie in [0, 1)")
        object.__setattr__(self, "adam_betas", betas)
        object.__setattr__(
            self,
            "adam_eps",
            _finite(self.adam_eps, "adam_eps", positive=True),
        )
        weight_decay = _finite(
            self.adam_weight_decay,
            "adam_weight_decay",
            positive=False,
        )
        if weight_decay < 0.0:
            raise ContractValidationError("adam_weight_decay must be non-negative")
        object.__setattr__(self, "adam_weight_decay", weight_decay)

    @property
    def config_hash(self) -> str:
        return hash_spec(self, "recipe")


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """Final immutable trace returned by one fresh or resumed segment."""

    step: int
    model_state_hash: str
    sampler_state: DeterministicSamplerState
    scheduler_state: TrainingSchedulerState
    adaptive_state: AdaptiveWeightState
    final_losses: Mapping[ObjectiveTerm, float]
    checkpoint: WrittenTrainingCheckpoint

    def __post_init__(self) -> None:
        _digest(self.model_state_hash, "model_state_hash")
        if type(self.sampler_state) is not DeterministicSamplerState:
            raise TypeError("sampler_state must be exact DeterministicSamplerState")
        if type(self.scheduler_state) is not TrainingSchedulerState:
            raise TypeError("scheduler_state must be exact TrainingSchedulerState")
        if type(self.adaptive_state) is not AdaptiveWeightState:
            raise TypeError("adaptive_state must be exact AdaptiveWeightState")
        if type(self.checkpoint) is not WrittenTrainingCheckpoint:
            raise TypeError("checkpoint must be exact WrittenTrainingCheckpoint")
        if not isinstance(self.final_losses, Mapping) or not self.final_losses:
            raise ValueError("final_losses must be a non-empty mapping")
        losses: dict[ObjectiveTerm, float] = {}
        for term, value in self.final_losses.items():
            if type(term) is not ObjectiveTerm:
                raise TypeError("final loss keys must be ObjectiveTerm values")
            losses[term] = _finite(value, f"{term.value} loss", positive=False)
            if losses[term] < 0.0:
                raise ValueError("final losses must be non-negative")
        object.__setattr__(self, "final_losses", MappingProxyType(losses))


def _scaled_registry(
    registry: ObjectiveRegistry,
    physics_weight_scale: float,
) -> ObjectiveRegistry:
    if type(registry) is not ObjectiveRegistry:
        raise TypeError("base_registry must be an exact ObjectiveRegistry")
    if (
        registry.mode is not ObjectiveMode.HYBRID
        or registry.equilibrium_form is not EquilibriumForm.STRONG
    ):
        raise ValueError("S13 adaptive training requires a strong HYBRID registry")
    weights = {
        term: weight if term in _DATA_TERMS else weight * physics_weight_scale
        for term, weight in registry.term_weights.items()
    }
    return ObjectiveRegistry(
        mode=registry.mode,
        equilibrium_form=registry.equilibrium_form,
        term_weights=weights,
    )


def _validate_model_profile(
    model: torch.nn.Module,
    binding: TrainingRunBinding,
) -> None:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    parameters = tuple(model.parameters())
    if not parameters:
        raise ValueError("model must contain trainable parameters")
    if any(parameter.device.type != "cpu" for parameter in parameters):
        raise ValueError("S13 exact training currently requires CPU parameters")
    expected_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }.get(binding.execution_profile.dtype)
    if expected_dtype is None or any(
        parameter.dtype is not expected_dtype for parameter in parameters
    ):
        raise ValueError("model dtype does not match the execution profile")


def _validate_execution_inputs(
    *,
    model: torch.nn.Module,
    population: TrainingPopulation,
    plan: DeterministicSamplerPlan,
    binding: TrainingRunBinding,
    config: DeterministicTrainerConfig,
    base_registry: ObjectiveRegistry,
    objective: ObjectiveBuilder,
) -> ObjectiveRegistry:
    if type(population) is not TrainingPopulation:
        raise TypeError("population must be an exact TrainingPopulation")
    if type(plan) is not DeterministicSamplerPlan:
        raise TypeError("plan must be an exact DeterministicSamplerPlan")
    if type(binding) is not TrainingRunBinding:
        raise TypeError("binding must be an exact TrainingRunBinding")
    if type(config) is not DeterministicTrainerConfig:
        raise TypeError("config must be an exact DeterministicTrainerConfig")
    if not callable(objective):
        raise TypeError("objective must be callable")
    if binding.dataset_snapshot_hash != population.dataset_snapshot_hash:
        raise ValueError("dataset snapshot does not match the training run binding")
    if binding.public_manifest_hash != population.public_manifest_hash:
        raise ValueError("public manifest does not match the training run binding")
    if binding.sampler_plan_hash != plan.plan_hash:
        raise ValueError("sampler plan does not match the training run binding")
    if binding.trainer_config_hash != config.config_hash:
        raise ValueError("trainer config does not match the training run binding")
    if binding.objective_registry_hash != hash_spec(base_registry, "recipe"):
        raise ValueError("objective registry does not match the training run binding")
    if binding.initial_weights_hash != model_state_hash(model):
        raise ValueError("initial weights do not match the training run binding")
    if plan.population_size != len(population.sample_ids):
        raise ValueError("sampler population_size does not match sample_ids")
    if len(plan.batches) != config.trial.optimizer_steps:
        raise ValueError("sampler plan length does not match optimizer_steps")
    if any(len(batch) != config.trial.case_batch_size for batch in plan.batches):
        raise ValueError("sampler batch size does not match case_batch_size")
    _validate_model_profile(model, binding)
    return _scaled_registry(base_registry, config.trial.physics_weight_scale)


def _unweighted_losses(
    registry: ObjectiveRegistry,
    residuals: ObjectiveResiduals,
) -> Mapping[ObjectiveTerm, torch.Tensor]:
    missing = tuple(
        term for term in registry.required_terms if term not in residuals.values
    )
    if missing:
        raise ValueError(
            "missing required objective residual: "
            + ", ".join(term.value for term in missing)
        )
    return MappingProxyType(
        {
            term: residuals.values[term].square().mean()
            for term in registry.required_terms
        }
    )


def _preflight_checkpoint_destination(path: Path) -> None:
    if not path.parent.exists() or not path.parent.is_dir():
        raise FileNotFoundError("checkpoint parent directory must already exist")
    if path.parent.is_symlink():
        raise ValueError("checkpoint parent directory must not be a symlink")
    if path.exists() or path.is_symlink():
        raise FileExistsError("checkpoint directory already exists")


def train_to_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    population: TrainingPopulation,
    plan: DeterministicSamplerPlan,
    binding: TrainingRunBinding,
    config: DeterministicTrainerConfig,
    base_registry: ObjectiveRegistry,
    objective: ObjectiveBuilder,
    target_step: int | None = None,
    resume_from: Path | None = None,
) -> TrainingRunResult:
    """Train one exact segment and persist its complete resumable state."""

    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a pathlib.Path")
    if resume_from is not None and not isinstance(resume_from, Path):
        raise TypeError("resume_from must be a pathlib.Path or None")
    _preflight_checkpoint_destination(checkpoint_path)
    registry = _validate_execution_inputs(
        model=model,
        population=population,
        plan=plan,
        binding=binding,
        config=config,
        base_registry=base_registry,
        objective=objective,
    )
    final_step = config.trial.optimizer_steps if target_step is None else target_step
    if (
        isinstance(final_step, bool)
        or not isinstance(final_step, int)
        or final_step <= 0
        or final_step > config.trial.optimizer_steps
    ):
        raise ValueError("target_step must lie inside the optimizer budget")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.trial.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.adam_weight_decay,
        foreach=False,
        fused=False,
    )
    if resume_from is None:
        torch.manual_seed(binding.seed)
        sampler_state = plan.initial_state
        scheduler_state = initial_scheduler_state(
            config.scheduler,
            learning_rate=config.trial.learning_rate,
        )
        adaptive_state = AdaptiveWeightState.initial()
    else:
        loaded = load_training_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            binding=binding,
        )
        sampler_state = loaded.sampler_state
        scheduler_state = loaded.scheduler_state
        adaptive_state = loaded.adaptive_state
    if sampler_state.next_step >= final_step:
        raise ValueError("target_step must be after the restored checkpoint step")

    final_losses: Mapping[ObjectiveTerm, float] | None = None
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    thread_count_before = torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    try:
        torch.set_num_threads(binding.execution_profile.blas_threads)
        try:
            model.train()
            while sampler_state.next_step < final_step:
                batch_indices, advanced_sampler = next_sampler_batch(
                    plan, sampler_state
                )
                completed_step = sampler_state.next_step + 1
                sample_ids = tuple(
                    population.sample_ids[index] for index in batch_indices
                )
                optimizer.zero_grad(set_to_none=True)
                residuals = objective(model, sample_ids, completed_step)
                if type(residuals) is not ObjectiveResiduals:
                    raise TypeError("objective must return exact ObjectiveResiduals")
                unweighted = _unweighted_losses(registry, residuals)
                gradients = (
                    measure_unweighted_gradient_statistics(
                        unweighted,
                        model.parameters(),
                    )
                    if config.adaptive.is_update_step(completed_step)
                    else {}
                )
                adaptive_update = update_adaptive_weights(
                    registry,
                    adaptive_state,
                    gradients,
                    config=config.adaptive,
                    step=completed_step,
                )
                evaluation = adaptive_update.registry.evaluate(
                    residuals,
                    role=EvidenceRole.TRAIN,
                )
                if not bool(torch.isfinite(evaluation.total).item()):
                    raise ValueError("training objective must be finite")
                evaluation.total.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                scheduler_state = advance_plateau_scheduler(
                    config.scheduler,
                    scheduler_state,
                    metric=float(evaluation.total.detach()),
                )
                optimizer.param_groups[0]["lr"] = scheduler_state.learning_rate
                sampler_state = advanced_sampler
                adaptive_state = adaptive_update.state
                final_losses = MappingProxyType(
                    {
                        term: float(loss.detach())
                        for term, loss in evaluation.losses.items()
                    }
                )
        finally:
            torch.set_num_threads(thread_count_before)
    finally:
        torch.use_deterministic_algorithms(deterministic_before)

    if final_losses is None:
        raise RuntimeError("training segment executed no optimizer step")
    checkpoint = write_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        binding=binding,
        step=final_step,
        sampler_state=sampler_state,
        scheduler_state=scheduler_state,
        adaptive_state=adaptive_state,
    )
    return TrainingRunResult(
        step=final_step,
        model_state_hash=model_state_hash(model),
        sampler_state=sampler_state,
        scheduler_state=scheduler_state,
        adaptive_state=adaptive_state,
        final_losses=final_losses,
        checkpoint=checkpoint,
    )


__all__ = (
    "DeterministicTrainerConfig",
    "ObjectiveBuilder",
    "TrainingPopulation",
    "TrainingRunResult",
    "model_state_hash",
    "train_to_checkpoint",
)
