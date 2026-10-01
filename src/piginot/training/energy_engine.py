"""Deterministic potential-energy training with exact resume.

This trainer is the energy-form counterpart of :func:`train_to_checkpoint`.
It reuses the S13 identity and persistence surface unchanged:

* :class:`TrainingRunBinding` binds data, architecture, objective, trainer
  configuration, sampler plan, initial weights, code revision, seed and
  execution profile into ``run_hash``;
* :class:`DeterministicSamplerPlan` fixes every case batch in advance;
* :func:`write_training_checkpoint` and :func:`load_training_checkpoint`
  persist and restore model, Adam moments, the torch CPU generator, sampler and
  schedule cursors, so a resumed run continues bit for bit.

The objective is evaluated one sample at a time and back-propagated with weight
``1 / batch``, which bounds memory to one sample's graph. The stratified
quadrature draws come from the torch CPU generator, whose state is part of
every checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Protocol

import torch

from piginot.contracts import hash_spec
from piginot.contracts.errors import ContractValidationError
from piginot.contracts.specs import SpecContract
from piginot.objectives.energy import PotentialEnergyTerms
from piginot.training.adaptive import AdaptiveWeightState
from piginot.training.checkpoint import (
    WrittenTrainingCheckpoint,
    load_training_checkpoint,
    model_state_hash,
    write_training_checkpoint,
)
from piginot.training.engine import TrainingPopulation
from piginot.training.identity import TrainingRunBinding
from piginot.training.search import GridSearchSpace, TrialConfig
from piginot.training.state import (
    DeterministicSamplerPlan,
    DeterministicSamplerState,
    TrainingSchedulerState,
    next_sampler_batch,
)

COSINE_TAIL_SCHEDULE_ID: Final = "constant-then-cosine-v1"


def _positive_float(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ContractValidationError(f"{label} must be finite and positive")
    return float(value)


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class CosineTailSchedule(SpecContract):
    """Constant learning rate followed by a cosine anneal to a final rate."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    peak_learning_rate: float
    final_learning_rate: float
    constant_steps: int
    total_steps: int
    schedule_id: str = COSINE_TAIL_SCHEDULE_ID

    def __post_init__(self) -> None:
        if self.schedule_id != COSINE_TAIL_SCHEDULE_ID:
            raise ContractValidationError(
                f"schedule_id must be {COSINE_TAIL_SCHEDULE_ID!r}"
            )
        peak = _positive_float(self.peak_learning_rate, "peak_learning_rate")
        final = _positive_float(self.final_learning_rate, "final_learning_rate")
        if final > peak:
            raise ContractValidationError(
                "final_learning_rate must not exceed peak_learning_rate"
            )
        constant = _nonnegative_int(self.constant_steps, "constant_steps")
        total = _nonnegative_int(self.total_steps, "total_steps")
        if total == 0 or constant > total:
            raise ContractValidationError(
                "schedule needs 0 <= constant_steps <= total_steps and total_steps > 0"
            )
        object.__setattr__(self, "peak_learning_rate", peak)
        object.__setattr__(self, "final_learning_rate", final)

    def learning_rate(self, step_index: int) -> float:
        """Return the rate applied by the optimizer step at ``step_index``."""

        index = _nonnegative_int(step_index, "step_index")
        if index < self.constant_steps:
            return self.peak_learning_rate
        span = max(1, self.total_steps - self.constant_steps)
        fraction = min(1.0, (index - self.constant_steps) / span)
        return self.final_learning_rate + 0.5 * (
            self.peak_learning_rate - self.final_learning_rate
        ) * (1.0 + math.cos(math.pi * fraction))


@dataclass(frozen=True, slots=True)
class EnergyTrainerConfig(SpecContract):
    """Complete energy-form optimizer policy bound by ``trainer_config_hash``."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    search_space: GridSearchSpace
    trial: TrialConfig
    schedule: CosineTailSchedule
    gradient_clip_norm: float = 1.0
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
        self.search_space.authorize(self.trial)
        if type(self.schedule) is not CosineTailSchedule:
            raise ContractValidationError("schedule must be a CosineTailSchedule")
        if self.schedule.total_steps != self.trial.optimizer_steps:
            raise ContractValidationError(
                "schedule total_steps differ from trial optimizer_steps"
            )
        if self.schedule.peak_learning_rate != self.trial.learning_rate:
            raise ContractValidationError(
                "schedule peak_learning_rate differs from the trial learning_rate"
            )
        object.__setattr__(
            self,
            "gradient_clip_norm",
            _positive_float(self.gradient_clip_norm, "gradient_clip_norm"),
        )
        if type(self.adam_betas) is not tuple or len(self.adam_betas) != 2:
            raise ContractValidationError("adam_betas must contain two values")
        betas = tuple(float(value) for value in self.adam_betas)
        if any(not math.isfinite(value) or not 0.0 <= value < 1.0 for value in betas):
            raise ContractValidationError("adam betas must lie in [0, 1)")
        object.__setattr__(self, "adam_betas", betas)
        object.__setattr__(self, "adam_eps", _positive_float(self.adam_eps, "adam_eps"))
        decay = float(self.adam_weight_decay)
        if not math.isfinite(decay) or decay < 0.0:
            raise ContractValidationError("adam_weight_decay must be non-negative")
        object.__setattr__(self, "adam_weight_decay", decay)

    @property
    def config_hash(self) -> str:
        return hash_spec(self, "recipe")


@dataclass(frozen=True, slots=True)
class EnergyStepRecord:
    """Batch-averaged diagnostics of one completed optimizer step."""

    step: int
    learning_rate: float
    total: float
    energy: float
    barrier: float
    minimum_determinant: float
    inadmissible_points: int
    gradient_norm: float


class EnergySampleObjective(Protocol):
    """Return the graph-connected energy terms of one registered TRAIN sample."""

    def __call__(
        self,
        model: torch.nn.Module,
        sample_id: str,
        step: int,
    ) -> PotentialEnergyTerms: ...


@dataclass(frozen=True, slots=True)
class EnergyTrainingResult:
    """Final immutable trace returned by one fresh or resumed segment."""

    step: int
    model_state_hash: str
    sampler_state: DeterministicSamplerState
    scheduler_state: TrainingSchedulerState
    history: tuple[EnergyStepRecord, ...]
    checkpoint: WrittenTrainingCheckpoint


def scheduler_state_for_step(
    schedule: CosineTailSchedule,
    step: int,
) -> TrainingSchedulerState:
    """Return the schedule cursor after ``step`` completed optimizer steps."""

    if type(schedule) is not CosineTailSchedule:
        raise TypeError("schedule must be a CosineTailSchedule")
    completed = _nonnegative_int(step, "step")
    return TrainingSchedulerState(
        scheduler_id=schedule.schedule_id,
        step=completed,
        learning_rate=schedule.learning_rate(completed),
        best_metric=None,
        bad_observations=0,
        cooldown_remaining=0,
        observation_count=completed,
    )


def _validate_inputs(
    *,
    model: torch.nn.Module,
    population: TrainingPopulation,
    plan: DeterministicSamplerPlan,
    binding: TrainingRunBinding,
    config: EnergyTrainerConfig,
    objective_hash: str,
) -> None:
    if type(population) is not TrainingPopulation:
        raise TypeError("population must be an exact TrainingPopulation")
    if type(plan) is not DeterministicSamplerPlan:
        raise TypeError("plan must be an exact DeterministicSamplerPlan")
    if type(binding) is not TrainingRunBinding:
        raise TypeError("binding must be an exact TrainingRunBinding")
    if type(config) is not EnergyTrainerConfig:
        raise TypeError("config must be an exact EnergyTrainerConfig")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if binding.dataset_snapshot_hash != population.dataset_snapshot_hash:
        raise ValueError("dataset snapshot does not match the training run binding")
    if binding.public_manifest_hash != population.public_manifest_hash:
        raise ValueError("public manifest does not match the training run binding")
    if binding.sampler_plan_hash != plan.plan_hash:
        raise ValueError("sampler plan does not match the training run binding")
    if binding.trainer_config_hash != config.config_hash:
        raise ValueError("trainer config does not match the training run binding")
    if binding.objective_registry_hash != objective_hash:
        raise ValueError("energy objective does not match the training run binding")
    if binding.initial_weights_hash != model_state_hash(model):
        raise ValueError("initial weights do not match the training run binding")
    if plan.population_size != len(population.sample_ids):
        raise ValueError("sampler population_size does not match sample_ids")
    if len(plan.batches) != config.trial.optimizer_steps:
        raise ValueError("sampler plan length does not match optimizer_steps")
    if any(len(batch) != config.trial.case_batch_size for batch in plan.batches):
        raise ValueError("sampler batch size does not match case_batch_size")
    parameters = tuple(model.parameters())
    if not parameters:
        raise ValueError("model must contain trainable parameters")
    expected_dtype = {"float32": torch.float32, "float64": torch.float64}.get(
        binding.execution_profile.dtype
    )
    if expected_dtype is None or any(
        parameter.dtype is not expected_dtype or parameter.device.type != "cpu"
        for parameter in parameters
    ):
        raise ValueError("model dtype/device does not match the execution profile")


def train_energy_to_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    population: TrainingPopulation,
    plan: DeterministicSamplerPlan,
    binding: TrainingRunBinding,
    config: EnergyTrainerConfig,
    objective: EnergySampleObjective,
    objective_hash: str,
    target_step: int | None = None,
    resume_from: Path | None = None,
    progress: Callable[[EnergyStepRecord], None] | None = None,
) -> EnergyTrainingResult:
    """Train one exact segment and persist its complete resumable state.

    ``model`` must carry the registered initial weights on every call; when
    ``resume_from`` is given the checkpoint replaces them, together with the
    optimizer moments, generator, sampler and schedule cursors.
    """

    if not isinstance(checkpoint_path, Path):
        raise TypeError("checkpoint_path must be a pathlib.Path")
    if resume_from is not None and not isinstance(resume_from, Path):
        raise TypeError("resume_from must be a pathlib.Path or None")
    if not callable(objective):
        raise TypeError("objective must be callable")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    if not checkpoint_path.parent.is_dir() or checkpoint_path.parent.is_symlink():
        raise FileNotFoundError("checkpoint parent directory must already exist")
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        raise FileExistsError("checkpoint directory already exists")
    _validate_inputs(
        model=model,
        population=population,
        plan=plan,
        binding=binding,
        config=config,
        objective_hash=objective_hash,
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
        lr=config.schedule.learning_rate(0),
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.adam_weight_decay,
        foreach=False,
        fused=False,
    )
    if resume_from is None:
        torch.manual_seed(binding.seed)
        sampler_state = plan.initial_state
        scheduler_state = scheduler_state_for_step(config.schedule, 0)
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
        if scheduler_state != scheduler_state_for_step(
            config.schedule, sampler_state.next_step
        ):
            raise ValueError("checkpoint schedule cursor differs from the policy")
    if sampler_state.next_step >= final_step:
        raise ValueError("target_step must be after the restored checkpoint step")

    history: list[EnergyStepRecord] = []
    batch_size = config.trial.case_batch_size
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    threads_before = torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    try:
        torch.set_num_threads(binding.execution_profile.blas_threads)
        model.train()
        while sampler_state.next_step < final_step:
            step_index = sampler_state.next_step
            learning_rate = config.schedule.learning_rate(step_index)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            batch_indices, advanced_sampler = next_sampler_batch(plan, sampler_state)
            completed_step = step_index + 1
            optimizer.zero_grad(set_to_none=True)
            totals: list[float] = []
            energies: list[float] = []
            barriers: list[float] = []
            minimum_determinant = math.inf
            inadmissible = 0
            for index in batch_indices:
                terms = objective(model, population.sample_ids[index], completed_step)
                if type(terms) is not PotentialEnergyTerms:
                    raise TypeError("objective must return PotentialEnergyTerms")
                if not bool(torch.isfinite(terms.total).item()):
                    raise ValueError("energy objective must be finite")
                (terms.total / batch_size).backward()  # type: ignore[no-untyped-call]
                totals.append(float(terms.total.detach()))
                energies.append(float(terms.energy.detach()))
                barriers.append(float(terms.barrier.detach()))
                minimum_determinant = min(
                    minimum_determinant, terms.minimum_determinant
                )
                inadmissible += terms.inadmissible_points
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
            )
            optimizer.step()
            sampler_state = advanced_sampler
            record = EnergyStepRecord(
                step=completed_step,
                learning_rate=learning_rate,
                total=math.fsum(totals) / len(totals),
                energy=math.fsum(energies) / len(energies),
                barrier=math.fsum(barriers) / len(barriers),
                minimum_determinant=minimum_determinant,
                inadmissible_points=inadmissible,
                gradient_norm=gradient_norm,
            )
            history.append(record)
            if progress is not None:
                progress(record)
    finally:
        torch.set_num_threads(threads_before)
        torch.use_deterministic_algorithms(deterministic_before)

    scheduler_state = scheduler_state_for_step(config.schedule, final_step)
    for group in optimizer.param_groups:
        group["lr"] = scheduler_state.learning_rate
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
    return EnergyTrainingResult(
        step=final_step,
        model_state_hash=model_state_hash(model),
        sampler_state=sampler_state,
        scheduler_state=scheduler_state,
        history=tuple(history),
        checkpoint=checkpoint,
    )


__all__ = (
    "COSINE_TAIL_SCHEDULE_ID",
    "CosineTailSchedule",
    "EnergySampleObjective",
    "EnergyStepRecord",
    "EnergyTrainerConfig",
    "EnergyTrainingResult",
    "scheduler_state_for_step",
    "train_energy_to_checkpoint",
)
