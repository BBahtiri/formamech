from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch

from piginot.contracts import ExecutionProfile, hash_spec
from piginot.doe import EvidenceRole
from piginot.objectives import (
    EquilibriumForm,
    ObjectiveMode,
    ObjectiveRegistry,
    ObjectiveResiduals,
    ObjectiveTerm,
)
from piginot.training.adaptive import AdaptiveWeightConfig
from piginot.training.engine import (
    DeterministicTrainerConfig,
    TrainingPopulation,
    model_state_hash,
    train_to_checkpoint,
)
from piginot.training.identity import TrainingRunBinding
from piginot.training.scheduler import PlateauSchedulerConfig
from piginot.training.search import GridSearchSpace, TrialConfig
from piginot.training.state import DeterministicSamplerPlan

_HASH = "0123456789abcdef" * 4


def _model() -> torch.nn.Linear:
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(0.25)
        model.bias.fill_(0.1)
    return model


def _registry() -> ObjectiveRegistry:
    return ObjectiveRegistry(
        mode=ObjectiveMode.HYBRID,
        equilibrium_form=EquilibriumForm.STRONG,
        term_weights={
            ObjectiveTerm.DATA_U: 1.0,
            ObjectiveTerm.DATA_P: 2.0,
            ObjectiveTerm.EQUILIBRIUM_STRONG: 3.0,
            ObjectiveTerm.CONSTITUTIVE: 4.0,
            ObjectiveTerm.ANGULAR: 5.0,
            ObjectiveTerm.DIRICHLET: 6.0,
            ObjectiveTerm.TRACTION_FREE: 7.0,
        },
    )


def _population(role: EvidenceRole = EvidenceRole.TRAIN) -> TrainingPopulation:
    return TrainingPopulation(
        dataset_snapshot_hash=_HASH,
        public_manifest_hash="1" * 64,
        role=role,
        sample_ids=tuple(f"train-{index}" for index in range(6)),
    )


def _plan() -> DeterministicSamplerPlan:
    return DeterministicSamplerPlan(
        sampler_id="fixture-complete-schedule-v1",
        population_size=6,
        batches=((0, 1), (2, 3), (4, 5), (1, 4), (0, 3), (2, 5)),
    )


def _trial() -> TrialConfig:
    return TrialConfig(
        learning_rate=1.0e-2,
        physics_weight_scale=1.0,
        optimizer_steps=6,
        case_batch_size=2,
    )


def _config() -> DeterministicTrainerConfig:
    trial = _trial()
    return DeterministicTrainerConfig(
        search_space=GridSearchSpace(
            learning_rates=(trial.learning_rate,),
            physics_weight_scales=(trial.physics_weight_scale,),
            optimizer_steps=trial.optimizer_steps,
            case_batch_size=trial.case_batch_size,
            maximum_trials=1,
        ),
        trial=trial,
        adaptive=AdaptiveWeightConfig(
            beta=0.1,
            warmup_steps=2,
            update_period=2,
        ),
        scheduler=PlateauSchedulerConfig(
            patience=2,
            reduction_factor=0.5,
            absolute_threshold=0.0,
            cooldown_steps=1,
            minimum_learning_rate=1.0e-5,
        ),
        gradient_clip_norm=10.0,
    )


def _profile(seed: int) -> ExecutionProfile:
    return ExecutionProfile(
        python_version="3.13.0",
        library_versions={"torch": torch.__version__, "safetensors": "0.8.1"},
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="torch-cpu-mt19937",
        seed=seed,
        container_digest=None,
    )


def _binding(model: torch.nn.Module, *, seed: int = 8128) -> TrainingRunBinding:
    population = _population()
    config = _config()
    registry = _registry()
    return TrainingRunBinding(
        dataset_snapshot_hash=population.dataset_snapshot_hash,
        public_manifest_hash=population.public_manifest_hash,
        normalization_hash="2" * 64,
        architecture_id="tiny-hybrid-fixture-v1",
        architecture_module_digest="blake2b256:" + "3" * 64,
        objective_registry_hash=hash_spec(registry, "recipe"),
        trainer_config_hash=config.config_hash,
        sampler_plan_hash=_plan().plan_hash,
        initial_weights_hash=model_state_hash(model),
        implementation_revision="8" * 40,
        seed=seed,
        execution_profile=_profile(seed),
    )


class _FixtureObjective:
    def __init__(self) -> None:
        self.call_count = 0
        self._x = {
            f"train-{index}": torch.tensor([0.25 * (index + 1)], dtype=torch.float64)
            for index in range(6)
        }

    def __call__(
        self,
        model: torch.nn.Module,
        sample_ids: tuple[str, ...],
        step: int,
    ) -> ObjectiveResiduals:
        del step
        self.call_count += 1
        inputs = torch.stack(tuple(self._x[sample_id] for sample_id in sample_ids))
        prediction = model(inputs)
        stochastic = prediction + 1.0e-3 * torch.rand_like(prediction)
        return ObjectiveResiduals(
            {
                ObjectiveTerm.DATA_U: stochastic - 0.5,
                ObjectiveTerm.DATA_P: stochastic + 0.2,
                ObjectiveTerm.EQUILIBRIUM_STRONG: stochastic + 0.4,
                ObjectiveTerm.CONSTITUTIVE: 1.1 * stochastic + 0.3,
                ObjectiveTerm.ANGULAR: 0.9 * stochastic + 0.5,
                ObjectiveTerm.DIRICHLET: 1.2 * stochastic + 0.6,
                ObjectiveTerm.TRACTION_FREE: 0.8 * stochastic + 0.7,
            }
        )


def _assert_model_equal(left: torch.nn.Module, right: torch.nn.Module) -> None:
    assert left.state_dict().keys() == right.state_dict().keys()
    for name, expected in left.state_dict().items():
        torch.testing.assert_close(right.state_dict()[name], expected, rtol=0, atol=0)


def test_same_seed_is_identical_and_halfway_resume_is_bitwise_exact(
    tmp_path: Path,
) -> None:
    uninterrupted_model = _model()
    binding = _binding(uninterrupted_model)
    uninterrupted = train_to_checkpoint(
        tmp_path / "uninterrupted",
        model=uninterrupted_model,
        population=_population(),
        plan=_plan(),
        binding=binding,
        config=_config(),
        base_registry=_registry(),
        objective=_FixtureObjective(),
    )

    repeated_model = _model()
    repeated = train_to_checkpoint(
        tmp_path / "repeated",
        model=repeated_model,
        population=_population(),
        plan=_plan(),
        binding=_binding(repeated_model),
        config=_config(),
        base_registry=_registry(),
        objective=_FixtureObjective(),
    )

    partial_model = _model()
    partial = train_to_checkpoint(
        tmp_path / "halfway",
        model=partial_model,
        population=_population(),
        plan=_plan(),
        binding=_binding(partial_model),
        config=_config(),
        base_registry=_registry(),
        objective=_FixtureObjective(),
        target_step=3,
    )
    resumed_model = _model()
    resumed = train_to_checkpoint(
        tmp_path / "resumed",
        model=resumed_model,
        population=_population(),
        plan=_plan(),
        binding=_binding(resumed_model),
        config=_config(),
        base_registry=_registry(),
        objective=_FixtureObjective(),
        resume_from=partial.checkpoint.path,
    )

    assert uninterrupted.step == repeated.step == resumed.step == 6
    assert partial.step == 3
    assert uninterrupted.model_state_hash == repeated.model_state_hash
    assert uninterrupted.model_state_hash == resumed.model_state_hash
    assert (
        uninterrupted.checkpoint.manifest.optimizer_state_hash
        == repeated.checkpoint.manifest.optimizer_state_hash
        == resumed.checkpoint.manifest.optimizer_state_hash
    )
    assert (
        uninterrupted.checkpoint.manifest.torch_cpu_rng_hash
        == repeated.checkpoint.manifest.torch_cpu_rng_hash
        == resumed.checkpoint.manifest.torch_cpu_rng_hash
    )
    assert uninterrupted.sampler_state == resumed.sampler_state
    assert uninterrupted.scheduler_state == resumed.scheduler_state
    assert uninterrupted.adaptive_state == resumed.adaptive_state
    assert uninterrupted.final_losses == resumed.final_losses
    _assert_model_equal(uninterrupted_model, repeated_model)
    _assert_model_equal(uninterrupted_model, resumed_model)


def test_existing_checkpoint_refuses_before_model_or_objective_mutation(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "already-present"
    destination.mkdir()
    model = _model()
    initial_hash = model_state_hash(model)
    objective = _FixtureObjective()

    with pytest.raises(FileExistsError, match="already exists"):
        train_to_checkpoint(
            destination,
            model=model,
            population=_population(),
            plan=_plan(),
            binding=_binding(model),
            config=_config(),
            base_registry=_registry(),
            objective=objective,
        )

    assert objective.call_count == 0
    assert model_state_hash(model) == initial_hash


def test_trainer_refuses_non_train_population_before_objective_access() -> None:
    with pytest.raises(PermissionError, match="DEVELOPMENT"):
        _population(EvidenceRole.DEVELOPMENT)
    with pytest.raises(PermissionError, match="HOLDOUT"):
        _population(EvidenceRole.HOLDOUT)


def test_trainer_refuses_identity_plan_and_search_drift(tmp_path: Path) -> None:
    model = _model()
    binding = _binding(model)
    config = _config()

    with pytest.raises(ValueError, match="dataset snapshot"):
        train_to_checkpoint(
            tmp_path / "dataset-drift",
            model=model,
            population=replace(_population(), dataset_snapshot_hash="9" * 64),
            plan=_plan(),
            binding=binding,
            config=config,
            base_registry=_registry(),
            objective=_FixtureObjective(),
        )
    with pytest.raises(ValueError, match="sampler plan"):
        train_to_checkpoint(
            tmp_path / "plan-drift",
            model=model,
            population=_population(),
            plan=replace(_plan(), sampler_id="changed-plan"),
            binding=binding,
            config=config,
            base_registry=_registry(),
            objective=_FixtureObjective(),
        )
    with pytest.raises(ValueError, match="trainer config"):
        train_to_checkpoint(
            tmp_path / "config-drift",
            model=model,
            population=_population(),
            plan=_plan(),
            binding=binding,
            config=replace(config, gradient_clip_norm=5.0),
            base_registry=_registry(),
            objective=_FixtureObjective(),
        )


def test_trainer_contracts_are_frozen_and_fail_closed() -> None:
    population = _population()
    config = _config()

    with pytest.raises(FrozenInstanceError):
        population.sample_ids = ("other",)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.gradient_clip_norm = 1.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="sorted and unique"):
        replace(population, sample_ids=("b", "a"))
    with pytest.raises(ValueError, match="optimizer_steps"):
        replace(config, trial=replace(config.trial, optimizer_steps=5))
