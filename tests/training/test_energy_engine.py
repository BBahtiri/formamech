from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from piginot.contracts import CanonicalQuantity, ExecutionProfile, MaterialSpec, Unit
from piginot.contracts.errors import ContractValidationError
from piginot.doe import EvidenceRole
from piginot.objectives import (
    PotentialEnergyConfig,
    PotentialEnergyTerms,
    TriangleDomain,
    plane_stress_potential_energy,
    stratified_triangle_quadrature,
)
from piginot.operators import (
    AxisAlignedTensionLifting,
    EnergyFormOperator,
    EnergyOperatorConfig,
    QueryFrame,
)
from piginot.training import (
    CosineTailSchedule,
    DeterministicSamplerPlan,
    EnergyTrainerConfig,
    GridSearchSpace,
    OutOfSearchSpaceError,
    TrainingPopulation,
    TrainingRunBinding,
    TrialConfig,
    load_training_checkpoint,
    model_state_hash,
    scheduler_state_for_step,
    train_energy_to_checkpoint,
)

_DTYPE = torch.float64
_OBJECTIVE_HASH = "a" * 64
_REVISION = "0123456789abcdef0123456789abcdef01234567"
_STEPS = 6
_MATERIAL = MaterialSpec(
    "energy-engine-test",
    "compressible_neo_hookean",
    {
        "elastic_modulus": CanonicalQuantity(760.0, Unit.MPA),
        "poisson_ratio": CanonicalQuantity(0.23, Unit.ONE),
    },
)


def _domain() -> TriangleDomain:
    vertices = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=_DTYPE,
    )
    cells = torch.tensor(
        [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]], dtype=torch.int64
    )
    return TriangleDomain(vertices, cells)


def _model() -> EnergyFormOperator:
    torch.manual_seed(5)
    config = EnergyOperatorConfig(
        parameter_names=("load",),
        hidden_dim=8,
        latent_token_count=2,
        parameter_hidden_dim=4,
        num_attention_heads=2,
        decoder_layers=1,
        positional_frequencies=2,
        output_init_std=0.05,
    )
    return EnergyFormOperator(config).to(_DTYPE)


class _Objective:
    def __init__(self) -> None:
        self.domain = _domain()
        self.loads = {"case-a": 0.1, "case-b": 0.2}

    def __call__(
        self, model: torch.nn.Module, sample_id: str, step: int
    ) -> PotentialEnergyTerms:
        assert isinstance(model, EnergyFormOperator)
        load = self.loads[sample_id]
        points, weights = stratified_triangle_quadrature(self.domain, 2)
        _, gradient = model.displacement_and_gradient(
            query_xy=points.unsqueeze(0),
            parameters=torch.tensor([[load]], dtype=_DTYPE),
            frame=QueryFrame(
                torch.zeros((1, 2), dtype=_DTYPE),
                torch.tensor([[2.0, 1.0]], dtype=_DTYPE),
            ),
            lifting=AxisAlignedTensionLifting(
                torch.tensor([[2.0, 1.0]], dtype=_DTYPE),
                torch.tensor([load], dtype=_DTYPE),
            ),
            create_graph=True,
        )
        return plane_stress_potential_energy(
            gradient[0],
            weights,
            _MATERIAL,
            config=PotentialEnergyConfig(points_per_cell=2),
            reference_area=self.domain.area,
        )


def _space(learning_rate: float = 1.0e-2) -> GridSearchSpace:
    return GridSearchSpace(
        learning_rates=(learning_rate,),
        physics_weight_scales=(1.0,),
        optimizer_steps=_STEPS,
        case_batch_size=2,
        maximum_trials=1,
    )


def _config() -> EnergyTrainerConfig:
    space = _space()
    (trial,) = space.trials
    return EnergyTrainerConfig(
        search_space=space,
        trial=trial,
        schedule=CosineTailSchedule(
            peak_learning_rate=1.0e-2,
            final_learning_rate=1.0e-4,
            constant_steps=3,
            total_steps=_STEPS,
        ),
    )


def _population(role: EvidenceRole = EvidenceRole.TRAIN) -> TrainingPopulation:
    return TrainingPopulation(
        dataset_snapshot_hash="1" * 64,
        public_manifest_hash="2" * 64,
        role=role,
        sample_ids=("case-a", "case-b"),
    )


def _plan() -> DeterministicSamplerPlan:
    return DeterministicSamplerPlan(
        sampler_id="energy-engine-test",
        population_size=2,
        batches=tuple((0, 1) if step % 2 == 0 else (1, 0) for step in range(_STEPS)),
    )


def _binding(model: EnergyFormOperator, **changes: object) -> TrainingRunBinding:
    values: dict[str, object] = {
        "dataset_snapshot_hash": "1" * 64,
        "public_manifest_hash": "2" * 64,
        "normalization_hash": "3" * 64,
        "architecture_id": model.architecture_id,
        "architecture_module_digest": "blake2b256:" + "4" * 64,
        "objective_registry_hash": _OBJECTIVE_HASH,
        "trainer_config_hash": _config().config_hash,
        "sampler_plan_hash": _plan().plan_hash,
        "initial_weights_hash": model_state_hash(model),
        "implementation_revision": _REVISION,
        "seed": 17,
        "execution_profile": ExecutionProfile(
            python_version="3.11.0",
            library_versions={"torch": "test"},
            blas_threads=1,
            device="cpu",
            dtype="float64",
            rng_algorithm="torch-cpu-mt19937",
            seed=17,
            container_digest=None,
        ),
    }
    values.update(changes)
    return TrainingRunBinding(**values)  # type: ignore[arg-type]


def _train(
    path: Path,
    *,
    target_step: int | None = None,
    resume_from: Path | None = None,
) -> tuple[str, list[float]]:
    model = _model()
    binding = _binding(model)
    result = train_energy_to_checkpoint(
        path,
        model=model,
        population=_population(),
        plan=_plan(),
        binding=binding,
        config=_config(),
        objective=_Objective(),
        objective_hash=_OBJECTIVE_HASH,
        target_step=target_step,
        resume_from=resume_from,
    )
    return result.model_state_hash, [record.total for record in result.history]


def test_schedule_is_constant_then_cosine() -> None:
    schedule = _config().schedule

    assert [schedule.learning_rate(step) for step in range(3)] == [1.0e-2] * 3
    assert schedule.learning_rate(4) == pytest.approx(
        1.0e-4 + 0.5 * (1.0e-2 - 1.0e-4) * (1.0 + math.cos(math.pi / 3.0))
    )
    assert schedule.learning_rate(6) == pytest.approx(1.0e-4)
    assert schedule.learning_rate(60) == pytest.approx(1.0e-4)
    state = scheduler_state_for_step(schedule, 2)
    assert (state.step, state.learning_rate, state.scheduler_id) == (
        2,
        1.0e-2,
        "constant-then-cosine-v1",
    )
    for changes in (
        {"final_learning_rate": 1.0},
        {"constant_steps": 7},
        {"total_steps": 0},
        {"peak_learning_rate": -1.0},
        {"schedule_id": "other"},
    ):
        with pytest.raises(ContractValidationError):
            replace(schedule, **changes)
    with pytest.raises(ContractValidationError):
        schedule.learning_rate(-1)


def test_trainer_config_refuses_out_of_space_trials_and_inconsistent_schedules() -> (
    None
):
    config = _config()
    outside = TrialConfig(
        learning_rate=2.0e-2,
        physics_weight_scale=1.0,
        optimizer_steps=_STEPS,
        case_batch_size=2,
    )
    with pytest.raises(OutOfSearchSpaceError):
        replace(config, trial=outside)
    with pytest.raises(ContractValidationError, match="total_steps"):
        replace(config, schedule=replace(config.schedule, total_steps=7))
    with pytest.raises(ContractValidationError, match="peak_learning_rate"):
        replace(config, schedule=replace(config.schedule, peak_learning_rate=2.0e-2))
    with pytest.raises(ContractValidationError):
        replace(config, gradient_clip_norm=0.0)
    with pytest.raises(ContractValidationError):
        replace(config, adam_betas=(0.9, 1.0))
    assert len(config.config_hash) == 64


def test_two_fresh_runs_are_bitwise_identical(tmp_path: Path) -> None:
    first_hash, first_losses = _train(tmp_path / "first")
    second_hash, second_losses = _train(tmp_path / "second")

    assert first_hash == second_hash
    assert first_losses == second_losses
    assert all(math.isfinite(value) for value in first_losses)
    first_files = sorted(p.name for p in (tmp_path / "first").iterdir())
    assert first_files == [
        "adaptive.json",
        "checkpoint.json",
        "model.safetensors",
        "optimizer.json",
        "state.safetensors",
    ]
    for name in first_files:
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "second" / name
        ).read_bytes()


def test_resume_at_half_budget_matches_the_uninterrupted_run(tmp_path: Path) -> None:
    straight_hash, straight_losses = _train(tmp_path / "straight")
    half_hash, half_losses = _train(tmp_path / "half", target_step=3)
    resumed_hash, resumed_losses = _train(
        tmp_path / "resumed", resume_from=tmp_path / "half"
    )

    assert half_hash != straight_hash
    assert resumed_hash == straight_hash
    assert half_losses + resumed_losses == straight_losses


def test_checkpoint_restores_model_and_schedule_cursor(tmp_path: Path) -> None:
    _train(tmp_path / "half", target_step=3)
    model = _model()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0e-2, foreach=False, fused=False
    )
    loaded = load_training_checkpoint(
        tmp_path / "half",
        model=model,
        optimizer=optimizer,
        binding=_binding(_model()),
    )
    assert loaded.manifest.step == 3
    assert loaded.scheduler_state == scheduler_state_for_step(_config().schedule, 3)
    assert model_state_hash(model) == loaded.manifest.model_state_hash


@pytest.mark.parametrize(
    "changes",
    [
        {"objective_registry_hash": "b" * 64},
        {"dataset_snapshot_hash": "5" * 64},
        {"implementation_revision": "f" * 40},
        {"normalization_hash": "6" * 64},
    ],
)
def test_any_identity_change_creates_a_new_run_hash(changes: dict[str, object]) -> None:
    model = _model()
    assert _binding(model, **changes).run_hash != _binding(model).run_hash


def test_profile_and_seed_changes_create_new_run_hashes() -> None:
    model = _model()
    base = _binding(model)
    threads = _binding(
        model,
        execution_profile=replace(base.execution_profile, blas_threads=2),
    )
    seed = _binding(
        model,
        seed=18,
        execution_profile=replace(base.execution_profile, seed=18),
    )
    assert len({base.run_hash, threads.run_hash, seed.run_hash}) == 3


def test_non_train_roles_cannot_enter_the_energy_trainer() -> None:
    for role in (EvidenceRole.DEVELOPMENT, EvidenceRole.HOLDOUT):
        with pytest.raises(PermissionError):
            _population(role)


def test_trainer_rejects_mismatched_bindings_and_inputs(tmp_path: Path) -> None:
    model = _model()
    common = {
        "model": model,
        "population": _population(),
        "plan": _plan(),
        "config": _config(),
        "objective": _Objective(),
    }
    with pytest.raises(ValueError, match="energy objective"):
        train_energy_to_checkpoint(
            tmp_path / "a",
            binding=_binding(model),
            objective_hash="c" * 64,
            **common,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="initial weights"):
        train_energy_to_checkpoint(
            tmp_path / "b",
            binding=_binding(model, initial_weights_hash="d" * 64),
            objective_hash=_OBJECTIVE_HASH,
            **common,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="target_step"):
        train_energy_to_checkpoint(
            tmp_path / "c",
            binding=_binding(model),
            objective_hash=_OBJECTIVE_HASH,
            target_step=_STEPS + 1,
            **common,  # type: ignore[arg-type]
        )
    with pytest.raises(FileNotFoundError):
        train_energy_to_checkpoint(
            tmp_path / "missing" / "d",
            binding=_binding(model),
            objective_hash=_OBJECTIVE_HASH,
            **common,  # type: ignore[arg-type]
        )
    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        train_energy_to_checkpoint(
            tmp_path / "exists",
            binding=_binding(model),
            objective_hash=_OBJECTIVE_HASH,
            **common,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="dtype"):
        train_energy_to_checkpoint(
            tmp_path / "e",
            model=model.to(torch.float32),
            population=_population(),
            plan=_plan(),
            binding=_binding(model.to(torch.float32)),
            config=_config(),
            objective=_Objective(),
            objective_hash=_OBJECTIVE_HASH,
        )


def test_non_finite_objective_aborts_before_any_checkpoint(tmp_path: Path) -> None:
    model = _model()

    def broken(
        model: torch.nn.Module, sample_id: str, step: int
    ) -> PotentialEnergyTerms:
        nan = torch.tensor(math.nan, dtype=_DTYPE)
        return PotentialEnergyTerms(nan, nan, nan, 1.0, 0)

    with pytest.raises(ValueError, match="finite"):
        train_energy_to_checkpoint(
            tmp_path / "nan",
            model=model,
            population=_population(),
            plan=_plan(),
            binding=_binding(model),
            config=_config(),
            objective=broken,
            objective_hash=_OBJECTIVE_HASH,
        )
    assert not (tmp_path / "nan").exists()


_SUBPROCESS_EXPORT = """
import sys
from pathlib import Path
sys.path.insert(0, {tests!r})
from training.test_energy_engine import _train
_train(Path(sys.argv[1]), target_step=2)
"""


def test_checkpoint_bytes_are_reproducible_across_processes(tmp_path: Path) -> None:
    tests_root = str(Path(__file__).resolve().parents[1])
    script = _SUBPROCESS_EXPORT.format(tests=tests_root)
    for name in ("one", "two"):
        completed = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / name)],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        assert completed.returncode == 0, completed.stderr
    for name in ("model.safetensors", "state.safetensors", "checkpoint.json"):
        assert (tmp_path / "one" / name).read_bytes() == (
            tmp_path / "two" / name
        ).read_bytes()


def test_trainer_validates_every_binding_and_plan_input(tmp_path: Path) -> None:
    model = _model()
    binding = _binding(model)
    base = {
        "model": model,
        "population": _population(),
        "plan": _plan(),
        "binding": binding,
        "config": _config(),
        "objective": _Objective(),
        "objective_hash": _OBJECTIVE_HASH,
    }
    short_plan = DeterministicSamplerPlan(
        sampler_id="energy-engine-test",
        population_size=2,
        batches=((0, 1),) * (_STEPS - 1),
    )
    single_plan = DeterministicSamplerPlan(
        sampler_id="energy-engine-test",
        population_size=2,
        batches=((0,),) * _STEPS,
    )
    wide_population = TrainingPopulation(
        dataset_snapshot_hash="1" * 64,
        public_manifest_hash="2" * 64,
        role=EvidenceRole.TRAIN,
        sample_ids=("case-a", "case-b", "case-c"),
    )
    cases: list[tuple[dict[str, object], type[Exception], str]] = [
        ({"population": object()}, TypeError, "population"),
        ({"plan": object()}, TypeError, "plan"),
        ({"binding": object()}, TypeError, "binding"),
        ({"config": object()}, TypeError, "config"),
        ({"model": object()}, TypeError, "torch module"),
        ({"objective": object()}, TypeError, "callable"),
        (
            {"binding": _binding(model, dataset_snapshot_hash="9" * 64)},
            ValueError,
            "dataset snapshot",
        ),
        (
            {"binding": _binding(model, public_manifest_hash="9" * 64)},
            ValueError,
            "public manifest",
        ),
        (
            {"binding": _binding(model, sampler_plan_hash="9" * 64)},
            ValueError,
            "sampler plan does not match",
        ),
        (
            {"binding": _binding(model, trainer_config_hash="9" * 64)},
            ValueError,
            "trainer config",
        ),
        (
            {"population": wide_population},
            ValueError,
            "population_size",
        ),
        (
            {
                "plan": short_plan,
                "binding": _binding(model, sampler_plan_hash=short_plan.plan_hash),
            },
            ValueError,
            "plan length",
        ),
        (
            {
                "plan": single_plan,
                "binding": _binding(model, sampler_plan_hash=single_plan.plan_hash),
            },
            ValueError,
            "batch size",
        ),
    ]
    for index, (changes, error, message) in enumerate(cases):
        with pytest.raises(error, match=message):
            train_energy_to_checkpoint(
                tmp_path / f"case-{index}",
                **{**base, **changes},  # type: ignore[arg-type]
            )
    with pytest.raises(TypeError, match="pathlib"):
        train_energy_to_checkpoint("not-a-path", **base)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="resume_from"):
        train_energy_to_checkpoint(
            tmp_path / "resume",
            resume_from="not-a-path",  # type: ignore[arg-type]
            **base,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="progress"):
        train_energy_to_checkpoint(
            tmp_path / "progress",
            progress=object(),  # type: ignore[arg-type]
            **base,  # type: ignore[arg-type]
        )


def test_trainer_rejects_parameterless_models_and_bad_contracts(tmp_path: Path) -> None:
    class _Empty(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("state", torch.zeros(1, dtype=_DTYPE))

    empty = _Empty()
    with pytest.raises(ValueError, match="trainable parameters"):
        train_energy_to_checkpoint(
            tmp_path / "empty",
            model=empty,
            population=_population(),
            plan=_plan(),
            binding=_binding(_model(), initial_weights_hash=model_state_hash(empty)),
            config=_config(),
            objective=_Objective(),
            objective_hash=_OBJECTIVE_HASH,
        )
    config = _config()
    with pytest.raises(ContractValidationError):
        replace(config, search_space=object())
    with pytest.raises(ContractValidationError):
        replace(config, trial=object())
    with pytest.raises(ContractValidationError):
        replace(config, schedule=object())
    with pytest.raises(ContractValidationError):
        replace(config, adam_betas=(0.9,))
    with pytest.raises(ContractValidationError):
        replace(config, adam_weight_decay=-1.0)
    with pytest.raises(TypeError):
        scheduler_state_for_step(object(), 0)  # type: ignore[arg-type]


def test_progress_callback_receives_every_step(tmp_path: Path) -> None:
    model = _model()
    records: list[int] = []
    train_energy_to_checkpoint(
        tmp_path / "progress",
        model=model,
        population=_population(),
        plan=_plan(),
        binding=_binding(model),
        config=_config(),
        objective=_Objective(),
        objective_hash=_OBJECTIVE_HASH,
        target_step=2,
        progress=lambda record: records.append(record.step),
    )
    assert records == [1, 2]
