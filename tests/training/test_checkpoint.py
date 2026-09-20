from __future__ import annotations

import hashlib
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from piginot.contracts import ExecutionProfile, canonical_json_bytes, hash_spec
from piginot.objectives import ObjectiveTerm
from piginot.training.adaptive import AdaptiveWeightState
from piginot.training.checkpoint import (
    TrainingCheckpointManifest,
    export_model_safetensors,
    load_training_checkpoint,
    write_training_checkpoint,
)
from piginot.training.identity import TrainingRunBinding
from piginot.training.state import (
    DeterministicSamplerPlan,
    DeterministicSamplerState,
    TrainingSchedulerState,
    next_sampler_batch,
)

_HASH = "0123456789abcdef" * 4


def _plan() -> DeterministicSamplerPlan:
    return DeterministicSamplerPlan(
        sampler_id="blake2b-batch-plan-v1",
        population_size=12,
        batches=(
            (4, 1),
            (0, 5),
            (2, 3),
            (8, 6),
            (11, 7),
            (9, 10),
        ),
    )


def _profile(seed: int = 1234) -> ExecutionProfile:
    return ExecutionProfile(
        python_version="3.13.0",
        library_versions={"torch": torch.__version__, "safetensors": "0.8.1"},
        blas_threads=1,
        device="cpu",
        dtype="float32",
        rng_algorithm="torch-cpu-mt19937",
        seed=seed,
        container_digest=None,
    )


def _binding(
    seed: int = 1234,
    *,
    sampler_plan_hash: str | None = None,
) -> TrainingRunBinding:
    return TrainingRunBinding(
        dataset_snapshot_hash=_HASH,
        public_manifest_hash="1" * 64,
        normalization_hash="2" * 64,
        architecture_id="tiny-linear-v1",
        architecture_module_digest="blake2b256:" + "3" * 64,
        objective_registry_hash="4" * 64,
        trainer_config_hash="5" * 64,
        sampler_plan_hash=(
            _plan().plan_hash if sampler_plan_hash is None else sampler_plan_hash
        ),
        initial_weights_hash="7" * 64,
        implementation_revision="8" * 40,
        seed=seed,
        execution_profile=_profile(seed),
    )


def _model() -> torch.nn.Linear:
    return torch.nn.Linear(1, 1)


def _optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float = 0.05,
    betas: tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Adam:
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        foreach=False,
        fused=False,
    )


def _scheduler(
    step: int = 0,
    learning_rate: float = 0.05,
) -> TrainingSchedulerState:
    return TrainingSchedulerState(
        scheduler_id="fit-loss-plateau-v1",
        step=step,
        learning_rate=learning_rate,
        best_metric=None if step == 0 else 0.25,
        bad_observations=0 if step == 0 else 2,
        cooldown_remaining=0,
        observation_count=0 if step == 0 else 3,
    )


def _adaptive(step: int = 0) -> AdaptiveWeightState:
    if step == 0:
        return AdaptiveWeightState.initial()
    return AdaptiveWeightState(
        multipliers={
            ObjectiveTerm.CONSTITUTIVE: 1.2,
            ObjectiveTerm.ANGULAR: 0.9,
            ObjectiveTerm.DIRICHLET: 1.1,
            ObjectiveTerm.TRACTION_FREE: 1.3,
        },
        update_count=1,
        last_update_step=step - 1,
    )


def _file_hash(path: Path) -> str:
    return hashlib.blake2b(path.read_bytes(), digest_size=32).hexdigest()


def _train_one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
    plan: DeterministicSamplerPlan,
    sampler: DeterministicSamplerState,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
) -> DeterministicSamplerState:
    batch, sampler = next_sampler_batch(plan, sampler)
    prediction = model(x[list(batch)])
    stochastic_offset = 0.01 * torch.rand_like(prediction)
    loss = torch.nn.functional.mse_loss(prediction + stochastic_offset, y[list(batch)])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    return sampler


def _assert_optimizer_states_equal(
    left: torch.optim.Adam,
    right: torch.optim.Adam,
) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state["param_groups"] == right_state["param_groups"]
    assert left_state["state"].keys() == right_state["state"].keys()
    for parameter_index, expected in left_state["state"].items():
        actual = right_state["state"][parameter_index]
        assert actual.keys() == expected.keys()
        for name, expected_value in expected.items():
            torch.testing.assert_close(actual[name], expected_value, rtol=0, atol=0)


def test_checkpoint_writes_complete_canonical_lineage_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    torch.manual_seed(101)
    binding = _binding()
    model = _model()
    optimizer = _optimizer(model)

    checkpoint = write_training_checkpoint(
        tmp_path / "step-00000000",
        model=model,
        optimizer=optimizer,
        binding=binding,
        step=0,
        sampler_state=_plan().initial_state,
        scheduler_state=_scheduler(),
        adaptive_state=_adaptive(),
    )

    manifest_path = checkpoint.path / "checkpoint.json"
    assert manifest_path.read_bytes() == checkpoint.manifest.to_bytes()
    assert checkpoint.manifest.checkpoint_id == "CP-TRAINING"
    assert checkpoint.manifest.run_hash == binding.run_hash
    assert checkpoint.manifest.dataset_snapshot_hash == binding.dataset_snapshot_hash
    assert checkpoint.manifest.execution_profile_hash == binding.execution_profile_hash
    assert checkpoint.manifest.sampler_plan_hash == binding.sampler_plan_hash
    assert checkpoint.manifest.manifest_hash == hash_spec(
        checkpoint.manifest,
        "content",
    )
    assert checkpoint.sampler_state == _plan().initial_state
    assert checkpoint.scheduler_state == _scheduler()
    assert checkpoint.adaptive_state == _adaptive()
    assert {path.name for path in checkpoint.path.iterdir()} == {
        "adaptive.json",
        "checkpoint.json",
        "model.safetensors",
        "optimizer.json",
        "state.safetensors",
    }

    with pytest.raises(FileExistsError, match="already exists"):
        write_training_checkpoint(
            checkpoint.path,
            model=model,
            optimizer=optimizer,
            binding=binding,
            step=0,
            sampler_state=_plan().initial_state,
            scheduler_state=_scheduler(),
            adaptive_state=_adaptive(),
        )


def test_model_export_is_model_only_identity_bound_and_write_once(
    tmp_path: Path,
) -> None:
    safetensors = pytest.importorskip("safetensors")
    torch.manual_seed(102)
    binding = _binding()
    model = _model()
    path = tmp_path / "model.safetensors"

    exported, digest = export_model_safetensors(
        path,
        model=model,
        binding=binding,
    )

    with safetensors.safe_open(exported, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(model.state_dict())
        metadata = handle.metadata()
    assert metadata is not None
    assert metadata["run_hash"] == binding.run_hash
    assert metadata["architecture_id"] == binding.architecture_id
    assert metadata["architecture_module_digest"] == binding.architecture_module_digest
    assert digest == _file_hash(path)
    with pytest.raises(FileExistsError):
        export_model_safetensors(path, model=model, binding=binding)


def test_interrupt_resume_restores_rng_adam_scheduler_sampler_and_adaptive_state(
    tmp_path: Path,
) -> None:
    plan = _plan()
    binding = _binding(seed=103)
    x = torch.linspace(-1.0, 1.0, 12, dtype=torch.float32).reshape(-1, 1)
    y = 2.0 * x + 1.0

    torch.manual_seed(103)
    template = _model()
    initial_state = {
        name: tensor.detach().clone() for name, tensor in template.state_dict().items()
    }
    initial_rng = torch.random.get_rng_state().clone()

    uninterrupted = _model()
    uninterrupted.load_state_dict(initial_state)
    uninterrupted_optimizer = _optimizer(uninterrupted)
    uninterrupted_sampler = plan.initial_state
    torch.random.set_rng_state(initial_rng)
    for _ in range(6):
        uninterrupted_sampler = _train_one_step(
            uninterrupted,
            uninterrupted_optimizer,
            plan,
            uninterrupted_sampler,
            x=x,
            y=y,
        )
    uninterrupted_next_random = torch.rand(4)

    interrupted = _model()
    interrupted.load_state_dict(initial_state)
    interrupted_optimizer = _optimizer(interrupted)
    interrupted_sampler = plan.initial_state
    torch.random.set_rng_state(initial_rng)
    for _ in range(3):
        interrupted_sampler = _train_one_step(
            interrupted,
            interrupted_optimizer,
            plan,
            interrupted_sampler,
            x=x,
            y=y,
        )
    checkpoint = write_training_checkpoint(
        tmp_path / "step-00000003",
        model=interrupted,
        optimizer=interrupted_optimizer,
        binding=binding,
        step=3,
        sampler_state=interrupted_sampler,
        scheduler_state=_scheduler(step=3),
        adaptive_state=_adaptive(step=3),
    )

    torch.rand(100)
    restored = _model()
    restored_optimizer = _optimizer(
        restored,
        learning_rate=0.9,
        betas=(0.7, 0.8),
    )
    loaded = load_training_checkpoint(
        checkpoint.path,
        model=restored,
        optimizer=restored_optimizer,
        binding=binding,
    )

    assert loaded.sampler_state == interrupted_sampler
    assert loaded.scheduler_state == _scheduler(step=3)
    assert loaded.adaptive_state == _adaptive(step=3)
    assert restored_optimizer.param_groups[0]["lr"] == 0.05
    assert restored_optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    resumed_sampler = loaded.sampler_state
    for _ in range(3, 6):
        resumed_sampler = _train_one_step(
            restored,
            restored_optimizer,
            plan,
            resumed_sampler,
            x=x,
            y=y,
        )
    resumed_next_random = torch.rand(4)

    assert resumed_sampler == uninterrupted_sampler
    for name, expected in uninterrupted.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[name], expected, rtol=0, atol=0
        )
    _assert_optimizer_states_equal(uninterrupted_optimizer, restored_optimizer)
    torch.testing.assert_close(
        resumed_next_random,
        uninterrupted_next_random,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("file_name", "expected_message"),
    (
        ("adaptive.json", "adaptive_state_hash"),
        ("optimizer.json", "optimizer_metadata_hash"),
        ("state.safetensors", "tensor_payload_hash"),
        ("model.safetensors", "model_export_hash"),
    ),
)
def test_checkpoint_rejects_every_tampered_payload(
    tmp_path: Path,
    file_name: str,
    expected_message: str,
) -> None:
    torch.manual_seed(104)
    model = _model()
    checkpoint = write_training_checkpoint(
        tmp_path / file_name.replace(".", "-"),
        model=model,
        optimizer=_optimizer(model),
        binding=_binding(),
        step=0,
        sampler_state=_plan().initial_state,
        scheduler_state=_scheduler(),
        adaptive_state=_adaptive(),
    )
    target = checkpoint.path / file_name
    target.write_bytes(target.read_bytes() + b"tampered")

    restored = _model()
    restored_optimizer = _optimizer(restored, learning_rate=0.9)
    before_model = {
        name: tensor.detach().clone() for name, tensor in restored.state_dict().items()
    }
    before_optimizer = restored_optimizer.state_dict()
    before_rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match=expected_message):
        load_training_checkpoint(
            checkpoint.path,
            model=restored,
            optimizer=restored_optimizer,
            binding=_binding(),
        )
    for name, expected in before_model.items():
        torch.testing.assert_close(restored.state_dict()[name], expected)
    assert restored_optimizer.state_dict() == before_optimizer
    torch.testing.assert_close(torch.random.get_rng_state(), before_rng)


def test_checkpoint_rejects_extra_directories_in_artifact_boundary(
    tmp_path: Path,
) -> None:
    model = _model()
    checkpoint = write_training_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=_optimizer(model),
        binding=_binding(),
        step=0,
        sampler_state=_plan().initial_state,
        scheduler_state=_scheduler(),
        adaptive_state=_adaptive(),
    )
    (checkpoint.path / "unexpected-directory").mkdir()

    restored = _model()
    with pytest.raises(ValueError, match="incomplete or contains extras"):
        load_training_checkpoint(
            checkpoint.path,
            model=restored,
            optimizer=_optimizer(restored),
            binding=_binding(),
        )


def test_checkpoint_refuses_binding_and_cursor_drift_before_writing(
    tmp_path: Path,
) -> None:
    model = _model()
    optimizer = _optimizer(model)
    with pytest.raises(ValueError, match="sampler plan"):
        write_training_checkpoint(
            tmp_path / "bad-plan",
            model=model,
            optimizer=optimizer,
            binding=_binding(sampler_plan_hash="9" * 64),
            step=0,
            sampler_state=_plan().initial_state,
            scheduler_state=_scheduler(),
            adaptive_state=_adaptive(),
        )
    with pytest.raises(ValueError, match="sampler next_step"):
        write_training_checkpoint(
            tmp_path / "bad-sampler-step",
            model=model,
            optimizer=optimizer,
            binding=_binding(),
            step=0,
            sampler_state=DeterministicSamplerState(_plan().plan_hash, 1),
            scheduler_state=_scheduler(),
            adaptive_state=_adaptive(),
        )
    with pytest.raises(ValueError, match="scheduler step"):
        write_training_checkpoint(
            tmp_path / "bad-scheduler-step",
            model=model,
            optimizer=optimizer,
            binding=_binding(),
            step=0,
            sampler_state=_plan().initial_state,
            scheduler_state=_scheduler(step=1),
            adaptive_state=_adaptive(),
        )
    with pytest.raises(ValueError, match="adaptive last_update_step"):
        write_training_checkpoint(
            tmp_path / "bad-adaptive-step",
            model=model,
            optimizer=optimizer,
            binding=_binding(),
            step=0,
            sampler_state=_plan().initial_state,
            scheduler_state=_scheduler(),
            adaptive_state=_adaptive(step=2),
        )
    with pytest.raises(ValueError, match="learning_rate"):
        write_training_checkpoint(
            tmp_path / "bad-learning-rate",
            model=model,
            optimizer=optimizer,
            binding=_binding(),
            step=0,
            sampler_state=_plan().initial_state,
            scheduler_state=_scheduler(learning_rate=0.01),
            adaptive_state=_adaptive(),
        )
    assert not any(tmp_path.iterdir())


def test_checkpoint_refuses_nonfinite_model_before_creating_directory(
    tmp_path: Path,
) -> None:
    model = _model()
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        write_training_checkpoint(
            tmp_path / "nonfinite",
            model=model,
            optimizer=_optimizer(model),
            binding=_binding(),
            step=0,
            sampler_state=_plan().initial_state,
            scheduler_state=_scheduler(),
            adaptive_state=_adaptive(),
        )
    assert not any(tmp_path.iterdir())


def test_checkpoint_rejects_wrong_binding_manifest_schema_and_model(
    tmp_path: Path,
) -> None:
    torch.manual_seed(105)
    model = _model()
    checkpoint = write_training_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=_optimizer(model),
        binding=_binding(),
        step=0,
        sampler_state=_plan().initial_state,
        scheduler_state=_scheduler(),
        adaptive_state=_adaptive(),
    )

    wrong_model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="parameter_names|model state"):
        load_training_checkpoint(
            checkpoint.path,
            model=wrong_model,
            optimizer=_optimizer(wrong_model),
            binding=_binding(),
        )
    restored = _model()
    with pytest.raises(ValueError, match="run_hash"):
        load_training_checkpoint(
            checkpoint.path,
            model=restored,
            optimizer=_optimizer(restored),
            binding=_binding(seed=999),
        )

    raw = checkpoint.manifest.to_bytes().replace(
        b'"checkpoint_id":"CP-TRAINING"',
        b'"checkpoint_id":"CP-MUTANT"',
    )
    (checkpoint.path / "checkpoint.json").write_bytes(raw)
    restored = _model()
    with pytest.raises(ValueError, match="CP-TRAINING|canonical"):
        load_training_checkpoint(
            checkpoint.path,
            model=restored,
            optimizer=_optimizer(restored),
            binding=_binding(),
        )


def test_manifest_refuses_unknown_fields_and_noncanonical_bytes() -> None:
    manifest = TrainingCheckpointManifest(
        run_hash="0" * 64,
        dataset_snapshot_hash="1" * 64,
        execution_profile_hash="2" * 64,
        sampler_plan_hash="3" * 64,
        step=0,
        tensor_payload_hash="4" * 64,
        model_state_hash="5" * 64,
        optimizer_state_hash="6" * 64,
        optimizer_metadata_hash="7" * 64,
        scheduler_state=_scheduler(),
        sampler_state=DeterministicSamplerState("3" * 64, 0),
        adaptive_state_hash="a" * 64,
        torch_cpu_rng_hash="b" * 64,
        model_export_hash="c" * 64,
    )
    assert manifest.to_bytes() == canonical_json_bytes(manifest)

    payload = {
        field.name: getattr(manifest, field.name)
        for field in fields(TrainingCheckpointManifest)
    }
    payload["absolute_path"] = "must-not-enter-canonical-manifest"
    with pytest.raises(Exception, match="unknown field"):
        TrainingCheckpointManifest.from_mapping(payload)
    with pytest.raises(Exception, match="canonical"):
        TrainingCheckpointManifest.from_bytes(manifest.to_bytes() + b"\n")
