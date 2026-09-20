from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import torch

from piginot.contracts import ExecutionProfile, hash_spec
from piginot.contracts.canonical import canonical_json_bytes
from piginot.training.checkpoint import (
    DeterministicSamplerState,
    TrainingSchedulerState,
    export_model_safetensors,
    load_training_checkpoint,
    next_sampler_batch,
    write_training_checkpoint,
)
from piginot.training.identity import TrainingRunBinding

_HASH = "0123456789abcdef" * 4


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


def _binding(seed: int = 1234) -> TrainingRunBinding:
    return TrainingRunBinding(
        dataset_snapshot_hash=_HASH,
        public_manifest_hash="1" * 64,
        normalization_hash="2" * 64,
        architecture_id="tiny-linear-v1",
        architecture_module_digest="blake2b256:" + "3" * 64,
        objective_registry_hash="4" * 64,
        trainer_config_hash="5" * 64,
        sampler_plan_hash="6" * 64,
        initial_weights_hash="7" * 64,
        implementation_revision="8" * 40,
        seed=seed,
        execution_profile=_profile(seed),
    )


def _model() -> torch.nn.Linear:
    return torch.nn.Linear(1, 1)


def _optimizer(model: torch.nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=0.05)


def _scheduler(step: int = 0, learning_rate: float = 0.05) -> TrainingSchedulerState:
    return TrainingSchedulerState(
        scheduler_id="constant-lr-v1",
        step=step,
        learning_rate=learning_rate,
    )


def _sampler(cursor: int = 0) -> DeterministicSamplerState:
    return DeterministicSamplerState(
        sampler_id="deterministic-index-v1",
        order=tuple(range(12)),
        cursor=cursor,
        epoch=0,
        batch_size=2,
    )


def _train_one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: DeterministicSamplerState,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
) -> DeterministicSamplerState:
    batch, sampler = next_sampler_batch(sampler)
    prediction = model(x[list(batch)])
    loss = torch.nn.functional.mse_loss(prediction, y[list(batch)])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    return sampler


def test_checkpoint_writes_canonical_manifest_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    torch.manual_seed(101)
    binding = _binding()
    model = _model()
    optimizer = _optimizer(model)
    sampler = _sampler()

    checkpoint = write_training_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        binding=binding,
        step=0,
        sampler_state=sampler,
        scheduler_state=_scheduler(),
    )

    manifest_path = tmp_path / "checkpoint" / "checkpoint.json"
    assert manifest_path.read_bytes() == checkpoint.manifest.to_bytes()
    assert checkpoint.manifest.checkpoint_id == "CP-TRAINING"
    assert checkpoint.manifest.run_hash == binding.run_hash
    assert checkpoint.manifest.dataset_snapshot_hash == binding.dataset_snapshot_hash
    assert checkpoint.manifest.execution_profile_hash == binding.execution_profile_hash
    assert checkpoint.manifest.tensor_payload_filename == "state.safetensors"
    assert checkpoint.manifest.model_export_filename == "model.safetensors"
    assert checkpoint.manifest.manifest_hash == hash_spec(
        checkpoint.manifest, "content"
    )

    with pytest.raises(FileExistsError, match="checkpoint directory already exists"):
        write_training_checkpoint(
            tmp_path / "checkpoint",
            model=model,
            optimizer=optimizer,
            binding=binding,
            step=0,
            sampler_state=sampler,
            scheduler_state=_scheduler(),
        )


def test_model_export_contains_only_model_tensors_and_identity_metadata(
    tmp_path: Path,
) -> None:
    safetensors_torch = pytest.importorskip("safetensors.torch")
    torch.manual_seed(102)
    binding = _binding()
    model = _model()

    path, digest = export_model_safetensors(
        tmp_path / "model.safetensors",
        model=model,
        binding=binding,
    )

    loaded = safetensors_torch.load_file(str(path), device="cpu")
    assert set(loaded) == set(model.state_dict())
    assert all(not name.startswith("optimizer::") for name in loaded)
    assert digest == checkpoint_file_hash(path)


def test_interrupt_resume_matches_uninterrupted_adam_training(
    tmp_path: Path,
) -> None:
    torch.manual_seed(103)
    binding = _binding(seed=103)
    x = torch.linspace(-1.0, 1.0, 12, dtype=torch.float32).reshape(-1, 1)
    y = 2.0 * x + 1.0

    uninterrupted = _model()
    uninterrupted_optimizer = _optimizer(uninterrupted)
    resumed_source = _model()
    resumed_source.load_state_dict(uninterrupted.state_dict())
    resumed_optimizer = _optimizer(resumed_source)
    sampler = _sampler()

    for _step in range(3):
        sampler = _train_one_step(
            uninterrupted,
            uninterrupted_optimizer,
            sampler,
            x=x,
            y=y,
        )
        _train_one_step(
            resumed_source,
            resumed_optimizer,
            _sampler(cursor=sampler.cursor - sampler.batch_size),
            x=x,
            y=y,
        )

    checkpoint = write_training_checkpoint(
        tmp_path / "resume",
        model=resumed_source,
        optimizer=resumed_optimizer,
        binding=binding,
        step=3,
        sampler_state=sampler,
        scheduler_state=_scheduler(step=3),
    )

    restored = _model()
    restored_optimizer = _optimizer(restored)
    loaded = load_training_checkpoint(
        checkpoint.path,
        model=restored,
        optimizer=restored_optimizer,
        binding=binding,
    )

    assert loaded.manifest == checkpoint.manifest
    assert loaded.sampler_state == sampler
    assert loaded.scheduler_state == _scheduler(step=3)

    for _step in range(3, 6):
        sampler = _train_one_step(
            uninterrupted,
            uninterrupted_optimizer,
            sampler,
            x=x,
            y=y,
        )
        loaded_sampler = _train_one_step(
            restored,
            restored_optimizer,
            loaded.sampler_state,
            x=x,
            y=y,
        )
        loaded = loaded.with_sampler_state(loaded_sampler)

    for name, expected in uninterrupted.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[name], expected, rtol=0, atol=0
        )


def test_checkpoint_rejects_wrong_binding_and_schema_drift(tmp_path: Path) -> None:
    torch.manual_seed(104)
    binding = _binding()
    model = _model()
    checkpoint = write_training_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=_optimizer(model),
        binding=binding,
        step=0,
        sampler_state=_sampler(),
        scheduler_state=_scheduler(),
    )

    with pytest.raises(ValueError, match="run_hash"):
        load_training_checkpoint(
            checkpoint.path,
            model=_model(),
            optimizer=_optimizer(_model()),
            binding=_binding(seed=999),
        )

    raw = checkpoint.manifest.to_bytes().replace(
        b'"checkpoint_id":"CP-TRAINING"', b'"checkpoint_id":"CP-MUTANT"'
    )
    (checkpoint.path / "checkpoint.json").write_bytes(raw)

    with pytest.raises(ValueError, match="CP-TRAINING|canonical"):
        load_training_checkpoint(
            checkpoint.path,
            model=_model(),
            optimizer=_optimizer(_model()),
            binding=binding,
        )


def test_checkpoint_rejects_sidecar_tampering(tmp_path: Path) -> None:
    torch.manual_seed(105)
    binding = _binding()
    model = _model()
    checkpoint = write_training_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=_optimizer(model),
        binding=binding,
        step=0,
        sampler_state=_sampler(),
        scheduler_state=_scheduler(),
    )

    (checkpoint.path / "scheduler.json").write_bytes(
        canonical_json_bytes(_scheduler(learning_rate=0.01))
    )

    with pytest.raises(ValueError, match="scheduler_state_hash"):
        load_training_checkpoint(
            checkpoint.path,
            model=model,
            optimizer=_optimizer(model),
            binding=binding,
        )


def checkpoint_file_hash(path: Path) -> str:
    import hashlib

    return hashlib.blake2b(path.read_bytes(), digest_size=32).hexdigest()


def test_manifest_refuses_unknown_fields() -> None:
    from piginot.training.checkpoint import TrainingCheckpointManifest

    payload = {
        field.name: getattr(
            TrainingCheckpointManifest(
                run_hash="0" * 64,
                dataset_snapshot_hash="1" * 64,
                execution_profile_hash="2" * 64,
                step=0,
                tensor_payload_hash="3" * 64,
                model_state_hash="4" * 64,
                optimizer_state_hash="5" * 64,
                scheduler_state_hash="6" * 64,
                sampler_state_hash="7" * 64,
                torch_cpu_rng_hash="8" * 64,
                model_export_hash="9" * 64,
            ),
            field.name,
        )
        for field in fields(TrainingCheckpointManifest)
    }
    payload["absolute_path"] = "must-not-enter-canonical-manifest"

    with pytest.raises(Exception, match="unknown field"):
        TrainingCheckpointManifest.from_mapping(payload)


def test_checkpoint_manifest_is_canonical_json() -> None:
    from piginot.training.checkpoint import TrainingCheckpointManifest

    manifest = TrainingCheckpointManifest(
        run_hash="0" * 64,
        dataset_snapshot_hash="1" * 64,
        execution_profile_hash="2" * 64,
        step=0,
        tensor_payload_hash="3" * 64,
        model_state_hash="4" * 64,
        optimizer_state_hash="5" * 64,
        scheduler_state_hash="6" * 64,
        sampler_state_hash="7" * 64,
        torch_cpu_rng_hash="8" * 64,
        model_export_hash="9" * 64,
    )

    assert manifest.to_bytes() == canonical_json_bytes(manifest)
