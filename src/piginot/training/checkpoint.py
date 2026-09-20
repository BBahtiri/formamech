"""Canonical S13 training checkpoints with safetensors-only tensor state."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch
from safetensors.torch import load_file, save_file

from piginot.contracts import hash_spec, strict_json_loads
from piginot.contracts.canonical import canonical_json_bytes
from piginot.contracts.errors import ContractValidationError
from piginot.contracts.specs import SpecContract
from piginot.training.identity import TrainingRunBinding

_DIGEST_BYTES = 32
_CHECKPOINT_ID = "CP-TRAINING"
_MANIFEST_FILENAME = "checkpoint.json"
_TENSOR_PAYLOAD_FILENAME = "state.safetensors"
_MODEL_EXPORT_FILENAME = "model.safetensors"
_CANONICAL_TYPE = "piginot.training.TrainingCheckpointManifest"


def _file_hash(path: Path) -> str:
    return hashlib.blake2b(path.read_bytes(), digest_size=_DIGEST_BYTES).hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    payload = bytes(cpu.reshape(-1).view(torch.uint8).tolist())
    metadata = f"{cpu.dtype}:{tuple(cpu.shape)}".encode()
    digest = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    digest.update(metadata)
    digest.update(payload)
    return digest.hexdigest()


def _state_hash(tensors: dict[str, torch.Tensor]) -> str:
    payload = {
        name: {
            "dtype": str(tensor.dtype),
            "shape": tuple(tensor.shape),
            "hash": _tensor_hash(tensor),
        }
        for name, tensor in sorted(tensors.items())
    }
    return hash_spec(payload, "content")


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ContractValidationError(f"{label} must be a 256-bit digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ContractValidationError(f"{label} must be lowercase hexadecimal")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    checked = _nonnegative_int(value, label)
    if checked == 0:
        raise ContractValidationError(f"{label} must be a positive integer")
    return checked


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{label} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class TrainingSchedulerState(SpecContract):
    """Small canonical scheduler cursor for deterministic resume."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    scheduler_id: str
    step: int
    learning_rate: float

    def __post_init__(self) -> None:
        _text(self.scheduler_id, "scheduler_id")
        _nonnegative_int(self.step, "step")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or float(self.learning_rate) <= 0.0
        ):
            raise ContractValidationError(
                "learning_rate must be strictly positive numeric"
            )
        object.__setattr__(self, "learning_rate", float(self.learning_rate))


@dataclass(frozen=True, slots=True)
class DeterministicSamplerState(SpecContract):
    """Canonical sequential sampler cursor over a fixed deterministic order."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    sampler_id: str
    order: tuple[int, ...]
    cursor: int
    epoch: int
    batch_size: int

    def __post_init__(self) -> None:
        _text(self.sampler_id, "sampler_id")
        order = tuple(self.order)
        if not order:
            raise ContractValidationError("order must be non-empty")
        if any(
            isinstance(index, bool) or not isinstance(index, int) for index in order
        ):
            raise ContractValidationError("order must contain integer indices")
        if len(set(order)) != len(order):
            raise ContractValidationError("order must not contain duplicates")
        cursor = _nonnegative_int(self.cursor, "cursor")
        if cursor > len(order):
            raise ContractValidationError("cursor must not exceed order length")
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "cursor", cursor)
        _nonnegative_int(self.epoch, "epoch")
        _positive_int(self.batch_size, "batch_size")


def next_sampler_batch(
    state: DeterministicSamplerState,
) -> tuple[tuple[int, ...], DeterministicSamplerState]:
    """Return the next deterministic batch and a new immutable sampler state."""

    if type(state) is not DeterministicSamplerState:
        raise TypeError("state must be an exact DeterministicSamplerState")
    next_cursor = state.cursor + state.batch_size
    if next_cursor < len(state.order):
        return state.order[state.cursor : next_cursor], DeterministicSamplerState(
            sampler_id=state.sampler_id,
            order=state.order,
            cursor=next_cursor,
            epoch=state.epoch,
            batch_size=state.batch_size,
        )
    if next_cursor == len(state.order):
        return state.order[state.cursor : next_cursor], DeterministicSamplerState(
            sampler_id=state.sampler_id,
            order=state.order,
            cursor=0,
            epoch=state.epoch + 1,
            batch_size=state.batch_size,
        )
    wrapped = next_cursor - len(state.order)
    batch = (*state.order[state.cursor :], *state.order[:wrapped])
    return batch, DeterministicSamplerState(
        sampler_id=state.sampler_id,
        order=state.order,
        cursor=wrapped,
        epoch=state.epoch + 1,
        batch_size=state.batch_size,
    )


@dataclass(frozen=True, slots=True)
class TrainingCheckpointManifest(SpecContract):
    """Canonical public manifest for one resumable S13 checkpoint."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    run_hash: str
    dataset_snapshot_hash: str
    execution_profile_hash: str
    step: int
    tensor_payload_hash: str
    model_state_hash: str
    optimizer_state_hash: str
    scheduler_state_hash: str
    sampler_state_hash: str
    torch_cpu_rng_hash: str
    model_export_hash: str
    checkpoint_id: str = _CHECKPOINT_ID
    tensor_payload_filename: str = _TENSOR_PAYLOAD_FILENAME
    model_export_filename: str = _MODEL_EXPORT_FILENAME

    def __post_init__(self) -> None:
        if self.checkpoint_id != _CHECKPOINT_ID:
            raise ContractValidationError("canonical checkpoint_id must be CP-TRAINING")
        for name in (
            "run_hash",
            "dataset_snapshot_hash",
            "execution_profile_hash",
            "tensor_payload_hash",
            "model_state_hash",
            "optimizer_state_hash",
            "scheduler_state_hash",
            "sampler_state_hash",
            "torch_cpu_rng_hash",
            "model_export_hash",
        ):
            _digest(getattr(self, name), name)
        _nonnegative_int(self.step, "step")
        if self.tensor_payload_filename != _TENSOR_PAYLOAD_FILENAME:
            raise ContractValidationError("canonical tensor payload filename mismatch")
        if self.model_export_filename != _MODEL_EXPORT_FILENAME:
            raise ContractValidationError("canonical model export filename mismatch")

    @property
    def manifest_hash(self) -> str:
        return hash_spec(self, "content")

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    @classmethod
    def from_bytes(cls, payload: bytes) -> TrainingCheckpointManifest:
        decoded = strict_json_loads(payload)
        if not isinstance(decoded, dict):
            raise ContractValidationError("checkpoint manifest must be a JSON object")
        canonical_type = decoded.pop("$type", None)
        if canonical_type != _CANONICAL_TYPE:
            raise ContractValidationError("canonical manifest type mismatch")
        manifest = cls.from_mapping(decoded)
        if manifest.to_bytes() != payload:
            raise ContractValidationError("checkpoint manifest is not canonical JSON")
        return manifest


@dataclass(frozen=True, slots=True)
class LoadedTrainingCheckpoint:
    """Loaded checkpoint plus mutable-free resume cursors."""

    path: Path
    manifest: TrainingCheckpointManifest
    sampler_state: DeterministicSamplerState
    scheduler_state: TrainingSchedulerState

    def with_sampler_state(
        self,
        sampler_state: DeterministicSamplerState,
    ) -> LoadedTrainingCheckpoint:
        if type(sampler_state) is not DeterministicSamplerState:
            raise TypeError("sampler_state must be an exact DeterministicSamplerState")
        return LoadedTrainingCheckpoint(
            path=self.path,
            manifest=self.manifest,
            sampler_state=sampler_state,
            scheduler_state=self.scheduler_state,
        )


@dataclass(frozen=True, slots=True)
class WrittenTrainingCheckpoint(LoadedTrainingCheckpoint):
    """Checkpoint returned immediately after a canonical write."""


TrainingCheckpoint = LoadedTrainingCheckpoint


def _model_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        f"model::{name}": tensor.detach().cpu()
        for name, tensor in sorted(model.state_dict().items())
    }


def _optimizer_tensors(optimizer: torch.optim.Optimizer) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    state_dict = optimizer.state_dict()
    for parameter_index, state in sorted(state_dict["state"].items()):
        if not isinstance(parameter_index, int) or not isinstance(state, dict):
            raise ContractValidationError("unsupported optimizer state schema")
        for name, value in sorted(state.items()):
            if not isinstance(value, torch.Tensor):
                raise ContractValidationError(
                    "optimizer state must be tensor-only for S13 checkpoints"
                )
            result[f"optimizer::{parameter_index}::{name}"] = value.detach().cpu()
    return result


def _write_scheduler_state(path: Path, state: TrainingSchedulerState) -> None:
    path.write_bytes(canonical_json_bytes(state))


def _write_sampler_state(path: Path, state: DeterministicSamplerState) -> None:
    path.write_bytes(canonical_json_bytes(state))


def _read_scheduler_state(path: Path) -> TrainingSchedulerState:
    payload = strict_json_loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ContractValidationError("scheduler state must be a JSON object")
    payload.pop("$type", None)
    return TrainingSchedulerState.from_mapping(payload)


def _read_sampler_state(path: Path) -> DeterministicSamplerState:
    payload = strict_json_loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ContractValidationError("sampler state must be a JSON object")
    payload.pop("$type", None)
    normalized_payload: dict[str, object] = dict(payload)
    if "order" in payload and isinstance(payload["order"], list):
        normalized_payload["order"] = tuple(payload["order"])
    return DeterministicSamplerState.from_mapping(normalized_payload)


def export_model_safetensors(
    path: Path,
    *,
    model: torch.nn.Module,
    binding: TrainingRunBinding,
) -> tuple[Path, str]:
    """Export model tensors only, with run identity in safetensors metadata."""

    if type(binding) is not TrainingRunBinding:
        raise TypeError("binding must be an exact TrainingRunBinding")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            key.removeprefix("model::"): value
            for key, value in _model_tensors(model).items()
        },
        str(path),
        metadata={
            "run_hash": binding.run_hash,
            "dataset_snapshot_hash": binding.dataset_snapshot_hash,
            "execution_profile_hash": binding.execution_profile_hash,
        },
    )
    return path, _file_hash(path)


def _optimizer_state_from_tensors(
    tensors: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    template = optimizer.state_dict()
    state: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        if not key.startswith("optimizer::"):
            continue
        _, parameter_index, name = key.split("::", maxsplit=2)
        state.setdefault(int(parameter_index), {})[name] = tensor.clone()
    return {"state": state, "param_groups": template["param_groups"]}


def write_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    binding: TrainingRunBinding,
    step: int,
    sampler_state: DeterministicSamplerState,
    scheduler_state: TrainingSchedulerState,
) -> WrittenTrainingCheckpoint:
    """Write one canonical, write-once training checkpoint directory."""

    if type(binding) is not TrainingRunBinding:
        raise TypeError("binding must be an exact TrainingRunBinding")
    if type(sampler_state) is not DeterministicSamplerState:
        raise TypeError("sampler_state must be an exact DeterministicSamplerState")
    if type(scheduler_state) is not TrainingSchedulerState:
        raise TypeError("scheduler_state must be an exact TrainingSchedulerState")
    checked_step = _nonnegative_int(step, "step")
    if path.exists():
        raise FileExistsError("checkpoint directory already exists")
    try:
        path.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError("checkpoint directory already exists") from exc

    model_tensors = _model_tensors(model)
    optimizer_tensors = _optimizer_tensors(optimizer)
    rng_state = torch.random.get_rng_state().cpu()
    payload_tensors = {
        **model_tensors,
        **optimizer_tensors,
        "torch_cpu_rng": rng_state,
    }
    payload_path = path / _TENSOR_PAYLOAD_FILENAME
    save_file(payload_tensors, str(payload_path))
    model_export_path, model_export_hash = export_model_safetensors(
        path / _MODEL_EXPORT_FILENAME,
        model=model,
        binding=binding,
    )
    if model_export_path.name != _MODEL_EXPORT_FILENAME:
        raise ContractValidationError("model export filename mismatch")

    scheduler_path = path / "scheduler.json"
    sampler_path = path / "sampler.json"
    _write_scheduler_state(scheduler_path, scheduler_state)
    _write_sampler_state(sampler_path, sampler_state)

    manifest = TrainingCheckpointManifest(
        run_hash=binding.run_hash,
        dataset_snapshot_hash=binding.dataset_snapshot_hash,
        execution_profile_hash=binding.execution_profile_hash,
        step=checked_step,
        tensor_payload_hash=_file_hash(payload_path),
        model_state_hash=_state_hash(model_tensors),
        optimizer_state_hash=_state_hash(optimizer_tensors),
        scheduler_state_hash=_file_hash(scheduler_path),
        sampler_state_hash=_file_hash(sampler_path),
        torch_cpu_rng_hash=_tensor_hash(rng_state),
        model_export_hash=model_export_hash,
    )
    (path / _MANIFEST_FILENAME).write_bytes(manifest.to_bytes())
    return WrittenTrainingCheckpoint(
        path=path,
        manifest=manifest,
        sampler_state=sampler_state,
        scheduler_state=scheduler_state,
    )


def load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    binding: TrainingRunBinding,
) -> LoadedTrainingCheckpoint:
    """Load a canonical checkpoint into ``model`` and ``optimizer``."""

    if type(binding) is not TrainingRunBinding:
        raise TypeError("binding must be an exact TrainingRunBinding")
    try:
        manifest = TrainingCheckpointManifest.from_bytes(
            (path / _MANIFEST_FILENAME).read_bytes()
        )
    except Exception as exc:
        raise ValueError("canonical checkpoint manifest is invalid") from exc
    if manifest.run_hash != binding.run_hash:
        raise ValueError("run_hash does not match the requested binding")
    if manifest.dataset_snapshot_hash != binding.dataset_snapshot_hash:
        raise ValueError("dataset snapshot does not match the requested binding")
    if manifest.execution_profile_hash != binding.execution_profile_hash:
        raise ValueError("execution profile does not match the requested binding")

    model_export_path = path / manifest.model_export_filename
    if _file_hash(model_export_path) != manifest.model_export_hash:
        raise ValueError("model_export_hash mismatch")
    payload_path = path / manifest.tensor_payload_filename
    if _file_hash(payload_path) != manifest.tensor_payload_hash:
        raise ValueError("tensor payload hash mismatch")
    tensors = load_file(str(payload_path), device="cpu")
    model_state = {
        key.removeprefix("model::"): value
        for key, value in tensors.items()
        if key.startswith("model::")
    }
    if _state_hash({f"model::{key}": value for key, value in model_state.items()}) != (
        manifest.model_state_hash
    ):
        raise ValueError("model_state_hash mismatch")
    optimizer_tensors = {
        key: value for key, value in tensors.items() if key.startswith("optimizer::")
    }
    if _state_hash(optimizer_tensors) != manifest.optimizer_state_hash:
        raise ValueError("optimizer_state_hash mismatch")
    if "torch_cpu_rng" not in tensors:
        raise ValueError("torch_cpu_rng_hash mismatch")
    if _tensor_hash(tensors["torch_cpu_rng"]) != manifest.torch_cpu_rng_hash:
        raise ValueError("torch_cpu_rng_hash mismatch")
    model.load_state_dict(model_state)
    optimizer.load_state_dict(_optimizer_state_from_tensors(tensors, optimizer))
    torch.random.set_rng_state(tensors["torch_cpu_rng"])

    scheduler_path = path / "scheduler.json"
    sampler_path = path / "sampler.json"
    if _file_hash(scheduler_path) != manifest.scheduler_state_hash:
        raise ValueError("scheduler_state_hash mismatch")
    if _file_hash(sampler_path) != manifest.sampler_state_hash:
        raise ValueError("sampler_state_hash mismatch")
    scheduler_state = _read_scheduler_state(scheduler_path)
    sampler_state = _read_sampler_state(sampler_path)
    return LoadedTrainingCheckpoint(
        path=path,
        manifest=manifest,
        sampler_state=sampler_state,
        scheduler_state=scheduler_state,
    )


__all__ = (
    "DeterministicSamplerState",
    "LoadedTrainingCheckpoint",
    "TrainingCheckpoint",
    "TrainingCheckpointManifest",
    "TrainingSchedulerState",
    "WrittenTrainingCheckpoint",
    "export_model_safetensors",
    "load_training_checkpoint",
    "next_sampler_batch",
    "write_training_checkpoint",
)
