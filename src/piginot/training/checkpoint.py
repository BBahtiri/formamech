"""Canonical, write-once S13 checkpoints with exact deterministic resume."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch
from safetensors import safe_open
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from piginot.contracts import hash_spec, strict_json_loads
from piginot.contracts.canonical import canonical_json_bytes
from piginot.contracts.errors import ContractValidationError
from piginot.contracts.specs import SpecContract
from piginot.training._checkpoint_codec import (
    _ADAPTIVE_FILENAME,
    _CHECKPOINT_ID,
    _EXPECTED_FILENAMES,
    _MANIFEST_FILENAME,
    _MANIFEST_TYPE,
    _MODEL_EXPORT_FILENAME,
    _MODEL_PREFIX,
    _OPTIMIZER_FILENAME,
    _OPTIMIZER_PREFIX,
    _RNG_KEY,
    _TENSOR_PAYLOAD_FILENAME,
    _digest,
    _exact_mapping,
    _file_hash_bytes,
    _new_artifact_path,
    _nonnegative_integer,
    _parse_adaptive,
    _parse_optimizer,
    _parse_sampler,
    _parse_scheduler,
    _state_hash,
    _tensor_hash,
    _write_once,
)
from piginot.training.adaptive import AdaptiveWeightState
from piginot.training.identity import TrainingRunBinding
from piginot.training.state import (
    AdamOptimizerState,
    DeterministicSamplerState,
    TrainingSchedulerState,
    capture_adam_optimizer_state,
)


@dataclass(frozen=True, slots=True)
class TrainingCheckpointManifest(SpecContract):
    """Canonical public lineage and integrity record for one checkpoint."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    run_hash: str
    dataset_snapshot_hash: str
    execution_profile_hash: str
    sampler_plan_hash: str
    step: int
    tensor_payload_hash: str
    model_state_hash: str
    optimizer_state_hash: str
    optimizer_metadata_hash: str
    scheduler_state: TrainingSchedulerState
    sampler_state: DeterministicSamplerState
    adaptive_state_hash: str
    torch_cpu_rng_hash: str
    model_export_hash: str
    checkpoint_id: str = _CHECKPOINT_ID
    tensor_payload_filename: str = _TENSOR_PAYLOAD_FILENAME
    model_export_filename: str = _MODEL_EXPORT_FILENAME
    optimizer_filename: str = _OPTIMIZER_FILENAME
    adaptive_filename: str = _ADAPTIVE_FILENAME

    def __post_init__(self) -> None:
        if self.checkpoint_id != _CHECKPOINT_ID:
            raise ContractValidationError("canonical checkpoint_id must be CP-TRAINING")
        for name in (
            "run_hash",
            "dataset_snapshot_hash",
            "execution_profile_hash",
            "sampler_plan_hash",
            "tensor_payload_hash",
            "model_state_hash",
            "optimizer_state_hash",
            "optimizer_metadata_hash",
            "adaptive_state_hash",
            "torch_cpu_rng_hash",
            "model_export_hash",
        ):
            _digest(getattr(self, name), name)
        _nonnegative_integer(self.step, "step")
        if type(self.scheduler_state) is not TrainingSchedulerState:
            raise ContractValidationError(
                "scheduler_state must be an exact TrainingSchedulerState"
            )
        if type(self.sampler_state) is not DeterministicSamplerState:
            raise ContractValidationError(
                "sampler_state must be an exact DeterministicSamplerState"
            )
        if self.sampler_state.plan_hash != self.sampler_plan_hash:
            raise ContractValidationError("sampler state does not match sampler plan")
        if self.sampler_state.next_step != self.step:
            raise ContractValidationError("sampler next_step does not match step")
        if self.scheduler_state.step != self.step:
            raise ContractValidationError("scheduler step does not match step")
        expected_names = (
            (_TENSOR_PAYLOAD_FILENAME, self.tensor_payload_filename),
            (_MODEL_EXPORT_FILENAME, self.model_export_filename),
            (_OPTIMIZER_FILENAME, self.optimizer_filename),
            (_ADAPTIVE_FILENAME, self.adaptive_filename),
        )
        if any(expected != actual for expected, actual in expected_names):
            raise ContractValidationError("canonical checkpoint filename mismatch")

    @property
    def scheduler_state_hash(self) -> str:
        return hash_spec(self.scheduler_state, "content")

    @property
    def sampler_state_hash(self) -> str:
        return hash_spec(self.sampler_state, "content")

    @property
    def manifest_hash(self) -> str:
        return hash_spec(self, "content")

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    @classmethod
    def from_bytes(cls, payload: bytes) -> TrainingCheckpointManifest:
        decoded = strict_json_loads(payload)
        mapping = _exact_mapping(
            decoded,
            canonical_type=_MANIFEST_TYPE,
            label="checkpoint manifest",
        )
        scheduler = _parse_scheduler(mapping.pop("scheduler_state", None))
        sampler = _parse_sampler(mapping.pop("sampler_state", None))
        manifest = cls.from_mapping(
            {
                **mapping,
                "scheduler_state": scheduler,
                "sampler_state": sampler,
            }
        )
        if manifest.to_bytes() != payload:
            raise ContractValidationError("checkpoint manifest is not canonical JSON")
        return manifest


@dataclass(frozen=True, slots=True)
class LoadedTrainingCheckpoint:
    """Validated checkpoint plus all immutable resume cursors."""

    path: Path
    manifest: TrainingCheckpointManifest
    sampler_state: DeterministicSamplerState
    scheduler_state: TrainingSchedulerState
    adaptive_state: AdaptiveWeightState


@dataclass(frozen=True, slots=True)
class WrittenTrainingCheckpoint(LoadedTrainingCheckpoint):
    """Checkpoint returned immediately after a canonical write."""


TrainingCheckpoint = LoadedTrainingCheckpoint


def _model_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    state = model.state_dict()
    if not state:
        raise ContractValidationError("model state must be non-empty")
    result: dict[str, torch.Tensor] = {}
    for name, tensor in sorted(state.items()):
        if not name or "::" in name:
            raise ContractValidationError("model state names must not contain '::'")
        if not isinstance(tensor, torch.Tensor) or tensor.layout is not torch.strided:
            raise ContractValidationError("model state must contain dense tensors")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ContractValidationError("model state must contain finite tensors")
        result[f"{_MODEL_PREFIX}{name}"] = tensor.detach().contiguous().cpu()
    return result


def model_state_hash(model: torch.nn.Module) -> str:
    """Hash the complete named model state using the checkpoint tensor framing."""

    return _state_hash(_model_tensors(model))


def _adam_state_tensors(
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
    metadata: AdamOptimizerState,
) -> dict[str, torch.Tensor]:
    named_parameters = dict(model.named_parameters())
    expected_state_keys = {"step", "exp_avg", "exp_avg_sq"}
    if metadata.amsgrad:
        expected_state_keys.add("max_exp_avg_sq")
    result: dict[str, torch.Tensor] = {}
    for name in metadata.state_parameter_names:
        parameter = named_parameters[name]
        state = optimizer.state.get(parameter)
        if not isinstance(state, dict) or set(state) != expected_state_keys:
            raise ContractValidationError("unsupported Adam state schema")
        for state_name in sorted(expected_state_keys):
            value = state[state_name]
            if not isinstance(value, torch.Tensor) or value.layout is not torch.strided:
                raise ContractValidationError("Adam state must contain dense tensors")
            if not bool(torch.isfinite(value).all().item()):
                raise ContractValidationError("Adam state must contain finite tensors")
            if state_name == "step" and (
                value.numel() != 1
                or float(value.item()) < 0.0
                or not float(value.item()).is_integer()
            ):
                raise ContractValidationError("Adam step state must be non-negative")
            result[f"{_OPTIMIZER_PREFIX}{name}::{state_name}"] = (
                value.detach().contiguous().cpu()
            )
    return result


def _model_export_bytes(
    *,
    model_tensors: dict[str, torch.Tensor],
    binding: TrainingRunBinding,
    model_state_hash: str,
) -> bytes:
    return save_safetensors(
        {
            name.removeprefix(_MODEL_PREFIX): tensor
            for name, tensor in model_tensors.items()
        },
        metadata={
            "run_hash": binding.run_hash,
            "dataset_snapshot_hash": binding.dataset_snapshot_hash,
            "execution_profile_hash": binding.execution_profile_hash,
            "architecture_id": binding.architecture_id,
            "architecture_module_digest": binding.architecture_module_digest,
            "model_state_hash": model_state_hash,
        },
    )


def _validate_binding(binding: TrainingRunBinding) -> None:
    if type(binding) is not TrainingRunBinding:
        raise TypeError("binding must be an exact TrainingRunBinding")
    if binding.execution_profile.device != "cpu":
        raise ContractValidationError(
            "S13 exact checkpoints currently require a CPU execution profile"
        )


def export_model_safetensors(
    path: Path,
    *,
    model: torch.nn.Module,
    binding: TrainingRunBinding,
) -> tuple[Path, str]:
    """Write one model-only, identity-bound safetensors export."""

    _validate_binding(binding)
    checked_path = _new_artifact_path(path, suffix=".safetensors")
    model_tensors = _model_tensors(model)
    payload = _model_export_bytes(
        model_tensors=model_tensors,
        binding=binding,
        model_state_hash=_state_hash(model_tensors),
    )
    _write_once(checked_path, payload)
    return checked_path, _file_hash_bytes(payload)


def _validate_resume_cursors(
    *,
    binding: TrainingRunBinding,
    step: int,
    sampler_state: DeterministicSamplerState,
    scheduler_state: TrainingSchedulerState,
    adaptive_state: AdaptiveWeightState,
) -> int:
    checked_step = _nonnegative_integer(step, "step")
    if type(sampler_state) is not DeterministicSamplerState:
        raise TypeError("sampler_state must be an exact DeterministicSamplerState")
    if type(scheduler_state) is not TrainingSchedulerState:
        raise TypeError("scheduler_state must be an exact TrainingSchedulerState")
    if type(adaptive_state) is not AdaptiveWeightState:
        raise TypeError("adaptive_state must be an exact AdaptiveWeightState")
    if sampler_state.plan_hash != binding.sampler_plan_hash:
        raise ValueError("sampler plan does not match the training run binding")
    if sampler_state.next_step != checked_step:
        raise ValueError("sampler next_step does not match checkpoint step")
    if scheduler_state.step != checked_step:
        raise ValueError("scheduler step does not match checkpoint step")
    if (
        adaptive_state.last_update_step is not None
        and adaptive_state.last_update_step > checked_step
    ):
        raise ValueError("adaptive last_update_step exceeds checkpoint step")
    return checked_step


def _checkpoint_payloads(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
    binding: TrainingRunBinding,
    step: int,
    sampler_state: DeterministicSamplerState,
    scheduler_state: TrainingSchedulerState,
    adaptive_state: AdaptiveWeightState,
) -> tuple[TrainingCheckpointManifest, dict[str, bytes]]:
    optimizer_metadata = capture_adam_optimizer_state(model, optimizer)
    if not math.isclose(
        scheduler_state.learning_rate,
        optimizer_metadata.learning_rate,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("scheduler learning_rate does not match Adam learning_rate")
    model_tensors = _model_tensors(model)
    optimizer_tensors = _adam_state_tensors(model, optimizer, optimizer_metadata)
    rng_state = torch.random.get_rng_state().detach().contiguous().cpu()
    if rng_state.dtype is not torch.uint8 or rng_state.ndim != 1:
        raise ContractValidationError("torch CPU RNG state schema is unsupported")
    state_tensors = {
        **model_tensors,
        **optimizer_tensors,
        _RNG_KEY: rng_state,
    }
    model_state_hash = _state_hash(model_tensors)
    state_payload = save_safetensors(
        state_tensors,
        metadata={"run_hash": binding.run_hash, "step": str(step)},
    )
    model_payload = _model_export_bytes(
        model_tensors=model_tensors,
        binding=binding,
        model_state_hash=model_state_hash,
    )
    optimizer_payload = canonical_json_bytes(optimizer_metadata)
    adaptive_payload = canonical_json_bytes(adaptive_state)
    manifest = TrainingCheckpointManifest(
        run_hash=binding.run_hash,
        dataset_snapshot_hash=binding.dataset_snapshot_hash,
        execution_profile_hash=binding.execution_profile_hash,
        sampler_plan_hash=binding.sampler_plan_hash,
        step=step,
        tensor_payload_hash=_file_hash_bytes(state_payload),
        model_state_hash=model_state_hash,
        optimizer_state_hash=_state_hash(optimizer_tensors),
        optimizer_metadata_hash=_file_hash_bytes(optimizer_payload),
        scheduler_state=scheduler_state,
        sampler_state=sampler_state,
        adaptive_state_hash=_file_hash_bytes(adaptive_payload),
        torch_cpu_rng_hash=_tensor_hash(rng_state),
        model_export_hash=_file_hash_bytes(model_payload),
    )
    return manifest, {
        _ADAPTIVE_FILENAME: adaptive_payload,
        _MODEL_EXPORT_FILENAME: model_payload,
        _OPTIMIZER_FILENAME: optimizer_payload,
        _TENSOR_PAYLOAD_FILENAME: state_payload,
        _MANIFEST_FILENAME: manifest.to_bytes(),
    }


def write_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
    binding: TrainingRunBinding,
    step: int,
    sampler_state: DeterministicSamplerState,
    scheduler_state: TrainingSchedulerState,
    adaptive_state: AdaptiveWeightState,
) -> WrittenTrainingCheckpoint:
    """Write one complete checkpoint directory; existing paths are never replaced."""

    _validate_binding(binding)
    checked_step = _validate_resume_cursors(
        binding=binding,
        step=step,
        sampler_state=sampler_state,
        scheduler_state=scheduler_state,
        adaptive_state=adaptive_state,
    )
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not path.parent.exists() or not path.parent.is_dir():
        raise FileNotFoundError("checkpoint parent directory must already exist")
    if path.parent.is_symlink():
        raise ValueError("checkpoint parent directory must not be a symlink")
    if path.exists() or path.is_symlink():
        raise FileExistsError("checkpoint directory already exists")
    manifest, payloads = _checkpoint_payloads(
        model=model,
        optimizer=optimizer,
        binding=binding,
        step=checked_step,
        sampler_state=sampler_state,
        scheduler_state=scheduler_state,
        adaptive_state=adaptive_state,
    )
    staging = path.parent / f".{path.name}.staging-{manifest.manifest_hash[:16]}"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError("checkpoint staging directory already exists")
    try:
        staging.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError("checkpoint staging directory already exists") from exc
    for filename in (
        _ADAPTIVE_FILENAME,
        _MODEL_EXPORT_FILENAME,
        _OPTIMIZER_FILENAME,
        _TENSOR_PAYLOAD_FILENAME,
        _MANIFEST_FILENAME,
    ):
        try:
            _write_once(staging / filename, payloads[filename])
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    try:
        staging.rename(path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return WrittenTrainingCheckpoint(
        path=path,
        manifest=manifest,
        sampler_state=sampler_state,
        scheduler_state=scheduler_state,
        adaptive_state=adaptive_state,
    )


def _read_exact_checkpoint_files(path: Path) -> dict[str, bytes]:
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not path.is_dir() or path.is_symlink():
        raise ValueError("checkpoint path must be a real directory")
    entries = tuple(path.iterdir())
    actual = {entry.name for entry in entries}
    if actual != _EXPECTED_FILENAMES:
        raise ValueError("checkpoint file set is incomplete or contains extras")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("checkpoint files must not be symlinks")
    return {name: (path / name).read_bytes() for name in _EXPECTED_FILENAMES}


def _validate_manifest_binding(
    manifest: TrainingCheckpointManifest,
    binding: TrainingRunBinding,
) -> None:
    comparisons = (
        ("run_hash", manifest.run_hash, binding.run_hash),
        (
            "dataset_snapshot_hash",
            manifest.dataset_snapshot_hash,
            binding.dataset_snapshot_hash,
        ),
        (
            "execution_profile_hash",
            manifest.execution_profile_hash,
            binding.execution_profile_hash,
        ),
        (
            "sampler_plan_hash",
            manifest.sampler_plan_hash,
            binding.sampler_plan_hash,
        ),
    )
    for label, actual, expected in comparisons:
        if actual != expected:
            raise ValueError(f"{label} does not match the requested binding")


def _validate_file_hashes(
    manifest: TrainingCheckpointManifest,
    payloads: dict[str, bytes],
) -> None:
    comparisons = (
        (
            "adaptive_state_hash",
            _file_hash_bytes(payloads[_ADAPTIVE_FILENAME]),
            manifest.adaptive_state_hash,
        ),
        (
            "optimizer_metadata_hash",
            _file_hash_bytes(payloads[_OPTIMIZER_FILENAME]),
            manifest.optimizer_metadata_hash,
        ),
        (
            "tensor_payload_hash",
            _file_hash_bytes(payloads[_TENSOR_PAYLOAD_FILENAME]),
            manifest.tensor_payload_hash,
        ),
        (
            "model_export_hash",
            _file_hash_bytes(payloads[_MODEL_EXPORT_FILENAME]),
            manifest.model_export_hash,
        ),
    )
    for label, actual, expected in comparisons:
        if actual != expected:
            raise ValueError(f"{label} mismatch")


def _validate_model_state(
    model: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    manifest: TrainingCheckpointManifest,
) -> dict[str, torch.Tensor]:
    expected = model.state_dict()
    model_tensors = {
        name.removeprefix(_MODEL_PREFIX): tensor
        for name, tensor in tensors.items()
        if name.startswith(_MODEL_PREFIX)
    }
    if set(model_tensors) != set(expected):
        raise ValueError("model state names do not match the requested model")
    for name, target in expected.items():
        source = model_tensors[name]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError("model state shape or dtype does not match the model")
        if (source.is_floating_point() or source.is_complex()) and not bool(
            torch.isfinite(source).all().item()
        ):
            raise ValueError("model state contains a non-finite tensor")
    framed = {f"{_MODEL_PREFIX}{name}": value for name, value in model_tensors.items()}
    if _state_hash(framed) != manifest.model_state_hash:
        raise ValueError("model_state_hash mismatch")
    return model_tensors


def _expected_optimizer_keys(metadata: AdamOptimizerState) -> set[str]:
    state_keys = {"step", "exp_avg", "exp_avg_sq"}
    if metadata.amsgrad:
        state_keys.add("max_exp_avg_sq")
    return {
        f"{_OPTIMIZER_PREFIX}{parameter_name}::{state_name}"
        for parameter_name in metadata.state_parameter_names
        for state_name in state_keys
    }


def _validate_optimizer_tensors(
    model: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    metadata: AdamOptimizerState,
    manifest: TrainingCheckpointManifest,
) -> dict[str, torch.Tensor]:
    optimizer_tensors = {
        name: tensor
        for name, tensor in tensors.items()
        if name.startswith(_OPTIMIZER_PREFIX)
    }
    if set(optimizer_tensors) != _expected_optimizer_keys(metadata):
        raise ValueError("optimizer state tensor schema mismatch")
    named_parameters = dict(model.named_parameters())
    for key, tensor in optimizer_tensors.items():
        _, parameter_name, state_name = key.split("::", maxsplit=2)
        parameter = named_parameters[parameter_name]
        if state_name == "step":
            if (
                tensor.numel() != 1
                or not bool(torch.isfinite(tensor).all().item())
                or float(tensor.item()) < 0.0
                or not float(tensor.item()).is_integer()
            ):
                raise ValueError("optimizer step tensor is invalid")
        else:
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise ValueError("optimizer moment shape or dtype mismatch")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError("optimizer moment contains a non-finite value")
    if _state_hash(optimizer_tensors) != manifest.optimizer_state_hash:
        raise ValueError("optimizer_state_hash mismatch")
    return optimizer_tensors


def _validate_model_export(
    path: Path,
    *,
    binding: TrainingRunBinding,
    manifest: TrainingCheckpointManifest,
    model_tensors: dict[str, torch.Tensor],
) -> None:
    expected_metadata = {
        "run_hash": binding.run_hash,
        "dataset_snapshot_hash": binding.dataset_snapshot_hash,
        "execution_profile_hash": binding.execution_profile_hash,
        "architecture_id": binding.architecture_id,
        "architecture_module_digest": binding.architecture_module_digest,
        "model_state_hash": manifest.model_state_hash,
    }
    with safe_open(path, framework="pt", device="cpu") as handle:
        if handle.metadata() != expected_metadata:
            raise ValueError("model export metadata mismatch")
        if set(handle.keys()) != set(model_tensors):
            raise ValueError("model export tensor names mismatch")
        for name, expected in model_tensors.items():
            actual = handle.get_tensor(name)
            if not torch.equal(actual, expected):
                raise ValueError("model export tensors do not match checkpoint state")


def _optimizer_state_dict(
    optimizer: torch.optim.Adam,
    metadata: AdamOptimizerState,
    tensors: dict[str, torch.Tensor],
) -> dict[str, object]:
    template = optimizer.state_dict()
    group = dict(template["param_groups"][0])
    parameter_ids = tuple(group["params"])
    group.update(
        {
            "lr": metadata.learning_rate,
            "betas": metadata.betas,
            "eps": metadata.eps,
            "weight_decay": metadata.weight_decay,
            "amsgrad": metadata.amsgrad,
            "maximize": metadata.maximize,
        }
    )
    state: dict[int, dict[str, torch.Tensor]] = {}
    by_name = dict(zip(metadata.parameter_names, parameter_ids, strict=True))
    for parameter_name in metadata.state_parameter_names:
        prefix = f"{_OPTIMIZER_PREFIX}{parameter_name}::"
        state[int(by_name[parameter_name])] = {
            key.removeprefix(prefix): value.clone()
            for key, value in tensors.items()
            if key.startswith(prefix)
        }
    return {"state": state, "param_groups": [group]}


def load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
    binding: TrainingRunBinding,
) -> LoadedTrainingCheckpoint:
    """Validate a complete checkpoint before restoring exact training state."""

    _validate_binding(binding)
    payloads = _read_exact_checkpoint_files(path)
    try:
        manifest = TrainingCheckpointManifest.from_bytes(payloads[_MANIFEST_FILENAME])
    except Exception as exc:
        raise ValueError("canonical checkpoint manifest is invalid") from exc
    _validate_manifest_binding(manifest, binding)
    _validate_file_hashes(manifest, payloads)
    try:
        adaptive_state = _parse_adaptive(payloads[_ADAPTIVE_FILENAME])
        optimizer_metadata = _parse_optimizer(payloads[_OPTIMIZER_FILENAME])
    except Exception as exc:
        raise ValueError("canonical checkpoint sidecar is invalid") from exc
    if (
        adaptive_state.last_update_step is not None
        and adaptive_state.last_update_step > manifest.step
    ):
        raise ValueError("adaptive state exceeds checkpoint step")
    current_optimizer = capture_adam_optimizer_state(model, optimizer)
    if current_optimizer.parameter_names != optimizer_metadata.parameter_names:
        raise ValueError("optimizer parameter_names do not match the model")
    if not math.isclose(
        manifest.scheduler_state.learning_rate,
        optimizer_metadata.learning_rate,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("scheduler and optimizer learning rates differ")
    try:
        with safe_open(
            path / manifest.tensor_payload_filename,
            framework="pt",
            device="cpu",
        ) as handle:
            if handle.metadata() != {
                "run_hash": binding.run_hash,
                "step": str(manifest.step),
            }:
                raise ValueError("checkpoint tensor payload metadata mismatch")
        tensors = load_safetensors(payloads[_TENSOR_PAYLOAD_FILENAME])
    except Exception as exc:
        raise ValueError("checkpoint tensor payload is invalid") from exc
    expected_all_keys = {
        *(f"{_MODEL_PREFIX}{name}" for name in model.state_dict()),
        *_expected_optimizer_keys(optimizer_metadata),
        _RNG_KEY,
    }
    if set(tensors) != expected_all_keys:
        raise ValueError("checkpoint tensor payload schema mismatch")
    model_state = _validate_model_state(model, tensors, manifest)
    optimizer_tensors = _validate_optimizer_tensors(
        model,
        tensors,
        optimizer_metadata,
        manifest,
    )
    rng_state = tensors[_RNG_KEY]
    if (
        rng_state.dtype is not torch.uint8
        or rng_state.ndim != 1
        or rng_state.shape != torch.random.get_rng_state().shape
        or _tensor_hash(rng_state) != manifest.torch_cpu_rng_hash
    ):
        raise ValueError("torch_cpu_rng_hash mismatch")
    _validate_model_export(
        path / manifest.model_export_filename,
        binding=binding,
        manifest=manifest,
        model_tensors=model_state,
    )

    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(
        _optimizer_state_dict(optimizer, optimizer_metadata, optimizer_tensors)
    )
    torch.random.set_rng_state(rng_state)
    return LoadedTrainingCheckpoint(
        path=path,
        manifest=manifest,
        sampler_state=manifest.sampler_state,
        scheduler_state=manifest.scheduler_state,
        adaptive_state=adaptive_state,
    )


__all__ = (
    "LoadedTrainingCheckpoint",
    "TrainingCheckpoint",
    "TrainingCheckpointManifest",
    "WrittenTrainingCheckpoint",
    "export_model_safetensors",
    "load_training_checkpoint",
    "model_state_hash",
    "write_training_checkpoint",
)
