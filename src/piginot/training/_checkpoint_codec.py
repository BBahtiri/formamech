"""Low-level canonical encoding helpers for S13 training checkpoints."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import torch
from safetensors.torch import save as save_safetensors

from piginot.contracts import hash_spec, strict_json_loads
from piginot.contracts.canonical import canonical_json_bytes
from piginot.contracts.errors import ContractValidationError
from piginot.objectives.models import ObjectiveTerm
from piginot.training.adaptive import AdaptiveWeightState
from piginot.training.state import (
    AdamOptimizerState,
    DeterministicSamplerState,
    TrainingSchedulerState,
)

_DIGEST_BYTES = 32
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_SAFE_INTEGER = 2**53 - 1
_CHECKPOINT_ID = "CP-TRAINING"
_MANIFEST_FILENAME = "checkpoint.json"
_TENSOR_PAYLOAD_FILENAME = "state.safetensors"
_MODEL_EXPORT_FILENAME = "model.safetensors"
_OPTIMIZER_FILENAME = "optimizer.json"
_ADAPTIVE_FILENAME = "adaptive.json"
_EXPECTED_FILENAMES = frozenset(
    {
        _ADAPTIVE_FILENAME,
        _MANIFEST_FILENAME,
        _MODEL_EXPORT_FILENAME,
        _OPTIMIZER_FILENAME,
        _TENSOR_PAYLOAD_FILENAME,
    }
)
_MANIFEST_TYPE = "piginot.training.TrainingCheckpointManifest"
_SCHEDULER_TYPE = "piginot.training.TrainingSchedulerState"
_SAMPLER_TYPE = "piginot.training.DeterministicSamplerState"
_OPTIMIZER_TYPE = "piginot.training.AdamOptimizerState"
_ADAPTIVE_TYPE = "piginot.training.AdaptiveWeightState"
_MODEL_PREFIX = "model::"
_OPTIMIZER_PREFIX = "optimizer::"
_RNG_KEY = "torch_cpu_rng"


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ContractValidationError(f"{label} must be a lowercase 256-bit digest")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > _MAX_SAFE_INTEGER
    ):
        raise ContractValidationError(
            f"{label} must be a non-negative RFC 8785-safe integer"
        )
    return value


def _file_hash_bytes(payload: bytes) -> str:
    return hashlib.blake2b(payload, digest_size=_DIGEST_BYTES).hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    payload = save_safetensors({"tensor": cpu})
    return hashlib.blake2b(payload, digest_size=_DIGEST_BYTES).hexdigest()


def _state_hash(tensors: dict[str, torch.Tensor]) -> str:
    description = {
        name: {
            "dtype": str(tensor.dtype),
            "shape": tuple(tensor.shape),
            "tensor_hash": _tensor_hash(tensor),
        }
        for name, tensor in sorted(tensors.items())
    }
    return hash_spec(description, "content")


def _write_once(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"artifact already exists: {path.name}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _new_artifact_path(path: Path, *, suffix: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if path.suffix != suffix:
        raise ValueError(f"artifact path must end in {suffix}")
    if not path.parent.exists() or not path.parent.is_dir():
        raise FileNotFoundError("artifact parent directory must already exist")
    if path.parent.is_symlink():
        raise ValueError("artifact parent directory must not be a symlink")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"artifact already exists: {path.name}")
    return path


def _exact_mapping(
    value: object,
    *,
    canonical_type: str,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{label} must be a JSON object")
    result: dict[str, object] = dict(value)
    if result.pop("$type", None) != canonical_type:
        raise ContractValidationError(f"canonical {label} type mismatch")
    return result


def _parse_scheduler(value: object) -> TrainingSchedulerState:
    payload = _exact_mapping(
        value,
        canonical_type=_SCHEDULER_TYPE,
        label="scheduler state",
    )
    return TrainingSchedulerState.from_mapping(payload)


def _parse_sampler(value: object) -> DeterministicSamplerState:
    payload = _exact_mapping(
        value,
        canonical_type=_SAMPLER_TYPE,
        label="sampler state",
    )
    return DeterministicSamplerState.from_mapping(payload)


def _parse_optimizer(payload: bytes) -> AdamOptimizerState:
    decoded = strict_json_loads(payload)
    mapping = _exact_mapping(
        decoded,
        canonical_type=_OPTIMIZER_TYPE,
        label="optimizer metadata",
    )
    for field_name in ("parameter_names", "state_parameter_names", "betas"):
        value = mapping.get(field_name)
        if isinstance(value, list):
            mapping[field_name] = tuple(value)
    result = AdamOptimizerState.from_mapping(mapping)
    if canonical_json_bytes(result) != payload:
        raise ContractValidationError("optimizer metadata is not canonical JSON")
    return result


def _parse_adaptive(payload: bytes) -> AdaptiveWeightState:
    decoded = strict_json_loads(payload)
    mapping = _exact_mapping(
        decoded,
        canonical_type=_ADAPTIVE_TYPE,
        label="adaptive state",
    )
    expected_fields = {"multipliers", "update_count", "last_update_step"}
    if set(mapping) != expected_fields:
        raise ContractValidationError("adaptive state fields are invalid")
    multipliers = mapping["multipliers"]
    if not isinstance(multipliers, dict):
        raise ContractValidationError("adaptive multipliers must be a JSON object")
    converted: dict[ObjectiveTerm, float] = {}
    try:
        for name, value in multipliers.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError
            converted[ObjectiveTerm(name)] = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError("adaptive multiplier term is invalid") from exc
    update_count = mapping["update_count"]
    last_update_step = mapping["last_update_step"]
    if isinstance(update_count, bool) or not isinstance(update_count, int):
        raise ContractValidationError("adaptive update_count must be an integer")
    if last_update_step is not None and (
        isinstance(last_update_step, bool) or not isinstance(last_update_step, int)
    ):
        raise ContractValidationError(
            "adaptive last_update_step must be null or an integer"
        )
    result = AdaptiveWeightState(
        multipliers=converted,
        update_count=update_count,
        last_update_step=last_update_step,
    )
    if canonical_json_bytes(result) != payload:
        raise ContractValidationError("adaptive state is not canonical JSON")
    return result


__all__ = (
    "_ADAPTIVE_FILENAME",
    "_CHECKPOINT_ID",
    "_EXPECTED_FILENAMES",
    "_MANIFEST_FILENAME",
    "_MANIFEST_TYPE",
    "_MODEL_EXPORT_FILENAME",
    "_MODEL_PREFIX",
    "_OPTIMIZER_FILENAME",
    "_OPTIMIZER_PREFIX",
    "_RNG_KEY",
    "_TENSOR_PAYLOAD_FILENAME",
    "_digest",
    "_exact_mapping",
    "_file_hash_bytes",
    "_new_artifact_path",
    "_nonnegative_integer",
    "_parse_adaptive",
    "_parse_optimizer",
    "_parse_sampler",
    "_parse_scheduler",
    "_state_hash",
    "_tensor_hash",
    "_write_once",
)
