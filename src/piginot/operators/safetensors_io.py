"""Fail-closed safetensors preflight and loading for S11 operators."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from piginot.operators.architecture import (
    ARCHITECTURE_ID,
    PiGinotOperator,
    module_digest,
)

_HEADER_PREFIX_BYTES = 8
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_SAFETENSORS_DTYPES: dict[torch.dtype, str] = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
}


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Expected tensor shape and dtype from an in-memory module schema."""

    shape: tuple[int, ...]
    dtype: torch.dtype


def state_dict_schema(
    value: nn.Module | Mapping[str, torch.Tensor],
) -> dict[str, TensorSpec]:
    """Return a schema without inspecting tensor values."""

    state: Mapping[str, torch.Tensor]
    if isinstance(value, nn.Module):
        state = value.state_dict()
    elif isinstance(value, Mapping):
        state = value
    else:
        raise TypeError("value must be a torch module or tensor mapping")
    result: dict[str, TensorSpec] = {}
    for name, tensor in sorted(state.items()):
        if type(name) is not str or not name:
            raise TypeError("state-dict keys must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("state-dict values must be torch tensors")
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must be a dense strided tensor")
        if tensor.dtype not in _SAFETENSORS_DTYPES:
            raise ValueError(f"{name} uses unsupported dtype {tensor.dtype}")
        result[name] = TensorSpec(tuple(tensor.shape), tensor.dtype)
    if not result:
        raise ValueError("state-dict schema must not be empty")
    return result


def load_operator_state_dict(
    path: str | Path,
    *,
    model: PiGinotOperator,
) -> dict[str, torch.Tensor]:
    """Load weights only after identity, key, shape and dtype preflight."""

    if type(model) is not PiGinotOperator:
        raise TypeError("model must be an exact PiGinotOperator")
    schema = state_dict_schema(model)
    expected_metadata = {
        "architecture_id": ARCHITECTURE_ID,
        "module_digest": module_digest(model),
    }
    header = _preflight_header(path, schema, expected_metadata)
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:
        raise RuntimeError("install piginot[ml] to load operator weights") from exc
    loaded = load_file(str(Path(path)), device="cpu")
    _validate_materialized(loaded, schema)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("safetensors metadata must be a JSON object")
    return dict(loaded)


def _preflight_header(
    path: str | Path,
    schema: Mapping[str, TensorSpec],
    expected_metadata: Mapping[str, str],
) -> dict[str, object]:
    candidate = Path(path)
    if candidate.suffix != ".safetensors":
        raise ValueError("model weights must use the .safetensors suffix")
    if not candidate.is_file():
        raise FileNotFoundError(f"safetensors file does not exist: {candidate}")
    file_size = candidate.stat().st_size
    with candidate.open("rb") as stream:
        prefix = stream.read(_HEADER_PREFIX_BYTES)
        if len(prefix) != _HEADER_PREFIX_BYTES:
            raise ValueError("invalid safetensors header prefix")
        header_size = int.from_bytes(prefix, "little", signed=False)
        if (
            header_size < 2
            or header_size > _MAX_HEADER_BYTES
            or _HEADER_PREFIX_BYTES + header_size > file_size
        ):
            raise ValueError("invalid safetensors header length")
        raw_header = stream.read(header_size)
    try:
        decoded = raw_header.decode("utf-8")
        header = json.loads(decoded, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid safetensors header JSON") from exc
    if type(header) is not dict:
        raise ValueError("safetensors header must be a JSON object")
    _validate_metadata(header.get("__metadata__"), expected_metadata)
    tensor_headers = {
        name: value for name, value in header.items() if name != "__metadata__"
    }
    expected_keys = set(schema)
    actual_keys = set(tensor_headers)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "safetensors keys do not match the module schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, spec in schema.items():
        _validate_tensor_header(name, tensor_headers[name], spec)
    return header


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _validate_metadata(
    value: object,
    expected: Mapping[str, str],
) -> None:
    if type(value) is not dict:
        raise ValueError("safetensors metadata is required")
    if any(
        type(key) is not str or type(item) is not str for key, item in value.items()
    ):
        raise ValueError("safetensors metadata must map strings to strings")
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(f"safetensors {name} does not match the module")


def _validate_tensor_header(name: str, value: object, spec: TensorSpec) -> None:
    if type(value) is not dict:
        raise ValueError(f"{name} tensor header must be a JSON object")
    dtype = value.get("dtype")
    expected_dtype = _SAFETENSORS_DTYPES[spec.dtype]
    if dtype != expected_dtype:
        raise ValueError(f"{name} has dtype {dtype!r}; expected {expected_dtype!r}")
    shape = value.get("shape")
    if (
        type(shape) is not list
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in shape
        )
        or tuple(shape) != spec.shape
    ):
        raise ValueError(f"{name} has shape {shape!r}; expected {spec.shape!r}")
    offsets = value.get("data_offsets")
    if (
        type(offsets) is not list
        or len(offsets) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in offsets
        )
        or offsets[0] > offsets[1]
    ):
        raise ValueError(f"{name} has invalid data_offsets")


def _validate_materialized(
    loaded: Mapping[str, torch.Tensor],
    schema: Mapping[str, TensorSpec],
) -> None:
    if set(loaded) != set(schema):
        raise ValueError("materialized safetensors keys changed after preflight")
    for name, tensor in loaded.items():
        spec = schema[name]
        if not isinstance(tensor, torch.Tensor) or tensor.layout is not torch.strided:
            raise ValueError(f"{name} did not materialize as a dense tensor")
        if tuple(tensor.shape) != spec.shape:
            raise ValueError(f"{name} shape changed after preflight")
        if tensor.dtype is not spec.dtype:
            raise ValueError(f"{name} dtype changed after preflight")


__all__ = (
    "TensorSpec",
    "load_operator_state_dict",
    "state_dict_schema",
)
