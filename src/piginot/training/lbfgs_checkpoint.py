"""Safe, immutable checkpoints for the distinct deterministic L-BFGS phase."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load, save

from piginot.contracts import canonical_json_bytes, strict_json_loads
from piginot.training._checkpoint_codec import _canonical_safetensors
from piginot.training.checkpoint import model_state_hash

SCHEMA = "piginot.lbfgs-stage-checkpoint-v1"


@dataclass(frozen=True, slots=True)
class EvaluatedPoint:
    model_hash: str
    value: float
    gradient: torch.Tensor
    admissible: bool = True


@dataclass(frozen=True, slots=True)
class StageCursor:
    iterations: int = 0
    evaluations: int = 0
    closure_calls: int = 0


def _pack(value: Any, tensors: dict[str, torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("checkpoint tensors must be finite")
        name = f"tensor-{len(tensors):06d}"
        tensors[name] = value.detach().cpu().contiguous().clone()
        return {"tensor": name}
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise ValueError("optimizer keys must be strings or integers")
        return {"mapping": [[key, _pack(item, tensors)] for key, item in value.items()]}
    if isinstance(value, (list, tuple)):
        tag = "list" if isinstance(value, list) else "tuple"
        return {tag: [_pack(item, tensors) for item in value]}
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("unsupported optimizer checkpoint value")


def _unpack(value: Any, tensors: dict[str, torch.Tensor], used: set[str]) -> Any:
    if not isinstance(value, dict):
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        raise ValueError("invalid checkpoint scalar")
    if set(value) == {"tensor"} and value["tensor"] in tensors:
        used.add(value["tensor"])
        return tensors[value["tensor"]].clone()
    if set(value) in ({"list"}, {"tuple"}):
        items = [_unpack(item, tensors, used) for item in next(iter(value.values()))]
        return items if "list" in value else tuple(items)
    if set(value) == {"mapping"}:
        pairs = value["mapping"]
        if any(type(key) not in (str, int) for key, _ in pairs):
            raise ValueError("invalid checkpoint mapping keys")
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("duplicate checkpoint mapping keys")
        return {key: _unpack(item, tensors, used) for key, item in pairs}
    raise ValueError("invalid checkpoint tree")


def save_stage_checkpoint(
    root: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.LBFGS,
    binding: str,
    config: dict[str, Any],
    cursor: StageCursor,
    point: EvaluatedPoint | None,
) -> Path:
    """Persist model, actual L-BFGS history, counters and accepted-point cache."""
    tensors: dict[str, torch.Tensor] = {}
    tree = _pack(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
            "gradient": None if point is None else point.gradient,
        },
        tensors,
    )
    payload = _canonical_safetensors(save(tensors))
    record = {
        "schema": SCHEMA,
        "optimizer_kind": "LBFGS",
        "binding": binding,
        "config": config,
        "iterations": cursor.iterations,
        "evaluations": cursor.evaluations,
        "closure_calls": cursor.closure_calls,
        "model_hash": model_state_hash(model),
        "loss": None if point is None else point.value,
        "admissible": None if point is None else point.admissible,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "tree": tree,
    }
    if point is not None and point.model_hash != record["model_hash"]:
        raise ValueError("cached gradient does not belong to accepted weights")
    metadata = canonical_json_bytes(record)
    folder = root / "checkpoints"
    folder.mkdir(exist_ok=True)
    path = (
        folder
        / f"iteration-{cursor.iterations:06d}-evaluation-{cursor.evaluations:06d}-closure-{cursor.closure_calls:06d}"
    )
    if path.exists():
        if (path / "checkpoint.json").read_bytes() != metadata or (
            path / "state.safetensors"
        ).read_bytes() != payload:
            raise FileExistsError("checkpoint identity already has different content")
        return path
    pending = Path(tempfile.mkdtemp(prefix=".lbfgs-checkpoint-", dir=folder))
    for name, content in (
        ("state.safetensors", payload),
        ("checkpoint.json", metadata),
    ):
        with (pending / name).open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    os.rename(pending, path)
    return path


def _validate_optimizer(saved: Any, optimizer: torch.optim.LBFGS) -> None:
    expected = optimizer.state_dict()["param_groups"]
    if (
        not isinstance(saved, dict)
        or set(saved) != {"state", "param_groups"}
        or saved["param_groups"] != expected
    ):
        raise ValueError("L-BFGS optimizer configuration differs")
    if set(saved["state"]) - {0}:
        raise ValueError("L-BFGS has one global optimizer state")
    allowed = {
        "func_evals",
        "n_iter",
        "d",
        "t",
        "old_dirs",
        "old_stps",
        "ro",
        "H_diag",
        "prev_flat_grad",
        "prev_loss",
        "al",
    }
    state = saved["state"].get(0, {})
    if not isinstance(state, dict) or set(state) - allowed:
        raise ValueError("unsupported L-BFGS state fields")
    for name in ("old_dirs", "old_stps", "ro", "al"):
        if name in state and (
            not isinstance(state[name], list)
            or len(state[name]) > expected[0]["history_size"]
        ):
            raise ValueError("invalid L-BFGS history length")


def load_stage_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.LBFGS,
    binding: str,
    config: dict[str, Any],
) -> tuple[StageCursor, EvaluatedPoint | None]:
    """Validate completely, then restore a bound L-BFGS checkpoint atomically."""
    if path.is_symlink() or any(
        (path / name).is_symlink() for name in ("checkpoint.json", "state.safetensors")
    ):
        raise ValueError("stage checkpoints cannot be symlinks")
    record = strict_json_loads((path / "checkpoint.json").read_bytes())
    if (
        not isinstance(record, dict)
        or record.get("schema") != SCHEMA
        or record.get("optimizer_kind") != "LBFGS"
        or record.get("binding") != binding
        or record.get("config") != config
    ):
        raise ValueError("checkpoint schema, phase, binding or configuration differs")
    counts = [
        record.get(name) for name in ("iterations", "evaluations", "closure_calls")
    ]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("invalid stage counters")
    cursor = StageCursor(*(cast(int, value) for value in counts))
    if (
        cursor.iterations > config["max_iterations"]
        or cursor.evaluations > config["max_evaluations"]
        or cursor.evaluations > cursor.closure_calls
    ):
        raise ValueError("checkpoint counters exceed the stage budget")
    payload = (path / "state.safetensors").read_bytes()
    if hashlib.sha256(payload).hexdigest() != record.get("payload_sha256"):
        raise ValueError("L-BFGS checkpoint payload digest differs")
    tensors = load(payload)
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("checkpoint tensors must be finite")
    used: set[str] = set()
    state = _unpack(record.get("tree"), tensors, used)
    if used != set(tensors) or set(state) != {"model", "optimizer", "rng", "gradient"}:
        raise ValueError("checkpoint tensor or state roster differs")
    expected = model.state_dict()
    if set(state["model"]) != set(expected) or any(
        state["model"][name].shape != value.shape
        or state["model"][name].dtype != value.dtype
        for name, value in expected.items()
    ):
        raise ValueError("checkpoint model layout differs")
    _validate_optimizer(state["optimizer"], optimizer)
    gradient, loss = state["gradient"], record.get("loss")
    if gradient is not None and (
        gradient.shape != (sum(p.numel() for p in model.parameters()),)
        or gradient.dtype != torch.float64
        or not isinstance(loss, (float, int))
        or isinstance(loss, bool)
        or not math.isfinite(loss)
    ):
        raise ValueError("invalid accepted-point gradient cache")
    if (
        (gradient is None) != (loss is None)
        or (cursor.iterations > 0 and gradient is None)
        or state["rng"].dtype != torch.uint8
        or state["rng"].shape != torch.get_rng_state().shape
    ):
        raise ValueError("invalid accepted-point cache or generator state")
    if gradient is not None and record.get("admissible") is not True:
        raise ValueError("an accepted checkpoint must be admissible")
    previous_model, previous_optimizer = (
        copy.deepcopy(expected),
        copy.deepcopy(optimizer.state_dict()),
    )
    try:
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if model_state_hash(model) != record["model_hash"]:
            raise ValueError("restored model identity differs")
    except BaseException:
        model.load_state_dict(previous_model)
        optimizer.load_state_dict(previous_optimizer)
        raise
    torch.set_rng_state(state["rng"])
    point = (
        None
        if gradient is None
        else EvaluatedPoint(
            str(record["model_hash"]), float(cast(float, loss)), gradient
        )
    )
    return cursor, point
