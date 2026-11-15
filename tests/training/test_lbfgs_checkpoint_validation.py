"""Malformed safe checkpoint data must fail before changing the caller model."""

import hashlib
import json

import pytest
import torch
from safetensors.torch import load, save

from piginot.training import lbfgs_checkpoint as codec
from piginot.training.lbfgs_stage import train_lbfgs_stage
from tests.training.test_lbfgs_stage import Rosenbrock, config, objective


@pytest.mark.parametrize(
    "change",
    [
        "counts",
        "budget",
        "model_hash",
        "admissible",
        "tree",
        "layout",
        "cache_shape",
        "rng_shape",
        "extra_state",
        "optimizer",
        "missing_cache",
        "nonfinite",
    ],
)
def test_checkpoint_semantics_are_validated_transactionally(tmp_path, change):
    root = tmp_path / "stage"
    model = Rosenbrock()
    saved = train_lbfgs_stage(
        root,
        model=model,
        objective=objective([]),
        binding="a" * 64,
        config=config(),
        target_iterations=2,
    )
    meta = saved.checkpoint / "checkpoint.json"
    path = saved.checkpoint / "state.safetensors"
    record = json.loads(meta.read_text())
    payload = path.read_bytes()
    incoming = Rosenbrock()
    if change == "counts":
        record["evaluations"] = -1
    if change == "budget":
        record["iterations"] = 100
    if change == "model_hash":
        record["model_hash"] = "f" * 64
    if change == "admissible":
        record["admissible"] = False
    if change == "tree":
        record["tree"] = {"unknown": 1}
    if change == "layout":
        incoming.point = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    if change in (
        "cache_shape",
        "rng_shape",
        "extra_state",
        "optimizer",
        "missing_cache",
    ):
        tensors = load(payload)
        state = codec._unpack(record["tree"], tensors, set())
        if change == "cache_shape":
            state["gradient"] = torch.ones(3, dtype=torch.float64)
        if change == "rng_shape":
            state["rng"] = torch.zeros(3, dtype=torch.uint8)
        if change == "extra_state":
            state["unexpected"] = 0
        if change == "optimizer":
            state["optimizer"]["state"][0]["unsupported"] = 1
        if change == "missing_cache":
            state["gradient"] = None
            record["loss"] = None
            record["admissible"] = None
        packed = {}
        record["tree"] = codec._pack(state, packed)
        payload = save(packed)
    if change == "nonfinite":
        tensors = load(payload)
        key = next(key for key, t in tensors.items() if t.dtype == torch.float64)
        tensors[key] = tensors[key] * float("nan")
        payload = save(tensors)
    path.write_bytes(payload)
    record["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    meta.write_text(json.dumps(record))
    before = incoming.point.detach().clone()
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        train_lbfgs_stage(
            root,
            model=incoming,
            objective=objective([]),
            binding="a" * 64,
            config=config(),
            resume_from=saved.checkpoint,
        )
    assert torch.equal(incoming.point, before)
    assert torch.equal(torch.get_rng_state(), rng)
