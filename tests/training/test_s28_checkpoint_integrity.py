"""Synthetic checkpoint corruption must refuse before restoring any state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import rfc8785
import torch
from safetensors import safe_open
from safetensors.torch import load, save

from piginot.training import checkpoint
from tests.training import test_checkpoint as cp


@pytest.fixture
def saved(tmp_path):
    model = cp._model()
    optimizer = cp._optimizer(model)
    # Explicit initial Adam moments exercise restoration without optimizer work.
    for parameter in model.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(0.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    arguments = {
        "model": model,
        "optimizer": optimizer,
        "binding": cp._binding(),
        "step": 0,
        "sampler_state": cp._plan().initial_state,
        "scheduler_state": cp._scheduler(),
        "adaptive_state": cp._adaptive(),
    }
    written = checkpoint.write_training_checkpoint(tmp_path / "checkpoint", **arguments)
    return written, arguments


def republish(written, filename, payload, field):
    (written.path / filename).write_bytes(payload)
    manifest = replace(
        written.manifest,
        **{field: hashlib.blake2b(payload, digest_size=32).hexdigest()},
    )
    (written.path / "checkpoint.json").write_bytes(manifest.to_bytes())


@pytest.mark.parametrize(
    "change",
    [
        "adaptive-fields",
        "adaptive-map",
        "adaptive-bool",
        "adaptive-term",
        "adaptive-count",
        "adaptive-last-step",
        "adaptive-future",
        "adaptive-noncanonical",
        "optimizer-noncanonical",
        "optimizer-names",
        "optimizer-rate",
        "model-shape",
        "model-nonfinite",
        "model-hash",
        "moment-shape",
        "moment-dtype",
        "moment-nonfinite",
        "moment-hash",
        "step-shape",
        "step-negative",
        "tensor-metadata",
        "tensor-schema",
        "tensor-unreadable",
        "rng-dtype",
        "rng-shape",
        "rng-hash",
        "export-metadata",
        "export-schema",
        "export-values",
    ],
)
def test_self_consistent_outer_hashes_do_not_authorize_corrupt_payload(saved, change):
    written, arguments = saved
    if change.startswith("adaptive-") or change.startswith("optimizer-"):
        adaptive = change.startswith("adaptive-")
        filename = "adaptive.json" if adaptive else "optimizer.json"
        field = "adaptive_state_hash" if adaptive else "optimizer_metadata_hash"
        document = json.loads((written.path / filename).read_bytes())
        if change == "adaptive-fields":
            document["unbound"] = 1
        elif change == "adaptive-map":
            document["multipliers"] = []
        elif change in {"adaptive-bool", "adaptive-term"}:
            document["multipliers"] = (
                {"constitutive": True}
                if change == "adaptive-bool"
                else {"unknown": 1.0}
            )
        elif change == "adaptive-count":
            document["update_count"] = 0.5
        elif change == "adaptive-last-step":
            document["last_update_step"] = 0.5
        elif change == "adaptive-future":
            document.update(update_count=1, last_update_step=1)
        elif change == "optimizer-names":
            document["parameter_names"] = list(reversed(document["parameter_names"]))
            document["state_parameter_names"] = list(document["parameter_names"])
        elif change == "optimizer-rate":
            document["learning_rate"] *= 2.0
        payload = (
            json.dumps(document, indent=2).encode()
            if "noncanonical" in change
            else rfc8785.dumps(document)
        )
        republish(written, filename, payload, field)
    elif change.startswith("export-"):
        filename = "model.safetensors"
        path = written.path / filename
        tensors = load(path.read_bytes())
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        if change == "export-metadata":
            metadata["architecture_id"] = "foreign-architecture"
        elif change == "export-schema":
            tensors.pop("bias")
        else:
            tensors["weight"] = tensors["weight"] + 1.0
        republish(
            written, filename, save(tensors, metadata=metadata), "model_export_hash"
        )
    elif change in {"model-hash", "moment-hash"}:
        field = "model_state_hash" if change == "model-hash" else "optimizer_state_hash"
        (written.path / "checkpoint.json").write_bytes(
            replace(written.manifest, **{field: "f" * 64}).to_bytes()
        )
    else:
        filename = "state.safetensors"
        tensors = load((written.path / filename).read_bytes())
        metadata = {"run_hash": arguments["binding"].run_hash, "step": "0"}
        if change == "model-shape":
            tensors["model::weight"] = torch.zeros((2, 2))
        elif change == "model-nonfinite":
            tensors["model::weight"].fill_(float("nan"))
        elif change == "moment-shape":
            tensors["optimizer::weight::exp_avg"] = torch.zeros((2, 2))
        elif change == "moment-dtype":
            tensors["optimizer::weight::exp_avg"] = tensors[
                "optimizer::weight::exp_avg"
            ].double()
        elif change == "moment-nonfinite":
            tensors["optimizer::weight::exp_avg"].fill_(float("inf"))
        elif change == "step-shape":
            tensors["optimizer::weight::step"] = torch.zeros(2)
        elif change == "step-negative":
            tensors["optimizer::weight::step"].fill_(-1.0)
        elif change == "tensor-metadata":
            metadata["run_hash"] = "f" * 64
        elif change == "tensor-schema":
            tensors["extra"] = torch.ones(1)
        elif change == "rng-dtype":
            tensors["torch_cpu_rng"] = tensors["torch_cpu_rng"].long()
        elif change == "rng-shape":
            tensors["torch_cpu_rng"] = tensors["torch_cpu_rng"][:-1]
        elif change == "rng-hash":
            tensors["torch_cpu_rng"][0] ^= 1
        payload = (
            b"unreadable safetensors"
            if change == "tensor-unreadable"
            else save(tensors, metadata=metadata)
        )
        republish(written, filename, payload, "tensor_payload_hash")

    restored = cp._model()
    optimizer = cp._optimizer(restored)
    before = {name: tensor.clone() for name, tensor in restored.state_dict().items()}
    rng = torch.get_rng_state().clone()
    previous_optimizer = optimizer.state_dict()
    with pytest.raises(ValueError):
        checkpoint.load_training_checkpoint(
            written.path,
            model=restored,
            optimizer=optimizer,
            binding=arguments["binding"],
        )
    assert all(
        torch.equal(tensor, restored.state_dict()[name])
        for name, tensor in before.items()
    )
    assert optimizer.state_dict() == previous_optimizer
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "change",
    [
        "sampler",
        "scheduler",
        "adaptive",
        "binding",
        "device",
        "step",
        "path",
        "missing-parent",
        "parent-link",
    ],
)
def test_checkpoint_publication_refuses_ambiguous_binding_or_cursors(saved, change):
    written, arguments = saved
    destination = written.path.with_name("never-written")
    arguments = dict(arguments)
    if change in {"sampler", "scheduler", "adaptive", "binding"}:
        arguments[
            {
                "sampler": "sampler_state",
                "scheduler": "scheduler_state",
                "adaptive": "adaptive_state",
                "binding": "binding",
            }[change]
        ] = object()
    elif change == "device":
        binding = arguments["binding"]
        arguments["binding"] = replace(
            binding, execution_profile=replace(binding.execution_profile, device="cuda")
        )
    elif change == "step":
        arguments["step"] = True
    elif change == "path":
        destination = str(destination)
    elif change == "missing-parent":
        destination = written.path.parent / "absent" / "child"
    else:
        link = written.path.with_name("link")
        link.symlink_to(written.path, target_is_directory=True)
        destination = link / "child"
    with pytest.raises((TypeError, ValueError, FileNotFoundError)):
        checkpoint.write_training_checkpoint(destination, **arguments)
    if isinstance(destination, Path):
        assert not destination.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("scheduler_state", object()),
        ("sampler_state", object()),
        ("sampler_plan_hash", "f" * 64),
        ("sampler_state", cp.DeterministicSamplerState(cp._plan().plan_hash, 1)),
        ("scheduler_state", cp._scheduler(step=1)),
        ("tensor_payload_filename", "foreign.safetensors"),
    ],
)
def test_manifest_refuses_inconsistent_cursors_and_payload_names(saved, field, value):
    written, _ = saved
    with pytest.raises((TypeError, ValueError)):
        replace(written.manifest, **{field: value})


@pytest.mark.parametrize(
    "change", ["noncanonical", "root-type", "scheduler-type", "sampler-type"]
)
def test_manifest_decoder_refuses_ambiguous_json(saved, change):
    written, _ = saved
    document = json.loads(written.manifest.to_bytes())
    if change == "root-type":
        document = []
    elif change == "scheduler-type":
        document["scheduler_state"] = []
    elif change == "sampler-type":
        document["sampler_state"] = []
    payload = (
        json.dumps(document, indent=2).encode()
        if change == "noncanonical"
        else rfc8785.dumps(document)
    )
    with pytest.raises(ValueError):
        checkpoint.TrainingCheckpointManifest.from_bytes(payload)


@pytest.mark.parametrize("change", ["type", "empty", "name", "tensor-type", "sparse"])
def test_weight_identity_refuses_unserializable_model_state(change):
    model = cp._model()
    if change == "type":
        model = object()
    elif change == "empty":
        model = torch.nn.Identity()
    elif change == "name":
        model.register_buffer("unbound::buffer", torch.ones(1))
    elif change == "sparse":
        model.register_buffer("sparse", torch.ones(2).to_sparse())
    else:

        class InvalidState(torch.nn.Module):
            def state_dict(self):
                return {"weight": "not a tensor"}

        model = InvalidState()
    with pytest.raises((TypeError, ValueError)):
        checkpoint.model_state_hash(model)
