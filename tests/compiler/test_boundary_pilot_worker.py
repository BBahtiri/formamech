"""Tiny worker integration and initializer import use no reference solves."""

import copy
import json
import time

import pytest
import torch
from safetensors.torch import save_file

from piginot.compiler import boundary_pilot as worker
from piginot.compiler import boundary_pilot_inputs as inputs
from piginot.compiler.boundary_objective import BoundaryPilotObjective
from piginot.training.checkpoint import model_state_hash
from piginot.training.lbfgs_stage import LBFGSConfig
from tests.compiler.test_execution_training import context as context


def test_worker_writes_bound_results_and_restores_process_guards(
    context, tmp_path, monkeypatch
):
    context.prepare()
    data = BoundaryPilotObjective(
        context,
        ("fit-001", "fit-002"),
        strain_scales={"fit-001": 0.02, "fit-002": 0.03},
        points_per_cell=2,
        chunk=3,
    )
    root = tmp_path / "run"
    root.mkdir()
    path = tmp_path / "protocol.json"
    path.write_text("{}")
    deadline = time.time() + 43200
    started = deadline - 43200
    (root / "job.json").write_text(
        json.dumps(
            {
                "schema": "piginot.boundary-pilot-job-v1",
                "status": "RUNNING",
                "protocol_sha256": worker.sha(path),
                "started_epoch": started,
                "deadline_epoch": deadline,
            }
        )
    )
    protocol = {
        "execution_output": str(root),
        "implementation": {"source_commit": "a" * 40},
    }
    proposal = {"budgets": {"maximum_wall_seconds_total": 43200}}
    initializer = {
        "training": {
            "checkpoint_path": str(tmp_path / "original/training/checkpoints/step"),
            "model_state_hash": model_state_hash(context.model),
        },
        "consumed_files_sha256": {},
    }
    monkeypatch.setattr(
        worker, "load_inputs", lambda p: (protocol, proposal, initializer)
    )
    monkeypatch.setattr(
        worker,
        "prepare_objective",
        lambda *args: (data, {"objective": data.document()}),
    )
    monkeypatch.setattr(
        worker, "LBFGSConfig", lambda: LBFGSConfig(max_iterations=1, max_evaluations=40)
    )
    originals = [
        (worker.EnergyTraining, "train"),
        (worker.ScikitFemBackend, "solve_many"),
        (worker.ScikitFemBackend, "solve_from"),
        (torch.optim.Adam, "step"),
    ]
    saved = [getattr(owner, name) for owner, name in originals]
    for (owner, name), value in zip(originals, saved, strict=True):
        monkeypatch.setattr(owner, name, value)
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    result = worker.run_pilot(path, root / "pilot", deadline_utc=deadline)
    assert result["optimizer_iterations"] <= 2 and result["new_fem_solves"] == 0
    assert (root / "pilot/assessment.json").is_file() and (
        root / "pilot/quadrature.safetensors"
    ).is_file()
    assert all(
        getattr(owner, name) == value
        for (owner, name), value in zip(originals, saved, strict=True)
    )
    assert before == (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
    )


def test_checked_initializer_import_keeps_original_normalization(
    context, tmp_path, monkeypatch
):
    original_values = context._values
    values = {
        name: {**original_values(name), "ux": 0.1, "uy": 0.2}
        for name in ("fit-001", "fit-002")
    }
    monkeypatch.setattr(context, "_values", lambda name: values[name])
    prepared = context.prepare()
    weights = tmp_path / "weights.safetensors"
    save_file(context.model.state_dict(), str(weights))
    initializer = {
        **{
            key: prepared[key]
            for key in ("normalization", "mesh_sha256", "dataset_snapshot_sha256")
        },
        "parent_protocol_path": str(tmp_path / "parent/protocol.json"),
        "weights_path": str(weights),
        "weights_sha256": inputs.sha(weights),
        "training": {"model_state_hash": model_state_hash(context.model)},
    }
    proposal = {
        "pilot_cases": {"ids": ["fit-001", "fit-002"], "values": values},
        "optimizer": {"gradient_point_chunk": 3},
    }
    monkeypatch.setattr(inputs, "build_plan", lambda *args: context.plan)
    monkeypatch.setattr(
        inputs, "resolve_configured_adapter", lambda *args: context.adapter
    )
    monkeypatch.setattr(inputs, "EnergyTraining", lambda *args: context)
    data, evidence = inputs.prepare_objective(
        {"implementation": {"source_commit": "a" * 40}},
        proposal,
        initializer,
        tmp_path / "out",
    )
    assert data.normalization == prepared["normalization"]
    assert evidence["initializer_weights_sha256"] == initializer["weights_sha256"]
    bad = copy.deepcopy(initializer)
    bad["normalization"]["mean"][0] += 1
    with pytest.raises(ValueError):
        inputs.prepare_objective(
            {"implementation": {"source_commit": "a" * 40}},
            proposal,
            bad,
            tmp_path / "other",
        )
