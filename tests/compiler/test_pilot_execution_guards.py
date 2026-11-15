"""A deterministic closure and the original coordinator deadline are mandatory."""

import json
import time

import pytest
import torch

from piginot.training.lbfgs_stage import train_lbfgs_stage
from tests.training.test_lbfgs_stage import Rosenbrock, config


def test_random_closures_are_rejected_and_restore_generator(tmp_path):
    before = torch.get_rng_state().clone()

    def random_objective(model):
        value = model.point.square().sum() + torch.rand((), dtype=torch.float64)
        value.backward()
        return float(value.detach())

    result = train_lbfgs_stage(
        tmp_path / "random",
        model=Rosenbrock(),
        objective=random_objective,
        binding="a" * 64,
        config=config(),
    )
    assert result.status == "ERROR" and result.evaluations == 1
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("change", ["none", "deadline", "protocol", "status", "future"])
def test_worker_requires_original_wall_clock_lease(tmp_path, change):
    from piginot.compiler.boundary_pilot import execution_deadline

    output = tmp_path / "pilot"
    now = time.time()
    end = now + 43200
    record = {
        "schema": "piginot.boundary-pilot-job-v1",
        "status": "RUNNING",
        "protocol_sha256": "a" * 64,
        "started_epoch": now,
        "deadline_epoch": end,
    }
    if change == "deadline":
        record["deadline_epoch"] = end + 1
    if change == "protocol":
        record["protocol_sha256"] = "b" * 64
    if change == "status":
        record["status"] = "COMPLETE"
    if change == "future":
        end = now + 86400
        record["deadline_epoch"] = end
    if change != "none":
        (tmp_path / "job.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        execution_deadline(output, "a" * 64, end, 43200)


def test_worker_keeps_elapsed_preparation_time_in_original_budget(
    tmp_path, monkeypatch
):
    from piginot.compiler.boundary_pilot import execution_deadline

    (tmp_path / "job.json").write_text(
        json.dumps(
            {
                "schema": "piginot.boundary-pilot-job-v1",
                "status": "RUNNING",
                "protocol_sha256": "a" * 64,
                "started_epoch": 100.0,
                "deadline_epoch": 43300.0,
            }
        )
    )
    monkeypatch.setattr(time, "time", lambda: 115.0)
    monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
    assert execution_deadline(tmp_path / "pilot", "a" * 64, 43300.0, 43200) == 44185.0
