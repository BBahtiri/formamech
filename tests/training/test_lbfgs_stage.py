"""Actual line-search, checkpoint and hard-budget behavior on a small problem."""

import json
import time
from dataclasses import replace

import pytest
import torch

from piginot.training.lbfgs_stage import LBFGSConfig, train_lbfgs_stage


class Rosenbrock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.point = torch.nn.Parameter(torch.tensor([-1.2, 1.0], dtype=torch.float64))


def objective(counter):
    def evaluate(model):
        counter.append(1)
        x, y = model.point
        value = 100 * (y - x * x) ** 2 + (1 - x) ** 2
        value.backward()
        return float(value.detach())

    return evaluate


def config(**changes):
    return replace(
        LBFGSConfig(
            max_iterations=8,
            max_evaluations=100,
            history_size=5,
            checkpoint_interval=2,
            tolerance_grad=1e-20,
            tolerance_change=1e-20,
        ),
        **changes,
    )


def test_deterministic_resume_restores_lbfgs_history_and_cached_gradient(tmp_path):
    first, second = Rosenbrock(), Rosenbrock()
    calls, resumed_calls = [], []
    whole = train_lbfgs_stage(
        tmp_path / "whole",
        model=first,
        objective=objective(calls),
        binding="a" * 64,
        config=config(),
    )
    half = train_lbfgs_stage(
        tmp_path / "split",
        model=second,
        objective=objective(resumed_calls),
        binding="a" * 64,
        config=config(),
        target_iterations=4,
    )
    # A different incoming model must be replaced by the bound stage checkpoint.
    with torch.no_grad():
        second.point.add_(7)
    resumed = train_lbfgs_stage(
        tmp_path / "split",
        model=second,
        objective=objective(resumed_calls),
        binding="a" * 64,
        config=config(),
        resume_from=half.checkpoint,
    )
    assert whole.status == resumed.status == "COMPLETE"
    assert whole.iterations == resumed.iterations == 8
    assert whole.evaluations == resumed.evaluations == len(calls) == len(resumed_calls)
    assert whole.closure_calls == resumed.closure_calls
    assert torch.equal(first.point, second.point)
    assert (whole.checkpoint / "state.safetensors").read_bytes() == (
        resumed.checkpoint / "state.safetensors"
    ).read_bytes()
    assert (whole.checkpoint / "checkpoint.json").read_bytes() == (
        resumed.checkpoint / "checkpoint.json"
    ).read_bytes()
    assert whole.model_hash == resumed.model_hash
    assert whole.closure_calls > whole.evaluations  # exact accepted-point reuse
    assert whole.loss < 24.2


def test_hard_evaluation_limit_rolls_back_unaccepted_trial_and_keeps_debits(tmp_path):
    model = Rosenbrock()
    initial = model.point.detach().clone()
    calls = []
    result = train_lbfgs_stage(
        tmp_path / "limited",
        model=model,
        objective=objective(calls),
        binding="a" * 64,
        config=config(max_evaluations=2),
    )
    assert result.status == "EVALUATION_LIMIT"
    assert result.evaluations == len(calls) == 2
    assert result.iterations == 0
    assert torch.equal(initial, model.point)
    record = json.loads((result.checkpoint / "checkpoint.json").read_text())
    assert record["evaluations"] == 2 and record["optimizer_kind"] == "LBFGS"


def test_wall_deadline_refuses_before_any_objective_evaluation(tmp_path):
    calls = []
    result = train_lbfgs_stage(
        tmp_path / "wall",
        model=Rosenbrock(),
        objective=objective(calls),
        binding="a" * 64,
        config=config(),
        deadline=time.monotonic() - 1,
    )
    assert result.status == "TIME_LIMIT"
    assert result.evaluations == 0 and not calls


@pytest.mark.parametrize("failure", ["nonfinite", "exception", "missing_gradient"])
def test_invalid_evaluation_stops_without_accepting_weights(tmp_path, failure):
    model = Rosenbrock()
    initial = model.point.detach().clone()

    def evaluate(model):
        if failure == "exception":
            raise RuntimeError("deliberate objective failure")
        if failure == "nonfinite":
            return float("nan")
        return 1.0

    result = train_lbfgs_stage(
        tmp_path / failure,
        model=model,
        objective=evaluate,
        binding="a" * 64,
        config=config(),
    )
    assert result.status == "ERROR" and result.evaluations == 1
    assert torch.equal(initial, model.point)
    assert result.checkpoint.is_dir()


def test_unverified_line_search_acceptance_is_rejected(tmp_path, monkeypatch):
    model = Rosenbrock()
    initial = model.point.detach().clone()

    def wrong_step(self, closure):
        closure()
        with torch.no_grad():
            model.point.add_(0.25)

    monkeypatch.setattr(torch.optim.LBFGS, "step", wrong_step)
    result = train_lbfgs_stage(
        tmp_path / "line",
        model=model,
        objective=objective([]),
        binding="a" * 64,
        config=config(),
    )
    assert result.status == "LINE_SEARCH_FAILURE"
    assert torch.equal(initial, model.point)


def test_inadmissible_wolfe_trial_is_not_published_as_accepted(tmp_path):
    from piginot.training.lbfgs_stage import GradientEvaluation

    model = Rosenbrock()
    initial = model.point.detach().clone()
    evaluate = objective([])

    def guarded(model):
        value = evaluate(model)
        return GradientEvaluation(value, admissible=torch.equal(model.point, initial))

    result = train_lbfgs_stage(
        tmp_path / "admissible",
        model=model,
        objective=guarded,
        binding="a" * 64,
        config=config(),
    )
    assert result.status == "LINE_SEARCH_FAILURE"
    assert torch.equal(initial, model.point)


@pytest.mark.parametrize("change", ["binding", "config", "payload", "schema"])
def test_foreign_or_corrupt_checkpoint_refuses_before_restoring_model(tmp_path, change):
    root = tmp_path / "original"
    model = Rosenbrock()
    saved = train_lbfgs_stage(
        root,
        model=model,
        objective=objective([]),
        binding="a" * 64,
        config=config(),
        target_iterations=2,
    )
    binding = "b" * 64 if change == "binding" else "a" * 64
    cfg = config(history_size=4) if change == "config" else config()
    if change == "payload":
        p = saved.checkpoint / "state.safetensors"
        p.write_bytes(p.read_bytes() + b"x")
    if change == "schema":
        p = saved.checkpoint / "checkpoint.json"
        d = json.loads(p.read_text())
        d["optimizer_kind"] = "Adam"
        p.write_text(json.dumps(d))
    incoming = Rosenbrock()
    initial = incoming.point.detach().clone()
    with pytest.raises(ValueError):
        train_lbfgs_stage(
            root,
            model=incoming,
            objective=objective([]),
            binding=binding,
            config=cfg,
            resume_from=saved.checkpoint,
        )
    assert torch.equal(initial, incoming.point)


def test_existing_output_requires_explicit_resume(tmp_path):
    root = tmp_path / "existing"
    train_lbfgs_stage(
        root,
        model=Rosenbrock(),
        objective=objective([]),
        binding="a" * 64,
        config=config(),
        target_iterations=2,
    )
    with pytest.raises(FileExistsError):
        train_lbfgs_stage(
            root,
            model=Rosenbrock(),
            objective=objective([]),
            binding="a" * 64,
            config=config(),
        )


def test_float32_model_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        train_lbfgs_stage(
            tmp_path / "dtype",
            model=Rosenbrock().float(),
            objective=objective([]),
            binding="a" * 64,
            config=config(),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_iterations", True),
        ("max_iterations", 0),
        ("max_evaluations", 0),
        ("history_size", 0),
        ("checkpoint_interval", 0),
        ("learning_rate", float("nan")),
        ("tolerance_grad", -1),
        ("tolerance_change", 0),
    ],
)
def test_stage_configuration_refuses_invalid_budgets(field, value):
    with pytest.raises(ValueError):
        replace(config(), **{field: value})
