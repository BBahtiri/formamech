"""Interruptions, counter preservation and absolute expiry across outer blocks."""

import json
import time

import pytest

from piginot.training.constrained_stage import run_constrained_stage
from piginot.training.constraint_budget import ConstraintUsage
from piginot.training.lbfgs_stage import train_lbfgs_stage
from tests.training.test_constrained_stage import Scalar, config, evaluator, meter
from tests.training.test_lbfgs_stage import Rosenbrock, objective
from tests.training.test_lbfgs_stage import config as inner_config


def test_interrupt_keeps_prior_block_and_current_reserved_cost(tmp_path):
    root = tmp_path / "interrupt"
    deadline = time.time() + 120
    first = run_constrained_stage(
        root,
        model=Scalar(),
        evaluator=evaluator(),
        budget=meter(deadline),
        config=config(),
        binding="a" * 64,
        target_blocks=1,
    )

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("test interruption")

    with pytest.raises(KeyboardInterrupt):
        run_constrained_stage(
            root,
            model=Scalar(),
            evaluator=interrupt,
            budget=meter(deadline),
            config=config(),
            binding="a" * 64,
            resume_from=first.checkpoint,
        )
    receipt = json.loads((root / "incomplete.json").read_text())["record"]
    assert receipt["status"] == "INTERRUPTED" and receipt["completed_blocks"] == 1
    assert (
        receipt["usage"]["gradient_evaluations"] == first.usage.gradient_evaluations + 1
    )
    assert (
        receipt["usage"]["forward_evaluations"] == first.usage.forward_evaluations + 2
    )
    with pytest.raises(ValueError, match="completed outer-block"):
        run_constrained_stage(
            root,
            model=Scalar(),
            evaluator=evaluator(),
            budget=meter(deadline),
            config=config(),
            binding="a" * 64,
            resume_from=first.checkpoint,
        )


def test_global_iteration_cap_is_not_renewed_by_outer_transition(tmp_path):
    result = run_constrained_stage(
        tmp_path / "iteration-cap",
        model=Scalar(),
        evaluator=evaluator(),
        budget=meter(max_iterations=1),
        config=config(),
        binding="a" * 64,
    )
    assert result.status == "EVALUATION_LIMIT" and result.usage.iterations == 1
    assert result.completed_blocks == 1


def test_expired_study_stops_without_an_objective_evaluation(tmp_path, monkeypatch):
    budget = meter()
    calls = []
    monkeypatch.setattr(time, "time", lambda: budget.deadline_utc + 1)
    result = run_constrained_stage(
        tmp_path / "expired",
        model=Scalar(),
        evaluator=evaluator(calls=calls),
        budget=budget,
        config=config(),
        binding="a" * 64,
    )
    assert result.status == "TIME_LIMIT" and not calls
    assert result.usage.gradient_evaluations == 0


def test_restoring_older_usage_cannot_erase_spent_cost():
    budget = meter()
    budget.reserve_training()
    before = budget.usage
    with pytest.raises(ValueError):
        budget.restore_usage(ConstraintUsage())
    assert budget.usage == before


def test_clock_jump_is_checked_before_an_accepted_gradient_cache_hit(
    tmp_path, monkeypatch
):
    clock = {"now": 100.0}
    accepted = []
    monkeypatch.setattr(time, "time", lambda: clock["now"])

    def progress(event):
        if event["event"] == "ACCEPTED_ITERATION":
            accepted.append(event)
            clock["now"] = 102.0

    result = train_lbfgs_stage(
        tmp_path / "cached",
        model=Rosenbrock(),
        objective=objective([]),
        binding="a" * 64,
        config=inner_config(),
        deadline_utc=101.0,
        progress=progress,
    )
    assert result.status == "TIME_LIMIT" and result.iterations == 1
    assert result.evaluations == accepted[0]["evaluations"]


def test_expiry_after_final_acceptance_is_not_reported_complete(tmp_path, monkeypatch):
    from dataclasses import replace

    clock = {"now": 100.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])

    def progress(event):
        if event["event"] == "ACCEPTED_ITERATION":
            clock["now"] = 102.0

    result = train_lbfgs_stage(
        tmp_path / "last",
        model=Rosenbrock(),
        objective=objective([]),
        binding="a" * 64,
        config=replace(inner_config(), max_iterations=1),
        deadline_utc=101.0,
        progress=progress,
    )
    assert result.status == "TIME_LIMIT" and result.iterations == 1


def test_outer_completion_checks_clock_after_inner_checkpoint_write(
    tmp_path, monkeypatch
):
    import piginot.training.constrained_stage as stage

    clock = {"now": 100.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    original = stage.train_lbfgs_stage

    def checkpoint_finishes_late(*args, **kwargs):
        result = original(*args, **kwargs)
        clock["now"] = 102.0
        return result

    monkeypatch.setattr(stage, "train_lbfgs_stage", checkpoint_finishes_late)
    result = stage.run_constrained_stage(
        tmp_path / "late-checkpoint",
        model=Scalar(),
        evaluator=evaluator(0.25),
        budget=meter(101.0),
        config=config(outer_blocks=1),
        binding="a" * 64,
    )
    assert result.status == "TIME_LIMIT" and result.completed_blocks == 0
    assert result.usage.gradient_evaluations > 0
