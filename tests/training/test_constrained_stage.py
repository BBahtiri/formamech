"""Known constraint problems, phase transitions and exact boundary resume."""

import hashlib
import json
import math
import time
from dataclasses import asdict, replace

import pytest
import torch

from piginot.contracts import canonical_json_bytes
from piginot.objectives.energy_constraint import ConstraintTerms
from piginot.training.checkpoint import model_state_hash
from piginot.training.constrained_stage import ConstrainedConfig, run_constrained_stage
from piginot.training.constraint_budget import ConstraintBudget, ConstraintLimits
from piginot.training.constraint_records import block_binding, read_record
from piginot.training.lbfgs_stage import LBFGSConfig


class Scalar(torch.nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.point = torch.nn.Parameter(torch.tensor([value], dtype=torch.float64))


def evaluator(target=2.0, *, infeasible=False, calls=None):
    def evaluate(model, *, energy_weight, gradients):
        if calls is not None:
            calls.append((energy_weight, gradients))
        x = model.point
        energy = x.square().sum() if not infeasible else 2 + 0 * x.sum()
        traction = (x - target).square().sum()
        if gradients:
            model.zero_grad(set_to_none=True)
            (energy_weight * energy + traction).backward()
        return ConstraintTerms(float(energy.detach()), float(traction.detach()))

    return evaluate


def config(**changes):
    return replace(
        ConstrainedConfig(
            reference_energy=1.0,
            inner=LBFGSConfig(
                max_iterations=10,
                max_evaluations=100,
                history_size=5,
                tolerance_grad=1e-12,
                tolerance_change=1e-14,
            ),
        ),
        **changes,
    )


def meter(deadline=None, **limits):
    return ConstraintBudget(
        ConstraintLimits(**limits), deadline_utc=deadline or time.time() + 120
    )


def test_known_feasible_interior_solution(tmp_path):
    model = Scalar()
    budget = meter()
    result = run_constrained_stage(
        tmp_path / "feasible",
        model=model,
        evaluator=evaluator(0.25),
        budget=budget,
        config=config(),
        binding="a" * 64,
    )
    assert result.status == "COMPLETE" and result.completed_blocks == 5
    assert result.train_feasible and model.point.item() == pytest.approx(0.25, abs=1e-7)
    assert result.usage.forward_evaluations == 2 * result.usage.gradient_evaluations


def test_active_constraint_approaches_known_boundary_optimum(tmp_path):
    model = Scalar()
    result = run_constrained_stage(
        tmp_path / "active",
        model=model,
        evaluator=evaluator(),
        budget=meter(),
        config=config(),
        binding="a" * 64,
    )
    assert result.status == "COMPLETE"
    assert model.point.item() == pytest.approx(1.005**0.5, abs=2e-5)
    assert result.state.multiplier > 0
    assert result.terms.bulk_energy <= 1.005 + 5e-5


def test_infeasible_problem_cannot_be_reported_as_feasible(tmp_path):
    model = Scalar()
    result = run_constrained_stage(
        tmp_path / "infeasible",
        model=model,
        evaluator=evaluator(0.0, infeasible=True),
        budget=meter(),
        config=config(),
        binding="a" * 64,
    )
    assert result.status == "COMPLETE" and not result.train_feasible
    assert result.terms.bulk_energy == 2.0 and result.state.penalty == 1e6


@pytest.mark.parametrize("limit", [1.005, 1.003])
def test_completed_block_resume_is_bitwise_identical_and_keeps_all_debits(
    tmp_path, limit
):
    deadline = time.time() + 120
    first, second = Scalar(), Scalar()
    declared = config(energy_ratio_limit=limit)
    whole = run_constrained_stage(
        tmp_path / "whole",
        model=first,
        evaluator=evaluator(),
        budget=meter(deadline),
        config=declared,
        binding="a" * 64,
    )
    part = run_constrained_stage(
        tmp_path / "split",
        model=second,
        evaluator=evaluator(),
        budget=meter(deadline),
        config=declared,
        binding="a" * 64,
        target_blocks=2,
    )
    assert part.status == "CHECKPOINTED" and part.completed_blocks == 2
    incoming = Scalar(999.0)
    resumed = run_constrained_stage(
        tmp_path / "split",
        model=incoming,
        evaluator=evaluator(),
        budget=meter(deadline),
        config=declared,
        binding="a" * 64,
        resume_from=part.checkpoint,
    )
    assert whole.status == resumed.status == "COMPLETE"
    assert whole.usage == resumed.usage and whole.state == resumed.state
    assert torch.equal(first.point, incoming.point)
    assert whole.checkpoint.read_bytes() == resumed.checkpoint.read_bytes()


def test_default_study_and_checkpoint_binding_keep_the_legacy_config(tmp_path):
    root, model, budget = tmp_path / "legacy", Scalar(), meter()
    declared = config()
    initial_hash = model_state_hash(model)
    outcome = run_constrained_stage(
        root,
        model=model,
        evaluator=evaluator(),
        budget=budget,
        config=declared,
        binding="a" * 64,
        target_blocks=1,
    )
    study = read_record(root / "study.json")
    assert study == {
        "schema": "piginot.constrained-study-v1",
        "binding": "a" * 64,
        "config": {
            "reference_energy": 1.0,
            "outer_blocks": 5,
            "inner": asdict(declared.inner),
        },
        "limits": asdict(budget.limits),
        "deadline_utc": budget.deadline_utc,
        "inner_stop_policy": "advance_on_evaluated_admissible_stall",
        "initial_model_hash": initial_hash,
        "initial_usage": {
            "iterations": 0,
            "gradient_evaluations": 0,
            "forward_evaluations": 0,
            "validation_evaluations": 0,
        },
    }
    outer = read_record(outcome.checkpoint)
    metadata = json.loads(
        (root / outer["inner_checkpoint"] / "checkpoint.json").read_text()
    )
    from piginot.objectives.energy_constraint import PenaltyState

    assert metadata["binding"] == block_binding(study, 1, PenaltyState())


def test_tighter_energy_boundary_is_bound_and_used_for_training(tmp_path):
    root, model = tmp_path / "tighter", Scalar()
    outcome = run_constrained_stage(
        root,
        model=model,
        evaluator=evaluator(),
        budget=meter(),
        config=config(energy_ratio_limit=1.003),
        binding="a" * 64,
    )
    assert outcome.status == "COMPLETE"
    assert model.point.item() == pytest.approx(1.003**0.5, abs=2e-5)
    assert outcome.terms.bulk_energy <= 1.003 + 5e-5
    assert read_record(root / "study.json")["config"]["energy_ratio_limit"] == 1.003


def test_training_feasibility_and_multiplier_use_the_declared_limit(tmp_path):
    def fixed_energy(model, *, energy_weight, gradients):
        traction = model.point.square().sum()
        if gradients:
            model.zero_grad(set_to_none=True)
            traction.backward()
        return ConstraintTerms(1.004, float(traction.detach()))

    results = []
    for limit in (1.005, 1.003):
        results.append(
            run_constrained_stage(
                tmp_path / str(limit),
                model=Scalar(),
                evaluator=fixed_energy,
                budget=meter(),
                config=config(outer_blocks=1, energy_ratio_limit=limit),
                binding="a" * 64,
            )
        )
    loose, tight = results
    assert loose.status == tight.status == "COMPLETE"
    assert loose.train_feasible and not tight.train_feasible
    assert loose.state.multiplier == 0 and tight.state.multiplier > 0


@pytest.mark.parametrize("old,new", [(1.005, 1.003), (1.003, 1.005), (1.003, 1.002)])
def test_a_changed_energy_limit_cannot_resume_an_old_objective(tmp_path, old, new):
    root, deadline = tmp_path / "bound", time.time() + 120
    saved = run_constrained_stage(
        root,
        model=Scalar(),
        evaluator=evaluator(),
        budget=meter(deadline),
        config=config(energy_ratio_limit=old),
        binding="a" * 64,
        target_blocks=1,
    )
    incoming, calls = Scalar(9.0), []
    before = incoming.point.detach().clone()
    rng = torch.get_rng_state().clone()
    budget = meter(deadline)
    with pytest.raises(ValueError, match="config"):
        run_constrained_stage(
            root,
            model=incoming,
            evaluator=evaluator(calls=calls),
            budget=budget,
            config=config(energy_ratio_limit=new),
            binding="a" * 64,
            resume_from=saved.checkpoint,
        )
    assert calls == [] and budget.usage.gradient_evaluations == 0
    assert torch.equal(incoming.point, before) and torch.equal(
        torch.get_rng_state(), rng
    )


@pytest.mark.parametrize("bad", [0, -1.0, math.inf, math.nan, True, None, "1.003"])
def test_invalid_config_energy_limit_refuses(bad):
    with pytest.raises(ValueError):
        config(energy_ratio_limit=bad)


def test_each_new_multiplier_gets_fresh_optimizer_and_gradient_cache(
    tmp_path, monkeypatch
):
    import piginot.training.constrained_stage as stage

    original = stage.train_lbfgs_stage
    observed = []

    def capture(root, **kwargs):
        assert kwargs.get("resume_from") is None
        observed.append((kwargs["binding"], kwargs["objective"].state))
        return original(root, **kwargs)

    monkeypatch.setattr(stage, "train_lbfgs_stage", capture)
    result = stage.run_constrained_stage(
        tmp_path / "reset",
        model=Scalar(),
        evaluator=evaluator(),
        budget=meter(),
        config=config(),
        binding="a" * 64,
    )
    assert result.completed_blocks == 5 and len({row[0] for row in observed}) == 5
    assert len({row[1] for row in observed}) > 1
    assert result.usage.gradient_evaluations >= 5


def test_global_caps_stop_without_resetting_a_new_block_allowance(tmp_path):
    calls = []
    result = run_constrained_stage(
        tmp_path / "limited",
        model=Scalar(),
        evaluator=evaluator(calls=calls),
        budget=meter(max_gradients=2, max_forwards=4),
        config=config(),
        binding="a" * 64,
    )
    assert result.status == "EVALUATION_LIMIT"
    assert (
        result.usage.gradient_evaluations == 2 and result.usage.forward_evaluations == 4
    )
    assert len(calls) == 4


@pytest.mark.parametrize(
    "change", ["binding", "deadline", "state", "counters", "path", "payload"]
)
def test_invalid_outer_checkpoint_refuses_before_model_or_rng_changes(tmp_path, change):
    root = tmp_path / "saved"
    deadline = time.time() + 120
    saved = run_constrained_stage(
        root,
        model=Scalar(),
        evaluator=evaluator(),
        budget=meter(deadline),
        config=config(),
        binding="a" * 64,
        target_blocks=1,
    )
    binding = "b" * 64 if change == "binding" else "a" * 64
    if change in ("state", "counters", "path"):
        envelope = json.loads(saved.checkpoint.read_text())
        record = envelope["record"]
        if change == "state":
            record["state_after"]["multiplier"] += 1
        if change == "counters":
            record["usage_after"]["gradient_evaluations"] = -1
        if change == "path":
            record["inner_checkpoint"] = "../outside"
        envelope["sha256"] = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
        saved.checkpoint.write_text(json.dumps(envelope))
    if change == "payload":
        payload = next((root / "block-000001").glob("checkpoints/*/state.safetensors"))
        payload.write_bytes(payload.read_bytes() + b"corrupt")
    incoming = Scalar(8.0)
    before = incoming.point.detach().clone()
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        run_constrained_stage(
            root,
            model=incoming,
            evaluator=evaluator(),
            budget=meter(deadline + 1 if change == "deadline" else deadline),
            config=config(),
            binding=binding,
            resume_from=saved.checkpoint,
        )
    assert torch.equal(incoming.point, before) and torch.equal(
        torch.get_rng_state(), rng
    )


def test_partial_block_failure_has_no_automatic_or_boundary_resume(tmp_path):
    root = tmp_path / "failed"

    def fail(*args, **kwargs):
        raise ValueError("deliberate failure")

    result = run_constrained_stage(
        root,
        model=Scalar(),
        evaluator=fail,
        budget=meter(),
        config=config(),
        binding="a" * 64,
    )
    assert result.status == "ERROR" and result.usage.gradient_evaluations == 1
    assert result.checkpoint is None and (root / "incomplete.json").exists()
    with pytest.raises(FileExistsError):
        run_constrained_stage(
            root,
            model=Scalar(),
            evaluator=evaluator(),
            budget=meter(),
            config=config(),
            binding="a" * 64,
        )
