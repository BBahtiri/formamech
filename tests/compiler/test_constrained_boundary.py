"""Actual constitutive gradients through the two-pass FIT-only adapter."""

import copy
import time

import pytest
import torch

from piginot.compiler.constrained_boundary import BoundaryConstraintEvaluator
from piginot.objectives.energy_constraint import PenaltyState, constraint_merit
from piginot.training.constrained_objective import ConstrainedObjective
from piginot.training.constraint_budget import ConstraintBudget, ConstraintLimits
from tests.compiler.test_boundary_objective import context as context
from tests.compiler.test_boundary_objective import prepared


@pytest.mark.parametrize("reference_ratio", [0.8, 2.0])
def test_global_merit_gradient_matches_actual_operator_finite_difference(
    context, reference_ratio
):
    data = prepared(context)
    model = data.new_model()
    meter = ConstraintBudget(ConstraintLimits(), deadline_utc=time.time() + 120)
    evaluate = BoundaryConstraintEvaluator(data, meter)
    initial = evaluate(model, energy_weight=1, gradients=False)
    reference = initial.bulk_energy * reference_ratio
    objective = ConstrainedObjective(evaluate, reference, PenaltyState(), meter)
    objective(model)
    parameter = list(model.parameters())[-1]
    actual = parameter.grad.flatten()[0].item()
    stencil = []
    for sign in (1, -1):
        candidate = copy.deepcopy(model)
        with torch.no_grad():
            list(candidate.parameters())[-1].flatten()[0] += sign * 1e-6
        terms = evaluate(candidate, energy_weight=1, gradients=False)
        stencil.append(constraint_merit(terms, reference, PenaltyState()).value)
    assert actual == pytest.approx((stencil[0] - stencil[1]) / 2e-6, rel=2e-5, abs=2e-5)
    assert (
        meter.usage.gradient_evaluations == 1 and meter.usage.forward_evaluations == 2
    )


def test_independent_validation_is_charged_and_does_not_update_parameters(context):
    data = prepared(context)
    model = data.new_model()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    meter = ConstraintBudget(
        ConstraintLimits(max_validations=1), deadline_utc=time.time() + 120
    )
    out = BoundaryConstraintEvaluator(data, meter).validate(model)
    assert out.admissible and meter.usage.validation_evaluations == 1
    assert out.value == pytest.approx(out.bulk_energy + out.guard)
    assert all(
        torch.equal(value, before[name]) for name, value in model.state_dict().items()
    )
    assert all(p.grad is None for p in model.parameters())


def test_zero_energy_coefficient_does_not_suppress_invalid_determinant_guard(
    context, monkeypatch
):
    data = prepared(context)
    model = data.new_model()
    parameter = next(model.parameters())

    def invalid_gradient(model, case, points, gradients):
        value = next(model.parameters()).flatten()[0]
        zero = 0 * value
        matrix = torch.stack((value - 2, zero, zero, zero)).reshape(2, 2)
        return matrix.expand(len(points), 2, 2)

    monkeypatch.setattr(data, "_gradient", invalid_gradient)
    a = data.evaluate(model, traction_weight=1, energy_weight=0, gradients=True)
    first = parameter.grad.clone()
    b = data.evaluate(model, traction_weight=1, energy_weight=3, gradients=True)
    assert not a.admissible and a.guard > 0 and b.guard == pytest.approx(a.guard)
    assert bool(torch.isfinite(first).all()) and float(first.abs().max()) > 0
    torch.testing.assert_close(parameter.grad, first, rtol=0, atol=0)


def test_tiny_physics_workflow_connects_adapter_outer_stage_and_safe_checkpoint(
    context, tmp_path
):
    from piginot.training.constrained_stage import (
        ConstrainedConfig,
        run_constrained_stage,
    )
    from piginot.training.lbfgs_stage import LBFGSConfig

    data = prepared(context)
    model = data.new_model()
    reference = data.evaluate(model, traction_weight=1, gradients=False).bulk_energy
    budget = ConstraintBudget(
        ConstraintLimits(max_iterations=1, max_gradients=20, max_forwards=40),
        deadline_utc=time.time() + 120,
    )
    result = run_constrained_stage(
        tmp_path / "physics",
        model=model,
        evaluator=BoundaryConstraintEvaluator(data, budget),
        budget=budget,
        config=ConstrainedConfig(
            reference,
            outer_blocks=1,
            inner=LBFGSConfig(max_iterations=1, max_evaluations=20),
        ),
        binding="a" * 64,
    )
    assert result.status == "COMPLETE" and result.completed_blocks == 1
    assert result.terms.admissible and result.checkpoint.is_file()
    assert result.usage.iterations <= 1
    assert result.usage.forward_evaluations == 2 * result.usage.gradient_evaluations
