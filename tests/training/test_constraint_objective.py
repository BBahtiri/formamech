"""Two-pass gradients use the global energy coefficient and unit guard."""

import copy
import math
import time

import pytest
import torch

from piginot.objectives.energy_constraint import (
    ConstraintTerms,
    PenaltyState,
    constraint_merit,
)
from piginot.training.constrained_objective import ConstrainedObjective
from piginot.training.constraint_budget import (
    ConstraintBudget,
    ConstraintLimits,
    ConstraintUsage,
)
from piginot.training.lbfgs_stage import EvaluationLimit, StageDeadline


class Point(torch.nn.Module):
    def __init__(self, values=(1.2, -0.3)):
        super().__init__()
        self.point = torch.nn.Parameter(torch.tensor(values, dtype=torch.float64))


def evaluator(calls, chunk=1):
    def evaluate(model, *, energy_weight, gradients):
        calls.append((energy_weight, gradients))
        energy, residual, guard = 0.0, 0.0, 0.0
        if gradients:
            model.zero_grad(set_to_none=True)
        for part in model.point.split(chunk):
            e = part.square().sum()
            t = (part - 0.5).square().sum()
            g = 2 * torch.relu(-part).square().sum()
            energy += float(e.detach())
            residual += float(t.detach())
            guard += float(g.detach())
            if gradients:
                (energy_weight * e + t + g).backward()
        return ConstraintTerms(energy, residual, guard)

    return evaluate


def budget(**limits):
    return ConstraintBudget(ConstraintLimits(**limits), deadline_utc=time.time() + 60)


@pytest.mark.parametrize("reference", [0.5, 10.0])
@pytest.mark.parametrize("chunk", [1, 2])
@pytest.mark.parametrize("limit", [1.005, 1.003])
def test_two_pass_gradient_matches_the_global_merit(reference, chunk, limit):
    model, calls = Point(), []
    state = PenaltyState(multiplier=0.0)
    meter = budget()
    objective = ConstrainedObjective(
        evaluator(calls, chunk), reference, state, meter, energy_ratio_limit=limit
    )
    observed = objective(model)
    actual = model.point.grad.clone()
    expected = []
    for index in range(2):
        values = []
        for sign in (1, -1):
            candidate = copy.deepcopy(model)
            with torch.no_grad():
                candidate.point[index] += sign * 1e-6
            terms = evaluator([], chunk)(candidate, energy_weight=1, gradients=False)
            values.append(
                constraint_merit(
                    terms, reference, state, energy_ratio_limit=limit
                ).value
            )
        expected.append((values[0] - values[1]) / 2e-6)
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=torch.float64), rtol=1e-7, atol=1e-6
    )
    assert observed.value == pytest.approx(
        constraint_merit(
            evaluator([], chunk)(model, energy_weight=1, gradients=False),
            reference,
            state,
            energy_ratio_limit=limit,
        ).value
    )
    assert len(calls) == 2 and calls[0] == (1.0, False) and calls[1][1] is True
    assert meter.usage.gradient_evaluations == 1
    assert meter.usage.forward_evaluations == 2


def test_inactive_constraint_keeps_the_guard_gradient():
    model = Point((-1.0,))
    observed = ConstrainedObjective(evaluator([]), 10.0, PenaltyState(), budget())(
        model
    )
    assert model.point.grad.item() == pytest.approx(-7.0)
    assert observed.value == pytest.approx(4.25)


def test_partial_failure_keeps_the_reserved_cost():
    meter = budget()
    calls = []

    def failure(model, *, energy_weight, gradients):
        calls.append(gradients)
        raise ValueError("synthetic value-pass failure")

    with pytest.raises(ValueError, match="synthetic"):
        ConstrainedObjective(failure, 1.0, PenaltyState(), meter)(Point())
    assert calls == [False]
    assert meter.usage == ConstraintUsage(gradient_evaluations=1, forward_evaluations=2)


def test_reservation_is_atomic_at_the_gradient_or_forward_cap():
    for limits in ({"max_gradients": 1}, {"max_forwards": 3}):
        meter = budget(**limits)
        meter.reserve_training()
        before = meter.usage
        with pytest.raises(EvaluationLimit):
            meter.reserve_training()
        assert meter.usage == before


def test_validation_and_iteration_counters_are_bounded():
    meter = budget(max_validations=1, max_iterations=1)
    meter.reserve_training()
    meter.accept_iteration()
    meter.reserve_validation()
    with pytest.raises(EvaluationLimit):
        meter.reserve_validation()
    with pytest.raises(EvaluationLimit):
        meter.accept_iteration()


def test_absolute_clock_expiry_spends_nothing(monkeypatch):
    meter = budget()
    monkeypatch.setattr(time, "time", lambda: meter.deadline_utc + 1)
    with pytest.raises(StageDeadline):
        meter.reserve_training()
    assert meter.usage == ConstraintUsage()


def test_an_already_accepted_iteration_is_debited_after_clock_expiry(monkeypatch):
    meter = budget()
    meter.reserve_training()
    monkeypatch.setattr(time, "time", lambda: meter.deadline_utc + 1)
    meter.accept_iteration()
    assert meter.usage.iterations == 1


@pytest.mark.parametrize("field", ["max_gradients", "max_forwards", "max_iterations"])
def test_invalid_limits_refuse(field):
    with pytest.raises(ValueError):
        ConstraintLimits(**{field: True})


def test_bad_restored_usage_refuses():
    with pytest.raises(ValueError):
        ConstraintBudget(
            ConstraintLimits(),
            deadline_utc=time.time() + 60,
            usage=ConstraintUsage(gradient_evaluations=1),
        )


@pytest.mark.parametrize("change", ["values", "admissibility", "weights"])
def test_disagreeing_passes_refuse_without_publishing_terms(change):
    calls = []

    def changed(model, *, energy_weight, gradients):
        calls.append(gradients)
        if gradients and change == "weights":
            with torch.no_grad():
                model.point.add_(1)
        return ConstraintTerms(
            0.5,
            0.3 if gradients and change == "values" else 0.2,
            admissible=not (gradients and change == "admissibility"),
        )

    objective = ConstrainedObjective(changed, 1.0, PenaltyState(), budget())
    model = Point()
    with pytest.raises(ValueError):
        objective(model)
    with pytest.raises(ValueError, match="no evaluated"):
        objective.terms_for(model)
    assert calls == [False, True]


def test_expiry_between_value_and_gradient_pass_stops_before_backward(monkeypatch):
    meter = budget()
    calls = []
    clock = {"now": time.time()}
    monkeypatch.setattr(time, "time", lambda: clock["now"])

    def expires(model, *, energy_weight, gradients):
        calls.append(gradients)
        clock["now"] = meter.deadline_utc + 1
        return ConstraintTerms(0.5, 0.2)

    with pytest.raises(StageDeadline):
        ConstrainedObjective(expires, 1.0, PenaltyState(), meter)(Point())
    assert calls == [False] and meter.usage.forward_evaluations == 2


@pytest.mark.parametrize("bad", [0, -1.0, math.inf, math.nan, True, None, "1.003"])
def test_invalid_energy_limit_refuses_before_reservation_or_evaluation(bad):
    calls, meter = [], budget()
    with pytest.raises(ValueError):
        ConstrainedObjective(
            evaluator(calls), 1.0, PenaltyState(), meter, energy_ratio_limit=bad
        )
    assert calls == [] and meter.usage == ConstraintUsage()
