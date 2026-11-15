"""Scalar inequality merit, derivatives and frozen multiplier updates."""

import math

import pytest
import torch

from piginot.objectives.energy_constraint import (
    ConstraintTerms,
    PenaltyState,
    constraint_merit,
    update_penalty,
)


@pytest.mark.parametrize("energy", [0.8, 1.0, 1.005, 1.02])
@pytest.mark.parametrize("multiplier", [0.0, 2.0])
def test_merit_and_energy_coefficient_match_autograd_and_finite_difference(
    energy, multiplier
):
    state = PenaltyState(multiplier=multiplier)
    value = torch.tensor(energy, dtype=torch.float64, requires_grad=True)
    violation = value - 1.005
    expected = (
        0.3
        + (torch.relu(multiplier + state.penalty * violation).square() - multiplier**2)
        / (2 * state.penalty)
        + 0.07
    )
    expected.backward()
    result = constraint_merit(ConstraintTerms(energy, 0.3, 0.07), 1.0, state)
    assert result.value == pytest.approx(float(expected.detach()), abs=1e-14)
    assert result.energy_weight == pytest.approx(float(value.grad), abs=1e-11)
    delta = 1e-7
    finite_difference = (
        constraint_merit(ConstraintTerms(energy + delta, 0.3, 0.07), 1.0, state).value
        - constraint_merit(ConstraintTerms(energy - delta, 0.3, 0.07), 1.0, state).value
    ) / (2 * delta)
    assert result.energy_weight == pytest.approx(finite_difference, abs=3e-5)


def test_coefficient_includes_reference_energy_normalization():
    state = PenaltyState(multiplier=1.0, penalty=10.0)
    out = constraint_merit(ConstraintTerms(0.4, 0.2), 0.2, state)
    assert out.constraint == pytest.approx(0.995)
    assert out.energy_weight == pytest.approx((1 + 10 * 0.995) / 0.2)


def test_inactive_constraint_retains_the_unit_guard():
    out = constraint_merit(ConstraintTerms(0.5, 0.2, 7.0, False), 1.0, PenaltyState())
    assert out.energy_weight == 0
    assert out.value == pytest.approx(7.2)


def test_multiplier_update_is_immutable_and_uses_the_declared_schedule():
    initial = PenaltyState()
    first = update_penalty(initial, 0.01)
    assert initial == PenaltyState()
    assert first.multiplier == pytest.approx(10.0)
    assert first.penalty == 10000 and first.previous_violation == 0.01
    improving = update_penalty(first, 0.004)
    assert improving.multiplier == pytest.approx(50.0)
    assert improving.penalty == 10000
    feasible = update_penalty(improving, -0.01)
    assert feasible.multiplier == 0 and feasible.previous_violation == 0
    capped = update_penalty(PenaltyState(penalty=1e6), 1.0)
    assert capped.penalty == 1e6


@pytest.mark.parametrize("bad", [-1.0, math.inf, math.nan, True])
def test_invalid_terms_and_reference_refuse(bad):
    with pytest.raises(ValueError):
        ConstraintTerms(bad, 0.1)
    with pytest.raises(ValueError):
        constraint_merit(ConstraintTerms(0.2, 0.1), bad, PenaltyState())


@pytest.mark.parametrize("bad", [0, -1.0, math.inf, math.nan, True])
def test_invalid_penalty_refuses(bad):
    with pytest.raises(ValueError):
        PenaltyState(penalty=bad)


def test_nonfinite_derived_merit_refuses():
    with pytest.raises(ValueError):
        constraint_merit(ConstraintTerms(1e300, 1.0), 1e-300, PenaltyState())


@pytest.mark.parametrize("limit", [1.005, 1.003])
@pytest.mark.parametrize("energy", [0.99, 1.004, 1.02])
@pytest.mark.parametrize("multiplier", [0.0, 2.0])
def test_declared_energy_limit_preserves_the_reference_and_exact_derivative(
    limit, energy, multiplier
):
    reference = 0.2
    state = PenaltyState(multiplier=multiplier)
    value = torch.tensor(reference * energy, dtype=torch.float64, requires_grad=True)
    active = torch.relu(multiplier + state.penalty * (value / reference - limit))
    expected = 0.3 + (active.square() - multiplier**2) / (2 * state.penalty) + 0.07
    expected.backward()
    result = constraint_merit(
        ConstraintTerms(float(value.detach()), 0.3, 0.07),
        reference,
        state,
        energy_ratio_limit=limit,
    )
    assert result.constraint == pytest.approx(energy - limit)
    assert result.value == pytest.approx(float(expected.detach()), abs=1e-14)
    assert result.energy_weight == pytest.approx(float(value.grad), abs=1e-11)
    delta = 1e-8
    differences = [
        constraint_merit(
            ConstraintTerms(reference * energy + sign * delta, 0.3, 0.07),
            reference,
            state,
            energy_ratio_limit=limit,
        ).value
        for sign in (1, -1)
    ]
    assert result.energy_weight == pytest.approx(
        (differences[0] - differences[1]) / (2 * delta), abs=2e-5
    )


def test_omitted_energy_limit_has_the_exact_legacy_value():
    terms, state = ConstraintTerms(0.21, 0.3, 0.07), PenaltyState(multiplier=2.0)
    assert constraint_merit(terms, 0.2, state) == constraint_merit(
        terms, 0.2, state, energy_ratio_limit=1.005
    )


@pytest.mark.parametrize("bad", [0, -1.0, math.inf, math.nan, True, None, "1.003"])
def test_invalid_energy_limit_refuses(bad):
    with pytest.raises(ValueError):
        constraint_merit(
            ConstraintTerms(0.2, 0.1), 1.0, PenaltyState(), energy_ratio_limit=bad
        )
