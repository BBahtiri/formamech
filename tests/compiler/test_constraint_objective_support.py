"""Weighted energy derivatives retain the original barrier and chunk integral."""

import time

import pytest
import torch

from piginot.training.lbfgs_stage import StageDeadline
from tests.compiler.test_boundary_objective import (
    context as context,
)
from tests.compiler.test_boundary_objective import (
    flatten_gradients,
    prepared,
)


@pytest.mark.parametrize("weight", [0.0, 0.3, 3.0])
def test_global_energy_weight_has_the_correct_combined_gradient(context, weight):
    data = prepared(context)
    model = data.new_model()
    data.evaluate(model, traction_weight=0, gradients=True)
    energy_gradient = flatten_gradients(model).clone()
    data.evaluate(model, traction_weight=1, gradients=True)
    combined_gradient = flatten_gradients(model).clone()
    out = data.evaluate(model, traction_weight=1, gradients=True, energy_weight=weight)
    torch.testing.assert_close(
        flatten_gradients(model),
        combined_gradient + (weight - 1) * energy_gradient,
        rtol=1e-9,
        atol=1e-9,
    )
    assert out.value == pytest.approx(weight * out.bulk_energy + out.traction_squared)


def test_absolute_utc_expiry_stops_inside_quadrature_chunks(context, monkeypatch):
    data = prepared(context)
    clock = iter((100.0, 102.0))
    monkeypatch.setattr(time, "time", lambda: next(clock))
    with pytest.raises(StageDeadline):
        data.evaluate(
            data.new_model(), traction_weight=1, gradients=True, deadline_utc=101.0
        )


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), True])
def test_bad_energy_weights_refuse_before_evaluation(context, bad):
    data = prepared(context)
    with pytest.raises(ValueError):
        data.evaluate(
            data.new_model(), traction_weight=1, gradients=True, energy_weight=bad
        )
