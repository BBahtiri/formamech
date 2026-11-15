"""FIT-only preparation and chunked gradients use the actual energy operator."""

import pytest
import torch

from piginot.compiler.boundary_objective import BoundaryPilotObjective
from tests.compiler.test_execution_training import context as context


def prepared(context, chunk=3):
    context.prepare()
    return BoundaryPilotObjective(
        context,
        ("fit-001", "fit-002"),
        strain_scales={"fit-001": 0.02, "fit-002": 0.03},
        points_per_cell=2,
        chunk=chunk,
    )


def flatten_gradients(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def test_chunked_bulk_and_boundary_gradient_match_full_evaluation(context):
    small = prepared(context, chunk=3)
    large = BoundaryPilotObjective(
        context,
        ("fit-001", "fit-002"),
        strain_scales={"fit-001": 0.02, "fit-002": 0.03},
        points_per_cell=2,
        chunk=1000,
    )
    a, b = small.new_model(), large.new_model()
    observed = small.evaluate(a, traction_weight=1.0, gradients=True)
    expected = large.evaluate(b, traction_weight=1.0, gradients=True)
    assert observed.value == pytest.approx(expected.value, rel=1e-11)
    torch.testing.assert_close(
        flatten_gradients(a), flatten_gradients(b), rtol=1e-10, atol=1e-10
    )
    assert observed.admissible and expected.admissible
    assert (
        small.document() == large.document()
    )  # chunk size does not alter the integral


def test_preparation_freezes_fit_scales_and_uses_independent_validation_points(context):
    data = prepared(context)
    assert data.case_ids == ("fit-001", "fit-002")
    before = data.document()
    a = data.new_model()
    b = data.new_model()
    assert all(
        torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters(), strict=True)
    )
    assert not torch.equal(
        data.cases[0].training_points, data.cases[0].validation_points
    )
    data.evaluate(a, traction_weight=0.0, gradients=False, validation=True)
    assert data.document() == before
    assert all(not case.stress_scales.requires_grad for case in data.cases)
    assert data.normalization == context.normalization


def test_free_traction_term_adds_a_nonnegative_consistency_residual(context):
    data = prepared(context)
    model = data.new_model()
    a = data.evaluate(model, traction_weight=0.0, gradients=False)
    b = data.evaluate(model, traction_weight=1.0, gradients=False)
    assert b.value >= a.value
    assert b.value - a.value == pytest.approx(b.traction_squared, rel=1e-10)
    assert b.bulk_energy == pytest.approx(a.bulk_energy)


@pytest.mark.parametrize(
    "ids", [("screen-001",), ("absent",), (), ("fit-001", "fit-001")]
)
def test_screen_or_missing_cases_never_enter_pilot_preparation(context, ids):
    context.prepare()
    with pytest.raises(ValueError):
        BoundaryPilotObjective(context, ids, strain_scales={name: 0.01 for name in ids})


def test_validation_reports_regional_physics_without_parameter_updates(context):
    data = prepared(context)
    model = data.new_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = data.evaluate(model, traction_weight=1.0, gradients=False, validation=True)
    assert len(out.regions) > 0 and out.minimum_determinant > 0
    assert out.traction_squared >= 0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -1.0])
def test_invalid_load_scales_refuse_before_training(context, bad):
    context.prepare()
    with pytest.raises(ValueError):
        BoundaryPilotObjective(context, ("fit-001",), strain_scales={"fit-001": bad})
