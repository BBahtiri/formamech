from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest
import torch

from campaigns.dogbone_v1 import s13_joint_hybrid_metrics as metrics
from campaigns.dogbone_v1 import s13_quadrature_metrics as quadrature
from campaigns.dogbone_v1.s11_contingency_models import model_state_hash
from campaigns.dogbone_v1.s13_overfit_diagnostic import classify_s13_overfit
from campaigns.dogbone_v1.s13_quadrature_audit import (
    FemQuadratureFields,
    QuadraturePrediction,
)
from campaigns.dogbone_v1.s13_spatial_audit import summarize_spatial_case
from piginot.contracts import canonical_json_bytes
from piginot.operators import OperatorOutput
from tests.campaigns.test_dogbone_s13_objective import _model
from tests.campaigns.test_dogbone_s13_overfit_diagnostic import _metrics
from tests.campaigns.test_dogbone_s13_screen import _case


def _reference() -> FemQuadratureFields:
    shape = (1792, 7)
    gradient = torch.zeros((*shape, 2, 2), dtype=torch.float64)
    gradient[..., 0, 0] = 0.2
    gradient[..., 1, 1] = 0.1
    return FemQuadratureFields(
        torch.arange(12544 * 2, dtype=torch.float64).reshape(*shape, 2) / 1000,
        torch.ones((*shape, 2), dtype=torch.float64),
        gradient,
        760 * gradient,
        torch.arange(1, 12545, dtype=torch.float64).reshape(shape) / 12544,
        torch.arange(1792) < 41,
        0.0,
    )


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    model, case = _model(), _case("candidate-002485")
    reference = _reference()
    vertex = _metrics(data_mse=0.1)
    prediction = QuadraturePrediction(
        reference.u_mm * 1.2, reference.gradient * 2, reference.raw_p_mpa * 1.3
    )
    calls: list[int] = []

    def predict(*args: object, **kwargs: Any) -> QuadraturePrediction:
        assert args[0] is model and args[1] is case
        assert args[2] is reference.xy_mm
        calls.append(kwargs["chunk_size"])
        return prediction

    def linear(gradient: torch.Tensor, *args: object) -> torch.Tensor:
        assert bool((torch.linalg.det(torch.eye(2) + gradient) > 1e-12).all())
        return 760 * gradient

    monkeypatch.setattr(metrics, "predict_quadrature", predict)
    monkeypatch.setattr(metrics, "_measure_case", lambda *args: vertex)
    monkeypatch.setattr(quadrature, "_material_stress", linear)
    predicted = case.tensors.target_P * 1.3
    residual = torch.full_like(predicted, 0.1)
    determinant = torch.ones(len(predicted), dtype=torch.float64)
    monkeypatch.setattr(
        metrics,
        "_vertex_prediction_and_physics",
        lambda *args: (predicted, residual, determinant),
    )
    return {
        "model": model,
        "case": case,
        "reference": reference,
        "original_s11_metrics": _metrics(data_mse=1.0),
        "pinned_v5_metrics": _metrics(data_mse=0.2),
        "prediction": prediction,
        "vertex": vertex,
        "spatial": summarize_spatial_case(
            predicted,
            case.tensors.target_P,
            case.tensors.qoi_indices,
            case.tensors.query_xy,
            residual,
            determinant,
        ),
        "calls": calls,
    }


def _snapshot(
    harness: dict[str, Any], arm_id: str = "candidate", step: int = 4000
) -> Any:
    return metrics.evaluate_joint_snapshot(
        **{
            key: harness[key]
            for key in (
                "model",
                "case",
                "reference",
                "original_s11_metrics",
                "pinned_v5_metrics",
            )
        },
        arm_id=arm_id,
        step=step,
    )


def test_snapshot_reuses_full_area_weighted_and_historical_vertex_metrics(
    harness: dict[str, Any],
) -> None:
    snapshot = _snapshot(harness)
    assert harness["calls"] == [512]
    assert snapshot.state_hash == model_state_hash(harness["model"])
    assert snapshot.vertex_metrics is harness["vertex"]
    assert snapshot.vertex_classification == classify_s13_overfit(
        metrics.build_s13_overfit_plan(version="v5"),
        harness["original_s11_metrics"],
        harness["vertex"],
    )
    assert snapshot.vertex_classification.data_mse_drop_fraction == pytest.approx(0.9)
    assert snapshot.additional_data_mse_drop_from_v5 == pytest.approx(0.5)
    assert snapshot.spatial_metrics == harness["spatial"]
    assert snapshot.spatial_unavailable_reason is None
    assert snapshot.quadrature_metrics.all_points.point_count == 12544
    assert snapshot.quadrature_metrics.root_incident.point_count == 287
    assert snapshot.quadrature_metrics.nonroot.point_count == 12257
    for population in (
        snapshot.quadrature_metrics.all_points,
        snapshot.quadrature_metrics.root_incident,
        snapshot.quadrature_metrics.nonroot,
    ):
        assert population.gradient_relative_h1 == pytest.approx(1.0)
        assert population.u_relative_l2 == pytest.approx(0.2)
        assert population.p_relative_l2 == pytest.approx(0.3)
        assert population.admissible_area_fraction == 1.0
        assert population.decomposition is not None
        assert population.decomposition.material_model_relative_l2 == pytest.approx(1)
    assert b'"vertex_classification"' in canonical_json_bytes(snapshot)
    with pytest.raises(FrozenInstanceError):
        snapshot.step = 1


def test_snapshot_preserves_rng_grads_parameters_optimizer_and_mixed_module_modes(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = harness["model"]
    model.train()
    next(module for module in model.modules() if module is not model).eval()
    modes = tuple(module.training for module in model.modules())
    parameters = tuple(parameter.clone() for parameter in model.parameters())
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradients = tuple(parameter.grad.clone() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters())
    optimizer_before = optimizer.state_dict()
    rng = torch.random.get_rng_state().clone()

    def measure(*args: object) -> Any:
        torch.rand(1)
        model.eval()
        return harness["vertex"]

    monkeypatch.setattr(metrics, "_measure_case", measure)
    _snapshot(harness)
    assert tuple(module.training for module in model.modules()) == modes
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert optimizer.state_dict() == optimizer_before
    for parameter, expected, gradient in zip(
        model.parameters(), parameters, gradients, strict=True
    ):
        assert torch.equal(parameter, expected)
        assert torch.equal(parameter.grad, gradient)


def test_invalid_vertices_do_not_hide_unconditional_metrics_or_change_classification(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    vertex = _metrics(data_mse=0.1, invalid_vertices=1, invalid=2)
    monkeypatch.setattr(metrics, "_measure_case", lambda *args: vertex)
    monkeypatch.setattr(
        metrics,
        "_vertex_prediction_and_physics",
        lambda *args: pytest.fail("invalid F"),
    )
    snapshot = _snapshot(harness)
    assert snapshot.vertex_metrics.invalid_vertex_count == 1
    assert snapshot.vertex_metrics.invalid_collocation_count == 2
    assert snapshot.vertex_classification.failed_criteria == (
        "final_invalid_collocation_count",
    )
    assert snapshot.spatial_metrics is None
    assert snapshot.spatial_unavailable_reason == "inadmissible_vertex_deformation"
    assert snapshot.quadrature_metrics.all_points.point_count == 12544


def test_snapshot_retains_invalid_quadrature_area_and_exact_point_ids(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction = harness["prediction"]
    gradient = prediction.gradient.clone()
    gradient[0, 0] = torch.tensor([[-1.2, 0.0], [0.0, 0.0]])
    actual = replace(prediction, gradient=gradient)
    monkeypatch.setattr(metrics, "predict_quadrature", lambda *args, **kwargs: actual)
    snapshot = _snapshot(harness)
    population = snapshot.quadrature_metrics.all_points
    assert population.point_count == 12544
    assert population.invalid_count == 1
    assert population.admissible_count == 12543
    point = population.invalid_points[0]
    assert (point.cell_index, point.quadrature_index) == (0, 0)
    assert point.determinant == pytest.approx(-0.2)
    assert point.area_weight_mm2 == harness["reference"].area_weights[0, 0]
    assert population.admissible_area_fraction == pytest.approx(
        1 - point.area_weight_mm2 / population.area_mm2
    )
    assert population.gradient_relative_h1 > 1


@pytest.mark.parametrize("invalid", ("case", "step", "arm", "layout", "s11", "v5"))
def test_snapshot_fails_closed_on_inputs_before_prediction(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    monkeypatch.setattr(
        metrics, "predict_quadrature", lambda *args, **kwargs: pytest.fail("predicted")
    )
    arguments = {
        key: harness[key]
        for key in (
            "model",
            "case",
            "reference",
            "original_s11_metrics",
            "pinned_v5_metrics",
        )
    }
    arguments.update(arm_id="candidate", step=4000)
    if invalid == "case":
        arguments["case"] = object()
    elif invalid == "step":
        arguments["step"] = True
    elif invalid == "arm":
        arguments["arm_id"] = "other"
    elif invalid == "layout":
        arguments["reference"] = replace(
            harness["reference"], xy_mm=torch.zeros(2, 7, 2)
        )
    elif invalid == "s11":
        arguments["original_s11_metrics"] = _metrics(data_mse=0)
    else:
        arguments["pinned_v5_metrics"] = object()
    with pytest.raises((TypeError, ValueError, PermissionError)):
        metrics.evaluate_joint_snapshot(**arguments)


def test_snapshot_restores_modes_and_rng_on_failure_and_rejects_state_mutation(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = harness["model"]
    model.train()
    modes = tuple(module.training for module in model.modules())
    rng = torch.random.get_rng_state().clone()

    def failing(*args: object) -> None:
        model.eval()
        torch.rand(1)
        raise ValueError("admissible kernel failed")

    monkeypatch.setattr(metrics, "_measure_case", failing)
    with pytest.raises(ValueError, match="kernel"):
        _snapshot(harness)
    assert tuple(module.training for module in model.modules()) == modes
    assert torch.equal(torch.random.get_rng_state(), rng)

    def mutating(*args: object) -> Any:
        with torch.no_grad():
            next(model.parameters()).add_(1)
        return harness["vertex"]

    monkeypatch.setattr(metrics, "_measure_case", mutating)
    with pytest.raises(RuntimeError, match="state changed"):
        _snapshot(harness)


def test_real_vertex_and_chunked_evaluation_use_each_models_current_joint_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, reference = _case("candidate-002485"), _reference()
    snapshots = []
    for value, arm in ((0.2, "candidate"), (0.1, "control")):
        model = _model()

        def affine(factor: float = value, **kwargs: Any) -> OperatorOutput:
            xy = kwargs["query_xy"]
            return OperatorOutput(
                u=factor * xy,
                P=torch.ones((*xy.shape[:-1], 2, 2), dtype=torch.float64) * factor,
            )

        monkeypatch.setattr(model, "forward", affine)
        monkeypatch.setattr(model, "decode_queries", affine)
        before = model_state_hash(model)
        snapshots.append(
            metrics.evaluate_joint_snapshot(
                model,
                case,
                reference,
                arm_id=arm,
                step=4000,
                original_s11_metrics=_metrics(data_mse=3),
                pinned_v5_metrics=_metrics(data_mse=2),
            )
        )
        assert model_state_hash(model) == before
    candidate, control = snapshots
    assert candidate.vertex_metrics != control.vertex_metrics
    assert (
        candidate.quadrature_metrics.all_points.gradient_relative_h1
        != control.quadrature_metrics.all_points.gradient_relative_h1
    )
    assert (
        candidate.quadrature_metrics.all_points.u_relative_l2
        != control.quadrature_metrics.all_points.u_relative_l2
    )
    assert (
        candidate.quadrature_metrics.all_points.p_relative_l2
        != control.quadrature_metrics.all_points.p_relative_l2
    )
    assert (
        candidate.quadrature_metrics.all_points.model_determinant_min
        != control.quadrature_metrics.all_points.model_determinant_min
    )


def test_zero_v5_mse_keeps_separate_incremental_reduction_unavailable(
    harness: dict[str, Any],
) -> None:
    harness = {**harness, "pinned_v5_metrics": _metrics(data_mse=0)}
    snapshot = _snapshot(harness)
    assert snapshot.additional_data_mse_drop_from_v5 is None
    assert snapshot.vertex_classification.data_mse_drop_fraction == pytest.approx(0.9)


def test_snapshot_and_endpoint_refuse_wrong_plan_or_snapshot_type(
    harness: dict[str, Any],
) -> None:
    with pytest.raises(TypeError, match="exact JointHybridPlan"):
        metrics.evaluate_joint_snapshot(
            harness["model"],
            harness["case"],
            harness["reference"],
            arm_id="candidate",
            step=4000,
            original_s11_metrics=harness["original_s11_metrics"],
            pinned_v5_metrics=harness["pinned_v5_metrics"],
            plan=object(),
        )
    snapshot = _snapshot(harness)
    with pytest.raises(TypeError, match="exact joint"):
        metrics.summarize_joint_endpoint(object(), snapshot, snapshot)


def _contrasts(population: Any, conditional: bool = False) -> dict[str, Any]:
    return {
        item.metric: item
        for item in (
            population.conditional if conditional else population.unconditional
        )
    }


def test_endpoint_signed_changes_ratios_and_zero_denominators_are_descriptive(
    harness: dict[str, Any],
) -> None:
    candidate = _snapshot(harness)
    baseline = candidate.quadrature_metrics

    def changed(population: Any) -> Any:
        return replace(
            population,
            u_relative_l2=0.4,
            gradient_relative_h1=2.0,
            p_relative_l2=0.0,
        )

    control = replace(
        candidate,
        arm_id="control",
        quadrature_metrics=replace(
            baseline,
            **{
                name: changed(getattr(baseline, name))
                for name in ("all_points", "root_incident", "nonroot")
            },
        ),
    )
    v5 = replace(control, arm_id="v5", step=0)
    summary = metrics.summarize_joint_endpoint(candidate, control, v5)
    assert summary.comparison_step == 4000
    for comparison in (summary.candidate_vs_control, summary.candidate_vs_pinned_v5):
        for population in (
            comparison.all_points,
            comparison.root_incident,
            comparison.nonroot,
        ):
            values = _contrasts(population)
            assert values["u_relative_l2"].signed_change == pytest.approx(-0.2)
            assert values["u_relative_l2"].ratio == pytest.approx(0.5)
            assert values["gradient_relative_h1"].signed_change == pytest.approx(-1)
            assert values["p_relative_l2"].ratio is None
            assert values["p_relative_l2"].ratio_unavailable_reason == "zero_reference"
            assert population.admissible_point_support_identical
    encoded = canonical_json_bytes(summary)
    assert b'"ratio":null' in encoded
    assert b'"repaired"' not in encoded and b'"success"' not in encoded
    with pytest.raises(FrozenInstanceError):
        summary.comparison_step = 1


def test_equal_count_and_area_do_not_establish_identical_conditional_support(
    harness: dict[str, Any],
) -> None:
    candidate = _snapshot(harness)
    q = candidate.quadrature_metrics
    point = quadrature.InvalidQuadraturePoint(0, 0, (1, 1), -0.2, 1.0)
    first = replace(q.all_points, invalid_count=1, invalid_points=(point,))
    second = replace(first, invalid_points=(replace(point, quadrature_index=1),))
    candidate = replace(candidate, quadrature_metrics=replace(q, all_points=first))
    control = replace(
        candidate, arm_id="control", quadrature_metrics=replace(q, all_points=second)
    )
    v5 = replace(candidate, arm_id="v5", step=0)
    summary = metrics.summarize_joint_endpoint(candidate, control, v5)
    unmatched = summary.candidate_vs_control.all_points
    assert not unmatched.admissible_point_support_identical
    assert _contrasts(unmatched)["gradient_relative_h1"].ratio == 1
    assert all(value.ratio is None for value in unmatched.conditional)
    assert all(
        value.ratio_unavailable_reason == "different_admissible_support"
        for value in unmatched.conditional
    )
    assert summary.candidate_vs_pinned_v5.all_points.admissible_point_support_identical
    assert (
        _contrasts(summary.candidate_vs_pinned_v5.all_points, True)[
            "material_model_relative_l2"
        ].ratio
        == 1
    )


def test_conditional_support_uses_ids_not_determinants_and_handles_no_admissible_points(
    harness: dict[str, Any],
) -> None:
    candidate = _snapshot(harness)
    q = candidate.quadrature_metrics
    point = quadrature.InvalidQuadraturePoint(0, 0, (1, 1), -0.2, 1.0)
    population = replace(q.all_points, invalid_points=(point,), invalid_count=1)
    candidate = replace(candidate, quadrature_metrics=replace(q, all_points=population))
    control = replace(
        candidate,
        arm_id="control",
        quadrature_metrics=replace(
            q,
            all_points=replace(
                population, invalid_points=(replace(point, determinant=-0.3),)
            ),
        ),
    )
    v5 = replace(candidate, arm_id="v5", step=0)
    assert metrics.summarize_joint_endpoint(
        candidate, control, v5
    ).candidate_vs_control.all_points.admissible_point_support_identical
    absent = replace(population, decomposition=None, admissible_count=0)
    control = replace(control, quadrature_metrics=replace(q, all_points=absent))
    result = metrics.summarize_joint_endpoint(
        candidate, control, v5
    ).candidate_vs_control.all_points
    assert all(
        value.ratio is None and value.signed_change is None
        for value in result.conditional
    )


@pytest.mark.parametrize(
    "invalid", ("candidate_step", "control_step", "v5_step", "arm", "reference")
)
def test_endpoint_requires_fixed_step_arm_ids_and_same_fem_reference(
    harness: dict[str, Any], invalid: str
) -> None:
    candidate = _snapshot(harness)
    control = replace(candidate, arm_id="control")
    v5 = replace(candidate, arm_id="v5", step=0)
    if invalid == "candidate_step":
        candidate = replace(candidate, step=3000)
    elif invalid == "control_step":
        control = replace(control, step=3000)
    elif invalid == "v5_step":
        v5 = replace(v5, step=4000)
    elif invalid == "arm":
        candidate = replace(candidate, arm_id="control")
    else:
        control = replace(control, quadrature_reference_hash="0" * 64)
    with pytest.raises(ValueError):
        metrics.summarize_joint_endpoint(candidate, control, v5)
