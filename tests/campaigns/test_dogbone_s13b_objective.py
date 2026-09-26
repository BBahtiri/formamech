from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from campaigns.dogbone_v1.s13_objective import (
    DogboneS13ObjectiveBuilder,
    DogboneS13TrainingCase,
)
from campaigns.dogbone_v1.s13_protocol import build_dogbone_s13_protocol
from campaigns.dogbone_v1.s13_screen import _predict_vertices
from piginot.objectives import ObjectiveTerm
from tests.campaigns.test_dogbone_s13_objective import _model, _training_case


def _case_with_distinct_root_peak() -> DogboneS13TrainingCase:
    case = _training_case()
    target_P = torch.zeros_like(case.tensors.target_P)
    target_P[0, 0, 0] = 100.0  # Outside the registered root QoI.
    target_P[2, 0, 0] = 0.2
    target_P[3, 0, 0] = 0.4
    return replace(case, tensors=replace(case.tensors, target_P=target_P))


def test_s13b_peak_residual_uses_only_registered_root_indices() -> None:
    case = _case_with_distinct_root_peak()
    model = _model()
    model.eval()
    builder = DogboneS13ObjectiveBuilder((case,), include_peak_loss=True)

    residuals = builder(model, (case.tensors.case_id,), 1)
    prediction = _predict_vertices(model, case)
    predicted_peak = torch.linalg.vector_norm(
        prediction.P[0, case.tensors.qoi_indices], dim=(-2, -1)
    ).amax()
    target_peak = torch.tensor(0.4, dtype=torch.float64)

    peak = residuals.values[ObjectiveTerm.DATA_PEAK]
    assert peak.shape == (1,)
    torch.testing.assert_close(peak[0], predicted_peak - target_peak)
    assert tuple(residuals.values)[:3] == (
        ObjectiveTerm.DATA_U,
        ObjectiveTerm.DATA_P,
        ObjectiveTerm.DATA_PEAK,
    )


def test_s13b_peak_ignores_larger_prediction_outside_qoi() -> None:
    case = _training_case()
    model = _model()
    model.eval()
    target_P = torch.zeros_like(case.tensors.target_P)
    target_P[0, 0, 0] = 0.4
    target_P[2, 0, 0] = 100.0
    case = replace(
        case,
        tensors=replace(
            case.tensors,
            target_P=target_P,
            qoi_indices=torch.tensor([0, 1], dtype=torch.int64),
        ),
    )
    prediction = _predict_vertices(model, case)
    predicted_norms = torch.linalg.vector_norm(prediction.P[0], dim=(-2, -1))
    assert predicted_norms[2] > predicted_norms[case.tensors.qoi_indices].amax()

    peak = DogboneS13ObjectiveBuilder((case,), include_peak_loss=True)(
        model, (case.tensors.case_id,), 1
    ).values[ObjectiveTerm.DATA_PEAK]

    torch.testing.assert_close(
        peak[0],
        predicted_norms[case.tensors.qoi_indices].amax()
        - torch.tensor(0.4, dtype=torch.float64),
    )


def test_s13b_peak_residual_has_stress_head_gradient() -> None:
    case = _case_with_distinct_root_peak()
    model = _model()
    builder = DogboneS13ObjectiveBuilder((case,), include_peak_loss=True)

    peak = builder(model, (case.tensors.case_id,), 1).values[ObjectiveTerm.DATA_PEAK]
    peak.square().mean().backward()  # type: ignore[no-untyped-call]

    gradient = model.p_head.weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all().item())
    assert bool((gradient != 0.0).any().item())


def test_s13b_peak_mode_requires_explicit_boolean() -> None:
    case = _case_with_distinct_root_peak()

    with pytest.raises(TypeError, match="include_peak_loss"):
        DogboneS13ObjectiveBuilder((case,), include_peak_loss=1)  # type: ignore[arg-type]


def test_legacy_s13_builder_still_excludes_peak_term() -> None:
    case = _case_with_distinct_root_peak()

    residuals = DogboneS13ObjectiveBuilder((case,))(
        _model(), (case.tensors.case_id,), 1
    )

    assert ObjectiveTerm.DATA_PEAK not in residuals.values


def test_legacy_s13_protocol_hash_is_unchanged() -> None:
    assert build_dogbone_s13_protocol().protocol_hash == (
        "256cf8748489ea84ed3277a380d599eb7dc4227efb693204c2c99cd83d47cced"
    )
