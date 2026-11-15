"""Refuse ambiguous tensor context and incomplete lifting evidence before use."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from piginot.operators import (
    BoundaryConditionMode,
    DirichletData,
    GeometryEncoding,
    LiftingVerification,
    OperatorOutput,
    PiGinotOperator,
)
from tests.operators.test_validation import _config, _inputs


@pytest.fixture
def model() -> PiGinotOperator:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2801)
        return PiGinotOperator(_config())


def _encoding() -> GeometryEncoding:
    return GeometryEncoding(torch.zeros(1, 2, 8), torch.zeros(1, 2), torch.ones(1, 2))


def _local_encoding() -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.zeros(1, 2, 8),
        "coordinate_min": torch.zeros(1, 2),
        "coordinate_span": torch.ones(1, 2),
        "normalized_boundary_xy": torch.zeros(1, 2, 2),
        "boundary_features": torch.zeros(1, 2, 8),
        "boundary_weights": torch.ones(1, 2),
        "boundary_mask": torch.ones(1, 2, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"coordinate_span": torch.ones(2, 2)}, "same shape"),
        (
            {"coordinate_min": torch.zeros(1, 2, dtype=torch.float64)},
            "dtype and device",
        ),
        ({"boundary_features": None}, "supplied together"),
        (
            {"boundary_features": torch.zeros(1, 3, 8)},
            "align with boundary coordinates",
        ),
        ({"boundary_features": torch.zeros(1, 2, 7)}, "token width"),
        ({"boundary_weights": torch.ones(1, 3)}, "align with boundary coordinates"),
        ({"boundary_weights": torch.ones(1, 2).to_sparse()}, "dense and strided"),
        ({"boundary_weights": torch.full((1, 2), float("nan"))}, "finite"),
        ({"boundary_weights": -torch.ones(1, 2)}, "non-negative"),
        ({"boundary_mask": torch.ones(1, 2)}, "boolean"),
        (
            {"boundary_mask": torch.ones(1, 2, dtype=torch.bool).to_sparse()},
            "dense and strided",
        ),
        (
            {"boundary_mask": torch.ones(1, 3, dtype=torch.bool)},
            "align with boundary coordinates",
        ),
        (
            {"boundary_mask": torch.ones(1, 2, dtype=torch.bool, device="meta")},
            "share a device",
        ),
        (
            {"boundary_mask": torch.zeros(1, 2, dtype=torch.bool)},
            "positive visible boundary weight",
        ),
    ],
)
def test_reused_geometry_encoding_requires_complete_aligned_context(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GeometryEncoding(**{**_local_encoding(), **changes})


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"boundary_xy": object()}, TypeError, "torch.Tensor"),
        ({"boundary_xy": torch.ones(1, 4, 2).to_sparse()}, ValueError, "dense strided"),
        ({"boundary_xy": torch.zeros(1, 4, 3)}, ValueError, "shape"),
        ({"boundary_xy": torch.zeros(1, 0, 2)}, ValueError, "non-empty"),
        ({"boundary_xy": torch.full((1, 4, 2), float("inf"))}, ValueError, "finite"),
        ({"boundary_normals": torch.ones(1, 3, 2)}, ValueError, "align"),
        ({"boundary_mask": object()}, TypeError, "torch.Tensor"),
        ({"boundary_mask": torch.ones(1, 3, dtype=torch.bool)}, ValueError, "shape"),
        (
            {"boundary_mask": torch.ones(1, 4, dtype=torch.bool, device="meta")},
            ValueError,
            "share a device",
        ),
        ({"boundary_weights": object()}, TypeError, "torch.Tensor"),
        (
            {"boundary_weights": torch.ones(1, 4).to_sparse()},
            ValueError,
            "dense and strided",
        ),
        ({"boundary_weights": torch.full((1, 4), float("nan"))}, ValueError, "finite"),
        (
            {"boundary_weights": torch.zeros(1, 4)},
            ValueError,
            "positive visible boundary weight",
        ),
    ],
)
def test_boundary_input_refusal_precedes_neural_encoding(
    model: PiGinotOperator,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid boundary data reached the neural encoder")

    monkeypatch.setattr(model.boundary_projection, "forward", forbidden)
    values = {
        key: value for key, value in _inputs().items() if key.startswith("boundary_")
    }
    with pytest.raises(error, match=message):
        model.encode_boundary(**{**values, **changes})


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"encoding": object()}, TypeError, "GeometryEncoding"),
        ({"query_xy": torch.zeros(2, 2, 2)}, ValueError, "batch dimension"),
        ({"parameters": object()}, TypeError, "torch.Tensor"),
        ({"parameters": torch.zeros(1, 3).to_sparse()}, ValueError, "dense strided"),
        (
            {"parameters": torch.zeros(1, 3, dtype=torch.int64)},
            ValueError,
            "float32 or float64",
        ),
        ({"parameters": torch.full((1, 3), float("nan"))}, ValueError, "finite"),
    ],
)
def test_query_input_refusal_precedes_neural_decoding(
    model: PiGinotOperator,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid query data reached the neural decoder")

    monkeypatch.setattr(model.query_projection, "forward", forbidden)
    values = {
        "encoding": _encoding(),
        "query_xy": torch.zeros(1, 2, 2),
        "parameters": torch.zeros(1, 3),
    }
    with pytest.raises(error, match=message):
        model.decode_queries(**{**values, **changes})


def test_dirichlet_tensor_types_and_distance_shape_are_unambiguous() -> None:
    with pytest.raises(TypeError, match="component_mask"):
        DirichletData(object(), torch.zeros(1, 2, 2), torch.zeros(1, 2, 2))
    with pytest.raises(ValueError, match="align"):
        DirichletData(
            torch.ones(1, 2, 2, dtype=torch.bool),
            torch.zeros(1, 2, 2),
            torch.zeros(1, 3, 2),
        )


def test_hard_mode_requires_a_proof_within_the_declared_tolerance() -> None:
    config = _config(boundary_condition_mode=BoundaryConditionMode.HARD_LIFTING)
    evidence = LiftingVerification(
        "public-fixture", "1" * 64, "2" * 64, (320, 4800), True, 2e-6
    )
    with pytest.raises(ValueError, match="exceeds tolerance"):
        PiGinotOperator(config, lifting_verification=evidence)
    with torch.random.fork_rng(devices=[]):
        hard_model = PiGinotOperator(
            config, lifting_verification=replace(evidence, max_boundary_residual=0.0)
        )
    with pytest.raises(ValueError, match="requires DirichletData"):
        hard_model.decode_queries(
            encoding=_encoding(),
            query_xy=torch.zeros(1, 2, 2),
            parameters=torch.zeros(1, 3),
        )


def test_residual_and_decoder_reject_untyped_boundary_data(
    model: PiGinotOperator,
) -> None:
    output = OperatorOutput(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2, 2))
    data = DirichletData(
        torch.ones(1, 2, 2, dtype=torch.bool),
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2, 2),
    )
    with pytest.raises(TypeError, match="OperatorOutput"):
        model.dirichlet_residual(object(), data)
    with pytest.raises(TypeError, match="DirichletData"):
        model.dirichlet_residual(output, object())
    with pytest.raises(TypeError, match="DirichletData"):
        model.decode_queries(
            encoding=_encoding(),
            query_xy=torch.zeros(1, 2, 2),
            parameters=torch.zeros(1, 3),
            dirichlet=object(),
        )


def test_local_stress_decoder_cannot_reuse_an_incomplete_encoding() -> None:
    with torch.random.fork_rng(devices=[]):
        model = PiGinotOperator(_config(local_boundary_kernel_scales=(0.2,)))
    with pytest.raises(ValueError, match="requires boundary encoding fields"):
        model.decode_queries(
            encoding=_encoding(),
            query_xy=torch.zeros(1, 2, 2),
            parameters=torch.zeros(1, 3),
        )
