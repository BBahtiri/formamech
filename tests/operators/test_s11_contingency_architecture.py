from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
import torch

from piginot.operators import (
    ARCHITECTURE_ID,
    LOCAL_STRESS_ARCHITECTURE_ID,
    GeometryEncoding,
    OperatorConfig,
    PiGinotOperator,
    module_digest,
)

PARAMETER_NAMES = (
    "L_total",
    "W_grip",
    "W_gauge",
    "R_fillet",
    "E",
    "nu",
    "nominal_axial_strain",
)
TAG_NAMES = ("symmetry_x", "symmetry_y", "grip", "traction_free")


def _config(**overrides: object) -> OperatorConfig:
    values: dict[str, object] = {
        "parameter_names": PARAMETER_NAMES,
        "boundary_tag_names": TAG_NAMES,
        "hidden_dim": 32,
        "latent_token_count": 8,
        "num_attention_heads": 4,
        "num_fourier_frequencies": 4,
        "encoder_self_attention_layers": 1,
        "decoder_cross_attention_layers": 2,
    }
    values.update(overrides)
    return OperatorConfig(**values)  # type: ignore[arg-type]


def _model(config: OperatorConfig) -> PiGinotOperator:
    torch.manual_seed(11120)
    return PiGinotOperator(config).to(dtype=torch.float64)


def _encoding(config: OperatorConfig) -> GeometryEncoding:
    return GeometryEncoding(
        tokens=torch.zeros(1, 8, config.hidden_dim, dtype=torch.float64),
        coordinate_min=torch.zeros(1, 2, dtype=torch.float64),
        coordinate_span=torch.ones(1, 2, dtype=torch.float64),
        normalized_boundary_xy=torch.tensor(
            [[[-1.0, 0.0], [1.0, 0.0]]], dtype=torch.float64
        ),
        boundary_features=torch.zeros(1, 2, config.hidden_dim, dtype=torch.float64),
        boundary_weights=torch.ones(1, 2, dtype=torch.float64),
        boundary_mask=torch.ones(1, 2, dtype=torch.bool),
    )


def _local_fields(
    encoding: GeometryEncoding,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fields = (
        encoding.normalized_boundary_xy,
        encoding.boundary_features,
        encoding.boundary_weights,
        encoding.boundary_mask,
    )
    assert all(field is not None for field in fields)
    return cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], fields)


def test_contingency_config_rejects_ambiguous_local_enrichment() -> None:
    for scales in ((0.2, 0.1), (0.1, 0.1), (0.0,), (float("inf"),)):
        with pytest.raises(ValueError, match="local_boundary_kernel_scales"):
            _config(local_boundary_kernel_scales=scales)

    for depth in (-1, True, 1.5):
        with pytest.raises(ValueError, match="stress_enrichment_layers"):
            _config(stress_enrichment_layers=depth)


def test_multiscale_context_is_query_local_and_stress_specific() -> None:
    config = _config(
        local_boundary_kernel_scales=(0.08, 0.25),
        stress_enrichment_layers=2,
    )
    model = _model(config)
    encoding = _encoding(config)
    _, boundary_features, _, _ = _local_fields(encoding)
    highlighted_features = boundary_features.clone()
    highlighted_features[:, 0] = 5.0
    highlighted = replace(encoding, boundary_features=highlighted_features)
    query_xy = torch.tensor([[[0.01, 0.5], [0.99, 0.5]]], dtype=torch.float64)
    parameters = torch.rand(1, len(PARAMETER_NAMES), dtype=torch.float64)

    baseline = model.decode_queries(
        encoding=encoding,
        query_xy=query_xy,
        parameters=parameters,
    )
    local_change = model.decode_queries(
        encoding=highlighted,
        query_xy=query_xy,
        parameters=parameters,
    )

    torch.testing.assert_close(local_change.u, baseline.u, rtol=0.0, atol=0.0)
    near_delta = torch.linalg.vector_norm(local_change.P[:, 0] - baseline.P[:, 0])
    far_delta = torch.linalg.vector_norm(local_change.P[:, 1] - baseline.P[:, 1])
    assert float(near_delta.item()) > 1.0e-4
    assert float(far_delta.item()) < float(near_delta.item()) * 1.0e-6


def test_local_context_preserves_permutation_and_mask_invariants() -> None:
    config = _config(
        local_boundary_kernel_scales=(0.08, 0.25),
        stress_enrichment_layers=2,
    )
    model = _model(config)
    encoding = _encoding(config)
    query_xy = torch.tensor([[[0.04, 0.5]]], dtype=torch.float64)
    parameters = torch.rand(1, len(PARAMETER_NAMES), dtype=torch.float64)
    expected = model.decode_queries(
        encoding=encoding,
        query_xy=query_xy,
        parameters=parameters,
    )
    boundary_xy, boundary_features, boundary_weights, boundary_mask = _local_fields(
        encoding
    )

    permutation = torch.tensor([1, 0])
    permuted = replace(
        encoding,
        normalized_boundary_xy=boundary_xy[:, permutation],
        boundary_features=boundary_features[:, permutation],
        boundary_weights=boundary_weights[:, permutation],
        boundary_mask=boundary_mask[:, permutation],
    )
    actual = model.decode_queries(
        encoding=permuted,
        query_xy=query_xy,
        parameters=parameters,
    )
    torch.testing.assert_close(
        actual.flat(), expected.flat(), rtol=1.0e-12, atol=1.0e-12
    )

    padded = replace(
        encoding,
        normalized_boundary_xy=torch.cat((boundary_xy, query_xy * 2.0 - 1.0), dim=1),
        boundary_features=torch.cat(
            (
                boundary_features,
                torch.full((1, 1, config.hidden_dim), 1.0e6, dtype=torch.float64),
            ),
            dim=1,
        ),
        boundary_weights=torch.cat(
            (boundary_weights, torch.ones(1, 1, dtype=torch.float64)), dim=1
        ),
        boundary_mask=torch.cat(
            (boundary_mask, torch.zeros(1, 1, dtype=torch.bool)), dim=1
        ),
    )
    masked = model.decode_queries(
        encoding=padded,
        query_xy=query_xy,
        parameters=parameters,
    )
    torch.testing.assert_close(
        masked.flat(), expected.flat(), rtol=1.0e-12, atol=1.0e-12
    )


def test_local_stress_variant_has_an_explicit_identity_and_digest() -> None:
    baseline = _model(_config())
    enriched = _model(
        _config(
            local_boundary_kernel_scales=(0.08, 0.25),
            stress_enrichment_layers=2,
        )
    )

    assert baseline.architecture_id == ARCHITECTURE_ID
    assert enriched.architecture_id == LOCAL_STRESS_ARCHITECTURE_ID
    assert module_digest(baseline) != module_digest(enriched)
