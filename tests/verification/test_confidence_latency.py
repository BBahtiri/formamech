from __future__ import annotations

import math

import pytest
import torch

from piginot.contracts import CanonicalQuantity, MaterialSpec, Unit
from piginot.operators import (
    AxisAlignedTensionLifting,
    EnergyFormOperator,
    EnergyOperatorConfig,
    QueryFrame,
    constitutive_in_plane_stress,
)
from piginot.verification import (
    INDICATOR_VARIANTS,
    QuantileCurve,
    boundary_traction_residual,
    cell_quadrature_points,
    equilibrium_residual,
    fit_quantile_curve,
    indicator_variants,
    measure_latency,
    spearman,
    speedup,
    vertex_mesh_size,
    vertex_patch_mean,
)
from piginot.verification.amortization import amortized_break_even_queries

_DTYPE = torch.float64
_EXTENT = torch.tensor([[20.0, 8.0]], dtype=_DTYPE)


def _material() -> MaterialSpec:
    return MaterialSpec(
        "confidence-test",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(760.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.23, Unit.ONE),
        },
    )


def _model(scale: float) -> EnergyFormOperator:
    torch.manual_seed(3)
    model = EnergyFormOperator(EnergyOperatorConfig(parameter_names=("a", "b"))).to(
        _DTYPE
    )
    with torch.no_grad():
        model.output.weight.normal_(0.0, scale)
    return model


def _inputs() -> dict[str, object]:
    return {
        "parameters": torch.tensor([[0.2, -0.4]], dtype=_DTYPE),
        "frame": QueryFrame(torch.zeros((1, 2), dtype=_DTYPE), _EXTENT),
        "lifting": AxisAlignedTensionLifting(
            _EXTENT, torch.tensor([2.0], dtype=_DTYPE)
        ),
        "material": _material(),
        "length_scale": 20.0,
    }


def _points(count: int = 12) -> torch.Tensor:
    generator = torch.Generator().manual_seed(5)
    return torch.rand((count, 2), generator=generator, dtype=_DTYPE) * torch.tensor(
        [20.0, 8.0], dtype=_DTYPE
    )


def test_uniform_deformation_has_no_equilibrium_residual() -> None:
    model = _model(0.0)
    residual = equilibrium_residual(model, _points(), **_inputs())  # type: ignore[arg-type]

    assert residual.shape == (12,)
    assert float(residual.max()) < 1e-12


def test_residual_matches_a_finite_difference_divergence() -> None:
    model = _model(0.05)
    inputs = _inputs()
    point = torch.tensor([[7.0, 3.0]], dtype=_DTYPE)
    residual = equilibrium_residual(model, point, **inputs)  # type: ignore[arg-type]

    def stress(x: torch.Tensor) -> torch.Tensor:
        _, gradient = model.displacement_and_gradient(
            query_xy=x.unsqueeze(0),
            parameters=inputs["parameters"],  # type: ignore[arg-type]
            frame=inputs["frame"],  # type: ignore[arg-type]
            lifting=inputs["lifting"],  # type: ignore[arg-type]
            create_graph=False,
        )
        return constitutive_in_plane_stress(gradient[0], _material())[0]

    step = 1e-5
    divergence = torch.zeros(2, dtype=_DTYPE)
    for column in range(2):
        shift = torch.zeros((1, 2), dtype=_DTYPE)
        shift[0, column] = step
        derivative = (stress(point + shift) - stress(point - shift)) / (2 * step)
        divergence += derivative[:, column]
    expected = float(torch.linalg.vector_norm(divergence)) * 20.0 / 760.0

    assert expected > 1e-6
    assert float(residual[0]) == pytest.approx(expected, rel=1e-5)


def test_variants_patch_means_and_mesh_size_on_two_cells() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=_DTYPE
    )
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]])

    points = cell_quadrature_points(vertices, cells)
    assert points.shape == (2, 3, 2)
    assert torch.allclose(points.mean(dim=1), vertices[cells].mean(dim=1))
    patch = vertex_patch_mean(4, cells, torch.tensor([1.0, 3.0], dtype=_DTYPE))
    assert patch.tolist() == [1.0, 2.0, 2.0, 3.0]
    size = vertex_mesh_size(vertices, cells)
    assert size[0] == pytest.approx(2.0)
    assert size[1] == pytest.approx(1.0 + math.sqrt(8.0) / 2.0)
    with pytest.raises(ValueError):
        vertex_patch_mean(5, cells, torch.tensor([1.0, 3.0], dtype=_DTYPE))

    variants = indicator_variants(
        lambda x: x[:, 0] + 1.0, vertices, cells, length_scale=2.0
    )
    assert tuple(variants) == INDICATOR_VARIANTS
    assert torch.allclose(variants["pointwise"], vertices[:, 0] + 1.0)
    assert torch.allclose(
        variants["patch-mean-times-h"], variants["patch-mean"] * size / 2.0
    )


def test_spearman_handles_ties_and_rejects_degenerate_input() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x * 10.0) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert spearman(torch.tensor([1.0, 1.0, 2.0]), torch.tensor([1.0, 2.0, 3.0])) == (
        pytest.approx(math.sqrt(3.0) / 2.0)
    )
    for bad in (
        (torch.tensor([1.0]), torch.tensor([1.0])),
        (x, torch.ones(4)),
        (x, torch.tensor([1.0, math.nan, 2.0, 3.0])),
        (x, torch.ones(3)),
    ):
        with pytest.raises(ValueError):
            spearman(*bad)


def test_quantile_curve_is_monotone_and_calibrated_on_its_own_data() -> None:
    generator = torch.Generator().manual_seed(11)
    indicator = torch.rand(2_000, generator=generator, dtype=_DTYPE)
    error = indicator * torch.rand(2_000, generator=generator, dtype=_DTYPE)
    curve = fit_quantile_curve(indicator, error, quantile=0.9, bins=20)

    assert len(curve.values) == 20
    assert list(curve.values) == sorted(curve.values)
    assert curve.coverage(indicator, error) == pytest.approx(0.9, abs=0.02)
    assert float(curve.predict(torch.tensor([10.0]))[0]) == curve.values[-1]
    noisy = fit_quantile_curve(
        torch.arange(6.0, dtype=_DTYPE),
        torch.tensor([3.0, 1.0, 2.0, 0.5, 4.0, 5.0], dtype=_DTYPE),
        quantile=0.5,
        bins=6,
    )
    assert list(noisy.values) == sorted(noisy.values)
    with pytest.raises(ValueError):
        QuantileCurve(0.9, (1.0, 2.0), (2.0, 1.0))
    with pytest.raises(ValueError):
        QuantileCurve(0.9, (), ())
    with pytest.raises(ValueError):
        fit_quantile_curve(indicator, error, quantile=1.0)
    with pytest.raises(ValueError):
        fit_quantile_curve(indicator[:5], error[:5], bins=10)
    with pytest.raises(ValueError):
        fit_quantile_curve(indicator, error[:10])


def test_residual_inputs_are_validated() -> None:
    model = _model(0.0)
    inputs = _inputs()
    with pytest.raises(ValueError):
        equilibrium_residual(model, torch.zeros(3, dtype=_DTYPE), **inputs)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        equilibrium_residual(model, _points(), **{**inputs, "length_scale": 0.0})  # type: ignore[arg-type]
    bad = MaterialSpec(
        "bad",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(0.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.23, Unit.ONE),
        },
    )
    with pytest.raises(ValueError):
        equilibrium_residual(model, _points(), **{**inputs, "material": bad})  # type: ignore[arg-type]


def test_latency_reports_median_and_p99_and_restores_threads() -> None:
    ticks = iter(range(0, 10_000_000_000, 1_000_000))
    before = torch.get_num_threads()
    result = measure_latency(
        lambda: None, repeats=5, warmup=1, threads=1, clock=lambda: next(ticks)
    )

    assert result.repeats == 5 and result.threads == 1
    assert result.median_seconds == pytest.approx(1e-3)
    assert result.p99_seconds == pytest.approx(1e-3)
    assert torch.get_num_threads() == before
    assert speedup(2.0, 0.01) == pytest.approx(200.0)
    with pytest.raises(ValueError):
        speedup(0.0, 1.0)
    with pytest.raises(ValueError):
        measure_latency(lambda: None, repeats=0)
    with pytest.raises(TypeError):
        measure_latency(None)  # type: ignore[arg-type]


def test_amortized_break_even_queries_formula_and_validation() -> None:
    summary = amortized_break_even_queries(
        training_wall_seconds=1000.0,
        calibration_cases=20,
        fem_seconds=100.0,
        inference_seconds=0.1,
        threads=1,
        precision="float64",
        device="cpu",
        dof_count=1938,
    )
    expected = (1000.0 + 20 * 100.0) / (100.0 - 0.1)
    assert summary.break_even_queries == pytest.approx(expected)
    assert summary.calibration_only_break_even_queries == pytest.approx(
        (20 * 100.0) / (100.0 - 0.1)
    )
    assert summary.online_speedup == pytest.approx(1000.0)
    doc = summary.document()
    assert doc["dof_count"] == 1938
    assert doc["break_even_queries"] == pytest.approx(expected)

    no_train = amortized_break_even_queries(
        training_wall_seconds=None,
        calibration_cases=64,
        fem_seconds=109.9,
        inference_seconds=0.109,
    )
    assert no_train.break_even_queries is None
    assert no_train.calibration_only_break_even_queries > 64.0

    with pytest.raises(ValueError):
        amortized_break_even_queries(
            training_wall_seconds=10.0,
            calibration_cases=5,
            fem_seconds=0.1,
            inference_seconds=0.2,
        )
    with pytest.raises(ValueError):
        amortized_break_even_queries(
            training_wall_seconds=-1.0,
            calibration_cases=5,
            fem_seconds=10.0,
            inference_seconds=0.1,
        )
    with pytest.raises(TypeError):
        amortized_break_even_queries(
            training_wall_seconds=None,
            calibration_cases=True,  # type: ignore[arg-type]
            fem_seconds=10.0,
            inference_seconds=0.1,
        )


def test_boundary_traction_residual_uses_outward_normals_and_free_components() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=_DTYPE
    )
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
    a, b, c, d = 3.0, 4.0, 5.0, 6.0
    stress = torch.tensor([[a, b], [c, d]], dtype=_DTYPE).expand(4, 2, 2).clone()
    facets = ((0, 1), (1, 3), (3, 2), (2, 0))
    free = ((0,), (0, 1), (), (1,))
    traction = boundary_traction_residual(
        stress, vertices, cells, facets, free, elastic_modulus=10.0
    )
    right = math.hypot(a, c)

    assert traction.tolist() == pytest.approx(
        [(b + c) / 20.0, (b + right) / 20.0, c / 20.0, right / 20.0]
    )
    variants = indicator_variants(
        lambda x: torch.zeros(x.shape[0], dtype=_DTYPE),
        vertices,
        cells,
        length_scale=1.0,
        traction=traction,
    )
    assert all(torch.allclose(v, traction) for v in variants.values())
    with pytest.raises(ValueError):
        indicator_variants(
            lambda x: x[:, 0], vertices, cells, length_scale=1.0, traction=traction[:3]
        )
    for bad in (
        {"vertex_stress": stress[:3]},
        {"free_components": free[:3]},
        {"elastic_modulus": 0.0},
        {"facets": ((0, 3), *facets[1:])},
        {"free_components": ((2,), *free[1:])},
    ):
        arguments = {
            "vertex_stress": stress,
            "vertices": vertices,
            "cells": cells,
            "facets": facets,
            "free_components": free,
            "elastic_modulus": 10.0,
            **bad,
        }
        modulus = arguments.pop("elastic_modulus")
        with pytest.raises(ValueError):
            boundary_traction_residual(**arguments, elastic_modulus=modulus)  # type: ignore[arg-type]
