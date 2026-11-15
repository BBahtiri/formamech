"""Physical boundary integration and gradient checks for the pilot objective."""

from dataclasses import replace

import pytest
import torch

from piginot.contracts import CanonicalMesh, Unit
from piginot.objectives.free_traction import (
    boundary_quadrature,
    boundary_stress_scales,
    free_traction_loss,
    regional_rms,
)


@pytest.fixture
def rectangle():
    return CanonicalMesh(
        mesh_id="traction-rectangle",
        nodes=((0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)),
        cells=((0, 1, 2), (0, 2, 3)),
        cell_type="triangle",
        boundary_facets=((1, 0), (2, 1), (2, 3), (3, 0)),
        semantic_tags={"bottom": (0,), "right": (1,), "top": (2,), "left": (3,)},
        region_ids=(1, 1),
        unit=Unit.MM,
        orientation="counterclockwise",
        topology_signature="rectangle",
        quality_metrics={},
        source_provenance={},
    )


def test_actual_material_normals_and_length_weights_ignore_facet_order(rectangle):
    q = boundary_quadrature(rectangle, ("left", "right", "bottom", "top"))
    expected = torch.tensor(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], dtype=torch.float64
    )
    torch.testing.assert_close(q.normals, expected.repeat_interleave(2, dim=0))
    torch.testing.assert_close(
        q.weights.reshape(4, 2).sum(1),
        torch.tensor([2.0, 2.0, 4.0, 4.0], dtype=torch.float64),
    )
    assert q.points.shape == (8, 2)
    assert not any(
        torch.equal(point, torch.tensor(node))
        for point in q.points
        for node in rectangle.nodes
    )


def test_constant_stress_traction_uses_tensor_times_reference_normal(rectangle):
    q = boundary_quadrature(rectangle, ("right", "top"))
    stress = torch.tensor([[2.0, 5.0], [3.0, 7.0]], dtype=torch.float64).expand(4, 2, 2)
    scales = torch.tensor([1.0, 2.0], dtype=torch.float64)
    # Right: (2,3); top: (5,7). Region means are weighted equally.
    assert float(free_traction_loss(stress, q, scales)) == pytest.approx(
        (13 + 74 / 4) / 2
    )


def test_zero_traction_and_material_unit_scaling(rectangle):
    q = boundary_quadrature(rectangle, ("top",))
    stress = torch.tensor([[2.0, 0.0], [0.0, 0.0]], dtype=torch.float64).expand(2, 2, 2)
    scales = torch.ones(1, dtype=torch.float64)
    assert free_traction_loss(stress, q, scales) == 0
    nonzero = stress + 1
    torch.testing.assert_close(
        free_traction_loss(nonzero, q, scales),
        free_traction_loss(100 * nonzero, q, 100 * scales),
    )


def test_surface_loss_gradient_matches_finite_differences(rectangle):
    q = boundary_quadrature(rectangle, ("top", "right"))
    stress = (
        torch.arange(16, dtype=torch.float64).reshape(4, 2, 2).div(10).requires_grad_()
    )
    scales = torch.tensor([2.0, 3.0], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda p: free_traction_loss(p, q, scales), (stress,)
    )


def test_fixed_scales_use_regional_tensor_rms_and_positive_floor(rectangle):
    q = boundary_quadrature(rectangle, ("top", "right"))
    stress = torch.ones((4, 2, 2), dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(
        regional_rms(stress, q), torch.full((2,), 2.0, dtype=torch.float64)
    )
    scales = boundary_stress_scales(stress, q, minimum=3.0)
    torch.testing.assert_close(scales, torch.full((2,), 3.0, dtype=torch.float64))
    assert not scales.requires_grad


def test_chunk_weights_preserve_the_global_surface_integral(rectangle):
    from piginot.objectives.free_traction import traction_point_weights

    q = boundary_quadrature(rectangle, ("right", "top"))
    stress = torch.arange(16, dtype=torch.float64).reshape(4, 2, 2)
    scales = torch.tensor([2.0, 3.0], dtype=torch.float64)
    weights = traction_point_weights(q, scales)
    traction = (stress @ q.normals.unsqueeze(-1)).squeeze(-1)
    chunked = sum(
        (traction[a:b].square().sum(1) * weights[a:b]).sum()
        for a, b in ((0, 1), (1, 3), (3, 4))
    )
    torch.testing.assert_close(chunked, free_traction_loss(stress, q, scales))


@pytest.mark.parametrize("regions", [(), ("absent",), ("top", "top")])
def test_missing_or_repeated_regions_refuse(rectangle, regions):
    with pytest.raises(ValueError):
        boundary_quadrature(rectangle, regions)


def test_interior_or_degenerate_facet_refuses(rectangle):
    interior = replace(
        rectangle, boundary_facets=((0, 2),), semantic_tags={"inside": (0,)}
    )
    with pytest.raises(ValueError):
        boundary_quadrature(interior, ("inside",))
    bad = replace(rectangle, boundary_facets=((0, 0),), semantic_tags={"bad": (0,)})
    with pytest.raises(ValueError):
        boundary_quadrature(bad, ("bad",))


@pytest.mark.parametrize(
    "kind", ["zero_scale", "nan", "wrong_shape", "dtype", "learnable_scale"]
)
def test_invalid_boundary_inputs_refuse(rectangle, kind):
    q = boundary_quadrature(rectangle, ("top",))
    stress = torch.ones((2, 2, 2), dtype=torch.float64)
    scales = torch.ones(1, dtype=torch.float64)
    if kind == "zero_scale":
        scales = scales * 0
    if kind == "nan":
        stress = stress * float("nan")
    if kind == "wrong_shape":
        stress = stress[:1]
    if kind == "dtype":
        stress = stress.float()
    if kind == "learnable_scale":
        scales.requires_grad_()
    with pytest.raises(ValueError):
        free_traction_loss(stress, q, scales)
