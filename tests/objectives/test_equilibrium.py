from __future__ import annotations

import pytest
import torch

from piginot.objectives import (
    CoordinateConversion,
    NondimensionalScales,
    displacement_gradient,
    strong_equilibrium_residual,
    weak_equilibrium_residual,
)

SCALES = NondimensionalScales(10.0, 2.0, 100.0)


def _stress_field(
    coordinates: torch.Tensor, conversion: CoordinateConversion
) -> torch.Tensor:
    physical = coordinates * coordinates.new_tensor(conversion.to_mm)
    x_coordinate = physical[..., 0]
    y_coordinate = physical[..., 1]
    return torch.stack(
        (
            2.0 * x_coordinate,
            3.0 * y_coordinate,
            -x_coordinate,
            4.0 * y_coordinate,
        ),
        dim=-1,
    ).reshape(*coordinates.shape[:-1], 2, 2)


def test_displacement_gradient_uses_the_same_per_axis_coordinate_conversion() -> None:
    coordinates = torch.tensor(
        [[[2.0, 0.3], [7.0, 0.4]]], dtype=torch.float64, requires_grad=True
    )
    conversion = CoordinateConversion((0.1, 1.0))
    physical = coordinates * coordinates.new_tensor(conversion.to_mm)
    displacement = torch.stack(
        (
            2.0 * physical[..., 0] + 3.0 * physical[..., 1],
            -physical[..., 0] + 4.0 * physical[..., 1],
        ),
        dim=-1,
    )

    gradient = displacement_gradient(displacement, coordinates, conversion)

    expected = torch.tensor(
        [[2.0, 3.0], [-1.0, 4.0]], dtype=torch.float64
    ).expand(1, 2, 2, 2)
    assert torch.equal(gradient, expected)


def test_strong_equilibrium_applies_componentwise_coordinate_conversion() -> None:
    coordinates = torch.tensor(
        [[[0.2, 0.3], [0.7, 0.4]]], dtype=torch.float64, requires_grad=True
    )
    conversion = CoordinateConversion((1.0, 1.0))
    stress = _stress_field(coordinates, conversion)
    body_force = torch.tensor([[[-5.0, -3.0], [-5.0, -3.0]]], dtype=torch.float64)

    residual = strong_equilibrium_residual(
        stress, coordinates, body_force, SCALES, conversion
    )

    assert residual.abs().max().item() < 1.0e-12


def test_strong_residual_is_invariant_to_x_only_coordinate_unit_scaling() -> None:
    base = torch.tensor(
        [[[0.2, 0.3], [0.7, 0.4]]], dtype=torch.float64, requires_grad=True
    )
    scaled = torch.tensor(
        [[[2.0, 0.3], [7.0, 0.4]]], dtype=torch.float64, requires_grad=True
    )
    base_conversion = CoordinateConversion((1.0, 1.0))
    scaled_conversion = CoordinateConversion((0.1, 1.0))
    body_force = torch.tensor([[[-4.5, -2.5], [-4.5, -2.5]]], dtype=torch.float64)

    base_residual = strong_equilibrium_residual(
        _stress_field(base, base_conversion),
        base,
        body_force,
        SCALES,
        base_conversion,
    )
    scaled_residual = strong_equilibrium_residual(
        _stress_field(scaled, scaled_conversion),
        scaled,
        body_force,
        SCALES,
        scaled_conversion,
    )

    assert torch.equal(base_residual, scaled_residual)
    assert base_residual.abs().max().item() > 0.0


def test_weak_galerkin_balance_includes_volume_force_and_boundary_traction() -> None:
    stress = torch.tensor([[[[2.0, 1.0], [3.0, 4.0]]]], dtype=torch.float64)
    body_force = torch.zeros((1, 1, 2), dtype=torch.float64)
    test_values = torch.ones((1, 1, 1), dtype=torch.float64)
    test_gradients = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
    volume_weights = torch.ones((1, 1), dtype=torch.float64)
    traction = torch.tensor([[[2.0, 3.0]]], dtype=torch.float64)
    boundary_test_values = torch.ones((1, 1, 1), dtype=torch.float64)
    boundary_weights = torch.ones((1, 1), dtype=torch.float64)

    exact = weak_equilibrium_residual(
        stress=stress,
        body_force=body_force,
        test_values=test_values,
        test_gradients=test_gradients,
        volume_weights=volume_weights,
        traction=traction,
        boundary_test_values=boundary_test_values,
        boundary_weights=boundary_weights,
        scales=SCALES,
    )
    perturbed = weak_equilibrium_residual(
        stress=stress,
        body_force=body_force,
        test_values=test_values,
        test_gradients=test_gradients,
        volume_weights=volume_weights,
        traction=-traction,
        boundary_test_values=boundary_test_values,
        boundary_weights=boundary_weights,
        scales=SCALES,
    )

    assert exact.shape == (1, 1, 2)
    assert exact.abs().max().item() < 1.0e-12
    assert perturbed.abs().max().item() > 1.0e-3


def test_weak_form_can_omit_natural_boundary_integral() -> None:
    residual = weak_equilibrium_residual(
        stress=torch.zeros((1, 2, 2, 2), dtype=torch.float64),
        body_force=torch.zeros((1, 2, 2), dtype=torch.float64),
        test_values=torch.ones((1, 2, 1), dtype=torch.float64),
        test_gradients=torch.ones((1, 2, 1, 2), dtype=torch.float64),
        volume_weights=torch.ones((1, 2), dtype=torch.float64),
        scales=SCALES,
    )

    assert torch.equal(residual, torch.zeros((1, 1, 2), dtype=torch.float64))


def test_equilibrium_shapes_fail_before_einsum_broadcasting() -> None:
    coordinates = torch.ones((1, 2, 2), dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="stress"):
        strong_equilibrium_residual(
            torch.ones((1, 2, 3, 3), dtype=torch.float64),
            coordinates,
            torch.ones((1, 2, 2), dtype=torch.float64),
            SCALES,
            CoordinateConversion((1.0, 1.0)),
        )
    with pytest.raises(ValueError, match="test_gradients"):
        weak_equilibrium_residual(
            stress=torch.zeros((1, 2, 2, 2), dtype=torch.float64),
            body_force=torch.zeros((1, 2, 2), dtype=torch.float64),
            test_values=torch.ones((1, 2, 1), dtype=torch.float64),
            test_gradients=torch.ones((1, 2, 2), dtype=torch.float64),
            volume_weights=torch.ones((1, 2), dtype=torch.float64),
            scales=SCALES,
        )
