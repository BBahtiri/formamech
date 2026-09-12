"""Strong and weak equilibrium residuals for S12."""

from __future__ import annotations

import torch

from piginot.objectives.models import CoordinateConversion, NondimensionalScales


def strong_equilibrium_residual(
    served_stress: torch.Tensor,
    coordinates: torch.Tensor,
    body_force: torch.Tensor,
    scales: NondimensionalScales,
    conversion: CoordinateConversion,
) -> torch.Tensor:
    """Return ``(Div_X P_served + B)/(P0/L0)`` at point-local queries."""

    _field(served_stress, "stress", trailing=(2, 2))
    _field(coordinates, "coordinates", trailing=(2,))
    _field(body_force, "body_force", trailing=(2,))
    if served_stress.shape[:-2] != coordinates.shape[:-1]:
        raise ValueError("stress and coordinates must align")
    if body_force.shape != coordinates.shape:
        raise ValueError("body_force and coordinates must align")
    _same_context(served_stress, coordinates, "stress", "coordinates")
    _same_context(body_force, coordinates, "body_force", "coordinates")
    if not coordinates.requires_grad:
        raise ValueError("coordinates must require gradients for strong equilibrium")
    divergence_components: list[torch.Tensor] = []
    for row in range(2):
        divergence = torch.zeros_like(coordinates[..., 0])
        for column in range(2):
            component = served_stress[..., row, column]
            if not component.requires_grad:
                raise ValueError("served stress must remain connected to coordinates")
            gradient = torch.autograd.grad(
                component,
                coordinates,
                grad_outputs=torch.ones_like(component),
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )[0]
            divergence = divergence + gradient[..., column] / conversion.to_mm[column]
        divergence_components.append(divergence)
    divergence_tensor = torch.stack(tuple(divergence_components), dim=-1)
    return (divergence_tensor + body_force) / scales.body_force_mpa_per_mm


def weak_equilibrium_residual(
    *,
    stress: torch.Tensor,
    body_force: torch.Tensor,
    test_values: torch.Tensor,
    test_gradients: torch.Tensor,
    volume_weights: torch.Tensor,
    scales: NondimensionalScales,
    traction: torch.Tensor | None = None,
    boundary_test_values: torch.Tensor | None = None,
    boundary_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Assemble Galerkin residuals for scalar test functions and vector balance."""

    _field(stress, "stress", trailing=(2, 2))
    _field(body_force, "body_force", trailing=(2,))
    _field(test_values, "test_values", trailing=())
    _field(test_gradients, "test_gradients", trailing=(2,))
    _field(volume_weights, "volume_weights", trailing=())
    if stress.ndim != 4 or body_force.ndim != 3:
        raise ValueError("stress and body_force must have batch and point dimensions")
    if test_values.ndim != 3:
        raise ValueError("test_values must have shape (batch, points, tests)")
    if test_gradients.ndim != 4:
        raise ValueError("test_gradients must have shape (batch, points, tests, 2)")
    if volume_weights.ndim != 2:
        raise ValueError("volume_weights must have shape (batch, points)")
    batch_points = stress.shape[:2]
    if body_force.shape[:2] != batch_points or test_values.shape[:2] != batch_points:
        raise ValueError("volume fields must share batch and point dimensions")
    if test_gradients.shape[:3] != test_values.shape:
        raise ValueError("test_gradients must align with test_values")
    if volume_weights.shape != batch_points:
        raise ValueError("volume_weights must align with volume points")
    for name, value in (
        ("body_force", body_force),
        ("test_values", test_values),
        ("test_gradients", test_gradients),
        ("volume_weights", volume_weights),
    ):
        _same_context(stress, value, "stress", name)
    internal = torch.einsum("bqij,bqaj,bq->bai", stress, test_gradients, volume_weights)
    external = torch.einsum("bqi,bqa,bq->bai", body_force, test_values, volume_weights)
    boundary_values = (traction, boundary_test_values, boundary_weights)
    if any(value is not None for value in boundary_values):
        if any(value is None for value in boundary_values):
            raise ValueError(
                "traction, boundary test values and weights are all required"
            )
        assert traction is not None
        assert boundary_test_values is not None
        assert boundary_weights is not None
        _field(traction, "traction", trailing=(2,))
        _field(boundary_test_values, "boundary_test_values", trailing=())
        _field(boundary_weights, "boundary_weights", trailing=())
        if (
            traction.ndim != 3
            or boundary_test_values.ndim != 3
            or boundary_weights.ndim != 2
        ):
            raise ValueError(
                "boundary fields must have batch, point and test dimensions"
            )
        if traction.shape[:2] != boundary_test_values.shape[:2]:
            raise ValueError("boundary traction and test values must align")
        if boundary_test_values.shape[0] != stress.shape[0]:
            raise ValueError("volume and boundary fields must share a batch dimension")
        if boundary_test_values.shape[2] != test_values.shape[2]:
            raise ValueError("volume and boundary fields must share test functions")
        if boundary_weights.shape != traction.shape[:2]:
            raise ValueError("boundary_weights must align with boundary points")
        for name, value in (
            ("traction", traction),
            ("boundary_test_values", boundary_test_values),
            ("boundary_weights", boundary_weights),
        ):
            _same_context(stress, value, "stress", name)
        external = external + torch.einsum(
            "bri,bra,br->bai", traction, boundary_test_values, boundary_weights
        )
    return (internal - external) / scales.weak_force_per_thickness_n_per_mm


def _field(value: torch.Tensor, label: str, *, trailing: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.layout is not torch.strided or value.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise ValueError(f"{label} must be a dense floating tensor")
    if trailing and (
        value.ndim < len(trailing) or tuple(value.shape[-len(trailing) :]) != trailing
    ):
        raise ValueError(f"{label} must end in shape {trailing}")
    if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{label} must be finite and non-empty")


def _same_context(
    left: torch.Tensor, right: torch.Tensor, left_name: str, right_name: str
) -> None:
    if left.dtype != right.dtype or left.device != right.device:
        raise ValueError(f"{left_name} and {right_name} must share dtype and device")


__all__ = ("strong_equilibrium_residual", "weak_equilibrium_residual")
