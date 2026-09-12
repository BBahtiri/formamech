"""Point-local differentiable kinematics for neural-operator queries."""

from __future__ import annotations

import torch

from piginot.objectives.models import CoordinateConversion


def displacement_gradient(
    displacement: torch.Tensor,
    coordinates: torch.Tensor,
    conversion: CoordinateConversion,
) -> torch.Tensor:
    """Return ``du_i/dX_J`` in canonical mm coordinates.

    The S11 decoder is point-local in its query axis. Summing each output component
    before differentiation therefore obtains the diagonal point Jacobians without
    materializing a quadratic point-by-point Jacobian.
    """

    _vector_field(displacement, "displacement")
    _vector_field(coordinates, "coordinates")
    if displacement.shape != coordinates.shape:
        raise ValueError("displacement and coordinates must have identical shapes")
    if (
        displacement.dtype != coordinates.dtype
        or displacement.device != coordinates.device
    ):
        raise ValueError("displacement and coordinates must share dtype and device")
    if not coordinates.requires_grad:
        raise ValueError(
            "coordinates must require gradients for displacement kinematics"
        )
    rows: list[torch.Tensor] = []
    for component in range(2):
        value = displacement[..., component]
        if not value.requires_grad:
            raise ValueError("displacement must remain connected to coordinates")
        gradient = torch.autograd.grad(
            value,
            coordinates,
            grad_outputs=torch.ones_like(value),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        rows.append(gradient / coordinates.new_tensor(conversion.to_mm))
    return torch.stack(tuple(rows), dim=-2)


def _vector_field(value: torch.Tensor, label: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if (
        value.layout is not torch.strided
        or value.dtype not in (torch.float32, torch.float64)
        or value.ndim < 2
        or value.shape[-1] != 2
    ):
        raise ValueError(f"{label} must be a dense floating vector field")
    if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{label} must be finite and non-empty")


__all__ = ("displacement_gradient",)
