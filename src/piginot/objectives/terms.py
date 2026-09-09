"""Differentiable, nondimensional data and local physics residuals."""

from __future__ import annotations

import torch

from piginot.contracts import MaterialSpec
from piginot.objectives.models import NondimensionalScales
from piginot.physics import first_piola, plane_stress_closure


def supervised_displacement_residual(
    predicted: torch.Tensor,
    target: torch.Tensor,
    scales: NondimensionalScales,
) -> torch.Tensor:
    _matching_fields(predicted, target, "displacement", trailing=(2,))
    return (predicted - target) / scales.displacement_mm


def supervised_stress_residual(
    predicted: torch.Tensor,
    target: torch.Tensor,
    scales: NondimensionalScales,
) -> torch.Tensor:
    _matching_fields(predicted, target, "stress", trailing=(2, 2))
    return (predicted - target) / scales.stress_mpa


def dirichlet_residual(
    predicted: torch.Tensor,
    prescribed: torch.Tensor,
    component_mask: torch.Tensor,
    scales: NondimensionalScales,
) -> torch.Tensor:
    _matching_fields(predicted, prescribed, "Dirichlet displacement", trailing=(2,))
    if not isinstance(component_mask, torch.Tensor):
        raise TypeError("component_mask must be a torch.Tensor")
    if component_mask.dtype is not torch.bool or component_mask.layout is not torch.strided:
        raise ValueError("component_mask must be a dense boolean tensor")
    if component_mask.shape != predicted.shape or component_mask.device != predicted.device:
        raise ValueError("component_mask must align with predicted displacement")
    if not bool(component_mask.any().item()):
        raise ValueError("component_mask must select at least one Dirichlet component")
    return ((predicted - prescribed) / scales.displacement_mm)[component_mask]


def traction_free_residual(
    served_stress: torch.Tensor,
    outward_normals: torch.Tensor,
    scales: NondimensionalScales,
) -> torch.Tensor:
    _field(served_stress, "served in-plane stress", trailing=(2, 2))
    _field(outward_normals, "outward_normals", trailing=(2,))
    if served_stress.shape[:-2] != outward_normals.shape[:-1]:
        raise ValueError("outward_normals must align with served stress points")
    _same_context(served_stress, outward_normals, "served_stress", "outward_normals")
    normal_length = torch.linalg.vector_norm(outward_normals, dim=-1)
    if not bool(torch.allclose(normal_length, torch.ones_like(normal_length), atol=1e-10, rtol=1e-10)):
        raise ValueError("outward_normals must be unit vectors")
    traction = torch.einsum("...ij,...j->...i", served_stress, outward_normals)
    return traction / scales.stress_mpa


def constitutive_residual(
    served_stress: torch.Tensor,
    displacement_gradient: torch.Tensor,
    material: MaterialSpec,
    scales: NondimensionalScales,
) -> torch.Tensor:
    if not isinstance(served_stress, torch.Tensor) or served_stress.shape[-2:] != (2, 2):
        raise ValueError("served stress must contain in-plane (2, 2) components only")
    _field(served_stress, "served in-plane stress", trailing=(2, 2))
    _field(displacement_gradient, "displacement_gradient", trailing=(2, 2))
    if served_stress.shape != displacement_gradient.shape:
        raise ValueError("served stress and displacement gradient must align")
    _same_context(served_stress, displacement_gradient, "served_stress", "displacement_gradient")
    identity = torch.eye(2, dtype=served_stress.dtype, device=served_stress.device)
    deformation_2d = identity + displacement_gradient
    thickness_stretch = plane_stress_closure(deformation_2d, material)
    deformation_3d = torch.zeros(
        (*deformation_2d.shape[:-2], 3, 3),
        dtype=deformation_2d.dtype,
        device=deformation_2d.device,
    )
    deformation_3d[..., :2, :2] = deformation_2d
    deformation_3d[..., 2, 2] = thickness_stretch
    material_stress = first_piola(deformation_3d, material)[..., :2, :2]
    return (served_stress - material_stress) / scales.stress_mpa


def angular_momentum_residual(
    served_stress: torch.Tensor,
    deformation_gradient_2d: torch.Tensor,
    scales: NondimensionalScales,
) -> torch.Tensor:
    _field(served_stress, "served in-plane stress", trailing=(2, 2))
    _field(deformation_gradient_2d, "deformation_gradient_2d", trailing=(2, 2))
    if served_stress.shape != deformation_gradient_2d.shape:
        raise ValueError("served stress and deformation gradient must align")
    _same_context(served_stress, deformation_gradient_2d, "served_stress", "deformation_gradient_2d")
    moment = served_stress @ deformation_gradient_2d.transpose(-2, -1)
    independent_skew = 0.5 * (moment[..., 0, 1] - moment[..., 1, 0])
    return independent_skew.unsqueeze(-1) / scales.stress_mpa


def _matching_fields(
    left: torch.Tensor,
    right: torch.Tensor,
    label: str,
    *,
    trailing: tuple[int, ...],
) -> None:
    _field(left, f"predicted {label}", trailing=trailing)
    _field(right, f"target {label}", trailing=trailing)
    if left.shape != right.shape:
        raise ValueError(f"predicted and target {label} must have identical shapes")
    _same_context(left, right, "predicted", "target")


def _field(value: torch.Tensor, label: str, *, trailing: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.layout is not torch.strided or value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{label} must be a dense floating tensor")
    if value.ndim < len(trailing) or tuple(value.shape[-len(trailing) :]) != trailing:
        raise ValueError(f"{label} must end in shape {trailing}")
    if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{label} must be finite and non-empty")


def _same_context(left: torch.Tensor, right: torch.Tensor, left_name: str, right_name: str) -> None:
    if left.dtype != right.dtype or left.device != right.device:
        raise ValueError(f"{left_name} and {right_name} must share dtype and device")


__all__ = (
    "angular_momentum_residual",
    "constitutive_residual",
    "dirichlet_residual",
    "supervised_displacement_residual",
    "supervised_stress_residual",
    "traction_free_residual",
)
