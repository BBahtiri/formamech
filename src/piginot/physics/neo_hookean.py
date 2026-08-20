"""Compressible Neo-Hookean energy, stress, and consistent tangent."""

from __future__ import annotations

import torch

from piginot.contracts import MaterialSpec
from piginot.physics._validation import (
    determinant_is_admissible,
    require_admissible,
    validate_material,
    validate_tensor,
)


def admissibility(deformation: torch.Tensor, material: MaterialSpec) -> bool:
    """Return whether all 3D deformation gradients are in the strict J domain."""

    validate_material(material)
    deformation = validate_tensor(deformation, (3, 3))
    determinant = torch.linalg.det(deformation)
    return bool(determinant_is_admissible(determinant).all().item())


def energy_density(deformation: torch.Tensor, material: MaterialSpec) -> torch.Tensor:
    """Evaluate compressible Neo-Hookean strain-energy density in MPa."""

    parameters = validate_material(material)
    deformation = validate_tensor(deformation, (3, 3))
    determinant = torch.linalg.det(deformation)
    require_admissible(determinant)

    log_j = torch.log(determinant)
    trace_c = torch.sum(deformation * deformation, dim=(-2, -1))
    return (
        0.5 * parameters.mu * (trace_c - 3.0)
        - parameters.mu * log_j
        + 0.5 * parameters.lam * log_j.square()
    )


def first_piola(deformation: torch.Tensor, material: MaterialSpec) -> torch.Tensor:
    """Evaluate the first Piola--Kirchhoff stress in MPa."""

    parameters = validate_material(material)
    deformation = validate_tensor(deformation, (3, 3))
    determinant = torch.linalg.det(deformation)
    require_admissible(determinant)

    inverse_transpose = torch.linalg.inv(deformation).transpose(-2, -1)
    coefficient = parameters.lam * torch.log(determinant) - parameters.mu
    stress: torch.Tensor = (
        parameters.mu * deformation + coefficient[..., None, None] * inverse_transpose
    )
    return stress


def consistent_tangent(
    deformation: torch.Tensor, material: MaterialSpec
) -> torch.Tensor:
    """Evaluate dP_iJ/dF_kL with layout ``(..., i, J, k, L)``."""

    parameters = validate_material(material)
    deformation = validate_tensor(deformation, (3, 3))
    determinant = torch.linalg.det(deformation)
    require_admissible(determinant)

    inverse_transpose = torch.linalg.inv(deformation).transpose(-2, -1)
    coefficient = parameters.lam * torch.log(determinant) - parameters.mu
    identity = torch.eye(3, dtype=deformation.dtype, device=deformation.device)
    material_term = parameters.mu * torch.einsum("ik,jl->ijkl", identity, identity)
    volumetric_term = parameters.lam * torch.einsum(
        "...ij,...kl->...ijkl", inverse_transpose, inverse_transpose
    )
    geometric_term = coefficient[..., None, None, None, None] * torch.einsum(
        "...il,...kj->...ijkl", inverse_transpose, inverse_transpose
    )
    return material_term + volumetric_term - geometric_term


__all__ = (
    "admissibility",
    "consistent_tangent",
    "energy_density",
    "first_piola",
)
