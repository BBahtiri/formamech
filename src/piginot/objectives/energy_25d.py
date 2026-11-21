"""2.5D finite-strain Deep Ritz potential-energy objectives (plane strain & axisymmetric).

Extends FormaMech's exact 2D plane-stress potential energy to:
- **Plane strain** (``2d-plane-strain-finite-strain``): ``F_33 == 1``, eliminating
  nested plane-stress closure Newton iterations while preserving full 3D
  hyperelastic constitutive models.
- **Axisymmetric** (``2d-axisymmetric-finite-strain``): cylindrical coordinates
  ``(R, Z)`` with ``R > 0``, hoop stretch ``F_tt = 1 + u_r / R``, and exact
  circumferential volume element ``2 * pi * R dR dZ``.
"""

from __future__ import annotations

import math

import torch

from piginot.contracts import MaterialSpec
from piginot.objectives.energy import PotentialEnergyConfig, PotentialEnergyTerms
from piginot.physics import energy_density, reference_modulus


def _embed_3d_diagonal(deformation_2d: torch.Tensor, f33: torch.Tensor) -> torch.Tensor:
    f3d = torch.zeros(
        (*deformation_2d.shape[:-2], 3, 3),
        dtype=deformation_2d.dtype,
        device=deformation_2d.device,
    )
    f3d[..., :2, :2] = deformation_2d
    f3d[..., 2, 2] = f33
    return f3d


def plane_strain_potential_energy(
    displacement_gradient: torch.Tensor,
    weights: torch.Tensor,
    material: MaterialSpec,
    *,
    config: PotentialEnergyConfig | None = None,
    reference_area: float | None = None,
) -> PotentialEnergyTerms:
    """Nondimensional plane-strain (``F_33 == 1``) finite-strain potential energy."""
    cfg = PotentialEnergyConfig() if config is None else config
    if displacement_gradient.ndim != 3 or displacement_gradient.shape[-2:] != (2, 2):
        raise ValueError("displacement_gradient must have shape (N, 2, 2)")
    if weights.ndim != 1 or weights.shape[0] != displacement_gradient.shape[0]:
        raise ValueError("weights must have shape (N,)")

    eye2 = torch.eye(
        2, dtype=displacement_gradient.dtype, device=displacement_gradient.device
    )
    f2d = eye2 + displacement_gradient
    det_f = f2d[:, 0, 0] * f2d[:, 1, 1] - f2d[:, 0, 1] * f2d[:, 1, 0]
    min_det = float(det_f.detach().min().item())
    inadmissible = det_f <= cfg.admissibility_floor
    inadmissible_count = int(inadmissible.sum().item())

    safe_f2d = torch.where(inadmissible[:, None, None], eye2.expand_as(f2d), f2d)
    f33 = torch.ones(f2d.shape[0], dtype=f2d.dtype, device=f2d.device)
    f3d = _embed_3d_diagonal(safe_f2d, f33)
    psi = energy_density(f3d, material)
    psi = torch.where(inadmissible, torch.zeros_like(psi), psi)
    area = (
        float(weights.detach().sum().item())
        if reference_area is None
        else reference_area
    )
    if area <= 0.0:
        raise ValueError("reference_area must be positive")
    modulus = reference_modulus(material)
    scale = modulus * area

    physical_energy = (weights * psi).sum()
    nondim_energy = physical_energy / scale
    barrier_deficit = torch.relu(cfg.barrier_threshold - det_f)
    barrier = cfg.barrier_weight * (
        (weights * (barrier_deficit * barrier_deficit)).sum() / area
    )
    total = nondim_energy + barrier
    return PotentialEnergyTerms(
        total=total,
        energy=nondim_energy,
        barrier=barrier,
        minimum_determinant=min_det,
        inadmissible_points=inadmissible_count,
    )


def axisymmetric_potential_energy(
    displacement: torch.Tensor,
    displacement_gradient: torch.Tensor,
    radii: torch.Tensor,
    weights: torch.Tensor,
    material: MaterialSpec,
    *,
    config: PotentialEnergyConfig | None = None,
    reference_volume: float | None = None,
) -> PotentialEnergyTerms:
    """Nondimensional axisymmetric finite-strain potential energy over ``2*pi*R dR dZ``.

    Args:
        displacement: ``(N, 2)`` tensor of ``(u_r, u_z)`` at quadrature points.
        displacement_gradient: ``(N, 2, 2)`` tensor ``[[du_r/dR, du_r/dZ], [du_z/dR, du_z/dZ]]``.
        radii: ``(N,)`` positive radial coordinates ``R > 0`` in mm.
        weights: ``(N,)`` 2D planar triangle quadrature weights ``dR dZ`` in mm^2.
    """
    cfg = PotentialEnergyConfig() if config is None else config
    if displacement.ndim != 2 or displacement.shape[-1] != 2:
        raise ValueError("displacement must have shape (N, 2)")
    if displacement_gradient.ndim != 3 or displacement_gradient.shape[-2:] != (2, 2):
        raise ValueError("displacement_gradient must have shape (N, 2, 2)")
    if radii.ndim != 1 or radii.shape[0] != displacement.shape[0]:
        raise ValueError("radii must have shape (N,)")
    if weights.ndim != 1 or weights.shape[0] != displacement.shape[0]:
        raise ValueError("weights must have shape (N,)")
    if bool((radii <= 0.0).any().item()):
        raise ValueError("axisymmetric quadrature radii must be strictly positive")

    eye2 = torch.eye(
        2, dtype=displacement_gradient.dtype, device=displacement_gradient.device
    )
    f2d = eye2 + displacement_gradient
    f_tt = 1.0 + displacement[:, 0] / radii
    det_2d = f2d[:, 0, 0] * f2d[:, 1, 1] - f2d[:, 0, 1] * f2d[:, 1, 0]
    det_3d = det_2d * f_tt
    min_det = float(det_3d.detach().min().item())
    inadmissible = (
        (det_2d <= cfg.admissibility_floor)
        | (f_tt <= cfg.admissibility_floor)
        | (det_3d <= cfg.admissibility_floor)
    )
    inadmissible_count = int(inadmissible.sum().item())

    safe_f2d = torch.where(inadmissible[:, None, None], eye2.expand_as(f2d), f2d)
    safe_ftt = torch.where(inadmissible, torch.ones_like(f_tt), f_tt)
    f3d = _embed_3d_diagonal(safe_f2d, safe_ftt)
    psi = energy_density(f3d, material)
    psi = torch.where(inadmissible, torch.zeros_like(psi), psi)

    vol_weights = (2.0 * math.pi) * radii * weights
    vol = (
        float(vol_weights.detach().sum().item())
        if reference_volume is None
        else reference_volume
    )
    if vol <= 0.0:
        raise ValueError("reference_volume must be positive")
    modulus = reference_modulus(material)
    scale = modulus * vol

    physical_energy = (vol_weights * psi).sum()
    nondim_energy = physical_energy / scale
    barrier_deficit = torch.relu(cfg.barrier_threshold - det_3d)
    barrier = cfg.barrier_weight * (
        (vol_weights * (barrier_deficit * barrier_deficit)).sum() / vol
    )
    total = nondim_energy + barrier
    return PotentialEnergyTerms(
        total=total,
        energy=nondim_energy,
        barrier=barrier,
        minimum_determinant=min_det,
        inadmissible_points=inadmissible_count,
    )


__all__ = (
    "axisymmetric_potential_energy",
    "plane_strain_potential_energy",
)
