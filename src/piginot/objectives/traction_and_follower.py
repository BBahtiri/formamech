"""Conservative dead tractions, non-conservative follower pressure, and DEM guards.

Enforces three strict continuum-mechanics rules for finite-strain hyperelasticity:

1. **Conservative dead tractions** (fixed nominal traction ``t_0`` on reference
   boundary patches ``Gamma_{0,N}`` of strictly positive length) enter the total
   potential energy via ``- int_{Gamma_{0,N}} t_0 . u dGamma``. Zero-measure
   point loads are explicitly refused (formulate as finite-width patch loads).
2. **Non-conservative open-boundary follower pressure** (``t = -p n`` tracking
   the deformed surface normal) has no scalar potential energy and is NEVER
   added to a potential-energy minimization loss. It is enforced strictly via
   the reference-configuration Piola traction residual
   ``P N + p cof(F_2D) N = 0`` (or discrete virtual work).
3. **Enclosed cavity pressure** (``-p Delta_Area(u)``) is permitted ONLY after
   verifying that the pressurized boundary facets form a closed 2-regular loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from piginot.contracts import CanonicalMesh, MaterialSpec
from piginot.objectives.energy import (
    PotentialEnergyConfig,
    PotentialEnergyTerms,
    TriangleDomain,
    plane_stress_potential_energy,
)
from piginot.objectives.free_traction import BoundaryQuadrature
from piginot.objectives.hierarchical_energy import deterministic_triangle_quadrature
from piginot.physics import reference_modulus


@dataclass(frozen=True, slots=True)
class QuadratureOvershootDiagnostic:
    """Multi-order Dunavant consistency check guarding against DEM energy collapse."""

    energy_degree_2: float
    energy_degree_5: float
    relative_discrepancy: float
    max_allowed_discrepancy: float
    passed: bool


def _young_scale(material: MaterialSpec) -> float:
    if material.properties.get("elastic_modulus") is not None:
        return float(material.properties["elastic_modulus"].value)
    return float(reference_modulus(material))


def dead_traction_potential_energy(
    displacement_gradient: torch.Tensor,
    volume_weights: torch.Tensor,
    material: MaterialSpec,
    *,
    config: PotentialEnergyConfig,
    reference_area: float,
    boundary_displacement: torch.Tensor,
    nominal_traction: torch.Tensor,
    boundary_weights: torch.Tensor,
    thickness: float = 1.0,
) -> PotentialEnergyTerms:
    """Total potential energy ``(int W dA - int t_0 . u dGamma) / (E A) + barrier``.

    Explicitly refuses zero-measure point loads (``sum(boundary_weights) <= 0``).
    """
    if boundary_displacement.ndim != 2 or boundary_displacement.shape[1] != 2:
        raise ValueError("boundary_displacement must have shape (b_points, 2)")
    if nominal_traction.shape != boundary_displacement.shape:
        raise ValueError("nominal_traction must align with boundary_displacement")
    if boundary_weights.shape != (boundary_displacement.shape[0],):
        raise ValueError("boundary_weights must have shape (b_points,)")
    if bool((boundary_weights < 0.0).any().item()) or not bool(
        torch.isfinite(boundary_weights).all().item()
    ):
        raise ValueError("boundary_weights must be finite and non-negative")
    total_len = float(boundary_weights.sum().item())
    if total_len <= 0.0:
        raise ValueError(
            "zero-measure point loads are unsupported; supply a finite-width "
            "distributed traction patch with positive boundary length"
        )
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("thickness must be finite and positive")

    base = plane_stress_potential_energy(
        displacement_gradient,
        volume_weights,
        material,
        config=config,
        reference_area=reference_area,
    )
    young = _young_scale(material)
    ext_work = (
        torch.sum(
            torch.sum(nominal_traction * boundary_displacement, dim=-1)
            * boundary_weights
        )
        * thickness
        / (young * float(reference_area))
    )
    energy = base.energy - ext_work
    total = energy + config.barrier_weight * base.barrier
    return PotentialEnergyTerms(
        total=total,
        energy=energy,
        barrier=base.barrier,
        minimum_determinant=base.minimum_determinant,
        inadmissible_points=base.inadmissible_points,
    )


def check_energy_quadrature_overshoot(
    grad_fn: Callable[[torch.Tensor], torch.Tensor],
    domain: TriangleDomain,
    material: MaterialSpec,
    *,
    config: PotentialEnergyConfig | None = None,
    max_relative_discrepancy: float = 0.05,
) -> QuadratureOvershootDiagnostic:
    """Compare degree-2 vs degree-5 Dunavant integrals to flag DEM overshoot."""
    cfg = PotentialEnergyConfig() if config is None else config
    if not math.isfinite(max_relative_discrepancy) or max_relative_discrepancy <= 0.0:
        raise ValueError("max_relative_discrepancy must be finite and positive")
    pts2, w2 = deterministic_triangle_quadrature(domain, degree=2)
    pts5, w5 = deterministic_triangle_quadrature(domain, degree=5)
    with torch.no_grad():
        e2 = float(
            plane_stress_potential_energy(
                grad_fn(pts2),
                w2,
                material,
                config=cfg,
                reference_area=domain.area,
            ).energy.item()
        )
        e5 = float(
            plane_stress_potential_energy(
                grad_fn(pts5),
                w5,
                material,
                config=cfg,
                reference_area=domain.area,
            ).energy.item()
        )
    denom = max(abs(e5), 1.0e-12)
    rel = abs(e2 - e5) / denom
    return QuadratureOvershootDiagnostic(
        energy_degree_2=e2,
        energy_degree_5=e5,
        relative_discrepancy=rel,
        max_allowed_discrepancy=max_relative_discrepancy,
        passed=rel <= max_relative_discrepancy,
    )


def follower_pressure_traction_residual(
    first_piola_2d: torch.Tensor,
    displacement_gradient_2d: torch.Tensor,
    rule: BoundaryQuadrature,
    *,
    pressure_mpa: float | torch.Tensor,
    stress_scale_mpa: float,
) -> torch.Tensor:
    """Reference-configuration Piola residual for non-conservative follower pressure.

    Enforces ``P N + p cof(F_2D) N = 0`` on open or closed pressurized reference
    facets via length-weighted mean squared normalized traction residual:

        L_follower = (1 / |Gamma|) int ||(P N + p cof(F_2D) N) / S||^2 dGamma.
    """
    if first_piola_2d.ndim != 3 or first_piola_2d.shape[1:] != (2, 2):
        raise ValueError("first_piola_2d must have shape (b_points, 2, 2)")
    if displacement_gradient_2d.shape != first_piola_2d.shape:
        raise ValueError("displacement_gradient_2d must align with first_piola_2d")
    if first_piola_2d.shape[0] != rule.points.shape[0]:
        raise ValueError("tensors must align with BoundaryQuadrature point count")
    if not math.isfinite(stress_scale_mpa) or stress_scale_mpa <= 0.0:
        raise ValueError("stress_scale_mpa must be finite and positive")

    eye2 = torch.eye(2, dtype=first_piola_2d.dtype, device=first_piola_2d.device)
    f2d = eye2 + displacement_gradient_2d
    # 2D cofactor matrix cof(F) = [[F22, -F21], [-F12, F11]]
    cof_row0 = torch.stack((f2d[:, 1, 1], -f2d[:, 1, 0]), dim=-1)
    cof_row1 = torch.stack((-f2d[:, 0, 1], f2d[:, 0, 0]), dim=-1)
    cof_f = torch.stack((cof_row0, cof_row1), dim=-2)

    normals = rule.normals.to(dtype=first_piola_2d.dtype, device=first_piola_2d.device)
    weights = rule.weights.to(dtype=first_piola_2d.dtype, device=first_piola_2d.device)
    pn = torch.einsum("nij,nj->ni", first_piola_2d, normals)
    cof_n = torch.einsum("nij,nj->ni", cof_f, normals)
    if isinstance(pressure_mpa, torch.Tensor):
        p_val = pressure_mpa.to(
            dtype=first_piola_2d.dtype, device=first_piola_2d.device
        ).reshape(-1, 1)
    else:
        p_val = torch.tensor(
            float(pressure_mpa),
            dtype=first_piola_2d.dtype,
            device=first_piola_2d.device,
        )
    residual = (pn + p_val * cof_n) / float(stress_scale_mpa)
    sq_norm = torch.sum(residual * residual, dim=-1)
    total_len = torch.sum(weights).clamp_min(1.0e-30)
    return torch.sum(sq_norm * weights) / total_len


def enclosed_cavity_pressure_energy(
    mesh: CanonicalMesh,
    cavity_tag: str,
    nodal_displacement: torch.Tensor,
    *,
    pressure_mpa: float,
    reference_area: float,
    young_modulus_mpa: float,
) -> torch.Tensor:
    """Conservative pressure potential ``-p * (A(u) - A_0) / (E A_ref)`` for closed cavities.

    Strictly verifies that ``cavity_tag`` forms closed 2-regular loops; open
    boundaries raise ``ValueError`` (use :func:`follower_pressure_traction_residual`).
    """
    if cavity_tag not in mesh.semantic_tags or not mesh.semantic_tags[cavity_tag]:
        raise ValueError(f"unknown or empty cavity boundary tag {cavity_tag!r}")
    facets = [mesh.boundary_facets[i] for i in mesh.semantic_tags[cavity_tag]]
    deg: dict[int, int] = {}
    for a, b in facets:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    if not all(v == 2 for v in deg.values()):
        raise ValueError(
            f"boundary {cavity_tag!r} is open; follower pressure on open "
            "boundaries is non-conservative and has no scalar potential energy"
        )
    if nodal_displacement.shape != (len(mesh.nodes), 2):
        raise ValueError("nodal_displacement must have shape (n_nodes, 2)")
    coords0 = torch.tensor(
        mesh.nodes,
        dtype=nodal_displacement.dtype,
        device=nodal_displacement.device,
    )
    coords = coords0 + nodal_displacement
    idx = torch.tensor(facets, dtype=torch.int64, device=nodal_displacement.device)
    x0_a, x0_b = coords0[idx[:, 0]], coords0[idx[:, 1]]
    x_a, x_b = coords[idx[:, 0]], coords[idx[:, 1]]
    # Signed enclosed area via Green's theorem 0.5 * sum (x_a y_b - y_a x_b)
    area0 = 0.5 * torch.sum(x0_a[:, 0] * x0_b[:, 1] - x0_a[:, 1] * x0_b[:, 0])
    area_def = 0.5 * torch.sum(x_a[:, 0] * x_b[:, 1] - x_a[:, 1] * x_b[:, 0])
    delta_area = torch.abs(area_def) - torch.abs(area0)
    return (-float(pressure_mpa) * delta_area) / (
        float(young_modulus_mpa) * float(reference_area)
    )


def discrete_fem_residual_loss(
    internal_force_free: torch.Tensor,
    external_force_free: torch.Tensor,
    *,
    force_scale: float = 1.0,
) -> torch.Tensor:
    """PFEM-style discrete virtual-work residual ``mean(((f_int - f_ext) / s)^2)``."""
    if internal_force_free.shape != external_force_free.shape:
        raise ValueError("internal and external free force vectors must align")
    if not math.isfinite(force_scale) or force_scale <= 0.0:
        raise ValueError("force_scale must be finite and positive")
    diff = (internal_force_free - external_force_free) / float(force_scale)
    return torch.mean(diff * diff)


__all__ = (
    "QuadratureOvershootDiagnostic",
    "check_energy_quadrature_overshoot",
    "dead_traction_potential_energy",
    "discrete_fem_residual_loss",
    "enclosed_cavity_pressure_energy",
    "follower_pressure_traction_residual",
)
