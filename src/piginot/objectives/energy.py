"""Total potential energy with stratified triangle quadrature.

The energy form (Deep Ritz) minimises ``Pi(u) = int W(F) dA`` over fields that
satisfy the essential conditions exactly. With displacement-controlled loading
and traction-free remaining edges there is no external work term, so the free
edge tractions are natural conditions of the minimum rather than penalties.

Two measures keep the discrete objective honest:

* the quadrature points are redrawn on every call: one or more uniformly
  distributed points per cell, weighted by the exact cell area, an unbiased
  stratified estimate of the integral that leaves no fixed point set for the
  network to exploit;
* a quadratic determinant barrier ``relu(J_min - J)^2`` acts below a positive
  threshold, while the material energy itself is only evaluated where the
  in-plane determinant is admissible, so the constitutive law is never
  extended to folded points.

The strain-energy density and the plane-stress closure are the verified S4
kernels, so the objective minimises exactly the energy used by the FEM solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Final

import torch

from piginot.contracts import MaterialSpec, hash_spec
from piginot.operators.energy_operator import embed_plane_stress_deformation
from piginot.physics import energy_density, plane_stress_closure, reference_modulus

POTENTIAL_ENERGY_POLICY_ID: Final = "potential-energy-stratified-triangles-v1"


@dataclass(frozen=True, slots=True)
class PotentialEnergyConfig:
    """Preregistered quadrature density and determinant-barrier policy."""

    __canonical_namespace__: ClassVar[str] = "piginot.objectives"

    points_per_cell: int = 3
    barrier_weight: float = 1.0e3
    barrier_threshold: float = 0.05
    admissibility_floor: float = 1.0e-6
    policy_id: str = POTENTIAL_ENERGY_POLICY_ID

    def __post_init__(self) -> None:
        if self.policy_id != POTENTIAL_ENERGY_POLICY_ID:
            raise ValueError(f"policy_id must be {POTENTIAL_ENERGY_POLICY_ID!r}")
        count = self.points_per_cell
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("points_per_cell must be a positive integer")
        for label in ("barrier_weight", "barrier_threshold", "admissibility_floor"):
            value = getattr(self, label)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{label} must be finite and positive")
            object.__setattr__(self, label, float(value))
        if self.admissibility_floor >= self.barrier_threshold:
            raise ValueError("admissibility_floor must lie below barrier_threshold")

    @property
    def config_hash(self) -> str:
        return hash_spec(self, "recipe")


@dataclass(frozen=True, slots=True)
class TriangleDomain:
    """Positively oriented triangle cells used as quadrature strata."""

    vertices: torch.Tensor
    cells: torch.Tensor

    def __post_init__(self) -> None:
        vertices = self.vertices
        cells = self.cells
        if not isinstance(vertices, torch.Tensor) or not isinstance(
            cells, torch.Tensor
        ):
            raise TypeError("vertices and cells must be torch.Tensor values")
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 2
            or vertices.shape[0] < 3
            or vertices.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(vertices).all().item())
        ):
            raise ValueError("vertices must be finite floating (count, 2) values")
        if cells.ndim != 2 or cells.shape[1] != 3 or cells.shape[0] == 0:
            raise ValueError("cells must have shape (count, 3)")
        if cells.dtype is not torch.int64:
            raise ValueError("cells must use torch.int64 vertex indices")
        if bool((cells < 0).any().item()) or bool(
            (cells >= vertices.shape[0]).any().item()
        ):
            raise ValueError("cells reference missing vertices")
        if bool((self.areas <= 0.0).any().item()):
            raise ValueError("cells must be positively oriented and non-degenerate")

    @property
    def corners(self) -> torch.Tensor:
        """Return cell corner coordinates with shape ``(cells, 3, 2)``."""

        return self.vertices[self.cells]

    @property
    def areas(self) -> torch.Tensor:
        corners = self.corners
        edge_a = corners[:, 1] - corners[:, 0]
        edge_b = corners[:, 2] - corners[:, 0]
        return 0.5 * (edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])

    @property
    def area(self) -> float:
        return float(self.areas.sum())


def stratified_triangle_quadrature(
    domain: TriangleDomain,
    points_per_cell: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw uniform points per cell and return ``(points, weights)``.

    Barycentric coordinates use the reflection construction, so every point is
    uniformly distributed inside its cell and each weight is ``area / k``. The
    estimator is unbiased for any integrable field and exact for affine fields
    in expectation; constants integrate exactly for every draw.
    """

    if type(domain) is not TriangleDomain:
        raise TypeError("domain must be a TriangleDomain")
    if (
        isinstance(points_per_cell, bool)
        or not isinstance(points_per_cell, int)
        or points_per_cell <= 0
    ):
        raise ValueError("points_per_cell must be a positive integer")
    dtype = domain.vertices.dtype
    cell_count = domain.cells.shape[0]
    draws = torch.rand(
        (cell_count, points_per_cell, 2),
        generator=generator,
        dtype=dtype,
        device=domain.vertices.device,
    )
    first = draws[..., 0]
    second = draws[..., 1]
    fold = (first + second) > 1.0
    first = torch.where(fold, 1.0 - first, first)
    second = torch.where(fold, 1.0 - second, second)
    barycentric = torch.stack((1.0 - first - second, first, second), dim=-1)
    points = torch.einsum("ckv,cvd->ckd", barycentric, domain.corners)
    weights = (domain.areas / points_per_cell).unsqueeze(1).expand(-1, points_per_cell)
    return points.reshape(-1, 2), weights.reshape(-1).clone()


@dataclass(frozen=True, slots=True)
class PotentialEnergyTerms:
    """Graph-connected nondimensional energy objective of one sample."""

    total: torch.Tensor
    energy: torch.Tensor
    barrier: torch.Tensor
    minimum_determinant: float
    inadmissible_points: int

    def __post_init__(self) -> None:
        for label in ("total", "energy", "barrier"):
            value = getattr(self, label)
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise ValueError(f"{label} must be a scalar tensor")
        if not math.isfinite(self.minimum_determinant):
            raise ValueError("minimum_determinant must be finite")
        if type(self.inadmissible_points) is not int or self.inadmissible_points < 0:
            raise ValueError("inadmissible_points must be a non-negative integer")


def plane_stress_potential_energy(
    displacement_gradient: torch.Tensor,
    weights: torch.Tensor,
    material: MaterialSpec,
    *,
    config: PotentialEnergyConfig,
    reference_area: float,
) -> PotentialEnergyTerms:
    """Return ``Pi / (E A) + w_b * mean(relu(J_min - J)^2)`` for one sample.

    ``displacement_gradient`` has shape ``(points, 2, 2)`` in mm/mm and
    ``weights`` are the matching physical quadrature weights in mm^2. The
    stored energy is evaluated only at points whose in-plane determinant exceeds
    ``admissibility_floor``; the barrier sees every point, using the smaller of
    the in-plane and closed 3D determinants where both exist.
    """

    if type(config) is not PotentialEnergyConfig:
        raise TypeError("config must be a PotentialEnergyConfig")
    if not isinstance(displacement_gradient, torch.Tensor) or (
        displacement_gradient.ndim != 3
        or displacement_gradient.shape[1:] != (2, 2)
        or displacement_gradient.shape[0] == 0
    ):
        raise ValueError("displacement_gradient must have shape (points, 2, 2)")
    if not isinstance(weights, torch.Tensor) or weights.shape != (
        displacement_gradient.shape[0],
    ):
        raise ValueError("weights must have shape (points,)")
    if (
        weights.dtype != displacement_gradient.dtype
        or weights.device != displacement_gradient.device
    ):
        raise ValueError("weights and displacement_gradient must share dtype/device")
    if not bool(torch.isfinite(displacement_gradient).all().item()):
        raise ValueError("displacement_gradient must be finite")
    if bool((weights < 0.0).any().item()) or not bool(
        torch.isfinite(weights).all().item()
    ):
        raise ValueError("weights must be finite and non-negative")
    area = float(reference_area)
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("reference_area must be finite and positive")
    if material.properties.get("elastic_modulus") is not None:
        young = float(material.properties["elastic_modulus"].value)
    else:
        young = reference_modulus(material)

    identity = torch.eye(
        2, dtype=displacement_gradient.dtype, device=displacement_gradient.device
    )
    deformation = identity + displacement_gradient
    determinant = (
        deformation[:, 0, 0] * deformation[:, 1, 1]
        - deformation[:, 0, 1] * deformation[:, 1, 0]
    )
    admissible = determinant > config.admissibility_floor
    safe_deformation = torch.where(
        admissible[:, None, None],
        deformation,
        identity.expand_as(deformation),
    )
    f33 = plane_stress_closure(safe_deformation, material)
    density = energy_density(
        embed_plane_stress_deformation(safe_deformation, f33),
        material,
    )
    density = torch.where(admissible, density, torch.zeros_like(density))
    energy = torch.sum(density * weights) / (young * area)

    closed_determinant = determinant * f33
    guard = torch.where(
        admissible,
        torch.minimum(determinant, closed_determinant),
        determinant,
    )
    barrier = torch.mean(torch.relu(config.barrier_threshold - guard).square())
    total = energy + config.barrier_weight * barrier
    return PotentialEnergyTerms(
        total=total,
        energy=energy,
        barrier=barrier,
        minimum_determinant=float(guard.detach().min()),
        inadmissible_points=int((~admissible).sum().item()),
    )


__all__ = (
    "POTENTIAL_ENERGY_POLICY_ID",
    "PotentialEnergyConfig",
    "PotentialEnergyTerms",
    "TriangleDomain",
    "plane_stress_potential_energy",
    "stratified_triangle_quadrature",
)
