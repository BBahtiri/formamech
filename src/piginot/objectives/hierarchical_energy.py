"""Hierarchical physics-consistency energy objective and quadrature checks (Feature A).

Evaluates the preregistered experimental label-free variational objective:

    L = L_Pi + lambda_h * L_cross_mesh + lambda_l * L_local_physics

No FEM training targets are used. ``L_Pi`` remains the primary area-weighted
plane-stress potential energy. ``L_cross_mesh`` measures consistency between
coarse-cell and nested 1->4 subdivided child-cell energy integrals on the exact
same physical domain. ``L_local_physics`` optionally emphasizes local energy
resolution on feature-weighted cells (for example from a traction-jump indicator).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Final

import torch

from piginot.contracts import MaterialSpec, hash_spec
from piginot.objectives.energy import (
    PotentialEnergyConfig,
    PotentialEnergyTerms,
    TriangleDomain,
    plane_stress_potential_energy,
    stratified_triangle_quadrature,
)
from piginot.operators.energy_operator import embed_plane_stress_deformation
from piginot.physics import energy_density, plane_stress_closure, reference_modulus

HIERARCHICAL_ENERGY_POLICY_ID: Final = "hierarchical-plane-stress-energy-v1"
SUPPORTED_QUADRATURE_ORDERS: Final = (1, 2, 5)

# Symmetric Gauss-Dunavant rules on the reference triangle (barycentric coords, weight).
_DUNAVANT_RULES: Final[dict[int, tuple[tuple[float, float, float, float], ...]]] = {
    1: ((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0),),
    2: (
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0),
        (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0, 1.0 / 3.0),
    ),
    5: (
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.225),
        (
            0.059715871789770,
            0.470142064105115,
            0.470142064105115,
            0.132394152788506,
        ),
        (
            0.470142064105115,
            0.059715871789770,
            0.470142064105115,
            0.132394152788506,
        ),
        (
            0.470142064105115,
            0.470142064105115,
            0.059715871789770,
            0.132394152788506,
        ),
        (
            0.797426985353087,
            0.101286507323456,
            0.101286507323456,
            0.125939180544827,
        ),
        (
            0.101286507323456,
            0.797426985353087,
            0.101286507323456,
            0.125939180544827,
        ),
        (
            0.101286507323456,
            0.101286507323456,
            0.797426985353087,
            0.125939180544827,
        ),
    ),
}


def deterministic_triangle_quadrature(
    domain: TriangleDomain,
    *,
    degree: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact Dunavant quadrature ``(points, weights)`` on ``domain``.

    ``degree`` must be one of ``1`` (centroid, 1 pt/cell), ``2`` (3 pts/cell),
    or ``5`` (7 pts/cell). Weights sum strictly to ``domain.area``.
    """
    if type(domain) is not TriangleDomain:
        raise TypeError("domain must be a TriangleDomain")
    if type(degree) is not int or degree not in _DUNAVANT_RULES:
        raise ValueError(
            f"degree must be one of {SUPPORTED_QUADRATURE_ORDERS}, got {degree!r}"
        )
    rule = _DUNAVANT_RULES[degree]
    dtype = domain.vertices.dtype
    device = domain.vertices.device
    bary = torch.tensor([(r[0], r[1], r[2]) for r in rule], dtype=dtype, device=device)
    rel_weights = torch.tensor([r[3] for r in rule], dtype=dtype, device=device)
    # Normalize slight rounding in tabulated degree-5 weights to exact unity.
    rel_weights = rel_weights / rel_weights.sum()
    points = torch.einsum("kv,cvd->ckd", bary, domain.corners)
    weights = domain.areas.unsqueeze(1) * rel_weights.unsqueeze(0)
    return points.reshape(-1, 2), weights.reshape(-1)


def subdivide_triangle_domain(domain: TriangleDomain) -> TriangleDomain:
    """Exact 1->4 midpoint subdivision of a ``TriangleDomain`` preserving orientation.

    Cell ``c`` maps to child cells ``4*c, 4*c+1, 4*c+2, 4*c+3``, so parent-child
    energy density comparisons align without nearest-node approximations.
    """
    if type(domain) is not TriangleDomain:
        raise TypeError("domain must be a TriangleDomain")
    cells_list = domain.cells.tolist()
    edges: dict[tuple[int, int], int] = {}
    edge_pairs: list[tuple[int, int]] = []
    n_verts = int(domain.vertices.shape[0])
    for a, b, c in cells_list:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            if key not in edges:
                edges[key] = n_verts + len(edge_pairs)
                edge_pairs.append(key)

    if edge_pairs:
        idx_tensor = torch.tensor(
            edge_pairs, dtype=torch.int64, device=domain.vertices.device
        )
        midpoints = 0.5 * (
            domain.vertices[idx_tensor[:, 0]] + domain.vertices[idx_tensor[:, 1]]
        )
        new_vertices = torch.cat((domain.vertices, midpoints), dim=0)
    else:  # pragma: no cover - TriangleDomain forbids empty cells
        new_vertices = domain.vertices

    new_cells: list[tuple[int, int, int]] = []
    for a, b, c in cells_list:
        ab = edges[(a, b) if a < b else (b, a)]
        bc = edges[(b, c) if b < c else (c, b)]
        ca = edges[(c, a) if c < a else (a, c)]
        new_cells.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    return TriangleDomain(
        vertices=new_vertices,
        cells=torch.tensor(new_cells, dtype=torch.int64, device=domain.cells.device),
    )


@dataclass(frozen=True, slots=True)
class HierarchicalEnergyConfig:
    """Preregistered configuration for the hierarchical physics-consistency arm."""

    __canonical_namespace__: ClassVar[str] = "piginot.objectives"

    base: PotentialEnergyConfig = PotentialEnergyConfig()
    cross_mesh_weight: float = 0.0
    local_physics_weight: float = 0.0
    quadrature_degree: int = 2
    use_deterministic_quadrature: bool = False
    policy_id: str = HIERARCHICAL_ENERGY_POLICY_ID

    def __post_init__(self) -> None:
        if type(self.base) is not PotentialEnergyConfig:
            raise TypeError("base must be a PotentialEnergyConfig")
        if self.policy_id != HIERARCHICAL_ENERGY_POLICY_ID:
            raise ValueError(f"policy_id must be {HIERARCHICAL_ENERGY_POLICY_ID!r}")
        for label in ("cross_mesh_weight", "local_physics_weight"):
            val = getattr(self, label)
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(float(val))
                or float(val) < 0.0
            ):
                raise ValueError(f"{label} must be finite and nonnegative")
            object.__setattr__(self, label, float(val))
        if (
            type(self.quadrature_degree) is not int
            or self.quadrature_degree not in SUPPORTED_QUADRATURE_ORDERS
        ):
            raise ValueError(
                f"quadrature_degree must be one of {SUPPORTED_QUADRATURE_ORDERS}"
            )
        if type(self.use_deterministic_quadrature) is not bool:
            raise TypeError("use_deterministic_quadrature must be a boolean")

    @property
    def config_hash(self) -> str:
        return hash_spec(self, "recipe")


@dataclass(frozen=True, slots=True)
class HierarchicalEnergyTerms:
    """Graph-connected variational objective plus cross-mesh and local physics terms."""

    total: torch.Tensor
    primary: PotentialEnergyTerms
    cross_mesh: torch.Tensor
    local_physics: torch.Tensor
    coarse_area: float
    fine_area: float

    def __post_init__(self) -> None:
        for label in ("total", "cross_mesh", "local_physics"):
            val = getattr(self, label)
            if not isinstance(val, torch.Tensor) or val.ndim != 0:
                raise ValueError(f"{label} must be a scalar tensor")


def _cell_energy_densities(
    gradient: torch.Tensor,
    weights: torch.Tensor,
    material: MaterialSpec,
    *,
    cell_count: int,
    points_per_cell: int,
    config: PotentialEnergyConfig,
    young: float,
) -> torch.Tensor:
    """Nondimensional area-averaged strain-energy density per triangle cell."""
    identity = torch.eye(2, dtype=gradient.dtype, device=gradient.device)
    deformation = identity + gradient
    det = (
        deformation[:, 0, 0] * deformation[:, 1, 1]
        - deformation[:, 0, 1] * deformation[:, 1, 0]
    )
    admissible = det > config.admissibility_floor
    safe = torch.where(
        admissible[:, None, None], deformation, identity.expand_as(deformation)
    )
    f33 = plane_stress_closure(safe, material)
    density = energy_density(embed_plane_stress_deformation(safe, f33), material)
    density = torch.where(admissible, density, torch.zeros_like(density))
    weighted = (density * weights).reshape(cell_count, points_per_cell).sum(dim=1)
    cell_areas = weights.reshape(cell_count, points_per_cell).sum(dim=1)
    return weighted / (young * cell_areas)


def hierarchical_plane_stress_energy(
    gradient_evaluator: Callable[[torch.Tensor], torch.Tensor],
    coarse_domain: TriangleDomain,
    material: MaterialSpec,
    *,
    config: HierarchicalEnergyConfig,
    fine_domain: TriangleDomain | None = None,
    cell_focus_weights: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> HierarchicalEnergyTerms:
    """Evaluate ``L_Pi + lambda_h * L_cross_mesh + lambda_l * L_local_physics``.

    ``gradient_evaluator`` maps query coordinates ``(N, 2)`` to displacement
    gradients ``(N, 2, 2)`` while retaining the autograd graph.
    """
    if type(coarse_domain) is not TriangleDomain:
        raise TypeError("coarse_domain must be a TriangleDomain")
    if type(config) is not HierarchicalEnergyConfig:
        raise TypeError("config must be a HierarchicalEnergyConfig")

    if config.use_deterministic_quadrature:
        c_pts, c_wts = deterministic_triangle_quadrature(
            coarse_domain, degree=config.quadrature_degree
        )
        c_ppc = len(_DUNAVANT_RULES[config.quadrature_degree])
    else:
        c_pts, c_wts = stratified_triangle_quadrature(
            coarse_domain, config.base.points_per_cell, generator=generator
        )
        c_ppc = config.base.points_per_cell

    coarse_grad = gradient_evaluator(c_pts)
    primary = plane_stress_potential_energy(
        coarse_grad,
        c_wts,
        material,
        config=config.base,
        reference_area=coarse_domain.area,
    )

    zero = primary.total.new_zeros(())
    if config.cross_mesh_weight == 0.0 and config.local_physics_weight == 0.0:
        return HierarchicalEnergyTerms(
            total=primary.total,
            primary=primary,
            cross_mesh=zero,
            local_physics=zero,
            coarse_area=coarse_domain.area,
            fine_area=coarse_domain.area,
        )

    subdivided = (
        subdivide_triangle_domain(coarse_domain) if fine_domain is None else fine_domain
    )
    if subdivided.cells.shape[0] != 4 * coarse_domain.cells.shape[0]:
        raise ValueError(
            "fine_domain must be an exact 1->4 subdivision of coarse_domain"
        )
    if abs(subdivided.area - coarse_domain.area) > max(
        1e-6 * coarse_domain.area, 1e-12
    ):
        raise ValueError("fine_domain area must match coarse_domain area")

    if config.use_deterministic_quadrature:
        f_pts, f_wts = deterministic_triangle_quadrature(
            subdivided, degree=config.quadrature_degree
        )
        f_ppc = len(_DUNAVANT_RULES[config.quadrature_degree])
    else:
        f_pts, f_wts = stratified_triangle_quadrature(
            subdivided, config.base.points_per_cell, generator=generator
        )
        f_ppc = config.base.points_per_cell

    fine_grad = gradient_evaluator(f_pts)
    young = (
        float(material.properties["elastic_modulus"].value)
        if material.properties.get("elastic_modulus") is not None
        else reference_modulus(material)
    )

    coarse_cell_e = _cell_energy_densities(
        coarse_grad,
        c_wts,
        material,
        cell_count=int(coarse_domain.cells.shape[0]),
        points_per_cell=c_ppc,
        config=config.base,
        young=young,
    )
    fine_cell_e = _cell_energy_densities(
        fine_grad,
        f_wts,
        material,
        cell_count=int(subdivided.cells.shape[0]),
        points_per_cell=f_ppc,
        config=config.base,
        young=young,
    )
    fine_areas_grouped = subdivided.areas.reshape(coarse_domain.cells.shape[0], 4)
    fine_e_grouped = fine_cell_e.reshape(coarse_domain.cells.shape[0], 4)
    restricted_fine_e = (fine_e_grouped * fine_areas_grouped).sum(
        dim=1
    ) / coarse_domain.areas

    # Area-weighted mean squared energy-density discrepancy across M0 <-> M1.
    cell_diff_sq = (coarse_cell_e - restricted_fine_e).square()
    area_weights = coarse_domain.areas / coarse_domain.areas.sum()
    cross_mesh = torch.sum(area_weights * cell_diff_sq)

    if config.local_physics_weight > 0.0:
        if cell_focus_weights is None:
            # Default: emphasize smaller (feature-refined) cells via inverse area.
            inv_area = 1.0 / coarse_domain.areas
            focus = inv_area / inv_area.sum()
        else:
            if cell_focus_weights.shape != (coarse_domain.cells.shape[0],):
                raise ValueError(
                    "cell_focus_weights must have shape (coarse_cell_count,)"
                )
            if (
                bool((cell_focus_weights < 0.0).any().item())
                or float(cell_focus_weights.sum()) <= 0.0
            ):
                raise ValueError(
                    "cell_focus_weights must be nonnegative with positive sum"
                )
            focus = cell_focus_weights.to(
                dtype=coarse_cell_e.dtype, device=coarse_cell_e.device
            )
            focus = focus / focus.sum()
        # Feature-weighted cross-mesh consistency (preserves real physical stress
        # gradients while concentrating quadrature consistency on refined cells).
        local_physics = torch.sum(focus * cell_diff_sq)
    else:
        local_physics = zero

    # Primary variational energy evaluated on the finer nested M1 domain when
    # hierarchical weights are active, so resolution improves around features
    # without introducing any FEM supervision.
    fine_primary = plane_stress_potential_energy(
        fine_grad,
        f_wts,
        material,
        config=config.base,
        reference_area=subdivided.area,
    )
    total = (
        fine_primary.total
        + config.cross_mesh_weight * cross_mesh
        + config.local_physics_weight * local_physics
    )
    return HierarchicalEnergyTerms(
        total=total,
        primary=fine_primary,
        cross_mesh=cross_mesh,
        local_physics=local_physics,
        coarse_area=coarse_domain.area,
        fine_area=subdivided.area,
    )


@dataclass(frozen=True, slots=True)
class QuadratureConvergenceReport:
    """Measured convergence of the plane-stress energy integral under densification."""

    energies_by_degree: dict[int, float]
    subdivided_energy: float
    degree_2_to_5_relative_diff: float
    mesh_subdivision_relative_diff: float
    converged: bool
    tolerance: float


def check_quadrature_energy_convergence(
    gradient_evaluator: Callable[[torch.Tensor], torch.Tensor],
    domain: TriangleDomain,
    material: MaterialSpec,
    *,
    base_config: PotentialEnergyConfig | None = None,
    relative_tolerance: float = 1e-3,
) -> QuadratureConvergenceReport:
    """Verify energy-integral convergence across quadrature degree and subdivision."""
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    effective_config = PotentialEnergyConfig() if base_config is None else base_config
    energies: dict[int, float] = {}
    with torch.no_grad():
        for degree in SUPPORTED_QUADRATURE_ORDERS:
            pts, wts = deterministic_triangle_quadrature(domain, degree=degree)
            terms = plane_stress_potential_energy(
                gradient_evaluator(pts),
                wts,
                material,
                config=effective_config,
                reference_area=domain.area,
            )
            energies[degree] = float(terms.energy.item())
        sub = subdivide_triangle_domain(domain)
        sub_pts, sub_wts = deterministic_triangle_quadrature(sub, degree=5)
        sub_energy = float(
            plane_stress_potential_energy(
                gradient_evaluator(sub_pts),
                sub_wts,
                material,
                config=effective_config,
                reference_area=sub.area,
            ).energy.item()
        )
    denom_deg = max(abs(energies[5]), 1e-15)
    denom_sub = max(abs(sub_energy), 1e-15)
    deg_diff = abs(energies[2] - energies[5]) / denom_deg
    sub_diff = abs(energies[5] - sub_energy) / denom_sub
    return QuadratureConvergenceReport(
        energies_by_degree=energies,
        subdivided_energy=sub_energy,
        degree_2_to_5_relative_diff=deg_diff,
        mesh_subdivision_relative_diff=sub_diff,
        converged=(deg_diff <= relative_tolerance and sub_diff <= relative_tolerance),
        tolerance=relative_tolerance,
    )


__all__ = (
    "HIERARCHICAL_ENERGY_POLICY_ID",
    "SUPPORTED_QUADRATURE_ORDERS",
    "HierarchicalEnergyConfig",
    "HierarchicalEnergyTerms",
    "QuadratureConvergenceReport",
    "check_quadrature_energy_convergence",
    "deterministic_triangle_quadrature",
    "hierarchical_plane_stress_energy",
    "subdivide_triangle_domain",
)
