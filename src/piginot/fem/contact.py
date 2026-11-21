"""Frictionless large-deformation rigid-indenter smooth barrier / AL contact.

Provides a clean, MIT-licensed 2D/2.5D frictionless contact formulation for
elastomeric indentation and seal benchmarks using smooth signed-distance gaps
``g(x) >= 0`` (outside the rigid obstacle):
- **Circular / cylindrical rigid punch**: ``g(x) = ||x - c|| - R``
- **Rigid half-plane**: ``g(x) = (x - p) . n``

Supports both smooth penalty/IPC-style C^2 contact energy and Augmented
Lagrangian multiplier updates with exact nodal force vectors and exact 2x2
nodal contact Hessians for quadratic Newton convergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RigidIndenterSpec:
    """Frictionless rigid obstacle defined by a smooth signed distance ``g(x)``."""

    kind: str  # "circle" or "half_plane"
    center: tuple[float, float]
    radius: float = 0.0
    outward_normal: tuple[float, float] = (0.0, 1.0)
    penalty_stiffness: float = 1.0e4
    barrier_distance: float = 0.0
    contact_boundary_tag: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("circle", "half_plane"):
            raise ValueError("kind must be 'circle' or 'half_plane'")
        if self.kind == "circle" and (
            not math.isfinite(self.radius) or self.radius <= 0.0
        ):
            raise ValueError("circle indenter requires positive finite radius")
        if self.kind == "half_plane":
            nx, ny = self.outward_normal
            norm = math.hypot(nx, ny)
            if norm <= 0.0:
                raise ValueError("half_plane outward_normal must be non-zero")
            object.__setattr__(self, "outward_normal", (nx / norm, ny / norm))
        if not math.isfinite(self.penalty_stiffness) or self.penalty_stiffness <= 0.0:
            raise ValueError("penalty_stiffness must be finite and positive")
        if not math.isfinite(self.barrier_distance) or self.barrier_distance < 0.0:
            raise ValueError("barrier_distance must be finite and non-negative")

    def gap_normal_curvature(
        self, deformed_xy: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``(gap, normal, grad_normal)`` at deformed points ``(N, 2)``.

        ``gap > 0`` means strictly outside the rigid obstacle (admissible);
        ``gap < 0`` means penetration. ``normal`` points from obstacle into body.
        """
        pts = np.asarray(deformed_xy, dtype=np.float64)
        n_pts = pts.shape[0]
        if self.kind == "half_plane":
            p0 = np.asarray(self.center, dtype=np.float64)
            n_vec = np.asarray(self.outward_normal, dtype=np.float64)
            gap = (pts - p0) @ n_vec
            normals = np.broadcast_to(n_vec, (n_pts, 2)).copy()
            grad_n = np.zeros((n_pts, 2, 2), dtype=np.float64)
            return gap, normals, grad_n

        c = np.asarray(self.center, dtype=np.float64)
        diff = pts - c
        dist = np.linalg.norm(diff, axis=1)
        safe_dist = np.maximum(dist, 1.0e-12)
        normals = diff / safe_dist[:, None]
        gap = dist - self.radius
        eye2 = np.eye(2, dtype=np.float64)[None, :, :]
        nnT = normals[:, :, None] * normals[:, None, :]
        grad_n = (eye2 - nnT) / safe_dist[:, None, None]
        return gap, normals, grad_n


@dataclass(frozen=True, slots=True)
class ContactVerificationReport:
    """Non-penetration and reaction summary of a converged contact solve."""

    min_gap_mm: float
    max_penetration_mm: float
    active_node_count: int
    total_contact_force_n: tuple[float, float]
    auglag_iterations: int
    admissible: bool

    def document(self) -> dict[str, Any]:
        return {
            "min_gap_mm": self.min_gap_mm,
            "max_penetration_mm": self.max_penetration_mm,
            "active_node_count": self.active_node_count,
            "total_contact_force_n": list(self.total_contact_force_n),
            "auglag_iterations": self.auglag_iterations,
            "admissible": self.admissible,
        }


def assemble_contact_terms(
    indenter: RigidIndenterSpec,
    reference_xy: FloatArray,
    displacement_xy: FloatArray,
    candidate_nodes: NDArray[np.int64],
    *,
    multipliers: FloatArray | None = None,
    weights: FloatArray | None = None,
) -> tuple[float, FloatArray, FloatArray, FloatArray]:
    """Assemble Augmented Lagrangian contact energy, nodal force, and nodal 2x2 Hessian.

    Uses the standard Rockafellar-Wets C^1 Augmented Lagrangian potential:
        ``E_c = sum_i w_i * (1/(2*rho)) * (max(0, lam_i - rho * (g_i - d_hat))^2 - lam_i^2)``
    whose gradient wrt ``u_i`` is ``f_i = -w_i * p_i * n_i`` with
    ``p_i = max(0, lam_i - rho * (g_i - d_hat))`` and exact 2x2 tangent
    ``K_ii = w_i * (rho * n_i n_i^T - p_i * grad(n_i))`` on active nodes.

    Returns ``(energy, force_nodal (N,2), hessian_blocks (M,2,2), updated_pressures (M,))``.
    """
    n_total = reference_xy.shape[0]
    m = candidate_nodes.shape[0]
    rho = indenter.penalty_stiffness
    d_hat = indenter.barrier_distance
    lam = (
        np.zeros(m, dtype=np.float64)
        if multipliers is None
        else np.asarray(multipliers, dtype=np.float64)
    )
    w = (
        np.ones(m, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )

    deformed = reference_xy[candidate_nodes] + displacement_xy[candidate_nodes]
    gap, normals, grad_n = indenter.gap_normal_curvature(deformed)
    shifted_gap = gap - d_hat
    raw_pressure = lam - rho * shifted_gap
    active = raw_pressure > 0.0
    pressure = np.where(active, raw_pressure, 0.0)

    energy = float(np.sum(w * (0.5 / rho) * (pressure * pressure - lam * lam)))
    force_nodal = np.zeros((n_total, 2), dtype=np.float64)
    # Residual contribution dE_c / du_i = -w_i * pressure_i * n_i
    force_nodal[candidate_nodes] = -(w * pressure)[:, None] * normals

    hess = np.zeros((m, 2, 2), dtype=np.float64)
    if np.any(active):
        nnT = normals[active, :, None] * normals[active, None, :]
        # Exact d(-p * n)/du = rho * n n^T - p * grad(n)
        hess[active] = w[active, None, None] * (
            rho * nnT - pressure[active, None, None] * grad_n[active]
        )
    return energy, force_nodal, hess, pressure


__all__ = (
    "ContactVerificationReport",
    "RigidIndenterSpec",
    "assemble_contact_terms",
)
