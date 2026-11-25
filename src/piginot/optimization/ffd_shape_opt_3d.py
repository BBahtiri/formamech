"""Trivariate Bernstein FFD Free-Form 3D Shape Optimization (V4-4).

Enables arbitrary non-parametric 3D mesh morphing of any :class:`CanonicalMesh3D`
via a tensor-product Bernstein polynomial Free-Form Deformation (FFD) cage
with **fail-closed Jacobian positivity enforcement** (``min_det_j > 0``,
refusing any self-intersecting control move) and exact 3D FEM verification.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from piginot.contracts import MaterialSpec, canonical_json_bytes
from piginot.fem.backend_3d import (
    DirichletBC3D,
    FemRequest3D,
    TorchFemBackend3D,
    TractionBC3D,
    _tet_quadrature,
    _tet_shape_grads_ref,
)
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D

FloatArray = NDArray[np.float64]

FFD_SHAPE_OPT_SCHEMA: Final = "formamech.ffd-shape-opt-3d-v1"


class FfdMeshInversionError(ValueError):
    """Fail-closed refusal when an FFD morph produces inverted 3D elements (det J <= 0)."""


def _bernstein_basis_1d(u: FloatArray, degree: int) -> FloatArray:
    """Evaluate Bernstein polynomials ``B_{i,n}(u)`` of shape ``(N_pts, degree + 1)``."""
    u_c = np.clip(u, 0.0, 1.0)
    out = np.empty((u_c.size, degree + 1), dtype=np.float64)
    for i in range(degree + 1):
        coeff = float(math.comb(degree, i))
        out[:, i] = coeff * (u_c**i) * ((1.0 - u_c) ** (degree - i))
    return out


def _hex8_gauss_grads_ref() -> tuple[FloatArray, FloatArray]:
    """2x2x2 Gauss-Legendre reference points & shape gradients ``(8, 8, 3)`` for HEX8."""
    g = 1.0 / math.sqrt(3.0)
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    pts = g * signs
    d_ref = np.empty((8, 8, 3), dtype=np.float64)
    for q in range(8):
        xi, eta, zeta = float(pts[q, 0]), float(pts[q, 1]), float(pts[q, 2])
        d_ref[q, :, 0] = (
            0.125 * signs[:, 0] * (1.0 + signs[:, 1] * eta) * (1.0 + signs[:, 2] * zeta)
        )
        d_ref[q, :, 1] = (
            0.125 * (1.0 + signs[:, 0] * xi) * signs[:, 1] * (1.0 + signs[:, 2] * zeta)
        )
        d_ref[q, :, 2] = (
            0.125 * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 1] * eta) * signs[:, 2]
        )
    return d_ref, np.ones(8, dtype=np.float64)


def _min_cell_det_j(
    nodes: FloatArray,
    cells: NDArray[np.int64],
    cell_type: CellType3D = CellType3D.TET4,
) -> float:
    """Compute minimum signed isoparametric Jacobian determinant across TET4/TET10/HEX8."""
    if cell_type is CellType3D.TET4:
        p0 = nodes[cells[:, 0]]
        p1 = nodes[cells[:, 1]]
        p2 = nodes[cells[:, 2]]
        p3 = nodes[cells[:, 3]]
        det_j = np.einsum("ei,ei->e", np.cross(p1 - p0, p2 - p0), p3 - p0)
        return float(np.min(det_j))
    if cell_type is CellType3D.TET10:
        q_pts, _ = _tet_quadrature(CellType3D.TET10)
        corners = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        check_pts = np.vstack((q_pts, corners))
        d_ref = _tet_shape_grads_ref(CellType3D.TET10, check_pts)
        x_e = nodes[cells]
        jac = np.einsum("eni,qna->eqia", x_e, d_ref)
        return float(np.min(np.linalg.det(jac)))
    if cell_type is CellType3D.HEX8:
        d_ref, _ = _hex8_gauss_grads_ref()
        x_e = nodes[cells]
        jac = np.einsum("eni,qna->eqia", x_e, d_ref)
        return float(np.min(np.linalg.det(jac)))
    raise ValueError(f"unsupported cell_type for FFD Jacobian check: {cell_type!r}")


def _total_mesh_volume(
    nodes: FloatArray,
    cells: NDArray[np.int64],
    cell_type: CellType3D = CellType3D.TET4,
) -> float:
    """Compute exact 3D mesh volume for TET4, TET10, or HEX8."""
    if cell_type is CellType3D.TET4:
        p0 = nodes[cells[:, 0]]
        p1 = nodes[cells[:, 1]]
        p2 = nodes[cells[:, 2]]
        p3 = nodes[cells[:, 3]]
        vols = np.einsum("ei,ei->e", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0
        return float(np.sum(vols))
    if cell_type is CellType3D.TET10:
        q_pts, w_ref = _tet_quadrature(CellType3D.TET10)
        d_ref = _tet_shape_grads_ref(CellType3D.TET10, q_pts)
        x_e = nodes[cells]
        jac = np.einsum("eni,qna->eqia", x_e, d_ref)
        return float(np.sum(np.linalg.det(jac) * w_ref[None, :]))
    if cell_type is CellType3D.HEX8:
        d_ref, w_ref = _hex8_gauss_grads_ref()
        x_e = nodes[cells]
        jac = np.einsum("eni,qna->eqia", x_e, d_ref)
        return float(np.sum(np.linalg.det(jac) * w_ref[None, :]))
    raise ValueError(f"unsupported cell_type for FFD volume check: {cell_type!r}")


def _min_tet_det_j(nodes: FloatArray, cells: NDArray[np.int64]) -> float:
    """Compute minimum signed Jacobian determinant ``det J = 6 * V_e`` across tets."""
    return _min_cell_det_j(nodes, cells, CellType3D.TET4)


def _total_tet_volume(nodes: FloatArray, cells: NDArray[np.int64]) -> float:
    return _total_mesh_volume(nodes, cells, CellType3D.TET4)


@dataclass(frozen=True, slots=True)
class FFDGrid3D:
    """Trivariate Bernstein polynomial Free-Form Deformation cage for 3D meshes."""

    origin_mm: tuple[float, float, float]
    extent_mm: tuple[float, float, float]
    nx: int = 3
    ny: int = 2
    nz: int = 2

    def __post_init__(self) -> None:
        if (
            type(self.nx) is not int
            or type(self.ny) is not int
            or type(self.nz) is not int
            or self.nx < 2
            or self.ny < 2
            or self.nz < 2
        ):
            raise ValueError("FFD control grid must have integer >= 2 points per axis")
        if any(not math.isfinite(float(o)) for o in self.origin_mm):
            raise ValueError("FFD cage origin must be finite")
        if any(not math.isfinite(float(e)) or e <= 1.0e-12 for e in self.extent_mm):
            raise ValueError("FFD cage extents must be finite and positive")

    @property
    def num_control_points(self) -> int:
        return self.nx * self.ny * self.nz

    @classmethod
    def from_mesh(
        cls, mesh: CanonicalMesh3D, *, nx: int = 3, ny: int = 2, nz: int = 2
    ) -> FFDGrid3D:
        nodes = np.asarray(mesh.nodes, dtype=np.float64)
        lo = nodes.min(axis=0)
        hi = nodes.max(axis=0)
        ext = hi - lo
        if np.any(~np.isfinite(ext)) or np.any(ext <= 1.0e-12):
            raise ValueError(
                "cannot construct 3D FFD cage on degenerate or flat bounding box"
            )
        return cls(
            origin_mm=(float(lo[0]), float(lo[1]), float(lo[2])),
            extent_mm=(float(ext[0]), float(ext[1]), float(ext[2])),
            nx=nx,
            ny=ny,
            nz=nz,
        )

    def basis_matrix(self, nodes_mm: FloatArray) -> FloatArray:
        """Return ``(N_nodes, N_ctrl)`` trivariate Bernstein evaluation matrix ``B``."""
        nodes = np.asarray(nodes_mm, dtype=np.float64)
        orig = np.asarray(self.origin_mm, dtype=np.float64)
        ext = np.asarray(self.extent_mm, dtype=np.float64)
        s = (nodes[:, 0] - orig[0]) / ext[0]
        t = (nodes[:, 1] - orig[1]) / ext[1]
        u = (nodes[:, 2] - orig[2]) / ext[2]

        bx = _bernstein_basis_1d(s, self.nx - 1)  # (N, nx)
        by = _bernstein_basis_1d(t, self.ny - 1)  # (N, ny)
        bz = _bernstein_basis_1d(u, self.nz - 1)  # (N, nz)
        # Partition of unity: B_ijk(s, t, u) = Bx_i(s) * By_j(t) * Bz_k(u)
        b_mat = (
            bx[:, :, None, None] * by[:, None, :, None] * bz[:, None, None, :]
        ).reshape(nodes.shape[0], self.num_control_points)
        return b_mat

    def morph_mesh(
        self,
        mesh: CanonicalMesh3D,
        delta_control_mm: FloatArray,
        *,
        mesh_id: str | None = None,
    ) -> tuple[CanonicalMesh3D, float]:
        """Deform ``mesh`` by ``delta_control_mm`` ``(N_ctrl, 3)`` with fail-closed det J > 0."""
        delta = np.asarray(delta_control_mm, dtype=np.float64).reshape(
            self.num_control_points, 3
        )
        if not np.all(np.isfinite(delta)):
            raise ValueError("delta_control_mm must contain only finite numbers")
        nodes0 = np.asarray(mesh.nodes, dtype=np.float64)
        b_mat = self.basis_matrix(nodes0)
        nodes_new = nodes0 + b_mat @ delta

        cells = np.asarray(mesh.cells, dtype=np.int64)
        min_det0 = _min_cell_det_j(nodes0, cells, mesh.cell_type)
        min_det = _min_cell_det_j(nodes_new, cells, mesh.cell_type)
        min_allowed = max(1.0e-10, 1.0e-4 * max(min_det0, 0.0))
        if not math.isfinite(min_det) or min_det <= min_allowed:
            raise FfdMeshInversionError(
                f"FFD morph refused: inverted or sliver element "
                f"(min_det_j={min_det:.3e} <= {min_allowed:.3e})"
            )
        morphed = CanonicalMesh3D(
            nodes=tuple((float(r[0]), float(r[1]), float(r[2])) for r in nodes_new),
            cells=mesh.cells,
            cell_type=mesh.cell_type,
            surface_triangles=mesh.surface_triangles,
            semantic_face_tags=mesh.semantic_face_tags,
            unit=mesh.unit,
            metadata={"source_mesh_id": mesh.mesh_id, "label": mesh_id or "ffd"},
        )
        return morphed, min_det


@dataclass(frozen=True, slots=True)
class FfdShapeOptResult:
    """Auditable FEM-verified 3D Bernstein FFD shape optimization output."""

    mesh_id: str
    initial_volume_mm3: float
    optimized_volume_mm3: float
    initial_max_displacement_mm: float
    optimized_max_displacement_mm: float
    min_det_j: float
    fem_evaluations: int
    rejected_inversions: int
    delta_control_mm: FloatArray
    optimized_mesh: CanonicalMesh3D
    receipt_sha256: str
    wall_time_s: float

    def document(self) -> dict[str, Any]:
        return {
            "schema": FFD_SHAPE_OPT_SCHEMA,
            "mesh_id": self.mesh_id,
            "initial_volume_mm3": float(self.initial_volume_mm3),
            "optimized_volume_mm3": float(self.optimized_volume_mm3),
            "volume_reduction_fraction": float(
                (self.initial_volume_mm3 - self.optimized_volume_mm3)
                / max(self.initial_volume_mm3, 1.0e-12)
            ),
            "initial_max_displacement_mm": float(self.initial_max_displacement_mm),
            "optimized_max_displacement_mm": float(self.optimized_max_displacement_mm),
            "min_det_j": float(self.min_det_j),
            "fem_evaluations": self.fem_evaluations,
            "rejected_inversions": self.rejected_inversions,
            "receipt_sha256": self.receipt_sha256,
            "wall_time_s": float(self.wall_time_s),
        }


def optimize_3d_ffd_shape(
    mesh: CanonicalMesh3D,
    material: MaterialSpec,
    *,
    dirichlet_bcs: Sequence[DirichletBC3D],
    traction_bcs: Sequence[TractionBC3D],
    max_displacement_limit_mm: float,
    grid_shape: tuple[int, int, int] = (3, 2, 2),
    max_move_fraction: float = 0.25,
    iterations: int = 8,
) -> FfdShapeOptResult:
    """Minimize 3D solid volume via Bernstein FFD subject to FEM displacement bound.

    Keeps the clamped ``i = 0`` control plane fixed and adjusts free transverse
    control planes (``y, z`` taper) with fail-closed ``min_det_j > 0`` and exact
    3D FEM verification on every accepted iterate.
    """
    t0 = time.perf_counter()
    if not math.isfinite(max_displacement_limit_mm) or max_displacement_limit_mm <= 0.0:
        raise ValueError("max_displacement_limit_mm must be finite and positive")
    if not math.isfinite(max_move_fraction) or not (0.0 < max_move_fraction < 0.95):
        raise ValueError("max_move_fraction must lie in (0, 0.95)")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be an integer >= 1")

    nx, ny, nz = grid_shape
    cage = FFDGrid3D.from_mesh(mesh, nx=nx, ny=ny, nz=nz)
    backend = TorchFemBackend3D()

    nodes0 = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    vol0 = _total_mesh_volume(nodes0, cells, mesh.cell_type)

    def _eval(m_cur: CanonicalMesh3D, tag: str) -> tuple[bool, float]:
        res = backend.solve(
            FemRequest3D(
                request_id=tag,
                mesh=m_cur,
                material=material,
                dirichlet_bcs=tuple(dirichlet_bcs),
                traction_bcs=tuple(traction_bcs),
                load_steps=1,
            )
        )
        ok = bool(res.converged) and math.isfinite(float(res.max_displacement_mm))
        return ok, float(res.max_displacement_mm)

    ok0, u0 = _eval(mesh, "ffd-init")
    if not ok0:
        raise RuntimeError(
            "initial 3D FEM evaluation did not converge for FFD shape optimization"
        )
    fem_evals = 1
    rejected_inv = 0

    # Symmetric transverse taper design variable alpha_i in [-max_move, +max_move]
    # for each axial control plane i = 1..nx-1: moves y,z control points inward/outward.
    n_planes = nx - 1
    alphas = np.zeros(n_planes, dtype=np.float64)
    best_alphas = alphas.copy()
    best_mesh = mesh
    best_vol = vol0
    best_u = u0
    best_det_j = _min_cell_det_j(nodes0, cells, mesh.cell_type)

    def _alphas_to_delta(a_vec: FloatArray) -> FloatArray:
        delta = np.zeros((nx, ny, nz, 3), dtype=np.float64)
        ly = cage.extent_mm[1]
        lz = cage.extent_mm[2]
        for i in range(1, nx):
            a = float(a_vec[i - 1])
            for j in range(ny):
                sy = (j / (ny - 1)) - 0.5  # -0.5 .. +0.5
                for k in range(nz):
                    sz = (k / (nz - 1)) - 0.5
                    delta[i, j, k, 1] = -a * ly * sy
                    delta[i, j, k, 2] = -a * lz * sz
        return delta.reshape(cage.num_control_points, 3)

    # Coordinate search with shrinking step size (guaranteed monotone feasible descent)
    step = min(0.12, max_move_fraction * 0.5)
    for it in range(iterations):
        improved = False
        for p_idx in range(n_planes):
            for sign in (1.0, -1.0):
                cand = best_alphas.copy()
                cand[p_idx] = float(
                    np.clip(
                        cand[p_idx] + sign * step,
                        -max_move_fraction,
                        max_move_fraction,
                    )
                )
                if abs(cand[p_idx] - best_alphas[p_idx]) < 1.0e-12:
                    continue
                try:
                    m_cand, det_j = cage.morph_mesh(
                        mesh, _alphas_to_delta(cand), mesh_id=f"{mesh.mesh_id}-ffd-{it}"
                    )
                except FfdMeshInversionError:
                    rejected_inv += 1
                    continue
                v_cand = _total_mesh_volume(
                    np.asarray(m_cand.nodes, dtype=np.float64), cells, mesh.cell_type
                )
                if v_cand >= best_vol - 1.0e-9 and best_u <= max_displacement_limit_mm:
                    continue
                ok_cand, u_cand = _eval(m_cand, f"ffd-eval-{fem_evals}")
                fem_evals += 1
                if not ok_cand:
                    rejected_inv += 1
                    continue
                feasible = u_cand <= max_displacement_limit_mm
                # Accept if feasible with smaller volume, or reduces infeasibility
                if feasible and (
                    best_u > max_displacement_limit_mm or v_cand < best_vol
                ):
                    best_alphas = cand
                    best_mesh = m_cand
                    best_vol = v_cand
                    best_u = u_cand
                    best_det_j = det_j
                    improved = True
                elif (
                    (not feasible)
                    and (best_u > max_displacement_limit_mm)
                    and (u_cand < best_u)
                ):
                    best_alphas = cand
                    best_mesh = m_cand
                    best_vol = v_cand
                    best_u = u_cand
                    best_det_j = det_j
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1.0e-3:
                break

    best_delta = _alphas_to_delta(best_alphas)
    delta_sha = hashlib.sha256(
        np.ascontiguousarray(best_delta + 0.0, dtype=np.float64).tobytes()
    ).hexdigest()
    receipt = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": FFD_SHAPE_OPT_SCHEMA,
                "source_mesh_id": mesh.mesh_id,
                "mesh_id": best_mesh.mesh_id,
                "max_displacement_limit_mm": float(max_displacement_limit_mm),
                "delta_control_sha256": delta_sha,
                "initial_volume_mm3": float(vol0),
                "optimized_volume_mm3": float(best_vol),
                "optimized_max_displacement_mm": float(best_u),
                "min_det_j": float(best_det_j),
                "fem_evaluations": fem_evals,
            }
        )
    ).hexdigest()

    return FfdShapeOptResult(
        mesh_id=best_mesh.mesh_id,
        initial_volume_mm3=float(vol0),
        optimized_volume_mm3=float(best_vol),
        initial_max_displacement_mm=float(u0),
        optimized_max_displacement_mm=float(best_u),
        min_det_j=float(best_det_j),
        fem_evaluations=fem_evals,
        rejected_inversions=rejected_inv,
        delta_control_mm=best_delta,
        optimized_mesh=best_mesh,
        receipt_sha256=receipt,
        wall_time_s=time.perf_counter() - t0,
    )


__all__ = (
    "FFD_SHAPE_OPT_SCHEMA",
    "FFDGrid3D",
    "FfdMeshInversionError",
    "FfdShapeOptResult",
    "optimize_3d_ffd_shape",
)
