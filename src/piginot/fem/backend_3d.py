"""Pure-MIT 3D finite-strain nonlinear TET10/TET4 FEM backends (M5 / F1).

Provides two independent 3D Total-Lagrangian finite-strain hyperelastic solvers
reusing :mod:`piginot.physics` exact 3x3 constitutive kernels (``energy_density``,
``first_piola``, ``consistent_tangent``) without plane-stress condensation:

* :class:`ScikitFemBackend3D` (``scikit-fem-3d``): built on ``skfem.MeshTet`` +
  ``ElementVector(ElementTetP1 | ElementTetP2)``.
* :class:`TorchFemBackend3D` (``torch-fem-3d``): independent vectorized
  isoparametric ``TET4`` / ``TET10`` reference-element implementation with
  positive-weight symmetric tetrahedral Gauss quadrature.

Both backends enforce :meth:`CanonicalMesh3D.validate_incompressibility_locking`
(refusing ``TET4`` for ``poisson_ratio >= 0.45``), guard ``det(F) > 0`` at every
3D quadrature point, support load-stepping continuation and warm-start from
neural predictions, and return results indexed identically in
:class:`~piginot.geometry.mesh3d.CanonicalMesh3D` node order.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import spsolve  # type: ignore[import-untyped]
from skfem import (  # type: ignore[import-untyped]
    BilinearForm,
    CellBasis,
    ElementTetP1,
    ElementTetP2,
    ElementVector,
    LinearForm,
    MeshTet,
    asm,
)
from skfem.helpers import grad  # type: ignore[import-untyped]

from piginot.contracts import MaterialSpec, canonical_json_bytes
from piginot.fem.protocol import BackendCapabilities
from piginot.geometry.mesh3d import (
    CanonicalMesh3D,
    CellType3D,
    VolumetricLockingError,
)
from piginot.physics import consistent_tangent, energy_density, first_piola

FloatArray = NDArray[np.float64]
FEM_REQUEST_3D_SCHEMA: Final = "formamech.fem-request-3d-v1"
FEM_RESULT_3D_SCHEMA: Final = "formamech.fem-result-3d-v1"


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


@dataclass(frozen=True, slots=True)
class DirichletBC3D:
    """Component-wise 3D Dirichlet boundary condition on one or more semantic faces."""

    name: str
    face_tags: tuple[str, ...]
    components: tuple[str, ...]  # subset of ("x", "y", "z")
    values_mm: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("DirichletBC3D name must be non-empty text")
        if not self.face_tags:
            raise ValueError("face_tags must be non-empty")
        if not self.components or any(
            c not in ("x", "y", "z") for c in self.components
        ):
            raise ValueError("components must be a non-empty subset of ('x', 'y', 'z')")
        if len(self.values_mm) != len(self.components):
            raise ValueError("values_mm must align with components")
        if any(not math.isfinite(float(v)) for v in self.values_mm):
            raise ValueError("values_mm must be finite")

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "face_tags": list(self.face_tags),
            "components": list(self.components),
            "values_mm": [float(v) for v in self.values_mm],
        }


@dataclass(frozen=True, slots=True)
class TractionBC3D:
    """Uniform dead surface traction vector (MPa = N/mm^2) on named 3D faces."""

    name: str
    face_tags: tuple[str, ...]
    traction_mpa: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("TractionBC3D name must be non-empty text")
        if not self.face_tags:
            raise ValueError("face_tags must be non-empty")
        if len(self.traction_mpa) != 3 or any(
            not math.isfinite(float(v)) for v in self.traction_mpa
        ):
            raise ValueError("traction_mpa must be 3 finite numbers (tx, ty, tz)")

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "face_tags": list(self.face_tags),
            "traction_mpa": [float(v) for v in self.traction_mpa],
        }


@dataclass(frozen=True, slots=True)
class FemRequest3D:
    """Content-addressed 3D finite-strain nonlinear solid mechanics FEM request."""

    request_id: str
    mesh: CanonicalMesh3D
    material: MaterialSpec
    dirichlet_bcs: tuple[DirichletBC3D, ...]
    traction_bcs: tuple[TractionBC3D, ...] = ()
    load_steps: int = 4
    solver_tolerance: float = 1.0e-9
    max_newton_iterations: int = 25

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be non-empty text")
        if self.mesh.cell_type not in (CellType3D.TET4, CellType3D.TET10):
            raise ValueError(
                f"3D tetrahedral solver requires tet4 or tet10, got {self.mesh.cell_type.value!r}"
            )
        if not self.dirichlet_bcs:
            raise ValueError("at least one DirichletBC3D is required")
        if type(self.load_steps) is not int or self.load_steps < 1:
            raise ValueError("load_steps must be an integer >= 1")
        if not math.isfinite(self.solver_tolerance) or self.solver_tolerance <= 0.0:
            raise ValueError("solver_tolerance must be finite and positive")
        if (
            type(self.max_newton_iterations) is not int
            or self.max_newton_iterations < 1
        ):
            raise ValueError("max_newton_iterations must be >= 1")
        for bc in self.dirichlet_bcs:
            for tag in bc.face_tags:
                if tag not in self.mesh.semantic_face_tags:
                    raise ValueError(
                        f"DirichletBC3D {bc.name!r} references missing face tag {tag!r}"
                    )
        for tbc in self.traction_bcs:
            for tag in tbc.face_tags:
                if tag not in self.mesh.semantic_face_tags:
                    raise ValueError(
                        f"TractionBC3D {tbc.name!r} references missing face tag {tag!r}"
                    )
        nu_prop = self.material.properties.get("poisson_ratio")
        if nu_prop is not None:
            self.mesh.validate_incompressibility_locking(float(nu_prop.value))

    @property
    def request_hash(self) -> str:
        return _sha256_doc(
            {
                "schema": FEM_REQUEST_3D_SCHEMA,
                "request_id": self.request_id,
                "mesh_id": self.mesh.mesh_id,
                "material": self.material.name,
                "dirichlet_bcs": [b.document() for b in self.dirichlet_bcs],
                "traction_bcs": [t.document() for t in self.traction_bcs],
                "load_steps": self.load_steps,
                "solver_tolerance": self.solver_tolerance,
            }
        )


@dataclass(frozen=True, slots=True)
class FemResult3D:
    """Converged (or failed) 3D finite-strain FEM solution indexed in ``mesh.nodes`` order."""

    request_id: str
    backend_name: str
    mesh_id: str
    converged: bool
    nodal_displacement: FloatArray  # (N_nodes, 3) mm
    nodal_von_mises_mpa: FloatArray  # (N_nodes,) MPa
    reactions_n: Mapping[str, tuple[float, float, float]]
    total_strain_energy_n_mm: float
    external_work_n_mm: float
    min_det_f: float
    max_displacement_mm: float
    peak_von_mises_mpa: float
    final_residual_norm: float
    total_newton_iterations: int
    wall_time_s: float
    reason: str = "converged"

    @property
    def result_hash(self) -> str:
        return _sha256_doc(
            {
                "schema": FEM_RESULT_3D_SCHEMA,
                "request_id": self.request_id,
                "backend_name": self.backend_name,
                "mesh_id": self.mesh_id,
                "converged": self.converged,
                "reactions_n": {
                    k: list(v) for k, v in sorted(self.reactions_n.items())
                },
                "total_strain_energy_n_mm": round(self.total_strain_energy_n_mm, 10),
                "min_det_f": round(self.min_det_f, 10),
                "peak_von_mises_mpa": round(self.peak_von_mises_mpa, 10),
            }
        )

    def summary_document(self) -> dict[str, Any]:
        return {
            "schema": FEM_RESULT_3D_SCHEMA,
            "request_id": self.request_id,
            "backend_name": self.backend_name,
            "mesh_id": self.mesh_id,
            "converged": self.converged,
            "reactions_n": {k: list(v) for k, v in sorted(self.reactions_n.items())},
            "total_strain_energy_n_mm": self.total_strain_energy_n_mm,
            "external_work_n_mm": self.external_work_n_mm,
            "min_det_f": self.min_det_f,
            "max_displacement_mm": self.max_displacement_mm,
            "peak_von_mises_mpa": self.peak_von_mises_mpa,
            "final_residual_norm": self.final_residual_norm,
            "total_newton_iterations": self.total_newton_iterations,
            "wall_time_s": self.wall_time_s,
            "reason": self.reason,
            "result_hash": self.result_hash,
        }


# ---------------------------------------------------------------------------
# Shared 3D surface-node, traction-load, and Cauchy stress helpers
# ---------------------------------------------------------------------------


def _build_edge_midside_map(mesh: CanonicalMesh3D) -> dict[tuple[int, int], int]:
    """Map undirected corner edge ``(u, v)`` to its TET10 midside node index."""
    edge_map: dict[tuple[int, int], int] = {}
    if mesh.cell_type is not CellType3D.TET10:
        return edge_map
    # CanonicalMesh3D TET10: (v0, v1, v2, v3, e01, e12, e20, e03, e13, e23)
    local_edges = ((0, 1, 4), (1, 2, 5), (2, 0, 6), (0, 3, 7), (1, 3, 8), (2, 3, 9))
    for cell in mesh.cells:
        for a_i, b_i, m_i in local_edges:
            u, v = cell[a_i], cell[b_i]
            key = (u, v) if u < v else (v, u)
            edge_map[key] = cell[m_i]
    return edge_map


def face_node_indices_3d(
    mesh: CanonicalMesh3D, face_tags: Sequence[str]
) -> tuple[int, ...]:
    """Return sorted node indices (corners + TET10 edge midpoints) on ``face_tags``."""
    edge_map = _build_edge_midside_map(mesh)
    nodes_found: set[int] = set()
    for tag in face_tags:
        for tri_idx in mesh.semantic_face_tags[tag]:
            a, b, c = mesh.surface_triangles[tri_idx]
            nodes_found.update((a, b, c))
            if edge_map:
                for u, v in ((a, b), (b, c), (c, a)):
                    key = (u, v) if u < v else (v, u)
                    mid = edge_map.get(key)
                    if mid is not None:
                        nodes_found.add(mid)
    return tuple(sorted(nodes_found))


def assemble_surface_tractions_3d(
    mesh: CanonicalMesh3D, traction_bcs: Sequence[TractionBC3D]
) -> FloatArray:
    """Assemble exact consistent nodal force vector ``(N_nodes, 3)`` for dead tractions."""
    n_nodes = len(mesh.nodes)
    f_ext = np.zeros((n_nodes, 3), dtype=np.float64)
    if not traction_bcs:
        return f_ext
    edge_map = _build_edge_midside_map(mesh)
    is_p2 = mesh.cell_type is CellType3D.TET10
    pts = np.asarray(mesh.nodes, dtype=np.float64)

    for tbc in traction_bcs:
        t_vec = np.asarray(tbc.traction_mpa, dtype=np.float64)
        for tag in tbc.face_tags:
            for tri_idx in mesh.semantic_face_tags[tag]:
                a, b, c = mesh.surface_triangles[tri_idx]
                pa, pb, pc = pts[a], pts[b], pts[c]
                area = 0.5 * float(np.linalg.norm(np.cross(pb - pa, pc - pa)))
                if area <= 0.0:
                    continue
                if is_p2:
                    # Exact P2 6-node triangle integral: corners = 0, midsides = A/3 each
                    mids: list[int] = []
                    for u, v in ((a, b), (b, c), (c, a)):
                        key = (u, v) if u < v else (v, u)
                        mid = edge_map.get(key)
                        if mid is not None:
                            mids.append(mid)
                    if len(mids) == 3:
                        for mid in mids:
                            f_ext[mid] += (area / 3.0) * t_vec
                        continue
                # Linear triangle fallback: A/3 on each corner
                for nid in (a, b, c):
                    f_ext[nid] += (area / 3.0) * t_vec
    return f_ext


def _cauchy_von_mises_3d(p_arr: FloatArray, f_arr: FloatArray) -> FloatArray:
    """Exact 3D Cauchy von Mises stress from 3x3 ``P`` and ``F``."""
    j = np.linalg.det(f_arr)
    j_safe = np.where(np.abs(j) > 1.0e-12, j, 1.0)
    sigma = np.einsum("...ij,...kj->...ik", p_arr, f_arr) / j_safe[..., None, None]
    tr = sigma[..., 0, 0] + sigma[..., 1, 1] + sigma[..., 2, 2]
    dev = sigma - (tr / 3.0)[..., None, None] * np.eye(3, dtype=np.float64)
    vm_sq = 1.5 * np.sum(dev * dev, axis=(-2, -1))
    return np.asarray(np.sqrt(np.maximum(vm_sq, 0.0)), dtype=np.float64)


def _project_cell_scalar_to_nodes_3d(
    mesh: CanonicalMesh3D, cell_values: FloatArray, cell_volumes: FloatArray
) -> FloatArray:
    n_nodes = len(mesh.nodes)
    num = np.zeros(n_nodes, dtype=np.float64)
    den = np.zeros(n_nodes, dtype=np.float64)
    for c_idx, cell in enumerate(mesh.cells):
        w = float(cell_volumes[c_idx])
        val = float(cell_values[c_idx])
        for nid in cell:
            num[nid] += w * val
            den[nid] += w
    return np.asarray(num / np.maximum(den, 1.0e-12), dtype=np.float64)


# ---------------------------------------------------------------------------
# Backend 1: TorchFemBackend3D (independent vectorized isoparametric TET4/TET10)
# ---------------------------------------------------------------------------


def _tet_quadrature(cell_type: CellType3D) -> tuple[FloatArray, FloatArray]:
    """Return strictly positive-weight symmetric Gauss quadrature ``(xi (Q,3), w (Q,))``."""
    if cell_type is CellType3D.TET4:
        return (
            np.array([[0.25, 0.25, 0.25]], dtype=np.float64),
            np.array([1.0 / 6.0], dtype=np.float64),
        )
    # 4-point symmetric Keast/Hammer rule (exact degree 2, all weights = +1/24)
    a = (5.0 + 3.0 * math.sqrt(5.0)) / 20.0
    b = (5.0 - math.sqrt(5.0)) / 20.0
    pts = np.array(
        [
            [b, b, b],
            [a, b, b],
            [b, a, b],
            [b, b, a],
        ],
        dtype=np.float64,
    )
    wts = np.full(4, 1.0 / 24.0, dtype=np.float64)
    return pts, wts


def _tet_shape_grads_ref(cell_type: CellType3D, xi_pts: FloatArray) -> FloatArray:
    """Reference shape-function gradients ``(Q, n_loc, 3)`` matching ``CanonicalMesh3D``."""
    q = xi_pts.shape[0]
    if cell_type is CellType3D.TET4:
        g = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return np.broadcast_to(g[None, :, :], (q, 4, 3)).copy()

    # TET10: (0,1,2,3, mid01, mid12, mid20, mid03, mid13, mid23)
    d_l = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    out = np.zeros((q, 10, 3), dtype=np.float64)
    for k in range(q):
        xi, eta, zeta = float(xi_pts[k, 0]), float(xi_pts[k, 1]), float(xi_pts[k, 2])
        lam = (1.0 - xi - eta - zeta, xi, eta, zeta)
        for a in range(4):
            out[k, a, :] = (4.0 * lam[a] - 1.0) * d_l[a]
        pairs = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
        for offset, (a, b) in enumerate(pairs, start=4):
            out[k, offset, :] = 4.0 * (lam[a] * d_l[b] + lam[b] * d_l[a])
    return out


class TorchFemBackend3D:
    """Independent vectorized 3D finite-strain ``TET4``/``TET10`` nonlinear FEM backend."""

    backend_name = "torch-fem-3d"
    backend_id = "torch-fem-3d-v1"
    capabilities = BackendCapabilities(
        spatial_dimensions=(3,),
        physics_modes=("3d-solid-finite-strain",),
        element_orders=(1, 2),
        supports_warm_start=True,
        supports_linearization=False,
        supports_dead_tractions=True,
        supports_follower_pressure=False,
        supports_contact=False,
        supports_gpu=False,
        supports_mpi=False,
        license_spdx="MIT",
    )

    def _precompute_geometry(
        self, mesh: CanonicalMesh3D
    ) -> tuple[
        NDArray[np.int64],
        FloatArray,
        FloatArray,
        FloatArray,
        NDArray[np.int64],
        NDArray[np.int64],
    ]:
        nodes = np.asarray(mesh.nodes, dtype=np.float64)
        cells = np.asarray(mesh.cells, dtype=np.int64)
        xi_pts, w_ref = _tet_quadrature(mesh.cell_type)
        d_ref = _tet_shape_grads_ref(mesh.cell_type, xi_pts)  # (Q, n_loc, 3)

        x_elem = nodes[cells]  # (E, n_loc, 3)
        # J_{e,q,i,alpha} = dX_i / dxi_alpha = sum_n X_{e,n,i} * dN_{q,n,alpha}
        jac = np.einsum("eni,qna->eqia", x_elem, d_ref)  # (E, Q, 3, 3)
        det_j = np.linalg.det(jac)  # (E, Q)
        if np.any(det_j <= 0.0):
            raise ValueError(
                "3D tetrahedral element has non-positive Jacobian determinant"
            )
        inv_j = np.linalg.inv(jac)  # (E, Q, 3, 3): dxi_alpha / dX_j
        # dN_dX_{e,q,n,j} = sum_alpha dN_{q,n,alpha} * inv_j_{e,q,alpha,j}
        d_phys = np.einsum("qna,eqaj->eqnj", d_ref, inv_j)  # (E, Q, n_loc, 3)
        w_vol = det_j * w_ref[None, :]  # (E, Q)
        cell_vols = np.sum(w_vol, axis=1)  # (E,)

        # Precompute sparse COO index arrays for vectorized global tangent assembly
        n_elem, n_loc = cells.shape
        n_edof = 3 * n_loc
        edofs = np.empty((n_elem, n_edof), dtype=np.int64)
        for c in range(3):
            edofs[:, c::3] = 3 * cells + c
        rows = np.repeat(edofs, n_edof, axis=1).reshape(-1)
        cols = np.tile(edofs, (1, n_edof)).reshape(-1)
        return cells, d_phys, w_vol, cell_vols, rows, cols

    def _assemble(
        self,
        u_flat: FloatArray,
        cells: NDArray[np.int64],
        d_phys: FloatArray,
        w_vol: FloatArray,
        rows: NDArray[np.int64],
        cols: NDArray[np.int64],
        material: MaterialSpec,
        n_dofs: int,
    ) -> tuple[FloatArray, csr_matrix, FloatArray, FloatArray, float, float]:
        u_nodal = u_flat.reshape(-1, 3)
        u_elem = u_nodal[cells]  # (E, n_loc, 3)
        grad_u = np.einsum("eni,eqnj->eqij", u_elem, d_phys)  # (E, Q, 3, 3)
        f_np = grad_u + np.eye(3, dtype=np.float64)[None, None, :, :]
        det_f = np.linalg.det(f_np)
        min_det = float(np.min(det_f))
        if min_det <= 1.0e-8 or not np.all(np.isfinite(f_np)):
            return (
                np.zeros(n_dofs, dtype=np.float64),
                csr_matrix((n_dofs, n_dofs), dtype=np.float64),
                np.zeros_like(f_np),
                f_np,
                float("inf"),
                min_det,
            )

        with torch.no_grad():
            f_t = torch.from_numpy(np.ascontiguousarray(f_np))
            p_t = first_piola(f_t, material)
            c_t = consistent_tangent(f_t, material)
            w_t = energy_density(f_t, material)
        p_np = p_t.numpy()  # (E, Q, 3, 3)
        c_np = c_t.numpy()  # (E, Q, 3, 3, 3, 3)
        w_np = w_t.numpy()  # (E, Q)
        total_energy = float(np.sum(w_np * w_vol))

        # Internal force per element: f_{e,n,i} = sum_{q,j} w_{e,q} * P_{e,q,i,j} * dN_{e,q,n,j}
        f_elem = np.einsum("eq,eqij,eqnj->eni", w_vol, p_np, d_phys)  # (E, n_loc, 3)
        f_int = np.zeros((n_dofs // 3, 3), dtype=np.float64)
        np.add.at(f_int, cells, f_elem)

        # Element tangent: K_{e, n, i, m, k} = sum_{q,j,l} w_{e,q} C_{e,q,i,j,k,l} dN_{e,q,n,j} dN_{e,q,m,l}
        # Split into two contractions for optimal BLAS speed
        tmp = np.einsum("eq,eqijkl,eqnj->eqnikl", w_vol, c_np, d_phys)
        k_elem = np.einsum("eqnikl,eqml->enimk", tmp, d_phys)
        n_elem, n_loc = cells.shape
        n_edof = 3 * n_loc
        k_flat = k_elem.reshape(n_elem, n_edof, n_edof).reshape(-1)
        k_mat = coo_matrix((k_flat, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
        return f_int.reshape(-1), k_mat, p_np, f_np, total_energy, min_det

    def solve(
        self,
        request: FemRequest3D,
        *,
        initial_displacement: FloatArray | None = None,
    ) -> FemResult3D:
        """Solve a 3D finite-strain request with load stepping and damped Newton."""
        t0 = time.perf_counter()
        mesh = request.mesh
        n_nodes = len(mesh.nodes)
        n_dofs = 3 * n_nodes
        comp_idx = {"x": 0, "y": 1, "z": 2}

        cells, d_phys, w_vol, cell_vols, rows, cols = self._precompute_geometry(mesh)
        f_ext_full = assemble_surface_tractions_3d(mesh, request.traction_bcs).reshape(
            -1
        )

        # Map constrained DOFs and per-BC DOF groups for reaction force extraction
        bc_dofs: dict[str, tuple[list[int], list[int], list[int]]] = {}
        target_dirichlet: dict[int, float] = {}
        for bc in request.dirichlet_bcs:
            nids = face_node_indices_3d(mesh, bc.face_tags)
            comp_lists: tuple[list[int], list[int], list[int]] = ([], [], [])
            for comp, val in zip(bc.components, bc.values_mm, strict=True):
                c = comp_idx[comp]
                for nid in nids:
                    dof = 3 * nid + c
                    target_dirichlet[dof] = float(val)
                    comp_lists[c].append(dof)
            bc_dofs[bc.name] = comp_lists

        constrained_mask = np.zeros(n_dofs, dtype=bool)
        for dof in target_dirichlet:
            constrained_mask[dof] = True
        free_dofs = np.nonzero(~constrained_mask)[0]

        if initial_displacement is not None:
            u = (
                np.asarray(initial_displacement, dtype=np.float64)
                .reshape(n_dofs)
                .copy()
            )
            factors = [1.0]
        else:
            u = np.zeros(n_dofs, dtype=np.float64)
            factors = [
                float(step) / float(request.load_steps)
                for step in range(1, request.load_steps + 1)
            ]

        total_iters = 0
        last_res = float("inf")
        last_energy = 0.0
        last_min_det = 1.0
        last_f_int: FloatArray = np.zeros(n_dofs, dtype=np.float64)
        last_p: FloatArray = np.zeros(
            (cells.shape[0], d_phys.shape[1], 3, 3), dtype=np.float64
        )
        last_f: FloatArray = np.zeros_like(last_p)

        for factor in factors:
            f_ext = factor * f_ext_full
            step_target = {dof: factor * val for dof, val in target_dirichlet.items()}

            # Linearized tangent predictor for smooth Dirichlet continuation
            du_d = np.zeros(n_dofs, dtype=np.float64)
            has_du_d = False
            for dof, val in step_target.items():
                diff = val - u[dof]
                if abs(diff) > 0.0:
                    du_d[dof] = diff
                    has_du_d = True

            if has_du_d and free_dofs.size > 0 and initial_displacement is None:
                f_int0, k0, _, _, _, min_det0 = self._assemble(
                    u, cells, d_phys, w_vol, rows, cols, request.material, n_dofs
                )
                if min_det0 > 1.0e-6:
                    rhs0 = -(f_int0 - f_ext + k0 @ du_d)[free_dofs]
                    du_pred = np.asarray(
                        spsolve(k0[free_dofs][:, free_dofs].tocsc(), rhs0),
                        dtype=np.float64,
                    )
                    if np.all(np.isfinite(du_pred)):
                        u_try = u.copy()
                        u_try[free_dofs] += du_pred
                        for dof, val in step_target.items():
                            u_try[dof] = val
                        _, _, _, _, _, det_try = self._assemble(
                            u_try,
                            cells,
                            d_phys,
                            w_vol,
                            rows,
                            cols,
                            request.material,
                            n_dofs,
                        )
                        if det_try > 1.0e-4:
                            u = u_try
            for dof, val in step_target.items():
                u[dof] = val

            step_converged = False
            for step_iter in range(1, request.max_newton_iterations + 1):
                total_iters += 1
                f_int, k_mat, p_np, f_np, energy, min_det = self._assemble(
                    u, cells, d_phys, w_vol, rows, cols, request.material, n_dofs
                )
                if min_det <= 1.0e-8:
                    return self._failed_result(
                        request,
                        "non-positive 3D Jacobian determinant det(F) <= 0",
                        total_iters,
                        time.perf_counter() - t0,
                    )
                res = f_int - f_ext
                res_free = res[free_dofs]
                res_norm = float(np.linalg.norm(res_free))
                pi_curr = energy - float(np.dot(f_ext, u))
                last_res = res_norm
                last_energy = energy
                last_min_det = min_det
                last_f_int = f_int
                last_p = p_np
                last_f = f_np

                if res_norm <= request.solver_tolerance:
                    step_converged = True
                    break

                du_free = np.asarray(
                    spsolve(k_mat[free_dofs][:, free_dofs].tocsc(), -res_free),
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(du_free)):
                    return self._failed_result(
                        request,
                        "non-finite 3D Newton linear solve increment",
                        total_iters,
                        time.perf_counter() - t0,
                    )

                # Potential-energy & det(F) > 0 backtracking (finite-rotation safe)
                alpha = 1.0
                accepted_ls = False
                for _ in range(10):
                    u_cand = u.copy()
                    u_cand[free_dofs] += alpha * du_free
                    u_ec = u_cand.reshape(-1, 3)[cells]
                    f_c = (
                        np.einsum("eni,eqnj->eqij", u_ec, d_phys)
                        + np.eye(3, dtype=np.float64)[None, None, :, :]
                    )
                    det_c = float(np.min(np.linalg.det(f_c)))
                    if det_c > 1.0e-5 and np.all(np.isfinite(f_c)):
                        if step_iter == 1 or alpha <= 0.125:
                            u = u_cand
                            accepted_ls = True
                            break
                        with torch.no_grad():
                            w_c = energy_density(
                                torch.from_numpy(np.ascontiguousarray(f_c)),
                                request.material,
                            ).numpy()
                        pi_c = float(np.sum(w_c * w_vol)) - float(np.dot(f_ext, u_cand))
                        if pi_c <= pi_curr + 1.0e-7 * max(abs(pi_curr), 1.0):
                            u = u_cand
                            accepted_ls = True
                            break
                    alpha *= 0.5
                if not accepted_ls:
                    return self._failed_result(
                        request,
                        "3D line search failed to preserve det(F) > 0",
                        total_iters,
                        time.perf_counter() - t0,
                    )

            if not step_converged:
                return self._failed_result(
                    request,
                    f"3D Newton did not converge at load factor {factor:.3f} (res={last_res:.3e})",
                    total_iters,
                    time.perf_counter() - t0,
                )

        # Post-process nodal von Mises stress & per-BC reaction forces
        vm_q = _cauchy_von_mises_3d(last_p, last_f)  # (E, Q)
        vm_cell = np.sum(vm_q * w_vol, axis=1) / np.maximum(cell_vols, 1.0e-12)
        vm_nodes = _project_cell_scalar_to_nodes_3d(mesh, vm_cell, cell_vols)

        reactions: dict[str, tuple[float, float, float]] = {}
        r_vec = last_f_int - f_ext_full
        for bc_name, (dx_l, dy_l, dz_l) in bc_dofs.items():
            reactions[bc_name] = (
                float(np.sum(r_vec[dx_l])) if dx_l else 0.0,
                float(np.sum(r_vec[dy_l])) if dy_l else 0.0,
                float(np.sum(r_vec[dz_l])) if dz_l else 0.0,
            )

        u_nodal = u.reshape(n_nodes, 3)
        ext_work = float(np.dot(f_ext_full, u))
        return FemResult3D(
            request_id=request.request_id,
            backend_name=self.backend_name,
            mesh_id=mesh.mesh_id,
            converged=True,
            nodal_displacement=u_nodal,
            nodal_von_mises_mpa=vm_nodes,
            reactions_n=reactions,
            total_strain_energy_n_mm=last_energy,
            external_work_n_mm=ext_work,
            min_det_f=last_min_det,
            max_displacement_mm=float(np.max(np.linalg.norm(u_nodal, axis=1))),
            peak_von_mises_mpa=float(np.max(vm_nodes)),
            final_residual_norm=last_res,
            total_newton_iterations=total_iters,
            wall_time_s=time.perf_counter() - t0,
            reason="converged",
        )

    def _failed_result(
        self, request: FemRequest3D, reason: str, iters: int, elapsed: float
    ) -> FemResult3D:
        n_nodes = len(request.mesh.nodes)
        return FemResult3D(
            request_id=request.request_id,
            backend_name=self.backend_name,
            mesh_id=request.mesh.mesh_id,
            converged=False,
            nodal_displacement=np.zeros((n_nodes, 3), dtype=np.float64),
            nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
            reactions_n={},
            total_strain_energy_n_mm=0.0,
            external_work_n_mm=0.0,
            min_det_f=0.0,
            max_displacement_mm=0.0,
            peak_von_mises_mpa=0.0,
            final_residual_norm=float("inf"),
            total_newton_iterations=iters,
            wall_time_s=elapsed,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Backend 2: ScikitFemBackend3D (independent skfem.MeshTet P1/P2 assembly)
# ---------------------------------------------------------------------------


@LinearForm  # type: ignore[untyped-decorator]
def _skfem_3d_internal_form(test: Any, data: Any) -> Any:
    return np.einsum("ij...,ij...->...", data.stress, grad(test))


@BilinearForm  # type: ignore[untyped-decorator]
def _skfem_3d_tangent_form(increment: Any, test: Any, data: Any) -> Any:
    return np.einsum(
        "ij...,ijkl...,kl...->...", grad(test), data.modulus, grad(increment)
    )


class ScikitFemBackend3D:
    """Independent ``skfem.MeshTet`` 3D finite-strain ``TET4``/``TET10`` nonlinear backend."""

    backend_name = "scikit-fem-3d"
    backend_id = "scikit-fem-3d-v1"
    capabilities = BackendCapabilities(
        spatial_dimensions=(3,),
        physics_modes=("3d-solid-finite-strain",),
        element_orders=(1, 2),
        supports_warm_start=True,
        supports_linearization=False,
        supports_dead_tractions=True,
        supports_follower_pressure=False,
        supports_contact=False,
        supports_gpu=False,
        supports_mpi=False,
        license_spdx="BSD-3-Clause",
    )

    @staticmethod
    def _build_skfem_basis_and_node_map(
        mesh: CanonicalMesh3D,
    ) -> tuple[Any, NDArray[np.int64], NDArray[np.int64]]:
        """Build ``skfem`` 3D basis and exact bidirectional node mapping to ``mesh.nodes``."""
        all_nodes = np.asarray(mesh.nodes, dtype=np.float64)
        cells = np.asarray(mesh.cells, dtype=np.int64)
        corner_indices = sorted({int(i) for cell in mesh.cells for i in cell[:4]})
        old_to_corner = {old: new for new, old in enumerate(corner_indices)}
        corner_pts = all_nodes[corner_indices].T  # (3, N_corners)
        corner_tets = np.array(
            [[old_to_corner[int(c[k])] for k in range(4)] for c in cells],
            dtype=np.int64,
        ).T  # (4, E)

        skmesh = MeshTet(corner_pts, corner_tets)
        scalar_elem = (
            ElementTetP1() if mesh.cell_type is CellType3D.TET4 else ElementTetP2()
        )
        basis = CellBasis(skmesh, ElementVector(scalar_elem))
        scalar_basis = CellBasis(skmesh, scalar_elem)

        # Match each scalar skfem DOF location to its canonical mesh.nodes index
        sk_locs = np.asarray(scalar_basis.doflocs, dtype=np.float64).T  # (N_sk, 3)
        # Quantized spatial lookup table for O(N) exact matching
        bbox_scale = max(float(np.ptp(all_nodes)), 1.0)
        tol = 1.0e-5 * bbox_scale

        def _key(pt: FloatArray) -> tuple[int, int, int]:
            return (
                int(round(float(pt[0]) / tol)),
                int(round(float(pt[1]) / tol)),
                int(round(float(pt[2]) / tol)),
            )

        canon_lookup: dict[tuple[int, int, int], list[int]] = {}
        for idx, pt in enumerate(all_nodes):
            canon_lookup.setdefault(_key(pt), []).append(idx)

        sk_to_canon = np.empty(sk_locs.shape[0], dtype=np.int64)
        canon_to_sk = np.full(all_nodes.shape[0], -1, dtype=np.int64)
        for sk_i, pt in enumerate(sk_locs):
            kx, ky, kz = _key(pt)
            best_idx = -1
            best_dist = float("inf")
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for cand in canon_lookup.get((kx + dx, ky + dy, kz + dz), ()):
                            dist = float(np.linalg.norm(all_nodes[cand] - pt))
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = cand
            if best_idx < 0 or best_dist > 1.0e-2 * bbox_scale:
                raise ValueError(
                    f"could not align skfem 3D DOF {sk_i} to CanonicalMesh3D nodes "
                    f"(min_dist={best_dist:.3e})"
                )
            sk_to_canon[sk_i] = best_idx
            canon_to_sk[best_idx] = sk_i

        # Build canonical-node -> vector-basis DOF index table (N_canon, 3)
        # In skfem ElementVector(scalar_elem), vector DOFs are interleaved by component
        # across nodal_dofs and edge_dofs: match via basis.doflocs directly!
        vec_locs = np.asarray(basis.doflocs, dtype=np.float64).T  # (3*N_sk, 3)
        canon_to_vec = np.full((all_nodes.shape[0], 3), -1, dtype=np.int64)
        # For each spatial location there are 3 consecutive or component-grouped DOFs;
        # use basis.nodal_dofs / basis.edge_dofs directly for exact component assignment!
        comp_dofs: list[NDArray[np.int64]] = []
        for comp in range(3):
            parts = [np.asarray(basis.nodal_dofs[comp], dtype=np.int64)]
            if mesh.cell_type is CellType3D.TET10:
                parts.append(np.asarray(basis.edge_dofs[comp], dtype=np.int64))
            comp_dofs.append(np.concatenate(parts))

        for comp in range(3):
            for v_dof in comp_dofs[comp]:
                pt = vec_locs[v_dof]
                kx, ky, kz = _key(pt)
                best_idx = -1
                best_dist = float("inf")
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            for cand in canon_lookup.get(
                                (kx + dx, ky + dy, kz + dz), ()
                            ):
                                dist = float(np.linalg.norm(all_nodes[cand] - pt))
                                if dist < best_dist:
                                    best_dist = dist
                                    best_idx = cand
                canon_to_vec[best_idx, comp] = int(v_dof)

        if np.any(canon_to_vec < 0):
            raise ValueError("incomplete skfem-to-CanonicalMesh3D 3D vector DOF map")
        return basis, canon_to_vec, sk_to_canon

    def solve(
        self,
        request: FemRequest3D,
        *,
        initial_displacement: FloatArray | None = None,
    ) -> FemResult3D:
        """Solve a 3D finite-strain request via ``skfem.MeshTet`` assembly."""
        t0 = time.perf_counter()
        mesh = request.mesh
        n_nodes = len(mesh.nodes)
        basis, canon_to_vec, _ = self._build_skfem_basis_and_node_map(mesh)
        n_dofs = int(basis.N)
        comp_idx = {"x": 0, "y": 1, "z": 2}
        dx_w = np.asarray(basis.dx, dtype=np.float64)  # (E, Q)
        cell_vols = np.sum(dx_w, axis=1)

        # Map consistent external nodal tractions into skfem DOF vector
        f_ext_nodal = assemble_surface_tractions_3d(mesh, request.traction_bcs)
        f_ext_full = np.zeros(n_dofs, dtype=np.float64)
        for nid in range(n_nodes):
            for c in range(3):
                f_ext_full[canon_to_vec[nid, c]] = f_ext_nodal[nid, c]

        bc_dofs: dict[str, tuple[list[int], list[int], list[int]]] = {}
        target_dirichlet: dict[int, float] = {}
        for bc in request.dirichlet_bcs:
            nids = face_node_indices_3d(mesh, bc.face_tags)
            comp_lists: tuple[list[int], list[int], list[int]] = ([], [], [])
            for comp, val in zip(bc.components, bc.values_mm, strict=True):
                c = comp_idx[comp]
                for nid in nids:
                    dof = int(canon_to_vec[nid, c])
                    target_dirichlet[dof] = float(val)
                    comp_lists[c].append(dof)
            bc_dofs[bc.name] = comp_lists

        constrained_mask = np.zeros(n_dofs, dtype=bool)
        for dof in target_dirichlet:
            constrained_mask[dof] = True
        free_dofs = np.nonzero(~constrained_mask)[0]

        if initial_displacement is not None:
            u_init_n = np.asarray(initial_displacement, dtype=np.float64).reshape(
                n_nodes, 3
            )
            u = np.zeros(n_dofs, dtype=np.float64)
            for nid in range(n_nodes):
                for c in range(3):
                    u[canon_to_vec[nid, c]] = u_init_n[nid, c]
            factors = [1.0]
        else:
            u = np.zeros(n_dofs, dtype=np.float64)
            factors = [
                float(step) / float(request.load_steps)
                for step in range(1, request.load_steps + 1)
            ]

        def _eval_state(
            u_vec: FloatArray,
        ) -> tuple[FloatArray, csr_matrix, FloatArray, FloatArray, float, float]:
            grad_u = np.asarray(basis.interpolate(u_vec).grad, dtype=np.float64)
            # skfem grad layout: (3, 3, E, Q) -> transpose to (E, Q, 3, 3)
            f_np = np.transpose(grad_u, (2, 3, 0, 1)) + np.eye(3, dtype=np.float64)
            min_det = float(np.min(np.linalg.det(f_np)))
            if min_det <= 1.0e-8 or not np.all(np.isfinite(f_np)):
                return (
                    np.zeros(n_dofs, dtype=np.float64),
                    csr_matrix((n_dofs, n_dofs), dtype=np.float64),
                    np.zeros_like(f_np),
                    f_np,
                    float("inf"),
                    min_det,
                )
            with torch.no_grad():
                f_t = torch.from_numpy(np.ascontiguousarray(f_np))
                p_np = first_piola(f_t, request.material).numpy()
                c_np = consistent_tangent(f_t, request.material).numpy()
                w_np = energy_density(f_t, request.material).numpy()
            total_e = float(np.sum(w_np * dx_w))
            s_field = np.transpose(p_np, (2, 3, 0, 1))
            m_field = np.transpose(c_np, (2, 3, 4, 5, 0, 1))
            f_int = np.asarray(
                asm(_skfem_3d_internal_form, basis, stress=s_field), dtype=np.float64
            )
            k_mat = asm(_skfem_3d_tangent_form, basis, modulus=m_field).tocsr()
            return f_int, k_mat, p_np, f_np, total_e, min_det

        total_iters = 0
        last_res = float("inf")
        last_energy = 0.0
        last_min_det = 1.0
        last_f_int: FloatArray = np.zeros(n_dofs, dtype=np.float64)
        last_p: FloatArray = np.zeros(
            (dx_w.shape[0], dx_w.shape[1], 3, 3), dtype=np.float64
        )
        last_f: FloatArray = np.zeros_like(last_p)

        for factor in factors:
            f_ext = factor * f_ext_full
            step_target = {dof: factor * val for dof, val in target_dirichlet.items()}
            du_d = np.zeros(n_dofs, dtype=np.float64)
            has_du_d = False
            for dof, val in step_target.items():
                diff = val - u[dof]
                if abs(diff) > 0.0:
                    du_d[dof] = diff
                    has_du_d = True
            if has_du_d and free_dofs.size > 0 and initial_displacement is None:
                f_int0, k0, _, _, _, min_det0 = _eval_state(u)
                if min_det0 > 1.0e-6:
                    rhs0 = -(f_int0 - f_ext + k0 @ du_d)[free_dofs]
                    du_pred = np.asarray(
                        spsolve(k0[free_dofs][:, free_dofs].tocsc(), rhs0),
                        dtype=np.float64,
                    )
                    if np.all(np.isfinite(du_pred)):
                        u_try = u.copy()
                        u_try[free_dofs] += du_pred
                        for dof, val in step_target.items():
                            u_try[dof] = val
                        grad_try = np.asarray(
                            basis.interpolate(u_try).grad, dtype=np.float64
                        )
                        f_try = np.transpose(grad_try, (2, 3, 0, 1)) + np.eye(
                            3, dtype=np.float64
                        )
                        if float(np.min(np.linalg.det(f_try))) > 1.0e-4:
                            u = u_try
            for dof, val in step_target.items():
                u[dof] = val

            step_converged = False
            for step_iter in range(1, request.max_newton_iterations + 1):
                total_iters += 1
                f_int, k_mat, p_np, f_np, energy, min_det = _eval_state(u)
                if min_det <= 1.0e-8:
                    return FemResult3D(
                        request_id=request.request_id,
                        backend_name=self.backend_name,
                        mesh_id=mesh.mesh_id,
                        converged=False,
                        nodal_displacement=np.zeros((n_nodes, 3), dtype=np.float64),
                        nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
                        reactions_n={},
                        total_strain_energy_n_mm=0.0,
                        external_work_n_mm=0.0,
                        min_det_f=min_det,
                        max_displacement_mm=0.0,
                        peak_von_mises_mpa=0.0,
                        final_residual_norm=float("inf"),
                        total_newton_iterations=total_iters,
                        wall_time_s=time.perf_counter() - t0,
                        reason="non-positive 3D det(F) <= 0",
                    )
                res_free = (f_int - f_ext)[free_dofs]
                res_norm = float(np.linalg.norm(res_free))
                pi_curr = energy - float(np.dot(f_ext, u))
                last_res = res_norm
                last_energy = energy
                last_min_det = min_det
                last_f_int = f_int
                last_p = p_np
                last_f = f_np
                if res_norm <= request.solver_tolerance:
                    step_converged = True
                    break

                du_free = np.asarray(
                    spsolve(k_mat[free_dofs][:, free_dofs].tocsc(), -res_free),
                    dtype=np.float64,
                )
                alpha = 1.0
                accepted = False
                for _ in range(10):
                    u_cand = u.copy()
                    u_cand[free_dofs] += alpha * du_free
                    grad_c = np.asarray(
                        basis.interpolate(u_cand).grad, dtype=np.float64
                    )
                    f_c = np.transpose(grad_c, (2, 3, 0, 1)) + np.eye(
                        3, dtype=np.float64
                    )
                    if float(np.min(np.linalg.det(f_c))) > 1.0e-5 and np.all(
                        np.isfinite(f_c)
                    ):
                        if step_iter == 1 or alpha <= 0.125:
                            u = u_cand
                            accepted = True
                            break
                        with torch.no_grad():
                            w_c = energy_density(
                                torch.from_numpy(np.ascontiguousarray(f_c)),
                                request.material,
                            ).numpy()
                        pi_c = float(np.sum(w_c * dx_w)) - float(np.dot(f_ext, u_cand))
                        if pi_c <= pi_curr + 1.0e-7 * max(abs(pi_curr), 1.0):
                            u = u_cand
                            accepted = True
                            break
                    alpha *= 0.5
                if not accepted:
                    u[free_dofs] += alpha * du_free

            if not step_converged:
                return FemResult3D(
                    request_id=request.request_id,
                    backend_name=self.backend_name,
                    mesh_id=mesh.mesh_id,
                    converged=False,
                    nodal_displacement=np.zeros((n_nodes, 3), dtype=np.float64),
                    nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
                    reactions_n={},
                    total_strain_energy_n_mm=0.0,
                    external_work_n_mm=0.0,
                    min_det_f=last_min_det,
                    max_displacement_mm=0.0,
                    peak_von_mises_mpa=0.0,
                    final_residual_norm=last_res,
                    total_newton_iterations=total_iters,
                    wall_time_s=time.perf_counter() - t0,
                    reason=f"skfem 3D Newton did not converge (res={last_res:.3e})",
                )

        # Map solution back to canonical mesh.nodes order
        u_nodal = np.zeros((n_nodes, 3), dtype=np.float64)
        for nid in range(n_nodes):
            for c in range(3):
                u_nodal[nid, c] = u[canon_to_vec[nid, c]]

        vm_q = _cauchy_von_mises_3d(last_p, last_f)
        vm_cell = np.sum(vm_q * dx_w, axis=1) / np.maximum(cell_vols, 1.0e-12)
        vm_nodes = _project_cell_scalar_to_nodes_3d(mesh, vm_cell, cell_vols)

        reactions: dict[str, tuple[float, float, float]] = {}
        r_vec = last_f_int - f_ext_full
        for bc_name, (dx_l, dy_l, dz_l) in bc_dofs.items():
            reactions[bc_name] = (
                float(np.sum(r_vec[dx_l])) if dx_l else 0.0,
                float(np.sum(r_vec[dy_l])) if dy_l else 0.0,
                float(np.sum(r_vec[dz_l])) if dz_l else 0.0,
            )

        return FemResult3D(
            request_id=request.request_id,
            backend_name=self.backend_name,
            mesh_id=mesh.mesh_id,
            converged=True,
            nodal_displacement=u_nodal,
            nodal_von_mises_mpa=vm_nodes,
            reactions_n=reactions,
            total_strain_energy_n_mm=last_energy,
            external_work_n_mm=float(np.dot(f_ext_full, u)),
            min_det_f=last_min_det,
            max_displacement_mm=float(np.max(np.linalg.norm(u_nodal, axis=1))),
            peak_von_mises_mpa=float(np.max(vm_nodes)),
            final_residual_norm=last_res,
            total_newton_iterations=total_iters,
            wall_time_s=time.perf_counter() - t0,
            reason="converged",
        )


def analytical_3d_uniaxial_hyperelastic(
    material: MaterialSpec, axial_stretch: float
) -> dict[str, float]:
    """Exact 3D finite-strain uniaxial response (``P_22 = P_33 = 0``) via scalar Newton."""
    if not math.isfinite(axial_stretch) or axial_stretch <= 0.0:
        raise ValueError("axial_stretch must be finite and positive")
    lam1 = float(axial_stretch)
    lam2 = 1.0 / math.sqrt(lam1)
    for _ in range(30):
        f = torch.diag(torch.tensor([lam1, lam2, lam2], dtype=torch.float64))
        p = first_piola(f, material)
        p22 = float(p[1, 1].item())
        if abs(p22) < 1.0e-13:
            break
        c = consistent_tangent(f, material)
        dp22_dlam2 = float((c[1, 1, 1, 1] + c[1, 1, 2, 2]).item())
        lam2 -= p22 / dp22_dlam2
    f = torch.diag(torch.tensor([lam1, lam2, lam2], dtype=torch.float64))
    p = first_piola(f, material)
    w = energy_density(f, material)
    det_f = lam1 * lam2 * lam2
    p11 = float(p[0, 0].item())
    sigma11 = p11 * lam1 / det_f
    return {
        "axial_stretch": lam1,
        "transverse_stretch": lam2,
        "det_f": det_f,
        "axial_first_piola_mpa": p11,
        "axial_cauchy_mpa": sigma11,
        "strain_energy_density_mpa": float(w.item()),
    }


__all__ = (
    "FEM_REQUEST_3D_SCHEMA",
    "FEM_RESULT_3D_SCHEMA",
    "DirichletBC3D",
    "FemRequest3D",
    "FemResult3D",
    "ScikitFemBackend3D",
    "TorchFemBackend3D",
    "TractionBC3D",
    "VolumetricLockingError",
    "analytical_3d_uniaxial_hyperelastic",
    "assemble_surface_tractions_3d",
    "face_node_indices_3d",
)
