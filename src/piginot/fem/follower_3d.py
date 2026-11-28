"""Total-Lagrangian 3D follower pressure, enclosed cavities & consistent tangents (V6-R1).

Implements exact finite-deformation configuration-dependent surface pressure
loading on 3D ``TET4`` (3-node linear triangle) and ``TET10`` (6-node quadratic
triangle) surface facets via Nanson's law:

    x(xi, eta) = sum_a N_a(xi, eta) * (X_a + u_a)
    g_1 = dx/dxi,   g_2 = dx/deta
    n da = (g_1 x g_2) dxi deta
    f_ext_a = int (-p) * N_a * (g_1 x g_2) dxi deta

with exact consistent load stiffness ``K_L = -df_ext / du``:

    (K_L)_{ab} = int p * N_a * (-[g_2]_x N_{b,xi} + [g_1]_x N_{b,eta}) dxi deta

where ``[v]_x`` is the skew-symmetric cross-product matrix (``[v]_x w = v x w``).
For enclosed cavities (``closed_cavity=True``), also supports ideal-gas / polytropic
pressure-volume coupling ``p(V) = p_0 * (V_0 / V)^gamma`` with exact rank-1 volume
stiffness ``(dp/dV) * (dV/du)^T``.
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

from piginot.contracts import MaterialSpec, canonical_json_bytes
from piginot.fem._similitude import resolve_similitude_material
from piginot.fem.backend_3d import (
    DirichletBC3D,
    TorchFemBackend3D,
    TractionBC3D,
    _build_edge_midside_map,
    _cauchy_von_mises_3d,
    _project_cell_scalar_to_nodes_3d,
    assemble_surface_tractions_3d,
    compute_3d_element_hotspots,
    face_node_indices_3d,
)
from piginot.fem.linear_solvers import LinearSolverConfig, solve_linear_system
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D
from piginot.physics import consistent_tangent, energy_density, first_piola

FloatArray = NDArray[np.float64]
FOLLOWER_3D_RESULT_SCHEMA: Final = "formamech.follower-pressure-3d-v1"


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


@dataclass(frozen=True, slots=True)
class FollowerPressureBC3D:
    """Configuration-dependent 3D surface pressure (MPa = N/mm^2) on named faces.

    Positive ``pressure_mpa`` acts inward along ``-n`` (compressing the surface
    along the deformed inward normal, matching standard solid-mechanics sign
    conventions for internal cavity inflation or external hydrostatic load).
    """

    name: str
    face_tags: tuple[str, ...]
    pressure_mpa: float
    closed_cavity: bool = False
    polytropic_gamma: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("FollowerPressureBC3D name must be non-empty text")
        if not self.face_tags:
            raise ValueError("face_tags must be non-empty")
        if not math.isfinite(self.pressure_mpa):
            raise ValueError("pressure_mpa must be finite")
        if not math.isfinite(self.polytropic_gamma) or self.polytropic_gamma < 0.0:
            raise ValueError("polytropic_gamma must be finite and nonnegative")
        if self.polytropic_gamma > 0.0 and not self.closed_cavity:
            raise ValueError("polytropic_gamma > 0 requires closed_cavity=True")

    def at_factor(self, factor: float) -> FollowerPressureBC3D:
        return FollowerPressureBC3D(
            name=self.name,
            face_tags=self.face_tags,
            pressure_mpa=self.pressure_mpa * float(factor),
            closed_cavity=self.closed_cavity,
            polytropic_gamma=self.polytropic_gamma,
        )

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "face_tags": list(self.face_tags),
            "pressure_mpa": float(self.pressure_mpa),
            "closed_cavity": self.closed_cavity,
            "polytropic_gamma": float(self.polytropic_gamma),
        }


@dataclass(frozen=True, slots=True)
class FollowerLoadAssembly3D:
    """Assembled 3D follower external force vector and exact load stiffness."""

    external_force: FloatArray  # (3*N,) N
    load_stiffness: csr_matrix  # (3*N, 3*N) N/mm, K_L = -d(f_ext)/du
    cavity_volumes_mm3: Mapping[str, float]
    effective_pressures_mpa: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class FollowerPressure3DResult:
    """Converged (or failed) 3D nonlinear follower-pressure FEM solution."""

    request_id: str
    mesh_id: str
    converged: bool
    nodal_displacement: FloatArray  # (N, 3) mm
    nodal_von_mises_mpa: FloatArray  # (N,) MPa
    quadrature_peak_von_mises_mpa: float
    reactions_n: Mapping[str, tuple[float, float, float]]
    cavity_volumes_mm3: Mapping[str, float]
    effective_pressures_mpa: Mapping[str, float]
    total_strain_energy_n_mm: float
    min_det_f: float
    max_displacement_mm: float
    peak_von_mises_mpa: float
    final_residual_norm: float
    total_newton_iterations: int
    wall_time_s: float
    reason: str = "converged"

    @property
    def result_hash(self) -> str:
        return _sha256_doc(self.summary_document(include_hash=False))

    def summary_document(self, *, include_hash: bool = True) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": FOLLOWER_3D_RESULT_SCHEMA,
            "request_id": self.request_id,
            "mesh_id": self.mesh_id,
            "converged": self.converged,
            "reactions_n": {k: list(v) for k, v in sorted(self.reactions_n.items())},
            "cavity_volumes_mm3": {
                k: float(v) for k, v in sorted(self.cavity_volumes_mm3.items())
            },
            "effective_pressures_mpa": {
                k: float(v) for k, v in sorted(self.effective_pressures_mpa.items())
            },
            "total_strain_energy_n_mm": (
                float(self.total_strain_energy_n_mm)
                if math.isfinite(self.total_strain_energy_n_mm)
                else None
            ),
            "min_det_f": float(self.min_det_f)
            if math.isfinite(self.min_det_f)
            else None,
            "max_displacement_mm": (
                float(self.max_displacement_mm)
                if math.isfinite(self.max_displacement_mm)
                else None
            ),
            "peak_von_mises_mpa": (
                float(self.peak_von_mises_mpa)
                if math.isfinite(self.peak_von_mises_mpa)
                else None
            ),
            "quadrature_peak_von_mises_mpa": (
                float(self.quadrature_peak_von_mises_mpa)
                if math.isfinite(self.quadrature_peak_von_mises_mpa)
                else None
            ),
            "final_residual_norm": (
                float(self.final_residual_norm)
                if math.isfinite(self.final_residual_norm)
                else None
            ),
            "total_newton_iterations": self.total_newton_iterations,
            "wall_time_s": float(self.wall_time_s),
            "reason": self.reason,
        }
        if include_hash:
            doc["result_hash"] = _sha256_doc(doc)
        return doc


def _tri_quadrature(is_p2: bool) -> tuple[FloatArray, FloatArray]:
    """Positive-weight symmetric Dunavant triangle quadrature on reference triangle."""
    if not is_p2:
        # 3-point degree-2 rule (exact for quadratic integrand N_a * (g1 x g2))
        pts = np.array(
            [[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]],
            dtype=np.float64,
        )
        wts: FloatArray = np.full(3, 1.0 / 6.0, dtype=np.float64)
        return pts, wts
    # 6-point degree-4 symmetric Dunavant rule (exact for P2 follower tangent)
    a1, b1 = 0.445948490915965, 0.108103018168070
    a2, b2 = 0.091576213509771, 0.816847572980459
    w1, w2 = 0.111690794839005, 0.054975871827661
    pts = np.array(
        [[a1, a1], [b1, a1], [a1, b1], [a2, a2], [b2, a2], [a2, b2]],
        dtype=np.float64,
    )
    wts = np.array([w1, w1, w1, w2, w2, w2], dtype=np.float64)
    return pts, wts


def _tri_shape_and_grads(
    is_p2: bool, xi_eta: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(N (Q, n_loc), dN_dxi (Q, n_loc), dN_deta (Q, n_loc))``."""
    q = int(xi_eta.shape[0])
    xi = xi_eta[:, 0]
    eta = xi_eta[:, 1]
    l0 = 1.0 - xi - eta
    l1 = xi
    l2 = eta
    if not is_p2:
        n_val = np.stack((l0, l1, l2), axis=1)
        dn_dxi = np.broadcast_to(
            np.array([-1.0, 1.0, 0.0], dtype=np.float64)[None, :], (q, 3)
        ).copy()
        dn_deta = np.broadcast_to(
            np.array([-1.0, 0.0, 1.0], dtype=np.float64)[None, :], (q, 3)
        ).copy()
        return n_val, dn_dxi, dn_deta

    # 6-node P2 triangle: (v0, v1, v2, e01, e12, e20)
    n_val = np.stack(
        (
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            4.0 * l0 * l1,
            4.0 * l1 * l2,
            4.0 * l2 * l0,
        ),
        axis=1,
    )
    dn_dxi = np.stack(
        (
            -(4.0 * l0 - 1.0),
            4.0 * l1 - 1.0,
            np.zeros(q, dtype=np.float64),
            4.0 * (l0 - l1),
            4.0 * l2,
            -4.0 * l2,
        ),
        axis=1,
    )
    dn_deta = np.stack(
        (
            -(4.0 * l0 - 1.0),
            np.zeros(q, dtype=np.float64),
            4.0 * l2 - 1.0,
            -4.0 * l1,
            4.0 * l1,
            4.0 * (l0 - l2),
        ),
        axis=1,
    )
    return n_val, dn_dxi, dn_deta


def _skew(v: FloatArray) -> FloatArray:
    """Return 3x3 skew-symmetric matrix ``[v]_x`` such that ``[v]_x w = v x w``."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def _facet_node_tuples(
    mesh: CanonicalMesh3D, face_tags: Sequence[str]
) -> tuple[list[tuple[int, ...]], bool]:
    """Build ordered 3-node or 6-node surface facet connectivity for ``face_tags``."""
    edge_map = _build_edge_midside_map(mesh)
    is_p2 = mesh.cell_type is CellType3D.TET10 and bool(edge_map)
    facets: list[tuple[int, ...]] = []
    seen_tris: set[int] = set()
    for tag in face_tags:
        if tag not in mesh.semantic_face_tags:
            raise ValueError(
                f"FollowerPressureBC3D references missing face tag {tag!r}"
            )
        for tid in mesh.semantic_face_tags[tag]:
            if tid in seen_tris:
                continue
            seen_tris.add(tid)
            a, b, c = mesh.surface_triangles[tid]
            if is_p2:
                e01 = edge_map[(a, b) if a < b else (b, a)]
                e12 = edge_map[(b, c) if b < c else (c, b)]
                e20 = edge_map[(c, a) if c < a else (a, c)]
                facets.append((a, b, c, e01, e12, e20))
            else:
                facets.append((a, b, c))
    return facets, is_p2


def compute_enclosed_surface_volume_3d(
    nodes_ref: FloatArray,
    u_nodal: FloatArray,
    facets: Sequence[tuple[int, ...]],
    *,
    is_p2: bool,
) -> tuple[float, FloatArray]:
    """Compute signed enclosed volume ``V`` and exact nodal gradient ``dV/du``."""
    n_nodes = int(nodes_ref.shape[0])
    x_def = nodes_ref + u_nodal
    pts_q, wts_q = _tri_quadrature(is_p2)
    n_val, dn_dxi, dn_deta = _tri_shape_and_grads(is_p2, pts_q)
    n_loc = int(n_val.shape[1])

    vol = 0.0
    dv_du = np.zeros((n_nodes, 3), dtype=np.float64)
    for f_nodes in facets:
        idx = list(f_nodes)
        xf = x_def[idx]  # (n_loc, 3)
        for q_i in range(len(wts_q)):
            wq = float(wts_q[q_i])
            n_q = n_val[q_i]
            dxi_q = dn_dxi[q_i]
            deta_q = dn_deta[q_i]
            x_q = n_q @ xf
            g1 = dxi_q @ xf
            g2 = deta_q @ xf
            g_cross = np.cross(g1, g2)
            vol += (wq / 3.0) * float(np.dot(x_q, g_cross))
            # Exact variation: delta V = int (delta x) . (g1 x g2) dxi deta
            # (valid for closed surfaces via divergence theorem; we also add exact
            # explicit boundary term so finite differences match even on open patches)
            x_cross_g2 = np.cross(x_q, g2)
            g1_cross_x = np.cross(g1, x_q)
            for a in range(n_loc):
                dv_du[idx[a]] += (wq / 3.0) * (
                    n_q[a] * g_cross
                    + dxi_q[a] * (-x_cross_g2)
                    + deta_q[a] * (-g1_cross_x)
                )
    return float(vol), dv_du.reshape(-1)


def assemble_follower_pressure_3d(
    mesh: CanonicalMesh3D,
    follower_bcs: Sequence[FollowerPressureBC3D],
    u_vec: FloatArray,
) -> FollowerLoadAssembly3D:
    """Assemble 3D configuration-dependent follower force and exact load stiffness."""
    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes
    nodes_ref = np.asarray(mesh.nodes, dtype=np.float64)
    u_nodal = u_vec.reshape(n_nodes, 3)
    x_def = nodes_ref + u_nodal

    f_ext = np.zeros(n_dofs, dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    cavity_vols: dict[str, float] = {}
    eff_pressures: dict[str, float] = {}

    for fbc in follower_bcs:
        facets, is_p2 = _facet_node_tuples(mesh, fbc.face_tags)
        pts_q, wts_q = _tri_quadrature(is_p2)
        n_val, dn_dxi, dn_deta = _tri_shape_and_grads(is_p2, pts_q)
        n_loc = int(n_val.shape[1])

        p_eff = float(fbc.pressure_mpa)
        dp_du: FloatArray | None = None
        if fbc.closed_cavity:
            v0, _ = compute_enclosed_surface_volume_3d(
                nodes_ref, np.zeros_like(u_nodal), facets, is_p2=is_p2
            )
            v_cur, dv_du = compute_enclosed_surface_volume_3d(
                nodes_ref, u_nodal, facets, is_p2=is_p2
            )
            # Cavity normals on internal cavity boundaries point INTO the cavity
            # (outward from the solid), so signed volume may be negative.
            sign_v = -1.0 if v0 < 0.0 else 1.0
            v0_abs = max(abs(v0), 1.0e-12)
            v_cur_abs = max(sign_v * v_cur, 1.0e-9 * v0_abs)
            cavity_vols[fbc.name] = float(v_cur_abs)
            if fbc.polytropic_gamma > 0.0:
                gamma = float(fbc.polytropic_gamma)
                ratio = v0_abs / v_cur_abs
                p_eff = float(fbc.pressure_mpa) * (ratio**gamma)
                dp_dv = -gamma * p_eff / v_cur_abs
                dp_du = (dp_dv * sign_v) * dv_du
        eff_pressures[fbc.name] = float(p_eff)

        # Unit-pressure force vector (for rank-1 polytropic stiffness if active)
        f_unit = np.zeros(n_dofs, dtype=np.float64)

        for f_nodes in facets:
            idx = list(f_nodes)
            xf = x_def[idx]  # (n_loc, 3)
            for q_i in range(len(wts_q)):
                wq = float(wts_q[q_i])
                n_q = n_val[q_i]
                dxi_q = dn_dxi[q_i]
                deta_q = dn_deta[q_i]
                g1 = dxi_q @ xf
                g2 = deta_q @ xf
                g_cross = np.cross(g1, g2)
                g1_skew = _skew(g1)
                g2_skew = _skew(g2)

                for a in range(n_loc):
                    na = float(n_q[a])
                    if abs(na) < 1.0e-15:
                        continue
                    f_a_unit = (-wq * na) * g_cross
                    nid_a = idx[a]
                    f_unit[3 * nid_a : 3 * nid_a + 3] += f_a_unit
                    f_ext[3 * nid_a : 3 * nid_a + 3] += p_eff * f_a_unit

                    # Exact K_L = -d(f_ext)/du
                    # d(g1 x g2)/du_b = -[g2]_x * dN_b/dxi + [g1]_x * dN_b/deta
                    # Since f_ext_a = -p * w * N_a * (g1 x g2),
                    # -df_ext_a/du_b = +p * w * N_a * (-[g2]_x dN_b/dxi + [g1]_x dN_b/deta)
                    scale = p_eff * wq * na
                    for b in range(n_loc):
                        k_ab = scale * (
                            (-float(dxi_q[b])) * g2_skew + float(deta_q[b]) * g1_skew
                        )
                        nid_b = idx[b]
                        for ci in range(3):
                            for cj in range(3):
                                val = float(k_ab[ci, cj])
                                if abs(val) > 1.0e-18:
                                    rows.append(3 * nid_a + ci)
                                    cols.append(3 * nid_b + cj)
                                    vals.append(val)

        if dp_du is not None:
            # f_ext = p(V(u)) * f_unit(u) => -d(f_ext)/du adds -(f_unit outer dp_du)
            nz_f = np.nonzero(np.abs(f_unit) > 1.0e-15)[0]
            nz_g = np.nonzero(np.abs(dp_du) > 1.0e-15)[0]
            for r_i in nz_f:
                fu_val = float(f_unit[r_i])
                for c_j in nz_g:
                    val = -fu_val * float(dp_du[c_j])
                    if abs(val) > 1.0e-18:
                        rows.append(int(r_i))
                        cols.append(int(c_j))
                        vals.append(val)

    k_load = coo_matrix((vals, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
    return FollowerLoadAssembly3D(
        external_force=f_ext,
        load_stiffness=k_load,
        cavity_volumes_mm3=cavity_vols,
        effective_pressures_mpa=eff_pressures,
    )


def solve_3d_follower_pressure(
    mesh: CanonicalMesh3D,
    material: MaterialSpec,
    *,
    dirichlet_bcs: Sequence[DirichletBC3D],
    follower_bcs: Sequence[FollowerPressureBC3D],
    traction_bcs: Sequence[TractionBC3D] = (),
    request_id: str = "follower-3d",
    load_steps: int = 4,
    solver_tolerance: float = 1.0e-8,
    max_newton_iterations: int = 25,
    linear_solver: LinearSolverConfig | None = None,
    initial_displacement_mm: FloatArray | None = None,
) -> FollowerPressure3DResult:
    """Solve 3D finite-strain nonlinear FEM under configuration-dependent follower pressure."""
    t0 = time.perf_counter()
    if mesh.cell_type not in (CellType3D.TET4, CellType3D.TET10):
        raise ValueError(f"requires tet4 or tet10, got {mesh.cell_type.value!r}")
    if not dirichlet_bcs:
        raise ValueError("at least one DirichletBC3D is required")
    nu_prop = material.properties.get("poisson_ratio")
    if nu_prop is not None:
        mesh.validate_incompressibility_locking(float(nu_prop.value))

    sim = resolve_similitude_material(material)
    mat_eff = sim.model_material
    s_fac = sim.scale_factor
    fbcs_eff = tuple(fbc.at_factor(s_fac) for fbc in follower_bcs)
    tol_eff = solver_tolerance * s_fac

    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes
    helper = TorchFemBackend3D()
    cells, d_phys, w_vol, cell_vols, rows, cols = helper._precompute_geometry(mesh)
    n_elem, n_loc = cells.shape
    n_edof = 3 * n_loc
    lin_cfg = LinearSolverConfig() if linear_solver is None else linear_solver

    f_dead_full = assemble_surface_tractions_3d(mesh, traction_bcs).reshape(-1) * s_fac
    comp_idx = {"x": 0, "y": 1, "z": 2}
    target_dirichlet: dict[int, float] = {}
    bc_dofs_by_name: dict[str, list[int]] = {}
    for bc in dirichlet_bcs:
        nids = face_node_indices_3d(mesh, bc.face_tags)
        dofs_this: list[int] = []
        for comp, val in zip(bc.components, bc.values_mm, strict=True):
            c = comp_idx[comp]
            for nid in nids:
                dof = 3 * nid + c
                target_dirichlet[dof] = float(val)
                dofs_this.append(dof)
        bc_dofs_by_name[bc.name] = dofs_this

    constrained = np.zeros(n_dofs, dtype=bool)
    for dof in target_dirichlet:
        constrained[dof] = True
    free_dofs = np.nonzero(~constrained)[0]

    def _assemble_internal(
        u_vec: FloatArray,
    ) -> tuple[FloatArray, csr_matrix, FloatArray, FloatArray, float, float]:
        u_e = u_vec.reshape(n_nodes, 3)[cells]
        f_np = (
            np.einsum("eni,eqnj->eqij", u_e, d_phys)
            + np.eye(3, dtype=np.float64)[None, None, :, :]
        )
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
            p_np = first_piola(f_t, mat_eff).numpy()
            c_np = consistent_tangent(f_t, mat_eff).numpy()
            w_np = energy_density(f_t, mat_eff).numpy()
        total_e = float(np.sum(w_np * w_vol))
        f_elem = np.einsum("eq,eqij,eqnj->eni", w_vol, p_np, d_phys)
        f_int = np.zeros((n_nodes, 3), dtype=np.float64)
        np.add.at(f_int, cells, f_elem)
        tmp = np.einsum("eq,eqijkl,eqnj->eqnikl", w_vol, c_np, d_phys)
        k_elem = np.einsum("eqnikl,eqml->enimk", tmp, d_phys)
        k_mat = coo_matrix(
            (k_elem.reshape(n_elem, n_edof, n_edof).reshape(-1), (rows, cols)),
            shape=(n_dofs, n_dofs),
        ).tocsr()
        return f_int.reshape(-1), k_mat, p_np, f_np, total_e, min_det

    u: FloatArray
    if initial_displacement_mm is not None:
        u = np.asarray(initial_displacement_mm, dtype=np.float64).reshape(n_dofs).copy()
    else:
        u = np.zeros(n_dofs, dtype=np.float64)

    total_newton = 0
    last_res = float("inf")
    last_energy = 0.0
    last_min_det = 1.0
    last_p: FloatArray = np.zeros((n_elem, d_phys.shape[1], 3, 3), dtype=np.float64)
    last_f: FloatArray = np.zeros_like(last_p)
    last_f_int: FloatArray = np.zeros(n_dofs, dtype=np.float64)
    last_f_ext: FloatArray = np.zeros(n_dofs, dtype=np.float64)
    last_vols: Mapping[str, float] = {}
    last_pressures: Mapping[str, float] = {}

    for step_idx in range(1, load_steps + 1):
        factor = float(step_idx) / float(load_steps)
        step_fbcs = tuple(fbc.at_factor(factor) for fbc in fbcs_eff)
        f_dead = factor * f_dead_full
        for dof, val in target_dirichlet.items():
            u[dof] = factor * val

        step_conv = False
        for _ in range(max_newton_iterations):
            total_newton += 1
            f_int, k_int, p_np, f_np, energy, min_det = _assemble_internal(u)
            if min_det <= 1.0e-8:
                return FollowerPressure3DResult(
                    request_id=request_id,
                    mesh_id=mesh.mesh_id,
                    converged=False,
                    nodal_displacement=np.zeros((n_nodes, 3), dtype=np.float64),
                    nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
                    quadrature_peak_von_mises_mpa=0.0,
                    reactions_n={},
                    cavity_volumes_mm3=last_vols,
                    effective_pressures_mpa={
                        k: v / s_fac for k, v in last_pressures.items()
                    },
                    total_strain_energy_n_mm=0.0,
                    min_det_f=min_det,
                    max_displacement_mm=0.0,
                    peak_von_mises_mpa=0.0,
                    final_residual_norm=float("inf"),
                    total_newton_iterations=total_newton,
                    wall_time_s=time.perf_counter() - t0,
                    reason="non-positive det(F) during 3D follower-pressure solve",
                )
            f_asm = assemble_follower_pressure_3d(mesh, step_fbcs, u)
            f_ext = f_dead + f_asm.external_force
            k_tot = (k_int + f_asm.load_stiffness).tocsr()

            res_free = (f_int - f_ext)[free_dofs]
            res_norm = float(np.linalg.norm(res_free))
            last_res = res_norm
            last_energy = energy
            last_min_det = min_det
            last_p = p_np
            last_f = f_np
            last_f_int = f_int
            last_f_ext = f_ext
            last_vols = f_asm.cavity_volumes_mm3
            last_pressures = f_asm.effective_pressures_mpa
            if res_norm <= tol_eff:
                step_conv = True
                break

            du_free, lin_rep = solve_linear_system(
                k_tot[free_dofs][:, free_dofs],
                -res_free,
                config=lin_cfg,
                is_symmetric=False,
            )
            if not lin_rep.converged:
                return FollowerPressure3DResult(
                    request_id=request_id,
                    mesh_id=mesh.mesh_id,
                    converged=False,
                    nodal_displacement=np.zeros((n_nodes, 3), dtype=np.float64),
                    nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
                    quadrature_peak_von_mises_mpa=0.0,
                    reactions_n={},
                    cavity_volumes_mm3=last_vols,
                    effective_pressures_mpa={
                        k: v / s_fac for k, v in last_pressures.items()
                    },
                    total_strain_energy_n_mm=0.0,
                    min_det_f=last_min_det,
                    max_displacement_mm=0.0,
                    peak_von_mises_mpa=0.0,
                    final_residual_norm=last_res / s_fac,
                    total_newton_iterations=total_newton,
                    wall_time_s=time.perf_counter() - t0,
                    reason=f"linear solve failed: {lin_rep.reason}",
                )

            alpha = 1.0
            accepted = False
            best_u: FloatArray | None = None
            best_res = float("inf")
            for _ls in range(10):
                u_cand = u.copy()
                u_cand[free_dofs] += alpha * du_free
                f_int_c, _, _, _, _, det_c = _assemble_internal(u_cand)
                if det_c > 1.0e-5:
                    f_asm_c = assemble_follower_pressure_3d(mesh, step_fbcs, u_cand)
                    cand_res = float(
                        np.linalg.norm(
                            (f_int_c - (f_dead + f_asm_c.external_force))[free_dofs]
                        )
                    )
                    if np.isfinite(cand_res):
                        if cand_res < best_res:
                            best_res = cand_res
                            best_u = u_cand
                        # Accept full quadratic Newton step even across transient geometric
                        # finite-rotation residual bumps if det(F) stays healthy.
                        if (
                            cand_res < (1.0 - 1.0e-4 * alpha) * res_norm
                            or cand_res <= tol_eff
                            or (alpha == 1.0 and cand_res <= 500.0 * max(res_norm, 1.0))
                        ):
                            u = u_cand
                            accepted = True
                            break
                alpha *= 0.5
            if not accepted:
                if best_u is not None:
                    u = best_u
                else:
                    break

        if not step_conv:
            return FollowerPressure3DResult(
                request_id=request_id,
                mesh_id=mesh.mesh_id,
                converged=False,
                nodal_displacement=np.zeros((n_nodes, 3), dtype=np.float64),
                nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
                quadrature_peak_von_mises_mpa=0.0,
                reactions_n={},
                cavity_volumes_mm3=last_vols,
                effective_pressures_mpa={
                    k: v / s_fac for k, v in last_pressures.items()
                },
                total_strain_energy_n_mm=0.0,
                min_det_f=last_min_det,
                max_displacement_mm=0.0,
                peak_von_mises_mpa=0.0,
                final_residual_norm=last_res / s_fac,
                total_newton_iterations=total_newton,
                wall_time_s=time.perf_counter() - t0,
                reason=f"Newton did not converge at step {step_idx} (res={last_res / s_fac:.3e})",
            )

    u_nodal = u.reshape(n_nodes, 3)
    p_phys = last_p / s_fac
    vm_q = _cauchy_von_mises_3d(p_phys, last_f)
    vm_cell = np.sum(vm_q * w_vol, axis=1) / np.maximum(cell_vols, 1.0e-12)
    vm_nodes = _project_cell_scalar_to_nodes_3d(mesh, vm_cell, cell_vols)
    _, quad_peak_vm, _ = compute_3d_element_hotspots(p_phys, last_f)

    r_full = (last_f_int - last_f_ext) / s_fac
    reactions: dict[str, tuple[float, float, float]] = {}
    for name, dofs in bc_dofs_by_name.items():
        rx = sum(float(r_full[d]) for d in dofs if d % 3 == 0)
        ry = sum(float(r_full[d]) for d in dofs if d % 3 == 1)
        rz = sum(float(r_full[d]) for d in dofs if d % 3 == 2)
        reactions[name] = (rx, ry, rz)

    return FollowerPressure3DResult(
        request_id=request_id,
        mesh_id=mesh.mesh_id,
        converged=True,
        nodal_displacement=u_nodal,
        nodal_von_mises_mpa=vm_nodes,
        quadrature_peak_von_mises_mpa=quad_peak_vm,
        reactions_n=reactions,
        cavity_volumes_mm3=last_vols,
        effective_pressures_mpa={k: v / s_fac for k, v in last_pressures.items()},
        total_strain_energy_n_mm=last_energy / s_fac,
        min_det_f=last_min_det,
        max_displacement_mm=float(np.max(np.linalg.norm(u_nodal, axis=1))),
        peak_von_mises_mpa=float(np.max(vm_nodes)),
        final_residual_norm=last_res / s_fac,
        total_newton_iterations=total_newton,
        wall_time_s=time.perf_counter() - t0,
        reason="converged",
    )


__all__ = (
    "FOLLOWER_3D_RESULT_SCHEMA",
    "FollowerLoadAssembly3D",
    "FollowerPressure3DResult",
    "FollowerPressureBC3D",
    "assemble_follower_pressure_3d",
    "compute_enclosed_surface_volume_3d",
    "solve_3d_follower_pressure",
)
