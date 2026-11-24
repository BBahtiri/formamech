"""3D steady/transient heat conduction & coupled thermo-mechanical FEM solver (V3-4C).

Uses exact Duffy-transformed degree-5 positive-weight tetrahedral quadrature for
``TET4``/``TET10`` thermal capacitance and conductivity assembly, and couples
temperature fields into 3D finite-strain mechanics via multiplicative thermal
deformation ``F = F_m F_th``, ``F_th = (1 + alpha * (T - T_0)) I``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix  # type: ignore[import-untyped]

from piginot.contracts import MaterialSpec
from piginot.fem.backend_3d import (
    DirichletBC3D,
    TorchFemBackend3D,
    TractionBC3D,
    _build_edge_midside_map,
    _cauchy_von_mises_3d,
    _project_cell_scalar_to_nodes_3d,
    _tet_shape_grads_ref,
    assemble_surface_tractions_3d,
    face_node_indices_3d,
)
from piginot.fem.linear_solvers import LinearSolverConfig, solve_linear_system
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D
from piginot.physics import consistent_tangent, energy_density, first_piola

FloatArray = NDArray[np.float64]
THERMO_MECH_SCHEMA: Final = "formamech.thermo-mechanical-3d-v1"


@dataclass(frozen=True, slots=True)
class ThermalMaterialSpec:
    """Isotropic thermal & thermo-elastic material parameters (mm-N-s-K consistent)."""

    conductivity_w_per_mm_k: float = 0.05  # e.g. 50 W/(m K) = 0.05 W/(mm K)
    density_t_per_mm3: float = 7.85e-9  # 7850 kg/m^3 = 7.85e-9 tonne/mm^3
    specific_heat_mj_per_t_k: float = 4.6e8  # 460 J/(kg K) = 4.6e8 mJ/(tonne K)
    thermal_expansion_per_k: float = 1.2e-5
    reference_temp_k: float = 293.15

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.conductivity_w_per_mm_k)
            or self.conductivity_w_per_mm_k <= 0.0
        ):
            raise ValueError("conductivity_w_per_mm_k must be positive")
        if not math.isfinite(self.density_t_per_mm3) or self.density_t_per_mm3 <= 0.0:
            raise ValueError("density_t_per_mm3 must be positive")
        if (
            not math.isfinite(self.specific_heat_mj_per_t_k)
            or self.specific_heat_mj_per_t_k <= 0.0
        ):
            raise ValueError("specific_heat_mj_per_t_k must be positive")
        if not math.isfinite(self.thermal_expansion_per_k):
            raise ValueError("thermal_expansion_per_k must be finite")


@dataclass(frozen=True, slots=True)
class TemperatureBC3D:
    """Dirichlet prescribed temperature (K) on named 3D faces."""

    name: str
    face_tags: tuple[str, ...]
    temperature_k: float


@dataclass(frozen=True, slots=True)
class ConvectionBC3D:
    """Robin convective cooling ``q_n = h (T - T_inf)`` on named 3D faces."""

    name: str
    face_tags: tuple[str, ...]
    h_w_per_mm2_k: float
    t_ambient_k: float


@dataclass(frozen=True, slots=True)
class ThermoMechanical3DResult:
    """Coupled 3D temperature, displacement, and Cauchy von Mises stress fields."""

    converged: bool
    mesh_id: str
    nodal_temperature_k: FloatArray
    nodal_displacement: FloatArray
    nodal_von_mises_mpa: FloatArray
    max_temperature_k: float
    min_temperature_k: float
    peak_von_mises_mpa: float
    max_displacement_mm: float
    min_det_f: float
    total_newton_iterations: int
    wall_time_s: float
    reason: str

    def document(self) -> dict[str, Any]:
        return {
            "schema": THERMO_MECH_SCHEMA,
            "converged": self.converged,
            "mesh_id": self.mesh_id,
            "max_temperature_k": float(self.max_temperature_k),
            "min_temperature_k": float(self.min_temperature_k),
            "peak_von_mises_mpa": float(self.peak_von_mises_mpa),
            "max_displacement_mm": float(self.max_displacement_mm),
            "min_det_f": float(self.min_det_f),
            "total_newton_iterations": self.total_newton_iterations,
            "wall_time_s": float(self.wall_time_s),
            "reason": self.reason,
        }


def duffy_tet_quadrature(order_1d: int = 3) -> tuple[FloatArray, FloatArray]:
    """Positive-weight degree-(2*order-1) reference tetrahedron quadrature via Duffy map."""
    g_pts, g_wts = np.polynomial.legendre.leggauss(order_1d)
    # Map [-1, 1] -> [0, 1]
    u_1d = 0.5 * (g_pts + 1.0)
    w_1d = 0.5 * g_wts
    pts: list[tuple[float, float, float]] = []
    wts: list[float] = []
    for u, wu in zip(u_1d, w_1d, strict=True):
        for v, wv in zip(u_1d, w_1d, strict=True):
            for w, ww in zip(u_1d, w_1d, strict=True):
                xi = float(u)
                eta = float((1.0 - u) * v)
                zeta = float((1.0 - u) * (1.0 - v) * w)
                jac = float(((1.0 - u) ** 2) * (1.0 - v))
                pts.append((xi, eta, zeta))
                wts.append(float(wu * wv * ww * jac))
    return np.asarray(pts, dtype=np.float64), np.asarray(wts, dtype=np.float64)


def tet_shape_values_ref(cell_type: CellType3D, xi_pts: FloatArray) -> FloatArray:
    """Evaluate ``TET4``/``TET10`` reference shape functions ``N (Q, n_loc)``."""
    xi = xi_pts[:, 0]
    eta = xi_pts[:, 1]
    zeta = xi_pts[:, 2]
    l0 = 1.0 - xi - eta - zeta
    l1, l2, l3 = xi, eta, zeta
    if cell_type is CellType3D.TET4:
        return np.stack((l0, l1, l2, l3), axis=-1)
    # CanonicalMesh3D TET10: (0,1,2,3, mid01, mid12, mid20, mid03, mid13, mid23)
    return np.stack(
        (
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            l3 * (2.0 * l3 - 1.0),
            4.0 * l0 * l1,
            4.0 * l1 * l2,
            4.0 * l2 * l0,
            4.0 * l0 * l3,
            4.0 * l1 * l3,
            4.0 * l2 * l3,
        ),
        axis=-1,
    )


def _assemble_thermal_matrices(
    mesh: CanonicalMesh3D, thermal: ThermalMaterialSpec
) -> tuple[csr_matrix, csr_matrix]:
    """Assemble global conductivity ``K_th`` and consistent capacitance ``C_th``."""
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    n_nodes = len(mesh.nodes)
    n_elem, n_loc = cells.shape

    order = 3 if mesh.cell_type is CellType3D.TET10 else 2
    xi_pts, w_ref = duffy_tet_quadrature(order)
    n_val = tet_shape_values_ref(mesh.cell_type, xi_pts)  # (Q, n_loc)
    d_ref = _tet_shape_grads_ref(mesh.cell_type, xi_pts)  # (Q, n_loc, 3)

    x_e = nodes[cells]  # (E, n_loc, 3)
    jac = np.einsum("eni,qna->eqia", x_e, d_ref)
    det_j = np.linalg.det(jac)
    inv_j = np.linalg.inv(jac)
    d_phys = np.einsum("qna,eqai->eqni", d_ref, inv_j)  # (E, Q, n_loc, 3)
    w_vol = det_j * w_ref[None, :]  # (E, Q)

    k_cond = float(thermal.conductivity_w_per_mm_k)
    # Convert mJ/(mm^3 K) -> J/(mm^3 K) so C_th/dt has units W/K matching K_th & K_conv
    rho_cp = (
        1.0e-3
        * float(thermal.density_t_per_mm3)
        * float(thermal.specific_heat_mj_per_t_k)
    )

    # K_elem: (E, n_loc, n_loc)
    k_elem = k_cond * np.einsum("eq,eqni,eqmi->enm", w_vol, d_phys, d_phys)
    c_elem = rho_cp * np.einsum("eq,qn,qm->enm", w_vol, n_val, n_val)

    ii = np.repeat(cells[:, :, None], n_loc, axis=2).reshape(-1)
    jj = np.repeat(cells[:, None, :], n_loc, axis=1).reshape(-1)
    k_mat = coo_matrix((k_elem.reshape(-1), (ii, jj)), shape=(n_nodes, n_nodes)).tocsr()
    c_mat = coo_matrix((c_elem.reshape(-1), (ii, jj)), shape=(n_nodes, n_nodes)).tocsr()
    return k_mat, c_mat


_P1_TRI_MASS: Final[FloatArray] = (
    np.array(
        [
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    / 12.0
)
_P1_TRI_LOAD: Final[FloatArray] = np.array([1.0, 1.0, 1.0], dtype=np.float64) / 3.0

# Exact consistent P2 (6-node: v0, v1, v2, m01, m12, m20) reference triangle integrals
# M_ab = (1 / A) * int_T N_a N_b dA,  b_a = (1 / A) * int_T N_a dA
_P2_TRI_MASS: Final[FloatArray] = (
    np.array(
        [
            [6.0, -1.0, -1.0, 0.0, -4.0, 0.0],
            [-1.0, 6.0, -1.0, 0.0, 0.0, -4.0],
            [-1.0, -1.0, 6.0, -4.0, 0.0, 0.0],
            [0.0, 0.0, -4.0, 32.0, 16.0, 16.0],
            [-4.0, 0.0, 0.0, 16.0, 32.0, 16.0],
            [0.0, -4.0, 0.0, 16.0, 16.0, 32.0],
        ],
        dtype=np.float64,
    )
    / 180.0
)
_P2_TRI_LOAD: Final[FloatArray] = (
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64) / 3.0
)


def _apply_convection_bcs(
    mesh: CanonicalMesh3D,
    convection_bcs: Sequence[ConvectionBC3D],
) -> tuple[csr_matrix, FloatArray]:
    """Assemble exact consistent P1/P2 surface Robin convection matrix and ambient load."""
    n_nodes = len(mesh.nodes)
    rhs = np.zeros(n_nodes, dtype=np.float64)
    if not convection_bcs:
        return csr_matrix((n_nodes, n_nodes), dtype=np.float64), rhs
    _, _, areas = mesh.surface_geometry()
    is_p2 = mesh.cell_type is CellType3D.TET10
    edge_map = _build_edge_midside_map(mesh) if is_p2 else {}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for cbc in convection_bcs:
        h = float(cbc.h_w_per_mm2_k)
        t_inf = float(cbc.t_ambient_k)
        for tag in cbc.face_tags:
            for tid in mesh.semantic_face_tags[tag]:
                area = float(areas[tid])
                if area <= 0.0:
                    continue
                a, b, c = mesh.surface_triangles[tid]
                loc_nodes: tuple[int, ...] = (a, b, c)
                m_ref = _P1_TRI_MASS
                b_ref = _P1_TRI_LOAD
                if is_p2:
                    mids: list[int] = []
                    for u, v in ((a, b), (b, c), (c, a)):
                        key = (u, v) if u < v else (v, u)
                        mid = edge_map.get(key)
                        if mid is not None:
                            mids.append(mid)
                    if len(mids) == 3:
                        loc_nodes = (a, b, c, mids[0], mids[1], mids[2])
                        m_ref = _P2_TRI_MASS
                        b_ref = _P2_TRI_LOAD
                scale = h * area
                n_loc = len(loc_nodes)
                for i in range(n_loc):
                    ni = loc_nodes[i]
                    rhs[ni] += scale * t_inf * float(b_ref[i])
                    for j in range(n_loc):
                        m_ij = float(m_ref[i, j])
                        if m_ij != 0.0:
                            rows.append(ni)
                            cols.append(loc_nodes[j])
                            vals.append(scale * m_ij)
    return coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr(), rhs


def solve_3d_steady_heat_conduction(
    mesh: CanonicalMesh3D,
    thermal: ThermalMaterialSpec,
    *,
    temperature_bcs: Sequence[TemperatureBC3D],
    convection_bcs: Sequence[ConvectionBC3D] = (),
    linear_solver: LinearSolverConfig | None = None,
) -> FloatArray:
    """Solve steady-state 3D heat conduction ``(K_th + K_conv) T = f_conv``."""
    k_mat, _ = _assemble_thermal_matrices(mesh, thermal)
    k_conv, f_conv = _apply_convection_bcs(mesh, convection_bcs)
    a_mat = (k_mat + k_conv).tocsr()
    n_nodes = len(mesh.nodes)

    fixed: dict[int, float] = {}
    for tbc in temperature_bcs:
        for nid in face_node_indices_3d(mesh, tbc.face_tags):
            fixed[nid] = float(tbc.temperature_k)
    mask = np.zeros(n_nodes, dtype=bool)
    t_vec = np.zeros(n_nodes, dtype=np.float64)
    for nid, val in fixed.items():
        mask[nid] = True
        t_vec[nid] = val
    free = np.nonzero(~mask)[0]
    if free.size == 0:
        return t_vec
    rhs = (f_conv - a_mat @ t_vec)[free]
    t_free, rep = solve_linear_system(
        a_mat[free][:, free], rhs, config=linear_solver, is_symmetric=True
    )
    if not rep.converged:
        raise RuntimeError(f"steady thermal solve failed: {rep.reason}")
    t_vec[free] = t_free
    return t_vec


def solve_3d_transient_heat_conduction(
    mesh: CanonicalMesh3D,
    thermal: ThermalMaterialSpec,
    *,
    initial_temperature_k: float | FloatArray,
    dt_s: float,
    num_steps: int,
    temperature_bcs: Sequence[TemperatureBC3D],
    convection_bcs: Sequence[ConvectionBC3D] = (),
    linear_solver: LinearSolverConfig | None = None,
) -> tuple[FloatArray, ...]:
    """Solve transient 3D heat conduction via unconditionally stable Backward Euler."""
    k_mat, c_mat = _assemble_thermal_matrices(mesh, thermal)
    k_conv, f_conv = _apply_convection_bcs(mesh, convection_bcs)
    n_nodes = len(mesh.nodes)
    a_mat = (c_mat / float(dt_s) + k_mat + k_conv).tocsr()

    t_curr = (
        np.full(n_nodes, float(initial_temperature_k), dtype=np.float64)
        if isinstance(initial_temperature_k, (int, float))
        else np.asarray(initial_temperature_k, dtype=np.float64).copy()
    )
    fixed: dict[int, float] = {}
    for tbc in temperature_bcs:
        for nid in face_node_indices_3d(mesh, tbc.face_tags):
            fixed[nid] = float(tbc.temperature_k)
    mask = np.zeros(n_nodes, dtype=bool)
    for nid in fixed:
        mask[nid] = True
    free = np.nonzero(~mask)[0]

    history: list[FloatArray] = [t_curr.copy()]
    for _ in range(num_steps):
        rhs_full = (c_mat @ t_curr) / float(dt_s) + f_conv
        t_dirichlet = np.zeros(n_nodes, dtype=np.float64)
        for nid, val in fixed.items():
            t_dirichlet[nid] = val
        t_next = t_dirichlet.copy()
        if free.size > 0:
            rhs_free = (rhs_full - a_mat @ t_dirichlet)[free]
            t_f, rep = solve_linear_system(
                a_mat[free][:, free], rhs_free, config=linear_solver, is_symmetric=True
            )
            if not rep.converged:
                raise RuntimeError(f"transient thermal solve failed: {rep.reason}")
            t_next[free] = t_f
        t_curr = t_next
        history.append(t_curr.copy())
    return tuple(history)


def solve_3d_thermo_mechanical(
    mesh: CanonicalMesh3D,
    material: MaterialSpec,
    thermal: ThermalMaterialSpec,
    *,
    dirichlet_bcs: Sequence[DirichletBC3D],
    temperature_bcs: Sequence[TemperatureBC3D] = (),
    convection_bcs: Sequence[ConvectionBC3D] = (),
    prescribed_temperature_field_k: FloatArray | None = None,
    traction_bcs: Sequence[TractionBC3D] = (),
    load_steps: int = 2,
    solver_tolerance: float = 1.0e-8,
    max_newton_iterations: int = 25,
    linear_solver: LinearSolverConfig | None = None,
) -> ThermoMechanical3DResult:
    """Solve coupled 3D finite-strain thermo-mechanics via ``F = F_m F_th``."""
    t0 = time.perf_counter()
    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes
    t_nodal: FloatArray
    if prescribed_temperature_field_k is not None:
        t_nodal = np.asarray(prescribed_temperature_field_k, dtype=np.float64).reshape(
            n_nodes
        )
    else:
        t_nodal = solve_3d_steady_heat_conduction(
            mesh,
            thermal,
            temperature_bcs=temperature_bcs,
            convection_bcs=convection_bcs,
            linear_solver=linear_solver,
        )

    helper = TorchFemBackend3D()
    cells, d_phys, w_vol, cell_vols, rows, cols = helper._precompute_geometry(mesh)
    n_elem, n_loc = cells.shape
    n_edof = 3 * n_loc
    # Evaluate temperature at the mechanical quadrature points
    from piginot.fem.backend_3d import _tet_quadrature

    xi_m, _ = _tet_quadrature(mesh.cell_type)
    n_m = tet_shape_values_ref(mesh.cell_type, xi_m)  # (Q, n_loc)
    t_q = np.einsum("qn,en->eq", n_m, t_nodal[cells])  # (E, Q)
    dt_q = t_q - float(thermal.reference_temp_k)
    lam_th = 1.0 + float(thermal.thermal_expansion_per_k) * dt_q  # (E, Q)
    if float(np.min(lam_th)) <= 0.1:
        raise ValueError("non-physical thermal stretch <= 0.1")

    f_ext_full = assemble_surface_tractions_3d(mesh, traction_bcs).reshape(-1)
    comp_idx = {"x": 0, "y": 1, "z": 2}
    target_d: dict[int, float] = {}
    for bc in dirichlet_bcs:
        nids = face_node_indices_3d(mesh, bc.face_tags)
        for comp, val in zip(bc.components, bc.values_mm, strict=True):
            c = comp_idx[comp]
            for nid in nids:
                target_d[3 * nid + c] = float(val)
    constrained = np.zeros(n_dofs, dtype=bool)
    for dof in target_d:
        constrained[dof] = True
    free_dofs = np.nonzero(~constrained)[0]

    u = np.zeros(n_dofs, dtype=np.float64)
    total_iters = 0
    last_res = float("inf")
    last_min_det = 1.0
    last_p = np.zeros((n_elem, d_phys.shape[1], 3, 3), dtype=np.float64)
    last_f = np.zeros_like(last_p)

    for step_idx in range(1, load_steps + 1):
        fac = float(step_idx) / float(load_steps)
        f_ext = fac * f_ext_full
        for dof, val in target_d.items():
            u[dof] = fac * val
        lam_step = 1.0 + fac * (lam_th - 1.0)
        j_th = lam_step**3  # det(F_th)
        inv_lam = 1.0 / lam_step

        step_conv = False
        for _ in range(max_newton_iterations):
            total_iters += 1
            u_e = u.reshape(n_nodes, 3)[cells]
            f_tot = (
                np.einsum("eni,eqnj->eqij", u_e, d_phys)
                + np.eye(3, dtype=np.float64)[None, None, :, :]
            )
            f_mech = f_tot * inv_lam[:, :, None, None]
            min_det = float(np.min(np.linalg.det(f_mech)))
            if min_det <= 1.0e-8:
                break
            with torch.no_grad():
                fm_t = torch.from_numpy(np.ascontiguousarray(f_mech))
                p_m = first_piola(fm_t, material).numpy()
                c_m = consistent_tangent(fm_t, material).numpy()
                _ = energy_density(fm_t, material)
            # Chain rule for W_ref(F) = J_th * W_m(F * F_th^{-1}):
            # P_ref = dW_ref/dF = J_th * inv_lam * P_m = lam_th^2 * P_m
            # C_ref = dP_ref/dF = J_th * inv_lam^2 * C_m = lam_th * C_m
            p_ref = (j_th * inv_lam)[:, :, None, None] * p_m
            c_ref = (j_th * inv_lam * inv_lam)[:, :, None, None, None, None] * c_m

            f_elem = np.einsum("eq,eqij,eqnj->eni", w_vol, p_ref, d_phys)
            f_int = np.zeros((n_nodes, 3), dtype=np.float64)
            np.add.at(f_int, cells, f_elem)
            tmp = np.einsum("eq,eqijkl,eqnj->eqnikl", w_vol, c_ref, d_phys)
            k_elem = np.einsum("eqnikl,eqml->enimk", tmp, d_phys)
            k_mat = coo_matrix(
                (k_elem.reshape(n_elem, n_edof, n_edof).reshape(-1), (rows, cols)),
                shape=(n_dofs, n_dofs),
            ).tocsr()

            res_free = (f_int.reshape(-1) - f_ext)[free_dofs]
            res_norm = float(np.linalg.norm(res_free))
            last_res = res_norm
            last_min_det = min_det
            last_p = p_ref
            last_f = f_tot
            if res_norm <= solver_tolerance:
                step_conv = True
                break
            du_free, lin_rep = solve_linear_system(
                k_mat[free_dofs][:, free_dofs],
                -res_free,
                config=linear_solver,
                is_symmetric=True,
            )
            if not lin_rep.converged:
                break
            u[free_dofs] += du_free

        if not step_conv:
            return ThermoMechanical3DResult(
                converged=False,
                mesh_id=mesh.mesh_id,
                nodal_temperature_k=t_nodal,
                nodal_displacement=u.reshape(n_nodes, 3),
                nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
                max_temperature_k=float(np.max(t_nodal)),
                min_temperature_k=float(np.min(t_nodal)),
                peak_von_mises_mpa=0.0,
                max_displacement_mm=0.0,
                min_det_f=last_min_det,
                total_newton_iterations=total_iters,
                wall_time_s=time.perf_counter() - t0,
                reason=f"thermo-mechanical Newton did not converge (res={last_res:.3e})",
            )

    u_nodal = u.reshape(n_nodes, 3)
    vm_q = _cauchy_von_mises_3d(last_p, last_f)
    vm_cell = np.sum(vm_q * w_vol, axis=1) / np.maximum(cell_vols, 1.0e-12)
    vm_nodes = _project_cell_scalar_to_nodes_3d(mesh, vm_cell, cell_vols)

    return ThermoMechanical3DResult(
        converged=True,
        mesh_id=mesh.mesh_id,
        nodal_temperature_k=t_nodal,
        nodal_displacement=u_nodal,
        nodal_von_mises_mpa=vm_nodes,
        max_temperature_k=float(np.max(t_nodal)),
        min_temperature_k=float(np.min(t_nodal)),
        peak_von_mises_mpa=float(np.max(vm_nodes)),
        max_displacement_mm=float(np.max(np.linalg.norm(u_nodal, axis=1))),
        min_det_f=last_min_det,
        total_newton_iterations=total_iters,
        wall_time_s=time.perf_counter() - t0,
        reason="converged",
    )


__all__ = (
    "THERMO_MECH_SCHEMA",
    "ConvectionBC3D",
    "TemperatureBC3D",
    "ThermalMaterialSpec",
    "ThermoMechanical3DResult",
    "duffy_tet_quadrature",
    "solve_3d_steady_heat_conduction",
    "solve_3d_thermo_mechanical",
    "solve_3d_transient_heat_conduction",
    "tet_shape_values_ref",
)
