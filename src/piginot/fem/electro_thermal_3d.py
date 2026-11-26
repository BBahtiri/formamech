"""Coupled Multi-Material 3D Electro-Thermal-Mechanical (ETM) FEM Solver (V5-5).

Pure-MIT 3D multiphysics finite-element engine targeting electronics packaging
(3D-IC chiplets, TIM/heat spreaders, organic substrates), EV battery cold
plates, and high-current busways:

1. **Electrical Conduction & Joule Heating** (:func:`solve_3d_electrical_conduction`):
   Solves ``div(sigma_e grad V) = 0`` with voltage and current-density BCs and
   computes exact quadrature-point Joule power density ``q_J = sigma_e |grad V|^2``.
2. **Multi-Material Orthotropic Steady & Transient Heat Conduction**
   (:func:`solve_3d_multi_material_heat_conduction`):
   Supports per-part orthotropic conductivity ``(k_x, k_y, k_z)``, volumetric
   power maps ``q(x)`` + Joule heat ``q_J``, interfacial thermal contact
   conductance ``h_c (T_1 - T_2)``, surface heat flux, and spatially varying
   convective CHT maps ``h(x), T_amb(x)``.
3. **Coupled Multi-Material Finite-Strain Thermo-Mechanical Warpage**
   (:func:`solve_3d_electro_thermal_mechanical`):
   Couples ``T(x)`` into per-part ``F = F_m F_th`` (``F_th = (1 + alpha_p (T - T_0)) I``)
   with exact consistent tangent assembly and unsmoothed element-bound stress
   hotspots.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix  # type: ignore[import-untyped]
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from piginot.contracts import MaterialSpec
from piginot.fem.backend_3d import (
    DirichletBC3D,
    TorchFemBackend3D,
    TractionBC3D,
    _build_edge_midside_map,
    _cauchy_von_mises_3d,
    _project_cell_scalar_to_nodes_3d,
    _tet_quadrature,
    _tet_shape_grads_ref,
    assemble_surface_tractions_3d,
    compute_3d_element_hotspots,
    face_node_indices_3d,
)
from piginot.fem.linear_solvers import LinearSolverConfig, solve_linear_system
from piginot.fem.thermo_mechanical_3d import (
    _P1_TRI_LOAD,
    _P1_TRI_MASS,
    _P2_TRI_LOAD,
    _P2_TRI_MASS,
    ConvectionBC3D,
    TemperatureBC3D,
    duffy_tet_quadrature,
    tet_shape_values_ref,
)
from piginot.geometry.assembly_3d import AssemblyMesh3D
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D
from piginot.physics import consistent_tangent, first_piola

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ETM_3D_RESULT_SCHEMA: Final = "formamech.electro-thermal-mechanical-3d-v1"


@dataclass(frozen=True, slots=True)
class ElectroThermalMaterialSpec:
    """Multi-physics material spec in consistent ``mm-N-s-K-W-V-A`` units.

    Unit guide:
    - ``conductivity_xyz_w_per_mm_k``: e.g. Copper = ``(0.40, 0.40, 0.40)`` (400 W/mK),
      Silicon = ``(0.148, 0.148, 0.148)``, FR4 PCB = ``(0.020, 0.020, 0.0004)``.
    - ``electrical_conductivity_s_per_mm``: e.g. Copper = ``5.8e4`` S/mm (5.8e7 S/m).
    - ``volumetric_heat_capacity_j_per_mm3_k``: ``rho * c_p`` in ``J/(mm^3 K)``,
      e.g. Steel ``7850 * 460 * 1e-9 = 3.611e-3`` J/(mm^3 K).
    """

    part_id: str
    conductivity_xyz_w_per_mm_k: tuple[float, float, float]
    volumetric_heat_capacity_j_per_mm3_k: float
    thermal_expansion_per_k: float
    reference_temp_k: float
    mechanical: MaterialSpec
    electrical_conductivity_s_per_mm: float = 0.0

    def __post_init__(self) -> None:
        if not self.part_id.strip():
            raise ValueError("part_id must be non-empty")
        if len(self.conductivity_xyz_w_per_mm_k) != 3 or any(
            not math.isfinite(k) or k <= 0.0 for k in self.conductivity_xyz_w_per_mm_k
        ):
            raise ValueError("conductivity_xyz_w_per_mm_k must be 3 positive floats")
        if (
            not math.isfinite(self.volumetric_heat_capacity_j_per_mm3_k)
            or self.volumetric_heat_capacity_j_per_mm3_k <= 0.0
        ):
            raise ValueError("volumetric_heat_capacity_j_per_mm3_k must be positive")
        if not math.isfinite(self.thermal_expansion_per_k):
            raise ValueError("thermal_expansion_per_k must be finite")
        if (
            not math.isfinite(self.electrical_conductivity_s_per_mm)
            or self.electrical_conductivity_s_per_mm < 0.0
        ):
            raise ValueError("electrical_conductivity_s_per_mm must be nonnegative")


@dataclass(frozen=True, slots=True)
class VoltageBC3D:
    """Dirichlet electrical potential (V) on named 3D faces."""

    name: str
    face_tags: tuple[str, ...]
    voltage_v: float


@dataclass(frozen=True, slots=True)
class CurrentFluxBC3D:
    """Inward surface electrical current density ``J_n`` (A/mm^2) on named 3D faces."""

    name: str
    face_tags: tuple[str, ...]
    current_density_a_per_mm2: float


@dataclass(frozen=True, slots=True)
class HeatFluxBC3D:
    """Inward surface heat flux ``q_n`` (W/mm^2) on named 3D faces."""

    name: str
    face_tags: tuple[str, ...]
    flux_w_per_mm2: float


@dataclass(frozen=True, slots=True)
class SpatialConvectionBC3D:
    """Spatially varying convective cooling ``q_n = h(x) (T - T_amb(x))``."""

    name: str
    face_tags: tuple[str, ...]
    h_fn: Callable[[FloatArray], FloatArray]  # (M, 3) -> (M,) W/(mm^2 K)
    t_amb_fn: Callable[[FloatArray], FloatArray]  # (M, 3) -> (M,) K


@dataclass(frozen=True, slots=True)
class VolumetricPowerSource3D:
    """3D volumetric heat generation ``q(x)`` (W/mm^3) over a part or whole mesh."""

    name: str
    power_density_w_per_mm3: float = 0.0
    part_id: str | None = None
    cell_indices: tuple[int, ...] | None = None
    spatial_fn: Callable[[FloatArray], FloatArray] | None = (
        None  # (E, 3) -> (E,) W/mm^3
    )


@dataclass(frozen=True, slots=True)
class ThermalContactInterface3D:
    """Interfacial thermal contact conductance ``h_c (T_master - T_slave)`` across assembly faces."""

    name: str
    master_face_tag: str
    slave_face_tag: str
    conductance_w_per_mm2_k: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.conductance_w_per_mm2_k)
            or self.conductance_w_per_mm2_k <= 0.0
        ):
            raise ValueError("conductance_w_per_mm2_k must be positive")


@dataclass(frozen=True, slots=True)
class Electrical3DResult:
    """3D electrical conduction potential and quadrature-point Joule heating density."""

    nodal_voltage_v: FloatArray
    quadrature_joule_w_per_mm3: FloatArray  # (E, Q)
    element_mean_joule_w_per_mm3: FloatArray  # (E,)
    total_joule_power_w: float
    wall_time_s: float


@dataclass(frozen=True, slots=True)
class ElectroThermalMechanical3DResult:
    """Coupled multi-material 3D ETM result with element-bound hotspots."""

    converged: bool
    mesh_id: str
    nodal_voltage_v: FloatArray | None
    total_joule_power_w: float
    nodal_temperature_k: FloatArray
    temperature_history_k: tuple[FloatArray, ...]
    nodal_displacement: FloatArray
    nodal_von_mises_mpa: FloatArray
    element_peak_von_mises_mpa: FloatArray
    quadrature_peak_von_mises_mpa: float
    max_temperature_k: float
    min_temperature_k: float
    max_displacement_mm: float
    min_det_f: float
    total_newton_iterations: int
    wall_time_s: float
    reason: str

    def document(self) -> dict[str, Any]:
        return {
            "schema": ETM_3D_RESULT_SCHEMA,
            "converged": self.converged,
            "mesh_id": self.mesh_id,
            "total_joule_power_w": float(self.total_joule_power_w),
            "max_temperature_k": float(self.max_temperature_k),
            "min_temperature_k": float(self.min_temperature_k),
            "max_displacement_mm": float(self.max_displacement_mm),
            "quadrature_peak_von_mises_mpa": float(self.quadrature_peak_von_mises_mpa),
            "min_det_f": float(self.min_det_f),
            "total_newton_iterations": self.total_newton_iterations,
            "wall_time_s": float(self.wall_time_s),
            "reason": self.reason,
        }


def _resolve_cell_materials(
    mesh: CanonicalMesh3D,
    materials: Sequence[ElectroThermalMaterialSpec],
    part_cell_map: Mapping[str, Sequence[int]] | None = None,
) -> list[ElectroThermalMaterialSpec]:
    """Map every cell index ``0..E-1`` to its :class:`ElectroThermalMaterialSpec`."""
    n_elem = len(mesh.cells)
    if not materials:
        raise ValueError("materials must be non-empty")
    if len(materials) == 1 and part_cell_map is None:
        return [materials[0]] * n_elem
    if part_cell_map is None:
        raise ValueError(
            "part_cell_map is required when multiple materials are provided"
        )
    by_id = {m.part_id: m for m in materials}
    assigned: list[ElectroThermalMaterialSpec | None] = [None] * n_elem
    for pid, cids in part_cell_map.items():
        if pid not in by_id:
            raise ValueError(
                f"part_id {pid!r} missing from ElectroThermalMaterialSpec list"
            )
        mat = by_id[pid]
        for cid in cids:
            assigned[int(cid)] = mat
    if any(m is None for m in assigned):
        raise ValueError("part_cell_map did not assign a material to every cell")
    return [m for m in assigned if m is not None]


def _tet_quad_geometry(
    mesh: CanonicalMesh3D,
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return ``(cells, n_val (Q, n_loc), d_phys (E, Q, n_loc, 3), w_vol (E, Q), q_xyz (E, Q, 3))``."""
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    order = 3 if mesh.cell_type is CellType3D.TET10 else 2
    xi_pts, w_ref = duffy_tet_quadrature(order)
    n_val = tet_shape_values_ref(mesh.cell_type, xi_pts)
    d_ref = _tet_shape_grads_ref(mesh.cell_type, xi_pts)
    x_e = nodes[cells]  # (E, n_loc, 3)
    jac = np.einsum("eni,qna->eqia", x_e, d_ref)
    det_j = np.linalg.det(jac)
    if np.any(~np.isfinite(det_j)) or np.any(det_j <= 0.0):
        raise ValueError("mesh contains non-positive tetrahedral Jacobian")
    inv_j = np.linalg.inv(jac)
    d_phys = np.einsum("qna,eqai->eqni", d_ref, inv_j)
    w_vol = det_j * w_ref[None, :]
    q_xyz = np.einsum("qn,eni->eqi", n_val, x_e)
    return cells, n_val, d_phys, w_vol, q_xyz


def _assemble_surface_flux_rhs(
    mesh: CanonicalMesh3D, face_tags: Sequence[str], flux_value: float
) -> FloatArray:
    n_nodes = len(mesh.nodes)
    rhs = np.zeros(n_nodes, dtype=np.float64)
    if abs(flux_value) <= 1.0e-30:
        return rhs
    _, _, areas = mesh.surface_geometry()
    is_p2 = mesh.cell_type is CellType3D.TET10
    edge_map = _build_edge_midside_map(mesh) if is_p2 else {}
    for tag in face_tags:
        for tid in mesh.semantic_face_tags[tag]:
            area = float(areas[tid])
            if area <= 0.0:
                continue
            a, b, c = mesh.surface_triangles[tid]
            loc: tuple[int, ...] = (a, b, c)
            b_ref = _P1_TRI_LOAD
            if is_p2:
                mids = [
                    edge_map.get((u, v) if u < v else (v, u))
                    for u, v in ((a, b), (b, c), (c, a))
                ]
                if all(m is not None for m in mids):
                    loc = (
                        a,
                        b,
                        c,
                        int(mids[0] or 0),
                        int(mids[1] or 0),
                        int(mids[2] or 0),
                    )
                    b_ref = _P2_TRI_LOAD
            for i, ni in enumerate(loc):
                rhs[ni] += flux_value * area * float(b_ref[i])
    return rhs


def solve_3d_electrical_conduction(
    mesh: CanonicalMesh3D,
    materials: Sequence[ElectroThermalMaterialSpec],
    *,
    part_cell_map: Mapping[str, Sequence[int]] | None = None,
    voltage_bcs: Sequence[VoltageBC3D],
    current_flux_bcs: Sequence[CurrentFluxBC3D] = (),
    linear_solver: LinearSolverConfig | None = None,
) -> Electrical3DResult:
    """Solve 3D electrical conduction ``div(sigma_e grad V) = 0`` and Joule heating ``q_J``."""
    t0 = time.perf_counter()
    cell_mats = _resolve_cell_materials(mesh, materials, part_cell_map)
    cells, _, d_phys, w_vol, _ = _tet_quad_geometry(mesh)
    n_nodes = len(mesh.nodes)
    n_loc = cells.shape[1]

    sigma_e = np.array(
        [float(m.electrical_conductivity_s_per_mm) for m in cell_mats], dtype=np.float64
    )
    if float(np.max(sigma_e)) <= 0.0:
        raise ValueError(
            "at least one part must have electrical_conductivity_s_per_mm > 0"
        )
    sigma_safe = np.maximum(sigma_e, 1.0e-18)
    k_elec = np.einsum("e,eq,eqni,eqmi->enm", sigma_safe, w_vol, d_phys, d_phys)
    ii = np.repeat(cells[:, :, None], n_loc, axis=2).reshape(-1)
    jj = np.repeat(cells[:, None, :], n_loc, axis=1).reshape(-1)
    k_mat = coo_matrix((k_elec.reshape(-1), (ii, jj)), shape=(n_nodes, n_nodes)).tocsr()

    rhs = np.zeros(n_nodes, dtype=np.float64)
    for cbc in current_flux_bcs:
        rhs += _assemble_surface_flux_rhs(
            mesh, cbc.face_tags, float(cbc.current_density_a_per_mm2)
        )

    fixed: dict[int, float] = {}
    for vbc in voltage_bcs:
        for nid in face_node_indices_3d(mesh, vbc.face_tags):
            fixed[nid] = float(vbc.voltage_v)
    if not fixed:
        raise ValueError(
            "at least one VoltageBC3D is required to anchor electrical potential"
        )

    mask = np.zeros(n_nodes, dtype=bool)
    v_vec = np.zeros(n_nodes, dtype=np.float64)
    for nid, val in fixed.items():
        mask[nid] = True
        v_vec[nid] = val
    free = np.nonzero(~mask)[0]
    if free.size > 0:
        rhs_free = (rhs - k_mat @ v_vec)[free]
        v_free, rep = solve_linear_system(
            k_mat[free][:, free], rhs_free, config=linear_solver, is_symmetric=True
        )
        if not rep.converged:
            raise RuntimeError(f"3D electrical solve failed: {rep.reason}")
        v_vec[free] = v_free

    # Compute quadrature-point electric field E = -grad(V) and Joule power q_J = sigma_e |grad V|^2
    v_e = v_vec[cells]  # (E, n_loc)
    grad_v = np.einsum("en,eqni->eqi", v_e, d_phys)  # (E, Q, 3)
    grad_v_sq = np.sum(grad_v * grad_v, axis=-1)  # (E, Q)
    q_joule = sigma_e[:, None] * grad_v_sq  # W / mm^3
    elem_vols = np.sum(w_vol, axis=1)
    elem_power = np.sum(q_joule * w_vol, axis=1)
    elem_mean_q = elem_power / np.maximum(elem_vols, 1.0e-15)
    total_w = float(np.sum(elem_power))

    return Electrical3DResult(
        nodal_voltage_v=v_vec,
        quadrature_joule_w_per_mm3=q_joule,
        element_mean_joule_w_per_mm3=elem_mean_q,
        total_joule_power_w=total_w,
        wall_time_s=time.perf_counter() - t0,
    )


def _assemble_multimat_thermal_matrices(
    mesh: CanonicalMesh3D,
    cell_mats: Sequence[ElectroThermalMaterialSpec],
) -> tuple[csr_matrix, csr_matrix, IntArray, FloatArray, FloatArray, FloatArray]:
    """Assemble orthotropic multi-material ``K_th`` (W/K) and consistent ``C_th`` (J/K)."""
    cells, n_val, d_phys, w_vol, q_xyz = _tet_quad_geometry(mesh)
    n_nodes = len(mesh.nodes)
    n_loc = cells.shape[1]

    k_xyz = np.array(
        [m.conductivity_xyz_w_per_mm_k for m in cell_mats], dtype=np.float64
    )  # (E, 3)
    rho_cp = np.array(
        [float(m.volumetric_heat_capacity_j_per_mm3_k) for m in cell_mats],
        dtype=np.float64,
    )
    # Orthotropic conductivity: sum_i k_i * dN_a/dx_i * dN_b/dx_i
    k_elem = np.einsum("ei,eq,eqni,eqmi->enm", k_xyz, w_vol, d_phys, d_phys)
    c_elem = np.einsum("e,eq,qn,qm->enm", rho_cp, w_vol, n_val, n_val)

    ii = np.repeat(cells[:, :, None], n_loc, axis=2).reshape(-1)
    jj = np.repeat(cells[:, None, :], n_loc, axis=1).reshape(-1)
    k_mat = coo_matrix((k_elem.reshape(-1), (ii, jj)), shape=(n_nodes, n_nodes)).tocsr()
    c_mat = coo_matrix((c_elem.reshape(-1), (ii, jj)), shape=(n_nodes, n_nodes)).tocsr()
    return k_mat, c_mat, cells, n_val, w_vol, q_xyz


def _assemble_spatial_convection(
    mesh: CanonicalMesh3D,
    convection_bcs: Sequence[ConvectionBC3D],
    spatial_convection_bcs: Sequence[SpatialConvectionBC3D],
) -> tuple[csr_matrix, FloatArray]:
    """Assemble uniform + spatially varying Robin convection ``h(x)(T - T_amb(x))``."""
    n_nodes = len(mesh.nodes)
    rhs = np.zeros(n_nodes, dtype=np.float64)
    if not convection_bcs and not spatial_convection_bcs:
        return csr_matrix((n_nodes, n_nodes), dtype=np.float64), rhs

    centroids, _, areas = mesh.surface_geometry()
    is_p2 = mesh.cell_type is CellType3D.TET10
    edge_map = _build_edge_midside_map(mesh) if is_p2 else {}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def _add_tri_conv(tid: int, h_val: float, t_inf: float) -> None:
        area = float(areas[tid])
        if area <= 0.0 or h_val <= 0.0:
            return
        a, b, c = mesh.surface_triangles[tid]
        loc: tuple[int, ...] = (a, b, c)
        m_ref = _P1_TRI_MASS
        b_ref = _P1_TRI_LOAD
        if is_p2:
            mids = [
                edge_map.get((u, v) if u < v else (v, u))
                for u, v in ((a, b), (b, c), (c, a))
            ]
            if all(m is not None for m in mids):
                loc = (a, b, c, int(mids[0] or 0), int(mids[1] or 0), int(mids[2] or 0))
                m_ref = _P2_TRI_MASS
                b_ref = _P2_TRI_LOAD
        scale = h_val * area
        for i, ni in enumerate(loc):
            rhs[ni] += scale * t_inf * float(b_ref[i])
            for j, nj in enumerate(loc):
                m_ij = float(m_ref[i, j])
                if m_ij != 0.0:
                    rows.append(ni)
                    cols.append(nj)
                    vals.append(scale * m_ij)

    for cbc in convection_bcs:
        for tag in cbc.face_tags:
            for tid in mesh.semantic_face_tags[tag]:
                _add_tri_conv(tid, float(cbc.h_w_per_mm2_k), float(cbc.t_ambient_k))

    for sbc in spatial_convection_bcs:
        tids: list[int] = []
        for tag in sbc.face_tags:
            tids.extend(mesh.semantic_face_tags[tag])
        if not tids:
            continue
        pts = np.asarray([centroids[t] for t in tids], dtype=np.float64)
        h_vals = np.asarray(sbc.h_fn(pts), dtype=np.float64).reshape(-1)
        t_vals = np.asarray(sbc.t_amb_fn(pts), dtype=np.float64).reshape(-1)
        for tid, hv, tv in zip(tids, h_vals.tolist(), t_vals.tolist(), strict=True):
            _add_tri_conv(tid, float(hv), float(tv))

    return coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr(), rhs


def _assemble_thermal_contact_interfaces(
    mesh: CanonicalMesh3D, interfaces: Sequence[ThermalContactInterface3D]
) -> csr_matrix:
    """Assemble symmetric interfacial thermal contact conductance ``h_c (T_m - T_s)``."""
    n_nodes = len(mesh.nodes)
    if not interfaces:
        return csr_matrix((n_nodes, n_nodes), dtype=np.float64)
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    _, _, areas = mesh.surface_geometry()
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for iface in interfaces:
        m_nodes = sorted(face_node_indices_3d(mesh, (iface.master_face_tag,)))
        s_nodes = sorted(face_node_indices_3d(mesh, (iface.slave_face_tag,)))
        if not m_nodes or not s_nodes:
            continue
        # Tributary area per slave node from slave face triangles
        trib_area: dict[int, float] = {nid: 0.0 for nid in s_nodes}
        for tid in mesh.semantic_face_tags[iface.slave_face_tag]:
            a, b, c = mesh.surface_triangles[tid]
            share = float(areas[tid]) / 3.0
            for nid in (a, b, c):
                if nid in trib_area:
                    trib_area[nid] += share
        m_pts = nodes[m_nodes]
        tree = cKDTree(m_pts)
        for s_id in s_nodes:
            a_i = trib_area.get(s_id, 0.0)
            if a_i <= 0.0:
                continue
            _, nearest = tree.query(nodes[s_id], k=1)
            m_id = m_nodes[int(nearest)]
            if m_id == s_id:
                continue
            g = float(iface.conductance_w_per_mm2_k) * a_i
            rows.extend((m_id, s_id, m_id, s_id))
            cols.extend((m_id, s_id, s_id, m_id))
            vals.extend((g, g, -g, -g))
    return coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()


def solve_3d_multi_material_heat_conduction(
    mesh: CanonicalMesh3D,
    materials: Sequence[ElectroThermalMaterialSpec],
    *,
    part_cell_map: Mapping[str, Sequence[int]] | None = None,
    temperature_bcs: Sequence[TemperatureBC3D] = (),
    convection_bcs: Sequence[ConvectionBC3D] = (),
    spatial_convection_bcs: Sequence[SpatialConvectionBC3D] = (),
    heat_flux_bcs: Sequence[HeatFluxBC3D] = (),
    volumetric_sources: Sequence[VolumetricPowerSource3D] = (),
    quadrature_joule_w_per_mm3: FloatArray | None = None,
    thermal_interfaces: Sequence[ThermalContactInterface3D] = (),
    dt_s: float | None = None,
    num_steps: int = 1,
    initial_temperature_k: float | FloatArray | None = None,
    time_varying_temp_bcs_fn: Callable[[float], Sequence[TemperatureBC3D]]
    | None = None,
    crank_nicolson: bool = True,
    linear_solver: LinearSolverConfig | None = None,
) -> tuple[FloatArray, tuple[FloatArray, ...]]:
    """Solve steady or transient multi-material 3D heat conduction with CHT & contact resistance."""
    cell_mats = _resolve_cell_materials(mesh, materials, part_cell_map)
    k_mat, c_mat, cells, n_val, w_vol, q_xyz = _assemble_multimat_thermal_matrices(
        mesh, cell_mats
    )
    k_conv, f_conv = _assemble_spatial_convection(
        mesh, convection_bcs, spatial_convection_bcs
    )
    k_iface = _assemble_thermal_contact_interfaces(mesh, thermal_interfaces)
    k_total = (k_mat + k_conv + k_iface).tocsr()
    n_nodes = len(mesh.nodes)
    n_elem, n_q = w_vol.shape

    # Assemble volumetric heat source vector f_vol (W)
    q_total = (
        np.asarray(quadrature_joule_w_per_mm3, dtype=np.float64).copy()
        if quadrature_joule_w_per_mm3 is not None
        else np.zeros((n_elem, n_q), dtype=np.float64)
    )
    for src in volumetric_sources:
        active_cells: list[int]
        if src.cell_indices is not None:
            active_cells = [int(i) for i in src.cell_indices]
        elif src.part_id is not None and part_cell_map is not None:
            active_cells = [int(i) for i in part_cell_map[src.part_id]]
        else:
            active_cells = list(range(n_elem))
        if src.spatial_fn is not None:
            pts_flat = q_xyz[active_cells].reshape(-1, 3)
            vals_flat = np.asarray(src.spatial_fn(pts_flat), dtype=np.float64).reshape(
                len(active_cells), n_q
            )
            q_total[active_cells] += vals_flat
        if abs(src.power_density_w_per_mm3) > 0.0:
            q_total[active_cells] += float(src.power_density_w_per_mm3)

    f_vol_elem = np.einsum("eq,eq,qn->en", q_total, w_vol, n_val)
    f_vol = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(f_vol, cells, f_vol_elem)

    f_flux = np.zeros(n_nodes, dtype=np.float64)
    for hbc in heat_flux_bcs:
        f_flux += _assemble_surface_flux_rhs(
            mesh, hbc.face_tags, float(hbc.flux_w_per_mm2)
        )

    f_rhs_base = f_conv + f_vol + f_flux

    def _dirichlet_map(tbcs: Sequence[TemperatureBC3D]) -> dict[int, float]:
        fixed: dict[int, float] = {}
        for tbc in tbcs:
            for nid in face_node_indices_3d(mesh, tbc.face_tags):
                fixed[nid] = float(tbc.temperature_k)
        return fixed

    if dt_s is None:
        fixed = _dirichlet_map(temperature_bcs)
        mask = np.zeros(n_nodes, dtype=bool)
        t_vec = np.zeros(n_nodes, dtype=np.float64)
        for nid, val in fixed.items():
            mask[nid] = True
            t_vec[nid] = val
        free = np.nonzero(~mask)[0]
        if free.size > 0:
            rhs_free = (f_rhs_base - k_total @ t_vec)[free]
            t_f, rep = solve_linear_system(
                k_total[free][:, free],
                rhs_free,
                config=linear_solver,
                is_symmetric=True,
            )
            if not rep.converged:
                raise RuntimeError(
                    f"steady multi-material thermal solve failed: {rep.reason}"
                )
            t_vec[free] = t_f
        return t_vec, (t_vec.copy(),)

    # Transient stepping: Crank-Nicolson (theta=0.5) or Backward Euler (theta=1.0)
    if dt_s <= 0.0 or num_steps < 1:
        raise ValueError("dt_s must be positive and num_steps >= 1 for transient solve")
    theta = 0.5 if crank_nicolson else 1.0
    a_lhs = (c_mat / float(dt_s) + theta * k_total).tocsr()
    b_rhs_mat = (c_mat / float(dt_s) - (1.0 - theta) * k_total).tocsr()

    t_curr: FloatArray
    if initial_temperature_k is None:
        t_curr = np.full(
            n_nodes, float(cell_mats[0].reference_temp_k), dtype=np.float64
        )
    elif isinstance(initial_temperature_k, (int, float)):
        t_curr = np.full(n_nodes, float(initial_temperature_k), dtype=np.float64)
    else:
        t_curr = np.asarray(initial_temperature_k, dtype=np.float64).copy()

    history: list[FloatArray] = [t_curr.copy()]
    for step in range(1, num_steps + 1):
        t_time = step * float(dt_s)
        tbcs_step = (
            time_varying_temp_bcs_fn(t_time)
            if time_varying_temp_bcs_fn is not None
            else temperature_bcs
        )
        fixed = _dirichlet_map(tbcs_step)
        mask = np.zeros(n_nodes, dtype=bool)
        t_dir = np.zeros(n_nodes, dtype=np.float64)
        for nid, val in fixed.items():
            mask[nid] = True
            t_dir[nid] = val
        free = np.nonzero(~mask)[0]
        t_next = t_dir.copy()
        if free.size > 0:
            rhs_full = b_rhs_mat @ t_curr + f_rhs_base
            rhs_free = (rhs_full - a_lhs @ t_dir)[free]
            t_f, rep = solve_linear_system(
                a_lhs[free][:, free], rhs_free, config=linear_solver, is_symmetric=True
            )
            if not rep.converged:
                raise RuntimeError(
                    f"transient thermal step {step} failed: {rep.reason}"
                )
            t_next[free] = t_f
        t_curr = t_next
        history.append(t_curr.copy())
    return t_curr, tuple(history)


def assemble_multi_material_thermo_mech_residual_and_tangent(
    mesh: CanonicalMesh3D,
    u_vec: FloatArray,
    nodal_temperature_k: FloatArray,
    materials: Sequence[ElectroThermalMaterialSpec],
    *,
    part_cell_map: Mapping[str, Sequence[int]] | None = None,
    scale_thermal: float = 1.0,
) -> tuple[FloatArray, csr_matrix, FloatArray, FloatArray, float]:
    """Assemble exact multi-material finite-strain ``F = F_m F_th`` internal force & tangent."""
    cell_mats = _resolve_cell_materials(mesh, materials, part_cell_map)
    helper = TorchFemBackend3D()
    cells, d_phys, w_vol, _, rows, cols = helper._precompute_geometry(mesh)
    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes
    n_elem, n_loc = cells.shape
    n_q = d_phys.shape[1]
    n_edof = 3 * n_loc

    xi_m, _ = _tet_quadrature(mesh.cell_type)
    n_m = tet_shape_values_ref(mesh.cell_type, xi_m)  # (Q, n_loc)
    t_q = np.einsum("qn,en->eq", n_m, nodal_temperature_k[cells])  # (E, Q)

    alphas = np.array(
        [float(m.thermal_expansion_per_k) for m in cell_mats], dtype=np.float64
    )
    t_refs = np.array([float(m.reference_temp_k) for m in cell_mats], dtype=np.float64)
    lam_th = 1.0 + scale_thermal * alphas[:, None] * (t_q - t_refs[:, None])
    if float(np.min(lam_th)) <= 0.1:
        raise ValueError("non-physical multi-material thermal stretch <= 0.1")

    u_nod = u_vec.reshape(n_nodes, 3)
    u_e = u_nod[cells]
    grad_u = np.einsum("eni,eqnj->eqij", u_e, d_phys)
    f_tot = np.eye(3, dtype=np.float64)[None, None, :, :] + grad_u
    min_det = float(np.min(np.linalg.det(f_tot)))

    inv_lam = 1.0 / lam_th
    f_mech = f_tot * inv_lam[:, :, None, None]

    # Group elements by unique mechanical MaterialSpec for vectorized batched autograd
    p_tot_all = np.zeros((n_elem, n_q, 3, 3), dtype=np.float64)
    a_tot_all = np.zeros((n_elem, n_q, 3, 3, 3, 3), dtype=np.float64)
    mat_groups: dict[int, tuple[MaterialSpec, list[int]]] = {}
    for eid, etm in enumerate(cell_mats):
        key = id(etm.mechanical)
        if key not in mat_groups:
            mat_groups[key] = (etm.mechanical, [])
        mat_groups[key][1].append(eid)

    for mech_mat, eids in mat_groups.values():
        fm_sub = f_mech[eids]
        il_sub = inv_lam[eids]
        lt_sub = lam_th[eids]
        fm_t = torch.as_tensor(fm_sub, dtype=torch.float64)
        with torch.no_grad():
            p_m = first_piola(fm_t, mech_mat).numpy()
            a_m = consistent_tangent(fm_t, mech_mat).numpy()
        j_th = lt_sub**3
        # Exact chain rule: W_tot = J_th * W_mech(F / lam_th)
        # => P_tot = J_th * (1/lam_th) * P_mech, A_tot = J_th * (1/lam_th^2) * A_mech
        p_tot_all[eids] = (j_th * il_sub)[:, :, None, None] * p_m
        a_tot_all[eids] = (j_th * il_sub * il_sub)[:, :, None, None, None, None] * a_m

    f_e = np.einsum("eq,eqij,eqaj->eai", w_vol, p_tot_all, d_phys)
    f_int = np.zeros(n_dofs, dtype=np.float64)
    for a_i in range(n_loc):
        for comp in range(3):
            np.add.at(f_int, 3 * cells[:, a_i] + comp, f_e[:, a_i, comp])

    k_e = np.einsum(
        "eq,eqijkl,eqaj,eqbl->eaibk", w_vol, a_tot_all, d_phys, d_phys
    ).reshape(n_elem, n_edof, n_edof)
    k_mat = coo_matrix((k_e.reshape(-1), (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
    return f_int, k_mat, p_tot_all, f_tot, min_det


def solve_3d_electro_thermal_mechanical(
    mesh_or_assembly: CanonicalMesh3D | AssemblyMesh3D,
    materials: Sequence[ElectroThermalMaterialSpec],
    *,
    part_cell_map: Mapping[str, Sequence[int]] | None = None,
    dirichlet_bcs: Sequence[DirichletBC3D],
    traction_bcs: Sequence[TractionBC3D] = (),
    voltage_bcs: Sequence[VoltageBC3D] = (),
    current_flux_bcs: Sequence[CurrentFluxBC3D] = (),
    temperature_bcs: Sequence[TemperatureBC3D] = (),
    convection_bcs: Sequence[ConvectionBC3D] = (),
    spatial_convection_bcs: Sequence[SpatialConvectionBC3D] = (),
    heat_flux_bcs: Sequence[HeatFluxBC3D] = (),
    volumetric_sources: Sequence[VolumetricPowerSource3D] = (),
    thermal_interfaces: Sequence[ThermalContactInterface3D] = (),
    prescribed_temperature_field_k: FloatArray | None = None,
    dt_s: float | None = None,
    num_steps: int = 1,
    initial_temperature_k: float | FloatArray | None = None,
    time_varying_temp_bcs_fn: Callable[[float], Sequence[TemperatureBC3D]]
    | None = None,
    load_steps: int = 2,
    solver_tolerance: float = 1.0e-8,
    max_newton_iterations: int = 25,
    linear_solver: LinearSolverConfig | None = None,
) -> ElectroThermalMechanical3DResult:
    """Solve coupled multi-material 3D Electro-Thermal-Mechanical equilibrium."""
    t0 = time.perf_counter()
    if isinstance(mesh_or_assembly, AssemblyMesh3D):
        mesh = mesh_or_assembly.mesh
        cell_map: Mapping[str, Sequence[int]] | None = (
            {p.part_id: p.cell_indices for p in mesh_or_assembly.parts}
            if part_cell_map is None
            else part_cell_map
        )
    else:
        mesh = mesh_or_assembly
        cell_map = part_cell_map

    # 1) Optional electrical conduction + Joule heating
    elec_v: FloatArray | None = None
    joule_q: FloatArray | None = None
    total_joule_w = 0.0
    if voltage_bcs:
        elec_res = solve_3d_electrical_conduction(
            mesh,
            materials,
            part_cell_map=cell_map,
            voltage_bcs=voltage_bcs,
            current_flux_bcs=current_flux_bcs,
            linear_solver=linear_solver,
        )
        elec_v = elec_res.nodal_voltage_v
        joule_q = elec_res.quadrature_joule_w_per_mm3
        total_joule_w = elec_res.total_joule_power_w

    # 2) Multi-material steady or transient heat conduction
    t_nodal: FloatArray
    if prescribed_temperature_field_k is not None:
        t_nodal = np.asarray(prescribed_temperature_field_k, dtype=np.float64).reshape(
            len(mesh.nodes)
        )
        t_hist: tuple[FloatArray, ...] = (t_nodal.copy(),)
    else:
        t_nodal, t_hist = solve_3d_multi_material_heat_conduction(
            mesh,
            materials,
            part_cell_map=cell_map,
            temperature_bcs=temperature_bcs,
            convection_bcs=convection_bcs,
            spatial_convection_bcs=spatial_convection_bcs,
            heat_flux_bcs=heat_flux_bcs,
            volumetric_sources=volumetric_sources,
            quadrature_joule_w_per_mm3=joule_q,
            thermal_interfaces=thermal_interfaces,
            dt_s=dt_s,
            num_steps=num_steps,
            initial_temperature_k=initial_temperature_k,
            time_varying_temp_bcs_fn=time_varying_temp_bcs_fn,
            linear_solver=linear_solver,
        )

    # 3) Coupled multi-material finite-strain mechanical warpage Newton loop
    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes
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

    u: FloatArray = np.zeros(n_dofs, dtype=np.float64)
    total_iters = 0
    converged = True
    reason = "converged"
    last_p: FloatArray = np.zeros((len(mesh.cells), 1, 3, 3), dtype=np.float64)
    last_f: FloatArray = np.zeros_like(last_p)
    last_min_det = 1.0

    for step in range(1, load_steps + 1):
        frac = step / float(load_steps)
        f_ext = frac * f_ext_full
        for dof, val in target_d.items():
            u[dof] = frac * val
        step_conv = False
        for _ in range(max_newton_iterations):
            total_iters += 1
            f_int, k_mat, p_np, f_np, min_det = (
                assemble_multi_material_thermo_mech_residual_and_tangent(
                    mesh,
                    u,
                    t_nodal,
                    materials,
                    part_cell_map=cell_map,
                    scale_thermal=frac,
                )
            )
            last_p, last_f, last_min_det = p_np, f_np, min_det
            res = f_int - f_ext
            res_free = res[free_dofs]
            scale = max(
                float(np.linalg.norm(f_ext[free_dofs])),
                float(np.linalg.norm(f_int[free_dofs])),
                1.0,
            )
            rel_res = float(np.linalg.norm(res_free)) / scale
            if rel_res <= solver_tolerance:
                step_conv = True
                break
            du_free, rep = solve_linear_system(
                k_mat[free_dofs][:, free_dofs],
                -res_free,
                config=linear_solver,
                is_symmetric=True,
            )
            if not rep.converged or not np.all(np.isfinite(du_free)):
                converged = False
                reason = f"ETM linear solve failed at step {step}: {rep.reason}"
                break
            # Armijo-like line search ensuring det(F) > 0
            alpha_ls = 1.0
            for _ls in range(8):
                u_try = u.copy()
                u_try[free_dofs] += alpha_ls * du_free
                _, _, _, _, det_try = (
                    assemble_multi_material_thermo_mech_residual_and_tangent(
                        mesh,
                        u_try,
                        t_nodal,
                        materials,
                        part_cell_map=cell_map,
                        scale_thermal=frac,
                    )
                )
                if det_try > 1.0e-6:
                    u = u_try
                    break
                alpha_ls *= 0.5
        if not step_conv:
            converged = False
            reason = f"ETM Newton loop did not converge at load step {step}"
            break

    # Compute nodal + unsmoothed element-bound quadrature stress hotspots
    helper = TorchFemBackend3D()
    _, _, _, cell_vols, _, _ = helper._precompute_geometry(mesh)
    vm_cell_mean = np.mean(_cauchy_von_mises_3d(last_p, last_f), axis=1)
    nodal_vm = _project_cell_scalar_to_nodes_3d(mesh, vm_cell_mean, cell_vols)
    elem_peak_vm, quad_peak_vm, _ = compute_3d_element_hotspots(last_p, last_f)
    u_nod = u.reshape(n_nodes, 3)

    return ElectroThermalMechanical3DResult(
        converged=converged,
        mesh_id=mesh.mesh_id,
        nodal_voltage_v=elec_v,
        total_joule_power_w=total_joule_w,
        nodal_temperature_k=t_nodal,
        temperature_history_k=t_hist,
        nodal_displacement=u_nod,
        nodal_von_mises_mpa=nodal_vm,
        element_peak_von_mises_mpa=elem_peak_vm,
        quadrature_peak_von_mises_mpa=quad_peak_vm,
        max_temperature_k=float(np.max(t_nodal)),
        min_temperature_k=float(np.min(t_nodal)),
        max_displacement_mm=float(np.max(np.linalg.norm(u_nod, axis=1))),
        min_det_f=float(last_min_det),
        total_newton_iterations=total_iters,
        wall_time_s=time.perf_counter() - t0,
        reason=reason,
    )


__all__ = (
    "ETM_3D_RESULT_SCHEMA",
    "CurrentFluxBC3D",
    "Electrical3DResult",
    "ElectroThermalMaterialSpec",
    "ElectroThermalMechanical3DResult",
    "HeatFluxBC3D",
    "SpatialConvectionBC3D",
    "ThermalContactInterface3D",
    "VoltageBC3D",
    "VolumetricPowerSource3D",
    "assemble_multi_material_thermo_mech_residual_and_tangent",
    "solve_3d_electrical_conduction",
    "solve_3d_electro_thermal_mechanical",
    "solve_3d_multi_material_heat_conduction",
)
