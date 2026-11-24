"""3D shear-locking-free DKT-CST flat shells and Timoshenko beams (V4-3).

Implements two workhorse thin-walled structural elements in consistent mm-N-MPa
units with 6 DOFs/node ``(u, v, w, rx, ry, rz)``:

* **DKT-CST 3-node flat triangular shell** (:func:`solve_3d_dkt_shell`):
  Constant-Strain-Triangle (CST) membrane + exact Batoz-Bathe-Ho (1980)
  Discrete Kirchhoff Triangle (DKT) plate bending + light drilling-rotation
  stabilization transformed into arbitrary 3D curved/folded shells via local
  orthonormal element frames. Zero transverse shear locking down to
  ``t / L <= 1e-4``.
* **3D Timoshenko beam element** (:func:`solve_3d_timoshenko_beam`):
  Exact closed-form 12x12 element stiffness matrix accounting for axial,
  torsional, bi-axial bending, and Timoshenko transverse shear deformation
  factors ``Phi_y, Phi_z``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix  # type: ignore[import-untyped]

from piginot.fem.linear_solvers import (
    LinearSolverConfig,
    LinearSolverMethod,
    solve_linear_system,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SHELL_3D_RESULT_SCHEMA: Final = "formamech.shell-3d-result-v1"
BEAM_3D_RESULT_SCHEMA: Final = "formamech.beam-3d-result-v1"


@dataclass(frozen=True, slots=True)
class ShellMesh3D:
    """3-node triangular 3D shell surface mesh with named boundary node sets."""

    mesh_id: str
    nodes: FloatArray  # (N, 3) in mm
    triangles: IntArray  # (E, 3) 0-based vertex indices
    thickness_mm: float
    boundary_node_sets: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must have shape (N, 3)")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (E, 3)")
        if not math.isfinite(self.thickness_mm) or self.thickness_mm <= 0.0:
            raise ValueError("thickness_mm must be finite and positive")


@dataclass(frozen=True, slots=True)
class ShellDirichletBC:
    """Nodal essential boundary condition on a named shell node set."""

    name: str
    node_set: str
    components: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    values: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class Shell3DResult:
    """Verified 3D DKT-CST shell linear-static solve output."""

    mesh_id: str
    thickness_mm: float
    youngs_modulus_mpa: float
    poisson_ratio: float
    nodal_dofs: FloatArray  # (N, 6): u, v, w, rx, ry, rz
    max_displacement_mm: float
    max_deflection_z_mm: float
    strain_energy_mj: float
    solver_method: str
    wall_time_s: float

    def document(self) -> dict[str, Any]:
        return {
            "schema": SHELL_3D_RESULT_SCHEMA,
            "mesh_id": self.mesh_id,
            "thickness_mm": float(self.thickness_mm),
            "youngs_modulus_mpa": float(self.youngs_modulus_mpa),
            "poisson_ratio": float(self.poisson_ratio),
            "node_count": int(self.nodal_dofs.shape[0]),
            "max_displacement_mm": float(self.max_displacement_mm),
            "max_deflection_z_mm": float(self.max_deflection_z_mm),
            "strain_energy_mj": float(self.strain_energy_mj),
            "solver_method": self.solver_method,
            "wall_time_s": float(self.wall_time_s),
        }


def make_rectangular_plate_shell_mesh(
    lx: float,
    ly: float,
    *,
    nx: int = 8,
    ny: int = 8,
    thickness_mm: float = 1.0,
    mesh_id: str = "plate-shell-3d",
) -> ShellMesh3D:
    """Generate a symmetric cross-diagonal triangular mesh on ``[0, lx] x [0, ly]``."""
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")
    xs = np.linspace(0.0, float(lx), nx + 1)
    ys = np.linspace(0.0, float(ly), ny + 1)
    coords: list[tuple[float, float, float]] = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            coords.append((float(xs[i]), float(ys[j]), 0.0))
    tris: list[tuple[int, int, int]] = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * (nx + 1) + i
            n10 = n00 + 1
            n01 = (j + 1) * (nx + 1) + i
            n11 = n01 + 1
            if (i + j) % 2 == 0:
                tris.append((n00, n10, n11))
                tris.append((n00, n11, n01))
            else:
                tris.append((n00, n10, n01))
                tris.append((n10, n11, n01))

    nodes = np.asarray(coords, dtype=np.float64)
    triangles = np.asarray(tris, dtype=np.int64)
    tol = 1.0e-9 * max(float(lx), float(ly), 1.0)
    x_min = tuple(int(k) for k in range(len(nodes)) if nodes[k, 0] <= tol)
    x_max = tuple(int(k) for k in range(len(nodes)) if abs(nodes[k, 0] - lx) <= tol)
    y_min = tuple(int(k) for k in range(len(nodes)) if nodes[k, 1] <= tol)
    y_max = tuple(int(k) for k in range(len(nodes)) if abs(nodes[k, 1] - ly) <= tol)
    perimeter = tuple(sorted(set(x_min) | set(x_max) | set(y_min) | set(y_max)))
    return ShellMesh3D(
        mesh_id=mesh_id,
        nodes=nodes,
        triangles=triangles,
        thickness_mm=float(thickness_mm),
        boundary_node_sets={
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "perimeter": perimeter,
        },
    )


def _dkt_bending_stiffness_local(
    xy: FloatArray,
    d_bend: FloatArray,
) -> tuple[FloatArray, float]:
    """Assemble exact 9x9 DKT local bending stiffness for DOFs ``(w, rx, ry)`` x 3."""
    x1, y1 = float(xy[0, 0]), float(xy[0, 1])
    x2, y2 = float(xy[1, 0]), float(xy[1, 1])
    x3, y3 = float(xy[2, 0]), float(xy[2, 1])

    j_mat = np.array([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]], dtype=np.float64)
    det_j = float(j_mat[0, 0] * j_mat[1, 1] - j_mat[0, 1] * j_mat[1, 0])
    if det_j <= 1.0e-15:
        raise ValueError("degenerate or inverted shell triangle encountered")
    area = 0.5 * det_j
    j_inv = np.linalg.inv(j_mat)

    # Map local bending DOFs q = [w1, rx1, ry1, w2, rx2, ry2, w3, rx3, ry3]
    # to 6-node quadratic fiber rotations beta_x = +ry, beta_y = -rx.
    m_x = np.zeros((6, 9), dtype=np.float64)
    m_y = np.zeros((6, 9), dtype=np.float64)
    for k in range(3):
        m_x[k, 3 * k + 2] = 1.0
        m_y[k, 3 * k + 1] = -1.0

    # Midside nodes: 3 -> edge (0,1), 4 -> edge (1,2), 5 -> edge (2,0)
    edges = ((3, 0, 1), (4, 1, 2), (5, 2, 0))
    for m_idx, i, j in edges:
        dx = float(xy[j, 0] - xy[i, 0])
        dy = float(xy[j, 1] - xy[i, 1])
        le = math.hypot(dx, dy)
        c = dx / le
        s = dy / le
        # beta_t(k) = c * beta_x(k) + s * beta_y(k)
        # beta_n(k) = -s * beta_x(k) + c * beta_y(k)
        bt_i = c * m_x[i] + s * m_y[i]
        bt_j = c * m_x[j] + s * m_y[j]
        bn_i = -s * m_x[i] + c * m_y[i]
        bn_j = -s * m_x[j] + c * m_y[j]

        # Discrete Kirchhoff midpoint constraint:
        # beta_t(m) = -3/(2 L) * (w_j - w_i) - 1/4 * (beta_t(i) + beta_t(j))
        # beta_n(m) = 1/2 * (beta_n(i) + beta_n(j))
        bt_m = -0.25 * (bt_i + bt_j)
        bt_m[3 * i] += 1.5 / le
        bt_m[3 * j] -= 1.5 / le
        bn_m = 0.5 * (bn_i + bn_j)

        m_x[m_idx] = c * bt_m - s * bn_m
        m_y[m_idx] = s * bt_m + c * bn_m

    # 3-point mid-edge Gauss quadrature (exact for quadratic curvature products)
    gauss_pts = ((0.5, 0.0), (0.5, 0.5), (0.0, 0.5))
    k_b = np.zeros((9, 9), dtype=np.float64)
    for xi, eta in gauss_pts:
        l1 = 1.0 - xi - eta
        l2 = xi
        l3 = eta
        # Reference gradients of 6-node quadratic basis [N1..N6] w.r.t. (xi, eta)
        dn_dxi = np.array(
            [
                1.0 - 4.0 * l1,
                4.0 * l2 - 1.0,
                0.0,
                4.0 * (l1 - l2),
                4.0 * l3,
                -4.0 * l3,
            ],
            dtype=np.float64,
        )
        dn_deta = np.array(
            [
                1.0 - 4.0 * l1,
                0.0,
                4.0 * l3 - 1.0,
                -4.0 * l2,
                4.0 * l2,
                4.0 * (l1 - l3),
            ],
            dtype=np.float64,
        )
        # Physical gradients: [dN/dx, dN/dy] = [dN/dxi, dN/deta] @ J^{-1}
        dn_ref = np.column_stack((dn_dxi, dn_deta))
        dn_xy = dn_ref @ j_inv
        dn_dx = dn_xy[:, 0]
        dn_dy = dn_xy[:, 1]

        b_mat = np.zeros((3, 9), dtype=np.float64)
        b_mat[0] = dn_dx @ m_x
        b_mat[1] = dn_dy @ m_y
        b_mat[2] = dn_dy @ m_x + dn_dx @ m_y
        k_b += (area / 3.0) * (b_mat.T @ d_bend @ b_mat)

    return k_b, area


def _cst_membrane_stiffness_local(
    xy: FloatArray, d_mem: FloatArray, area: float
) -> FloatArray:
    """Assemble exact 6x6 CST local membrane stiffness for DOFs ``(u, v)`` x 3."""
    x1, y1 = float(xy[0, 0]), float(xy[0, 1])
    x2, y2 = float(xy[1, 0]), float(xy[1, 1])
    x3, y3 = float(xy[2, 0]), float(xy[2, 1])
    b_vec = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64) / (2.0 * area)
    c_vec = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64) / (2.0 * area)
    b_m = np.zeros((3, 6), dtype=np.float64)
    for k in range(3):
        b_m[0, 2 * k] = b_vec[k]
        b_m[1, 2 * k + 1] = c_vec[k]
        b_m[2, 2 * k] = c_vec[k]
        b_m[2, 2 * k + 1] = b_vec[k]
    return area * (b_m.T @ d_mem @ b_m)


def _shell_local_frame(p0: FloatArray, p1: FloatArray, p2: FloatArray) -> FloatArray:
    """Build 3x3 orthogonal rotation matrix ``R`` (rows = local e1, e2, e3)."""
    e1 = p1 - p0
    n1 = float(np.linalg.norm(e1))
    if n1 <= 1.0e-15:
        raise ValueError("degenerate shell triangle edge length")
    e1 = e1 / n1
    e3 = np.cross(e1, p2 - p0)
    n3 = float(np.linalg.norm(e3))
    if n3 <= 1.0e-15:
        raise ValueError("collinear shell triangle vertices")
    e3 = e3 / n3
    e2 = np.cross(e3, e1)
    return np.vstack((e1, e2, e3))


def solve_3d_dkt_shell(
    mesh: ShellMesh3D,
    *,
    youngs_modulus_mpa: float,
    poisson_ratio: float,
    dirichlet_bcs: Sequence[ShellDirichletBC],
    uniform_pressure_mpa: float = 0.0,
    nodal_loads_n: FloatArray | None = None,
    solver_config: LinearSolverConfig | None = None,
) -> Shell3DResult:
    """Solve a 3D flat/curved DKT-CST shell problem with 6 DOFs per node."""
    t0 = time.perf_counter()
    e_mod = float(youngs_modulus_mpa)
    nu = float(poisson_ratio)
    t_mm = float(mesh.thickness_mm)
    if e_mod <= 0.0 or not (-0.1 <= nu < 0.5):
        raise ValueError("invalid elastic constants for shell solve")

    d_unit = np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]],
        dtype=np.float64,
    ) / (1.0 - nu * nu)
    d_mem = (e_mod * t_mm) * d_unit
    d_bend = (e_mod * (t_mm**3) / 12.0) * d_unit

    n_nodes = int(mesh.nodes.shape[0])
    n_dofs = 6 * n_nodes
    f_global = np.zeros(n_dofs, dtype=np.float64)
    if nodal_loads_n is not None:
        arr_load = np.asarray(nodal_loads_n, dtype=np.float64).reshape(n_nodes, 6)
        f_global += arr_load.reshape(-1)

    rows_list: list[IntArray] = []
    cols_list: list[IntArray] = []
    vals_list: list[FloatArray] = []

    mem_idx = np.array([0, 1, 6, 7, 12, 13], dtype=np.int64)
    bend_idx = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16], dtype=np.int64)
    drill_idx = np.array([5, 11, 17], dtype=np.int64)

    for tri in mesh.triangles:
        n0, n1, n2 = int(tri[0]), int(tri[1]), int(tri[2])
        p0, p1, p2 = mesh.nodes[n0], mesh.nodes[n1], mesh.nodes[n2]
        r_mat = _shell_local_frame(p0, p1, p2)
        xy = np.vstack(
            (
                np.zeros(2, dtype=np.float64),
                (r_mat[:2] @ (p1 - p0)),
                (r_mat[:2] @ (p2 - p0)),
            )
        )
        k_b, area = _dkt_bending_stiffness_local(xy, d_bend)
        k_m = _cst_membrane_stiffness_local(xy, d_mem, area)

        k_loc = np.zeros((18, 18), dtype=np.float64)
        k_loc[np.ix_(mem_idx, mem_idx)] += k_m
        k_loc[np.ix_(bend_idx, bend_idx)] += k_b
        # Small drilling rotation regularization around local normal e3
        k_drill = 1.0e-4 * e_mod * t_mm * area
        k_loc[drill_idx, drill_idx] += k_drill

        # Block-diagonal 18x18 rotation matrix T mapping global -> local DOFs
        t_mat = np.zeros((18, 18), dtype=np.float64)
        for blk in range(6):
            s = 3 * blk
            t_mat[s : s + 3, s : s + 3] = r_mat
        k_glob = t_mat.T @ k_loc @ t_mat

        dof_ids = np.array(
            [6 * n0 + c for c in range(6)]
            + [6 * n1 + c for c in range(6)]
            + [6 * n2 + c for c in range(6)],
            dtype=np.int64,
        )
        ii, jj = np.meshgrid(dof_ids, dof_ids, indexing="ij")
        rows_list.append(ii.reshape(-1))
        cols_list.append(jj.reshape(-1))
        vals_list.append(k_glob.reshape(-1))

        if uniform_pressure_mpa != 0.0:
            # Consistent lumped nodal force along local +e3 normal
            f_tri = (uniform_pressure_mpa * area / 3.0) * r_mat[2]
            for nid in (n0, n1, n2):
                f_global[6 * nid : 6 * nid + 3] += f_tri

    k_mat: csr_matrix = coo_matrix(
        (
            np.concatenate(vals_list),
            (np.concatenate(rows_list), np.concatenate(cols_list)),
        ),
        shape=(n_dofs, n_dofs),
    ).tocsr()

    constrained = np.zeros(n_dofs, dtype=bool)
    u_prescribed = np.zeros(n_dofs, dtype=np.float64)
    for bc in dirichlet_bcs:
        if bc.node_set not in mesh.boundary_node_sets:
            raise ValueError(f"unknown shell boundary node_set {bc.node_set!r}")
        for nid in mesh.boundary_node_sets[bc.node_set]:
            for comp, val in zip(bc.components, bc.values, strict=True):
                dof = 6 * int(nid) + int(comp)
                constrained[dof] = True
                u_prescribed[dof] = float(val)

    free = np.nonzero(~constrained)[0]
    if free.size == 0:
        raise ValueError("all shell DOFs are constrained")

    rhs_f = f_global[free] - (k_mat[free][:, constrained] @ u_prescribed[constrained])
    k_ff = k_mat[free][:, free].tocsr()
    cfg = (
        LinearSolverConfig(method=LinearSolverMethod.DIRECT)
        if solver_config is None
        else solver_config
    )
    u_f, rep = solve_linear_system(k_ff, rhs_f, config=cfg, is_symmetric=True)
    if not rep.converged:
        raise RuntimeError(f"shell linear solve did not converge: {rep.reason}")

    u_full = u_prescribed.copy()
    u_full[free] = u_f
    nodal = u_full.reshape(n_nodes, 6)
    disp_mag = np.linalg.norm(nodal[:, :3], axis=1)
    energy = 0.5 * float(u_full @ (k_mat @ u_full))
    return Shell3DResult(
        mesh_id=mesh.mesh_id,
        thickness_mm=t_mm,
        youngs_modulus_mpa=e_mod,
        poisson_ratio=nu,
        nodal_dofs=nodal,
        max_displacement_mm=float(np.max(disp_mag)),
        max_deflection_z_mm=float(np.max(np.abs(nodal[:, 2]))),
        strain_energy_mj=energy,
        solver_method=rep.method_used,
        wall_time_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# 3D Timoshenko Beam Element
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BeamSection3D:
    """Cross-section properties for a 3D Timoshenko beam element (mm units)."""

    area_mm2: float
    iy_mm4: float
    iz_mm4: float
    j_torsion_mm4: float
    shear_factor_ky: float = 5.0 / 6.0
    shear_factor_kz: float = 5.0 / 6.0

    @classmethod
    def rectangular(cls, width_y_mm: float, height_z_mm: float) -> BeamSection3D:
        b = float(width_y_mm)
        h = float(height_z_mm)
        area = b * h
        iy = b * (h**3) / 12.0
        iz = h * (b**3) / 12.0
        a_l, b_s = max(b, h), min(b, h)
        # Roark rectangular torsion constant approximation
        j_t = (
            a_l
            * (b_s**3)
            * (1.0 / 3.0 - 0.21 * (b_s / a_l) * (1.0 - (b_s / a_l) ** 4 / 12.0))
        )
        return cls(
            area_mm2=area,
            iy_mm4=iy,
            iz_mm4=iz,
            j_torsion_mm4=max(j_t, 1.0e-9),
        )


@dataclass(frozen=True, slots=True)
class Beam3DResult:
    """Verified 3D Timoshenko beam / frame linear-static solve output."""

    node_count: int
    element_count: int
    nodal_dofs: FloatArray  # (N, 6): u, v, w, rx, ry, rz
    max_displacement_mm: float
    strain_energy_mj: float
    wall_time_s: float

    def document(self) -> dict[str, Any]:
        return {
            "schema": BEAM_3D_RESULT_SCHEMA,
            "node_count": self.node_count,
            "element_count": self.element_count,
            "max_displacement_mm": float(self.max_displacement_mm),
            "strain_energy_mj": float(self.strain_energy_mj),
            "wall_time_s": float(self.wall_time_s),
        }


def _timoshenko_beam_local_stiffness(
    length_mm: float,
    e_mpa: float,
    g_mpa: float,
    section: BeamSection3D,
) -> FloatArray:
    """Exact 12x12 local Timoshenko beam stiffness in ``(u, v, w, rx, ry, rz)``."""
    le = float(length_mm)
    if le <= 1.0e-12:
        raise ValueError("zero-length beam element")
    ea_l = e_mpa * section.area_mm2 / le
    gj_l = g_mpa * section.j_torsion_mm4 / le

    phi_y = (
        12.0
        * e_mpa
        * section.iz_mm4
        / (section.shear_factor_ky * g_mpa * section.area_mm2 * le * le)
    )
    phi_z = (
        12.0
        * e_mpa
        * section.iy_mm4
        / (section.shear_factor_kz * g_mpa * section.area_mm2 * le * le)
    )
    cz = e_mpa * section.iz_mm4 / ((1.0 + phi_y) * (le**3))
    cy = e_mpa * section.iy_mm4 / ((1.0 + phi_z) * (le**3))

    k = np.zeros((12, 12), dtype=np.float64)
    # Axial (0, 6)
    k[0, 0] = k[6, 6] = ea_l
    k[0, 6] = k[6, 0] = -ea_l
    # Torsion (3, 9)
    k[3, 3] = k[9, 9] = gj_l
    k[3, 9] = k[9, 3] = -gj_l

    # Bending in x-y plane (v1=1, rz1=5, v2=7, rz2=11)
    idx_xy = [1, 5, 7, 11]
    k_xy = cz * np.array(
        [
            [12.0, 6.0 * le, -12.0, 6.0 * le],
            [6.0 * le, (4.0 + phi_y) * le * le, -6.0 * le, (2.0 - phi_y) * le * le],
            [-12.0, -6.0 * le, 12.0, -6.0 * le],
            [6.0 * le, (2.0 - phi_y) * le * le, -6.0 * le, (4.0 + phi_y) * le * le],
        ],
        dtype=np.float64,
    )
    k[np.ix_(idx_xy, idx_xy)] += k_xy

    # Bending in x-z plane (w1=2, ry1=4, w2=8, ry2=10); right-hand rule dw/dx = -ry
    idx_xz = [2, 4, 8, 10]
    k_xz = cy * np.array(
        [
            [12.0, -6.0 * le, -12.0, -6.0 * le],
            [-6.0 * le, (4.0 + phi_z) * le * le, 6.0 * le, (2.0 - phi_z) * le * le],
            [-12.0, 6.0 * le, 12.0, 6.0 * le],
            [-6.0 * le, (2.0 - phi_z) * le * le, 6.0 * le, (4.0 + phi_z) * le * le],
        ],
        dtype=np.float64,
    )
    k[np.ix_(idx_xz, idx_xz)] += k_xz
    return k


def _beam_rotation_matrix(p0: FloatArray, p1: FloatArray) -> FloatArray:
    """Construct orthonormal 3x3 frame (e1 along beam axis, e2, e3 principal)."""
    vec = p1 - p0
    le = float(np.linalg.norm(vec))
    e1 = vec / le
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(e1, ref))) > 0.99:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e2 = np.cross(ref, e1)
    e2 /= float(np.linalg.norm(e2))
    e3 = np.cross(e1, e2)
    return np.vstack((e1, e2, e3))


def solve_3d_timoshenko_beam(
    nodes_mm: FloatArray,
    elements: IntArray,
    *,
    section: BeamSection3D,
    youngs_modulus_mpa: float,
    poisson_ratio: float,
    constrained_dofs: Mapping[int, float],
    nodal_loads_n: FloatArray,
) -> Beam3DResult:
    """Solve a 3D Timoshenko beam/frame structure with exact shear deformation."""
    t0 = time.perf_counter()
    nodes = np.asarray(nodes_mm, dtype=np.float64)
    elems = np.asarray(elements, dtype=np.int64)
    n_nodes = int(nodes.shape[0])
    n_dofs = 6 * n_nodes
    g_mpa = float(youngs_modulus_mpa) / (2.0 * (1.0 + float(poisson_ratio)))

    rows_l: list[IntArray] = []
    cols_l: list[IntArray] = []
    vals_l: list[FloatArray] = []
    for elem in elems:
        n0, n1 = int(elem[0]), int(elem[1])
        p0, p1 = nodes[n0], nodes[n1]
        le = float(np.linalg.norm(p1 - p0))
        k_loc = _timoshenko_beam_local_stiffness(le, youngs_modulus_mpa, g_mpa, section)
        r_mat = _beam_rotation_matrix(p0, p1)
        t_mat = np.zeros((12, 12), dtype=np.float64)
        for blk in range(4):
            s = 3 * blk
            t_mat[s : s + 3, s : s + 3] = r_mat
        k_glob = t_mat.T @ k_loc @ t_mat
        dofs = np.array(
            [6 * n0 + c for c in range(6)] + [6 * n1 + c for c in range(6)],
            dtype=np.int64,
        )
        ii, jj = np.meshgrid(dofs, dofs, indexing="ij")
        rows_l.append(ii.reshape(-1))
        cols_l.append(jj.reshape(-1))
        vals_l.append(k_glob.reshape(-1))

    k_mat: csr_matrix = coo_matrix(
        (
            np.concatenate(vals_l),
            (np.concatenate(rows_l), np.concatenate(cols_l)),
        ),
        shape=(n_dofs, n_dofs),
    ).tocsr()
    f_vec = np.asarray(nodal_loads_n, dtype=np.float64).reshape(n_dofs)

    constrained = np.zeros(n_dofs, dtype=bool)
    u_pres = np.zeros(n_dofs, dtype=np.float64)
    for dof_idx, val in constrained_dofs.items():
        constrained[int(dof_idx)] = True
        u_pres[int(dof_idx)] = float(val)
    free = np.nonzero(~constrained)[0]
    rhs = f_vec[free] - (k_mat[free][:, constrained] @ u_pres[constrained])
    u_f, rep = solve_linear_system(
        k_mat[free][:, free].tocsr(),
        rhs,
        config=LinearSolverConfig(method=LinearSolverMethod.DIRECT),
        is_symmetric=True,
    )
    if not rep.converged:
        raise RuntimeError(f"beam linear solve failed: {rep.reason}")
    u_full = u_pres.copy()
    u_full[free] = u_f
    nodal = u_full.reshape(n_nodes, 6)
    energy = 0.5 * float(u_full @ (k_mat @ u_full))
    return Beam3DResult(
        node_count=n_nodes,
        element_count=int(elems.shape[0]),
        nodal_dofs=nodal,
        max_displacement_mm=float(np.max(np.linalg.norm(nodal[:, :3], axis=1))),
        strain_energy_mj=energy,
        wall_time_s=time.perf_counter() - t0,
    )


__all__ = (
    "BEAM_3D_RESULT_SCHEMA",
    "SHELL_3D_RESULT_SCHEMA",
    "Beam3DResult",
    "BeamSection3D",
    "Shell3DResult",
    "ShellDirichletBC",
    "ShellMesh3D",
    "make_rectangular_plate_shell_mesh",
    "solve_3d_dkt_shell",
    "solve_3d_timoshenko_beam",
)
