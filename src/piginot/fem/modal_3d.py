"""3D consistent mass assembly, prestressed modal analysis & frequency sensitivity (V3-4C).

Solves the generalized sparse structural eigenproblem::

    K_t(u_0) phi_k = omega_k^2 M phi_k

in consistent mm-N-s-tonne units (where ``MPa = N/mm^2`` and ``density`` is in
``tonne/mm^3`` = ``1e-12 * kg/m^3``, yielding angular frequencies ``omega_k`` in
``rad/s`` and natural frequencies ``f_k = omega_k / (2 pi)`` in ``Hz``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix, diags  # type: ignore[import-untyped]
from scipy.sparse.linalg import eigsh  # type: ignore[import-untyped]

from piginot.contracts import MaterialSpec
from piginot.fem.backend_3d import (
    DirichletBC3D,
    TorchFemBackend3D,
    _tet_shape_grads_ref,
    face_node_indices_3d,
)
from piginot.fem.thermo_mechanical_3d import (
    duffy_tet_quadrature,
    tet_shape_values_ref,
)
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D

FloatArray = NDArray[np.float64]
MODAL_3D_RESULT_SCHEMA: Final = "formamech.modal-3d-result-v1"


@dataclass(frozen=True, slots=True)
class Modal3DResult:
    """Verified 3D structural modal / eigenfrequency analysis output."""

    mesh_id: str
    density_t_per_mm3: float
    prestressed: bool
    frequencies_hz: tuple[float, ...]
    angular_frequencies_rad_s: tuple[float, ...]
    mode_shapes_nodal: FloatArray  # (num_modes, N_nodes, 3), M-orthonormal
    participation_factors: FloatArray  # (num_modes, 3) along x, y, z
    total_mass_tonne: float
    wall_time_s: float

    def document(self) -> dict[str, Any]:
        return {
            "schema": MODAL_3D_RESULT_SCHEMA,
            "mesh_id": self.mesh_id,
            "density_t_per_mm3": float(self.density_t_per_mm3),
            "prestressed": self.prestressed,
            "frequencies_hz": [float(f) for f in self.frequencies_hz],
            "angular_frequencies_rad_s": [
                float(w) for w in self.angular_frequencies_rad_s
            ],
            "participation_factors": self.participation_factors.tolist(),
            "total_mass_tonne": float(self.total_mass_tonne),
            "wall_time_s": float(self.wall_time_s),
        }


def assemble_3d_mass_matrix(
    mesh: CanonicalMesh3D,
    density_t_per_mm3: float,
    *,
    lumped: bool = False,
) -> csr_matrix:
    """Assemble exact consistent (or row-sum lumped) 3D ``TET4``/``TET10`` mass matrix ``M``."""
    if not math.isfinite(density_t_per_mm3) or density_t_per_mm3 <= 0.0:
        raise ValueError("density_t_per_mm3 must be finite and positive")
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes
    n_elem, n_loc = cells.shape

    order = 3 if mesh.cell_type is CellType3D.TET10 else 2
    xi_pts, w_ref = duffy_tet_quadrature(order)
    n_val = tet_shape_values_ref(mesh.cell_type, xi_pts)  # (Q, n_loc)
    d_ref = _tet_shape_grads_ref(mesh.cell_type, xi_pts)

    x_e = nodes[cells]
    jac = np.einsum("eni,qna->eqia", x_e, d_ref)
    det_j = np.linalg.det(jac)
    w_vol = det_j * w_ref[None, :]  # (E, Q)

    # Scalar element mass M_s[e, a, b] = rho * sum_q w_vol[e, q] N_a(q) N_b(q)
    m_scalar = float(density_t_per_mm3) * np.einsum(
        "eq,qa,qb->eab", w_vol, n_val, n_val
    )
    ii_s = np.repeat(cells[:, :, None], n_loc, axis=2).reshape(-1)
    jj_s = np.repeat(cells[:, None, :], n_loc, axis=1).reshape(-1)
    m_s_mat = coo_matrix(
        (m_scalar.reshape(-1), (ii_s, jj_s)), shape=(n_nodes, n_nodes)
    ).tocsr()

    if lumped:
        row_sums = np.asarray(m_s_mat.sum(axis=1), dtype=np.float64).reshape(-1)
        diag_3 = np.repeat(row_sums, 3)
        return diags(diag_3, format="csr")

    # Expand scalar nodal mass identically onto x, y, z DOFs (3*i + c, 3*j + c)
    coo_s = m_s_mat.tocoo()
    rows_all: list[NDArray[np.int64]] = []
    cols_all: list[NDArray[np.int64]] = []
    vals_all: list[FloatArray] = []
    for c in range(3):
        rows_all.append(3 * coo_s.row.astype(np.int64) + c)
        cols_all.append(3 * coo_s.col.astype(np.int64) + c)
        vals_all.append(coo_s.data.astype(np.float64))
    return coo_matrix(
        (
            np.concatenate(vals_all),
            (np.concatenate(rows_all), np.concatenate(cols_all)),
        ),
        shape=(n_dofs, n_dofs),
    ).tocsr()


def solve_3d_modal_analysis(
    mesh: CanonicalMesh3D,
    material: MaterialSpec,
    *,
    density_t_per_mm3: float,
    dirichlet_bcs: Sequence[DirichletBC3D],
    num_modes: int = 6,
    prestress_displacement: FloatArray | None = None,
    lumped_mass: bool = False,
    shift_sigma: float = 0.0,
) -> Modal3DResult:
    """Compute the lowest ``num_modes`` natural frequencies & M-orthonormal mode shapes."""
    t0 = time.perf_counter()
    if type(num_modes) is not int or num_modes < 1:
        raise ValueError("num_modes must be >= 1")
    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes

    helper = TorchFemBackend3D()
    cells, d_phys, w_vol, _, rows, cols = helper._precompute_geometry(mesh)
    u0 = (
        np.zeros(n_dofs, dtype=np.float64)
        if prestress_displacement is None
        else np.asarray(prestress_displacement, dtype=np.float64).reshape(n_dofs)
    )
    _, k_mat, _, _, _, min_det = helper._assemble(
        u0, cells, d_phys, w_vol, rows, cols, material, n_dofs
    )
    if min_det <= 1.0e-8:
        raise ValueError(
            "inadmissible prestress deformation (det F <= 0) for modal solve"
        )
    k_sym = (0.5 * (k_mat + k_mat.T)).tocsr()
    m_mat = assemble_3d_mass_matrix(mesh, density_t_per_mm3, lumped=lumped_mass)

    comp_idx = {"x": 0, "y": 1, "z": 2}
    constrained = np.zeros(n_dofs, dtype=bool)
    for bc in dirichlet_bcs:
        nids = face_node_indices_3d(mesh, bc.face_tags)
        for comp in bc.components:
            c = comp_idx[comp]
            for nid in nids:
                constrained[3 * nid + c] = True
    free = np.nonzero(~constrained)[0]
    if free.size <= num_modes:
        raise ValueError(
            f"free DOF count {free.size} must exceed num_modes={num_modes}"
        )

    k_ff = k_sym[free][:, free].tocsc()
    m_ff = m_mat[free][:, free].tocsc()

    eigvals, eigvecs = eigsh(
        k_ff,
        k=num_modes,
        M=m_ff,
        sigma=float(shift_sigma),
        which="LM",
    )
    order = np.argsort(eigvals)
    eigvals = np.maximum(np.asarray(eigvals[order], dtype=np.float64), 0.0)
    eigvecs = np.asarray(eigvecs[:, order], dtype=np.float64)

    omegas = np.sqrt(eigvals)
    freqs = omegas / (2.0 * math.pi)

    modes_full = np.zeros((num_modes, n_dofs), dtype=np.float64)
    participation = np.zeros((num_modes, 3), dtype=np.float64)
    unit_rigid = np.zeros((n_dofs, 3), dtype=np.float64)
    for c in range(3):
        unit_rigid[c::3, c] = 1.0
    m_rigid = np.asarray(m_mat @ unit_rigid, dtype=np.float64)  # (n_dofs, 3)

    for k in range(num_modes):
        phi_f = eigvecs[:, k]
        # Enforce strict M-orthonormality phi^T M phi = 1
        m_norm = math.sqrt(max(float(phi_f @ (m_ff @ phi_f)), 1.0e-30))
        phi_f = phi_f / m_norm
        modes_full[k, free] = phi_f
        participation[k, :] = modes_full[k] @ m_rigid

    # Total mass = (1/3) * sum(M_x + M_y + M_z) = e_x^T M e_x
    total_mass = float(unit_rigid[:, 0] @ m_rigid[:, 0])

    return Modal3DResult(
        mesh_id=mesh.mesh_id,
        density_t_per_mm3=float(density_t_per_mm3),
        prestressed=prestress_displacement is not None,
        frequencies_hz=tuple(float(f) for f in freqs),
        angular_frequencies_rad_s=tuple(float(w) for w in omegas),
        mode_shapes_nodal=modes_full.reshape(num_modes, n_nodes, 3),
        participation_factors=participation,
        total_mass_tonne=total_mass,
        wall_time_s=time.perf_counter() - t0,
    )


def analytical_eigenfrequency_sensitivity(
    omega_rad_s: float,
    mode_shape_full: FloatArray,
    dk_dp: csr_matrix,
    dm_dp: csr_matrix | None = None,
) -> float:
    """Exact first-order modal sensitivity ``d(omega)/dp`` for M-normalized mode ``phi``.

    Formula::

        d(omega)/dp = (1 / (2 * omega)) * phi^T (dK/dp - omega^2 dM/dp) phi
    """
    phi = np.asarray(mode_shape_full, dtype=np.float64).reshape(-1)
    num = float(phi @ (dk_dp @ phi))
    if dm_dp is not None:
        num -= (omega_rad_s * omega_rad_s) * float(phi @ (dm_dp @ phi))
    return num / max(2.0 * omega_rad_s, 1.0e-15)


__all__ = (
    "MODAL_3D_RESULT_SCHEMA",
    "Modal3DResult",
    "analytical_eigenfrequency_sensitivity",
    "assemble_3d_mass_matrix",
    "solve_3d_modal_analysis",
)
