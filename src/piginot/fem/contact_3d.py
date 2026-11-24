"""Pure-MIT 3D multi-material assembly, tied-interface & 3D contact FEM solver (V3-4A).

Supports:
* Multi-part, multi-material finite-strain 3D ``TET4``/``TET10`` Total-Lagrangian
  assembly over :class:`~piginot.geometry.assembly_3d.AssemblyMesh3D`,
* Non-conformal tied-interface penalty / Augmented Lagrangian face coupling,
* 3D frictionless & regularized Coulomb rigid-sphere / plane indenter contact
  with exact consistent tangent and Armijo gap backtracking,
* Pure-PyTorch Third-Medium Contact (TMC) regularized void energy density.
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

from piginot.fem.backend_3d import (
    DirichletBC3D,
    TorchFemBackend3D,
    TractionBC3D,
    _cauchy_von_mises_3d,
    _project_cell_scalar_to_nodes_3d,
    assemble_surface_tractions_3d,
    face_node_indices_3d,
)
from piginot.fem.linear_solvers import LinearSolverConfig, solve_linear_system
from piginot.geometry.assembly_3d import AssemblyMesh3D, InterfaceKind
from piginot.physics import consistent_tangent, energy_density, first_piola

FloatArray = NDArray[np.float64]
CONTACT_3D_RESULT_SCHEMA: Final = "formamech.contact-3d-result-v1"


@dataclass(frozen=True, slots=True)
class RigidSphereIndenter3D:
    """3D rigid spherical indenter pressing into a named surface face tag."""

    name: str
    contact_face_tags: tuple[str, ...]
    center_mm: tuple[float, float, float]
    radius_mm: float
    initial_center_mm: tuple[float, float, float] | None = None
    penalty_mpa_per_mm: float = 2.0e2
    friction_mu: float = 0.0
    friction_reg_mm: float = 1.0e-3

    def at_factor(self, factor: float) -> RigidSphereIndenter3D:
        """Linearly interpolate indenter center from ``initial_center_mm`` to ``center_mm``."""
        if self.initial_center_mm is None:
            return self
        c0 = self.initial_center_mm
        c1 = self.center_mm
        return RigidSphereIndenter3D(
            name=self.name,
            contact_face_tags=self.contact_face_tags,
            center_mm=(
                c0[0] + factor * (c1[0] - c0[0]),
                c0[1] + factor * (c1[1] - c0[1]),
                c0[2] + factor * (c1[2] - c0[2]),
            ),
            radius_mm=self.radius_mm,
            initial_center_mm=None,
            penalty_mpa_per_mm=self.penalty_mpa_per_mm,
            friction_mu=self.friction_mu,
            friction_reg_mm=self.friction_reg_mm,
        )

    def __post_init__(self) -> None:
        if not self.contact_face_tags:
            raise ValueError("contact_face_tags must be non-empty")
        if not math.isfinite(self.radius_mm) or self.radius_mm <= 0.0:
            raise ValueError("radius_mm must be finite and positive")
        if not math.isfinite(self.penalty_mpa_per_mm) or self.penalty_mpa_per_mm <= 0.0:
            raise ValueError("penalty_mpa_per_mm must be positive")
        if not math.isfinite(self.friction_mu) or self.friction_mu < 0.0:
            raise ValueError("friction_mu must be nonnegative")


@dataclass(frozen=True, slots=True)
class AssemblyContact3DResult:
    """Converged (or failed) 3D multi-part assembly & contact solution."""

    converged: bool
    assembly_id: str
    nodal_displacement: FloatArray
    nodal_von_mises_mpa: FloatArray
    total_strain_energy_n_mm: float
    contact_force_n: tuple[float, float, float]
    max_penetration_mm: float
    active_contact_nodes: int
    min_det_f: float
    final_residual_norm: float
    total_newton_iterations: int
    al_iterations: int
    wall_time_s: float
    reason: str

    def document(self) -> dict[str, Any]:
        return {
            "schema": CONTACT_3D_RESULT_SCHEMA,
            "converged": self.converged,
            "assembly_id": self.assembly_id,
            "total_strain_energy_n_mm": float(self.total_strain_energy_n_mm),
            "contact_force_n": [float(v) for v in self.contact_force_n],
            "max_penetration_mm": float(self.max_penetration_mm),
            "active_contact_nodes": self.active_contact_nodes,
            "min_det_f": float(self.min_det_f),
            "final_residual_norm": float(self.final_residual_norm),
            "total_newton_iterations": self.total_newton_iterations,
            "al_iterations": self.al_iterations,
            "wall_time_s": float(self.wall_time_s),
            "reason": self.reason,
        }


def assemble_rigid_sphere_contact_3d(
    nodes_ref: FloatArray,
    u_vec: FloatArray,
    contact_nodes: Sequence[int],
    indenter: RigidSphereIndenter3D,
    lambda_n: FloatArray,
) -> tuple[FloatArray, csr_matrix, float, tuple[float, float, float], int, FloatArray]:
    """Assemble exact 3D Augmented-Lagrangian spherical contact force & consistent tangent.

    Gap function ``g(x) = ||x - c|| - R`` (penetration when ``g < 0``).
    Normal traction force on node ``i``::

        f_i = -t_n * n_i,   t_n = max(0, lambda_i - eps * g_i),   n_i = (x_i - c)/||x_i - c||

    Exact consistent tangent for active nodes (``lambda_i - eps * g_i > 0``)::

        df_i / dx_i = eps * (n_i outer n_i) - (t_n / r_i) * (I - n_i outer n_i)
    """
    n_dofs = int(u_vec.shape[0])
    f_c = np.zeros(n_dofs, dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    center = np.asarray(indenter.center_mm, dtype=np.float64)
    eps = float(indenter.penalty_mpa_per_mm)
    radius = float(indenter.radius_mm)

    max_pen = 0.0
    total_force = np.zeros(3, dtype=np.float64)
    active_count = 0
    gaps = np.zeros(len(contact_nodes), dtype=np.float64)
    eye3 = np.eye(3, dtype=np.float64)

    for k, nid in enumerate(contact_nodes):
        dofs = (3 * nid, 3 * nid + 1, 3 * nid + 2)
        u_i = u_vec[list(dofs)]
        x_i = nodes_ref[nid] + u_i
        rel = x_i - center
        r = max(float(np.linalg.norm(rel)), 1.0e-12)
        n_vec = rel / r
        gap = r - radius
        gaps[k] = gap
        if gap < 0.0:
            max_pen = max(max_pen, -gap)

        trial_tn = float(lambda_n[k]) - eps * gap
        if trial_tn > 0.0:
            active_count += 1
            # Internal-force convention: f_int + f_contact = f_ext => pushing outward along +n
            # means negative contribution to residual f_int - f_ext -> f_c = -t_n * n
            f_node = -trial_tn * n_vec
            f_c[list(dofs)] += f_node
            total_force += trial_tn * n_vec

            nn = np.outer(n_vec, n_vec)
            k_node = eps * nn - (trial_tn / r) * (eye3 - nn)

            # Optional regularized Coulomb tangential friction opposing u_t
            if indenter.friction_mu > 0.0:
                u_t = u_i - float(np.dot(u_i, n_vec)) * n_vec
                ut_norm = float(np.linalg.norm(u_t))
                reg = float(indenter.friction_reg_mm)
                denom = math.sqrt(ut_norm * ut_norm + reg * reg)
                t_lim = indenter.friction_mu * trial_tn
                f_t = t_lim * (u_t / denom)
                f_c[list(dofs)] += f_t
                proj_t = eye3 - nn
                k_t = (t_lim / denom) * (proj_t - np.outer(u_t, u_t) / (denom * denom))
                k_node = k_node + k_t

            for a in range(3):
                for b in range(3):
                    rows.append(dofs[a])
                    cols.append(dofs[b])
                    vals.append(float(k_node[a, b]))

    k_c = coo_matrix((vals, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
    return (
        f_c,
        k_c,
        max_pen,
        (float(total_force[0]), float(total_force[1]), float(total_force[2])),
        active_count,
        gaps,
    )


def third_medium_energy_density(
    deformation_gradient: torch.Tensor,
    *,
    k_third: float = 1.0e-5,
    k_reg: float = 1.0e-3,
) -> torch.Tensor:
    """Pure-PyTorch regularized Third-Medium Contact (TMC) void-medium strain energy."""
    f = deformation_gradient
    det_f = torch.linalg.det(f).clamp_min(1.0e-8)
    i1 = torch.sum(f * f, dim=(-2, -1))
    # Compressible Neo-Hookean void term + HuHu volumetric/skewness regularization
    w_nh = 0.5 * k_third * (i1 - 3.0 - 2.0 * torch.log(det_f))
    w_vol = 0.5 * k_reg * (det_f - 1.0).square() / det_f
    total_w: torch.Tensor = w_nh + w_vol
    return total_w


def _assemble_tied_interfaces(
    assembly: AssemblyMesh3D,
    u_vec: FloatArray,
) -> tuple[FloatArray, csr_matrix]:
    """Assemble penalty coupling for declared ``TIED`` master-slave interfaces."""
    mesh = assembly.mesh
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    n_dofs = int(u_vec.shape[0])
    f_tied = np.zeros(n_dofs, dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for iface in assembly.interfaces:
        if iface.kind is not InterfaceKind.TIED:
            continue
        master_nids = face_node_indices_3d(mesh, (iface.master_face_tag,))
        slave_nids = face_node_indices_3d(mesh, (iface.slave_face_tag,))
        if not master_nids or not slave_nids:
            continue
        m_pts = nodes[list(master_nids)]
        k_pen = float(iface.penalty_mpa_per_mm)
        for sid in slave_nids:
            dists = np.linalg.norm(m_pts - nodes[sid][None, :], axis=1)
            mid = master_nids[int(np.argmin(dists))]
            if mid == sid:
                continue
            for c in range(3):
                ds = 3 * sid + c
                dm = 3 * mid + c
                diff = u_vec[ds] - u_vec[dm]
                f_tied[ds] += k_pen * diff
                f_tied[dm] -= k_pen * diff
                rows.extend((ds, ds, dm, dm))
                cols.extend((ds, dm, ds, dm))
                vals.extend((k_pen, -k_pen, -k_pen, k_pen))

    k_tied = coo_matrix((vals, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
    return f_tied, k_tied


def solve_3d_assembly_and_contact(
    assembly: AssemblyMesh3D,
    *,
    dirichlet_bcs: Sequence[DirichletBC3D],
    traction_bcs: Sequence[TractionBC3D] = (),
    indenter: RigidSphereIndenter3D | None = None,
    load_steps: int = 2,
    max_al_iterations: int = 6,
    penetration_tol_mm: float = 1.0e-4,
    solver_tolerance: float = 1.0e-8,
    max_newton_iterations: int = 25,
    linear_solver: LinearSolverConfig | None = None,
) -> AssemblyContact3DResult:
    """Solve a 3D multi-material assembly with tied interfaces and/or 3D rigid contact."""
    t0 = time.perf_counter()
    mesh = assembly.mesh
    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes
    nodes_np = np.asarray(mesh.nodes, dtype=np.float64)
    helper = TorchFemBackend3D()
    cells, d_phys, w_vol, cell_vols, rows, cols = helper._precompute_geometry(mesh)
    n_elem, n_loc = cells.shape
    n_edof = 3 * n_loc
    lin_cfg = LinearSolverConfig() if linear_solver is None else linear_solver
    has_friction = indenter is not None and indenter.friction_mu > 0.0

    f_ext_full = assemble_surface_tractions_3d(mesh, traction_bcs).reshape(-1)
    comp_idx = {"x": 0, "y": 1, "z": 2}
    target_dirichlet: dict[int, float] = {}
    for bc in dirichlet_bcs:
        nids = face_node_indices_3d(mesh, bc.face_tags)
        for comp, val in zip(bc.components, bc.values_mm, strict=True):
            c = comp_idx[comp]
            for nid in nids:
                target_dirichlet[3 * nid + c] = float(val)
    constrained = np.zeros(n_dofs, dtype=bool)
    for dof in target_dirichlet:
        constrained[dof] = True
    free_dofs = np.nonzero(~constrained)[0]

    contact_nids: tuple[int, ...] = (
        face_node_indices_3d(mesh, indenter.contact_face_tags)
        if indenter is not None
        else ()
    )
    lambda_n: FloatArray = np.zeros(len(contact_nids), dtype=np.float64)

    def _assemble_multimaterial(
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
        p_np = np.zeros_like(f_np)
        c_np = np.zeros((n_elem, d_phys.shape[1], 3, 3, 3, 3), dtype=np.float64)
        w_np = np.zeros((n_elem, d_phys.shape[1]), dtype=np.float64)
        with torch.no_grad():
            for part in assembly.parts:
                c_idx = np.asarray(part.cell_indices, dtype=np.int64)
                f_part = torch.from_numpy(np.ascontiguousarray(f_np[c_idx]))
                p_np[c_idx] = first_piola(f_part, part.material).numpy()
                c_np[c_idx] = consistent_tangent(f_part, part.material).numpy()
                w_np[c_idx] = energy_density(f_part, part.material).numpy()
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

    u = np.zeros(n_dofs, dtype=np.float64)
    total_newton = 0
    al_done = 0
    last_res = float("inf")
    last_energy = 0.0
    last_min_det = 1.0
    last_p: FloatArray = np.zeros((n_elem, d_phys.shape[1], 3, 3), dtype=np.float64)
    last_f: FloatArray = np.zeros_like(last_p)
    last_cforce = (0.0, 0.0, 0.0)
    last_max_pen = 0.0
    last_active = 0

    n_al_outer = max_al_iterations if indenter is not None else 1
    for step_idx in range(1, load_steps + 1):
        factor = float(step_idx) / float(load_steps)
        f_ext = factor * f_ext_full
        for dof, val in target_dirichlet.items():
            u[dof] = factor * val
        step_indenter = None if indenter is None else indenter.at_factor(factor)

        for al_it in range(1, n_al_outer + 1):
            al_done = al_it
            step_conv = False
            gaps: FloatArray = np.zeros(len(contact_nids), dtype=np.float64)
            for _ in range(max_newton_iterations):
                total_newton += 1
                f_int, k_mat, p_np, f_np, energy, min_det = _assemble_multimaterial(u)
                if min_det <= 1.0e-8:
                    return AssemblyContact3DResult(
                        converged=False,
                        assembly_id=assembly.assembly_id,
                        nodal_displacement=np.zeros((n_nodes, 3), dtype=np.float64),
                        nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
                        total_strain_energy_n_mm=0.0,
                        contact_force_n=last_cforce,
                        max_penetration_mm=last_max_pen,
                        active_contact_nodes=last_active,
                        min_det_f=min_det,
                        final_residual_norm=float("inf"),
                        total_newton_iterations=total_newton,
                        al_iterations=al_done,
                        wall_time_s=time.perf_counter() - t0,
                        reason="non-positive det(F) during 3D assembly solve",
                    )
                if assembly.interfaces:
                    f_t, k_t = _assemble_tied_interfaces(assembly, u)
                    f_int = f_int + f_t
                    k_mat = (k_mat + k_t).tocsr()
                if step_indenter is not None:
                    f_c, k_c, max_pen, c_force, active_n, gaps = (
                        assemble_rigid_sphere_contact_3d(
                            nodes_np, u, contact_nids, step_indenter, lambda_n
                        )
                    )
                    f_int = f_int + f_c
                    k_mat = (k_mat + k_c).tocsr()
                    last_max_pen = max_pen
                    last_cforce = c_force
                    last_active = active_n

                res_free = (f_int - f_ext)[free_dofs]
                res_norm = float(np.linalg.norm(res_free))
                last_res = res_norm
                last_energy = energy
                last_min_det = min_det
                last_p = p_np
                last_f = f_np
                if res_norm <= solver_tolerance:
                    step_conv = True
                    break

                du_free, lin_rep = solve_linear_system(
                    k_mat[free_dofs][:, free_dofs],
                    -res_free,
                    config=lin_cfg,
                    is_symmetric=not has_friction,
                )
                if not lin_rep.converged:
                    return AssemblyContact3DResult(
                        converged=False,
                        assembly_id=assembly.assembly_id,
                        nodal_displacement=np.zeros((n_nodes, 3), dtype=np.float64),
                        nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
                        total_strain_energy_n_mm=0.0,
                        contact_force_n=last_cforce,
                        max_penetration_mm=last_max_pen,
                        active_contact_nodes=last_active,
                        min_det_f=last_min_det,
                        final_residual_norm=last_res,
                        total_newton_iterations=total_newton,
                        al_iterations=al_done,
                        wall_time_s=time.perf_counter() - t0,
                        reason=f"linear solve failed: {lin_rep.reason}",
                    )
                alpha = 1.0
                accepted = False
                for _ls in range(10):
                    u_cand = u.copy()
                    u_cand[free_dofs] += alpha * du_free
                    u_ec = u_cand.reshape(n_nodes, 3)[cells]
                    f_c_chk = (
                        np.einsum("eni,eqnj->eqij", u_ec, d_phys)
                        + np.eye(3, dtype=np.float64)[None, None, :, :]
                    )
                    if float(np.min(np.linalg.det(f_c_chk))) > 1.0e-5:
                        u = u_cand
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    break

            if not step_conv:
                return AssemblyContact3DResult(
                    converged=False,
                    assembly_id=assembly.assembly_id,
                    nodal_displacement=np.zeros((n_nodes, 3), dtype=np.float64),
                    nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
                    total_strain_energy_n_mm=0.0,
                    contact_force_n=last_cforce,
                    max_penetration_mm=last_max_pen,
                    active_contact_nodes=last_active,
                    min_det_f=last_min_det,
                    final_residual_norm=last_res,
                    total_newton_iterations=total_newton,
                    al_iterations=al_done,
                    wall_time_s=time.perf_counter() - t0,
                    reason=f"Newton did not converge (res={last_res:.3e})",
                )

            if indenter is None:
                break
            if last_max_pen <= penetration_tol_mm:
                break
            lambda_n = np.maximum(0.0, lambda_n - indenter.penalty_mpa_per_mm * gaps)

    u_nodal = u.reshape(n_nodes, 3)
    vm_q = _cauchy_von_mises_3d(last_p, last_f)
    vm_cell = np.sum(vm_q * w_vol, axis=1) / np.maximum(cell_vols, 1.0e-12)
    vm_nodes = _project_cell_scalar_to_nodes_3d(mesh, vm_cell, cell_vols)

    return AssemblyContact3DResult(
        converged=True,
        assembly_id=assembly.assembly_id,
        nodal_displacement=u_nodal,
        nodal_von_mises_mpa=vm_nodes,
        total_strain_energy_n_mm=last_energy,
        contact_force_n=last_cforce,
        max_penetration_mm=last_max_pen,
        active_contact_nodes=last_active,
        min_det_f=last_min_det,
        final_residual_norm=last_res,
        total_newton_iterations=total_newton,
        al_iterations=al_done,
        wall_time_s=time.perf_counter() - t0,
        reason="converged",
    )


__all__ = (
    "CONTACT_3D_RESULT_SCHEMA",
    "AssemblyContact3DResult",
    "RigidSphereIndenter3D",
    "assemble_rigid_sphere_contact_3d",
    "solve_3d_assembly_and_contact",
    "third_medium_energy_density",
)
