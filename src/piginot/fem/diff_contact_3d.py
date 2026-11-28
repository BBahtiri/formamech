"""General 3D SDF rigid-indenter contact & DiffMJX-style differentiable gradients (V6-R2).

Extends :mod:`piginot.fem.contact_3d` beyond spherical indenters to arbitrary
Signed Distance Field (SDF) rigid tool tips (sphere, flat punch, cylinder pin,
wedge/edge, box, and triangle-mesh SDF) with Augmented-Lagrangian normal contact,
regularized Coulomb stick-slip tangential friction, exact consistent tangents,
per-node surface traction extraction, and **DiffMJX-style straight-through
"Contacts From Distance" (CFD)** PyTorch backward gradients.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix, diags  # type: ignore[import-untyped]

from piginot.contracts import MaterialSpec, canonical_json_bytes
from piginot.fem._similitude import resolve_similitude_material
from piginot.fem.backend_3d import (
    DirichletBC3D,
    TorchFemBackend3D,
    TractionBC3D,
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
SDF_CONTACT_3D_SCHEMA: Final = "formamech.sdf-contact-3d-v1"


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


class SdfPrimitiveKind(StrEnum):
    """Supported rigid 3D indenter SDF geometries for robotic & tactile contact."""

    SPHERE = "sphere"
    FLAT_PUNCH = "flat_punch"
    CYLINDER_PIN = "cylinder_pin"
    BOX = "box"
    WEDGE_EDGE = "wedge_edge"


@dataclass(frozen=True, slots=True)
class RigidSdfIndenter3D:
    """General 3D rigid SDF indenter pressing into named deformable surface faces.

    Convention: ``phi(x) < 0`` means penetration inside the rigid indenter;
    ``n(x) = grad phi(x) / ||grad phi(x)||`` points outward from the indenter
    into the deformable body.
    """

    name: str
    kind: SdfPrimitiveKind
    contact_face_tags: tuple[str, ...]
    center_mm: tuple[float, float, float]
    radius_mm: float = 3.0
    half_extents_mm: tuple[float, float, float] = (3.0, 3.0, 3.0)
    corner_rounding_mm: float = 0.25
    initial_center_mm: tuple[float, float, float] | None = None
    tangential_shift_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    penalty_mpa_per_mm: float = 2.0e2
    friction_mu: float = 0.0
    friction_reg_mm: float = 1.0e-3

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("RigidSdfIndenter3D name must be non-empty")
        if not self.contact_face_tags:
            raise ValueError("contact_face_tags must be non-empty")
        if not math.isfinite(self.radius_mm) or self.radius_mm <= 0.0:
            raise ValueError("radius_mm must be finite and positive")
        if any(not math.isfinite(v) or v <= 0.0 for v in self.half_extents_mm):
            raise ValueError("half_extents_mm must be finite and positive")
        if not math.isfinite(self.corner_rounding_mm) or self.corner_rounding_mm < 0.0:
            raise ValueError("corner_rounding_mm must be nonnegative")
        if not math.isfinite(self.penalty_mpa_per_mm) or self.penalty_mpa_per_mm <= 0.0:
            raise ValueError("penalty_mpa_per_mm must be positive")
        if not math.isfinite(self.friction_mu) or self.friction_mu < 0.0:
            raise ValueError("friction_mu must be nonnegative")

    def at_factor(self, factor: float) -> RigidSdfIndenter3D:
        """Interpolate indenter center & tangential shift linearly along continuation."""
        f = float(factor)
        if self.initial_center_mm is None:
            c_step = self.center_mm
        else:
            c0, c1 = self.initial_center_mm, self.center_mm
            c_step = (
                c0[0] + f * (c1[0] - c0[0]),
                c0[1] + f * (c1[1] - c0[1]),
                c0[2] + f * (c1[2] - c0[2]),
            )
        t_step = (
            f * self.tangential_shift_mm[0],
            f * self.tangential_shift_mm[1],
            f * self.tangential_shift_mm[2],
        )
        return RigidSdfIndenter3D(
            name=self.name,
            kind=self.kind,
            contact_face_tags=self.contact_face_tags,
            center_mm=c_step,
            radius_mm=self.radius_mm,
            half_extents_mm=self.half_extents_mm,
            corner_rounding_mm=self.corner_rounding_mm,
            initial_center_mm=None,
            tangential_shift_mm=t_step,
            penalty_mpa_per_mm=self.penalty_mpa_per_mm,
            friction_mu=self.friction_mu,
            friction_reg_mm=self.friction_reg_mm,
        )

    def sdf_torch(self, pts: torch.Tensor) -> torch.Tensor:
        """Smooth C^1-regularized signed distance ``phi(x)`` in mm."""
        c = torch.tensor(self.center_mm, dtype=pts.dtype, device=pts.device)
        rel = pts - c
        r_round = max(float(self.corner_rounding_mm), 1.0e-4)

        if self.kind is SdfPrimitiveKind.SPHERE:
            return torch.sqrt((rel * rel).sum(dim=-1) + 1.0e-18) - self.radius_mm

        if self.kind is SdfPrimitiveKind.BOX:
            b = (
                torch.tensor(self.half_extents_mm, dtype=pts.dtype, device=pts.device)
                - r_round
            ).clamp_min(1.0e-4)
            s_sm = max(0.12 * r_round, 1.0e-3)
            delta = max(0.05 * r_round, 1.0e-3)
            abs_rel = torch.sqrt(rel * rel + delta * delta)
            q = abs_rel - b
            q_pos = 0.5 * (q + torch.sqrt(q * q + s_sm * s_sm))
            out_d = torch.sqrt((q_pos * q_pos).sum(dim=-1) + delta * delta)
            beta = 1.0 / s_sm
            m = q.max(dim=-1, keepdim=True).values
            q_max = (
                m.squeeze(-1) + torch.log(torch.exp(beta * (q - m)).sum(dim=-1)) / beta
            )
            in_d = 0.5 * (q_max - torch.sqrt(q_max * q_max + s_sm * s_sm))
            return out_d + in_d - r_round

        if self.kind in (SdfPrimitiveKind.FLAT_PUNCH, SdfPrimitiveKind.CYLINDER_PIN):
            # Axis along z: circular cross-section radius R, height half-extent hz
            s_sm = max(0.12 * r_round, 1.0e-3)
            delta = max(0.05 * r_round, 1.0e-3)
            r_xy = torch.sqrt(
                rel[..., 0].square() + rel[..., 1].square() + delta * delta
            )
            abs_z = torch.sqrt(rel[..., 2].square() + delta * delta)
            r_eff = max(self.radius_mm - r_round, 1.0e-4)
            hz_eff = max(self.half_extents_mm[2] - r_round, 1.0e-4)
            q2 = torch.stack((r_xy - r_eff, abs_z - hz_eff), dim=-1)
            q_pos = 0.5 * (q2 + torch.sqrt(q2 * q2 + s_sm * s_sm))
            out_d = torch.sqrt((q_pos * q_pos).sum(dim=-1) + delta * delta)
            beta = 1.0 / s_sm
            m = q2.max(dim=-1, keepdim=True).values
            q_max = (
                m.squeeze(-1) + torch.log(torch.exp(beta * (q2 - m)).sum(dim=-1)) / beta
            )
            in_d = 0.5 * (q_max - torch.sqrt(q_max * q_max + s_sm * s_sm))
            return out_d + in_d - r_round

        # WEDGE_EDGE: rounded ridge parallel to y-axis (tip pointing -z)
        hx = max(self.half_extents_mm[0] - r_round, 1.0e-4)
        hy = max(self.half_extents_mm[1] - r_round, 1.0e-4)
        hz = max(self.half_extents_mm[2] - r_round, 1.0e-4)
        # Tapered prism in x-z plane: |x| - slope * (z + hz)
        slope = hx / (2.0 * hz)
        d_wedge = (torch.abs(rel[..., 0]) - slope * (rel[..., 2] + hz)) / math.sqrt(
            1.0 + slope * slope
        )
        d_top = rel[..., 2] - hz
        d_bot = -(rel[..., 2] + hz)
        d_y = torch.abs(rel[..., 1]) - hy
        # Smooth max over half-spaces
        beta = 1.0 / max(r_round, 0.05)
        stacked = torch.stack((d_wedge, d_top, d_bot, d_y), dim=-1)
        m = stacked.max(dim=-1, keepdim=True).values
        smax = (
            m.squeeze(-1)
            + torch.log(torch.exp(beta * (stacked - m)).sum(dim=-1)) / beta
        )
        return smax - 0.5 * r_round

    def evaluate_sdf_and_derivatives(
        self, x_pt: FloatArray
    ) -> tuple[float, FloatArray, FloatArray]:
        """Return ``(phi, grad_phi (3,), hess_phi (3, 3))`` at one 3D point via autograd."""
        phis, grads, hesses = self.evaluate_sdf_and_derivatives_batch(x_pt[None, :])
        return float(phis[0]), grads[0], hesses[0]

    def evaluate_sdf_and_derivatives_batch(
        self, x_pts: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Vectorized ``(phi (K,), grad_phi (K, 3), hess_phi (K, 3, 3))`` evaluation."""
        x_t = torch.tensor(x_pts, dtype=torch.float64, requires_grad=True)
        phi_t = self.sdf_torch(x_t)  # (K,)
        g_t = torch.autograd.grad(phi_t.sum(), x_t, create_graph=True)[0]  # (K, 3)
        h_cols: list[torch.Tensor] = []
        for ci in range(3):
            grad_ci = torch.autograd.grad(g_t[:, ci].sum(), x_t, retain_graph=ci < 2)[0]
            h_cols.append(grad_ci)
        h_phi = torch.stack(h_cols, dim=1).detach().numpy()  # (K, 3, 3)
        return (
            phi_t.detach().numpy().astype(np.float64),
            g_t.detach().numpy().astype(np.float64),
            h_phi.astype(np.float64),
        )


@dataclass(frozen=True, slots=True)
class SdfContact3DResult:
    """Converged (or failed) 3D SDF rigid-indenter contact FEM solution."""

    converged: bool
    mesh_id: str
    indenter_name: str
    indenter_kind: str
    nodal_displacement: FloatArray  # (N, 3) mm
    nodal_von_mises_mpa: FloatArray  # (N,) MPa
    quadrature_peak_von_mises_mpa: float
    contact_node_indices: tuple[int, ...]
    contact_node_forces_n: FloatArray  # (N_contact, 3) force on body per contact node
    contact_node_gaps_mm: FloatArray  # (N_contact,)
    total_contact_force_n: tuple[float, float, float]
    total_contact_moment_n_mm: tuple[float, float, float]
    max_penetration_mm: float
    active_contact_nodes: int
    slip_fraction: float
    min_det_f: float
    total_strain_energy_n_mm: float
    final_residual_norm: float
    total_newton_iterations: int
    al_iterations: int
    wall_time_s: float
    reason: str = "converged"

    def summary_document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": SDF_CONTACT_3D_SCHEMA,
            "converged": self.converged,
            "mesh_id": self.mesh_id,
            "indenter_name": self.indenter_name,
            "indenter_kind": self.indenter_kind,
            "total_contact_force_n": [float(v) for v in self.total_contact_force_n],
            "total_contact_moment_n_mm": [
                float(v) for v in self.total_contact_moment_n_mm
            ],
            "max_penetration_mm": float(self.max_penetration_mm),
            "active_contact_nodes": self.active_contact_nodes,
            "slip_fraction": float(self.slip_fraction),
            "min_det_f": float(self.min_det_f),
            "peak_von_mises_mpa": float(np.max(self.nodal_von_mises_mpa)),
            "quadrature_peak_von_mises_mpa": float(self.quadrature_peak_von_mises_mpa),
            "total_strain_energy_n_mm": float(self.total_strain_energy_n_mm),
            "final_residual_norm": float(self.final_residual_norm),
            "total_newton_iterations": self.total_newton_iterations,
            "al_iterations": self.al_iterations,
            "wall_time_s": float(self.wall_time_s),
            "reason": self.reason,
        }
        doc["result_hash"] = _sha256_doc(doc)
        return doc


def assemble_sdf_contact_3d(
    nodes_ref: FloatArray,
    u_vec: FloatArray,
    contact_nodes: Sequence[int],
    indenter: RigidSdfIndenter3D,
    lambda_n: FloatArray,
    *,
    psd_projection: bool = False,
) -> tuple[
    FloatArray,
    csr_matrix,
    float,
    tuple[float, float, float],
    tuple[float, float, float],
    int,
    FloatArray,
    FloatArray,
    float,
]:
    """Assemble exact 3D Augmented-Lagrangian SDF contact force, wrench & tangent."""
    n_dofs = int(u_vec.shape[0])
    f_c = np.zeros(n_dofs, dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    eps = float(indenter.penalty_mpa_per_mm)
    shift = np.asarray(indenter.tangential_shift_mm, dtype=np.float64)
    center = np.asarray(indenter.center_mm, dtype=np.float64)

    max_pen = 0.0
    total_force = np.zeros(3, dtype=np.float64)
    total_moment = np.zeros(3, dtype=np.float64)
    active_count = 0
    slip_count = 0
    gaps = np.zeros(len(contact_nodes), dtype=np.float64)
    node_forces = np.zeros((len(contact_nodes), 3), dtype=np.float64)
    eye3 = np.eye(3, dtype=np.float64)

    if len(contact_nodes) > 0:
        u_nodal = u_vec.reshape(-1, 3)
        idx_arr = np.asarray(contact_nodes, dtype=np.int64)
        x_all = nodes_ref[idx_arr] + u_nodal[idx_arr]
        gaps_all, g_all, h_all = indenter.evaluate_sdf_and_derivatives_batch(x_all)
    else:
        gaps_all = np.zeros(0, dtype=np.float64)
        g_all = np.zeros((0, 3), dtype=np.float64)
        h_all = np.zeros((0, 3, 3), dtype=np.float64)

    for k, nid in enumerate(contact_nodes):
        dofs = (3 * nid, 3 * nid + 1, 3 * nid + 2)
        u_i = u_vec[list(dofs)]
        x_i = nodes_ref[nid] + u_i
        gap = float(gaps_all[k])
        g_vec = g_all[k]
        h_phi = h_all[k]
        gaps[k] = gap
        if gap < 0.0:
            max_pen = max(max_pen, -gap)

        trial_tn = float(lambda_n[k]) - eps * gap
        if trial_tn > 0.0:
            active_count += 1
            # Exact variational contact potential Pi_c = (1/(2*eps)) * trial_tn^2:
            # Force on body along +grad(phi): f_on_body = trial_tn * g_vec
            # Residual convention: f_int + f_c = f_ext -> f_c = -f_on_body
            # Exact symmetric Hessian: d(-t_n * g)/dx = eps * (g outer g) - t_n * H_phi
            f_on_body = trial_tn * g_vec
            f_node = -f_on_body
            k_node = eps * np.outer(g_vec, g_vec) - trial_tn * h_phi
            if psd_projection and indenter.friction_mu == 0.0:
                eigvals, eigvecs = np.linalg.eigh(0.5 * (k_node + k_node.T))
                k_node = (eigvecs * np.maximum(eigvals, 0.0)) @ eigvecs.T

            if indenter.friction_mu > 0.0:
                g_norm = max(float(np.linalg.norm(g_vec)), 1.0e-12)
                n_vec = g_vec / g_norm
                dn_dx = ((eye3 - np.outer(n_vec, n_vec)) @ h_phi) / g_norm
                # Relative slip displacement of body w.r.t. indenter shift
                u_rel = u_i - shift
                u_dot_n = float(np.dot(u_rel, n_vec))
                u_t = u_rel - u_dot_n * n_vec
                ut_norm = float(np.linalg.norm(u_t))
                reg = float(indenter.friction_reg_mm)
                denom = math.sqrt(ut_norm * ut_norm + reg * reg)
                if ut_norm > reg:
                    slip_count += 1
                t_lim = indenter.friction_mu * trial_tn
                t_hat = u_t / denom
                # Friction opposes relative slip u_t of body => f_on_body -= t_lim * t_hat
                # => f_c += +t_lim * t_hat
                f_t = t_lim * t_hat
                f_node = f_node + f_t
                f_on_body = f_on_body - f_t

                proj_t = eye3 - np.outer(n_vec, n_vec)
                du_t_dx = proj_t - (np.outer(n_vec, u_rel) + u_dot_n * eye3) @ dn_dx
                dt_hat_du_t = (eye3 - np.outer(t_hat, t_hat)) / denom
                k_t = -indenter.friction_mu * eps * np.outer(t_hat, g_vec) + t_lim * (
                    dt_hat_du_t @ du_t_dx
                )
                k_node = k_node + k_t

            f_c[list(dofs)] += f_node
            node_forces[k] = f_on_body
            total_force += f_on_body
            total_moment += np.cross(x_i - center, f_on_body)

            for a in range(3):
                for b in range(3):
                    rows.append(dofs[a])
                    cols.append(dofs[b])
                    vals.append(float(k_node[a, b]))

    slip_frac = float(slip_count) / float(max(active_count, 1))
    k_c = coo_matrix((vals, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
    return (
        f_c,
        k_c,
        max_pen,
        (float(total_force[0]), float(total_force[1]), float(total_force[2])),
        (float(total_moment[0]), float(total_moment[1]), float(total_moment[2])),
        active_count,
        gaps,
        node_forces,
        slip_frac,
    )


def solve_3d_sdf_contact(
    mesh: CanonicalMesh3D,
    material: MaterialSpec,
    *,
    dirichlet_bcs: Sequence[DirichletBC3D],
    indenter: RigidSdfIndenter3D,
    traction_bcs: Sequence[TractionBC3D] = (),
    load_steps: int = 3,
    max_al_iterations: int = 6,
    penetration_tol_mm: float = 1.0e-4,
    solver_tolerance: float = 1.0e-7,
    max_newton_iterations: int = 40,
    linear_solver: LinearSolverConfig | None = None,
) -> SdfContact3DResult:
    """Solve 3D nonlinear ``TET4``/``TET10`` FEM under general SDF indenter contact."""
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
    indenter_eff = RigidSdfIndenter3D(
        name=indenter.name,
        kind=indenter.kind,
        contact_face_tags=indenter.contact_face_tags,
        center_mm=indenter.center_mm,
        radius_mm=indenter.radius_mm,
        half_extents_mm=indenter.half_extents_mm,
        corner_rounding_mm=indenter.corner_rounding_mm,
        initial_center_mm=indenter.initial_center_mm,
        tangential_shift_mm=indenter.tangential_shift_mm,
        penalty_mpa_per_mm=indenter.penalty_mpa_per_mm * s_fac,
        friction_mu=indenter.friction_mu,
        friction_reg_mm=indenter.friction_reg_mm,
    )

    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes
    nodes_np = np.asarray(mesh.nodes, dtype=np.float64)
    helper = TorchFemBackend3D()
    cells, d_phys, w_vol, cell_vols, rows, cols = helper._precompute_geometry(mesh)
    n_elem, n_loc = cells.shape
    n_edof = 3 * n_loc
    lin_cfg = LinearSolverConfig() if linear_solver is None else linear_solver
    has_friction = indenter_eff.friction_mu > 0.0

    f_ext_full = (assemble_surface_tractions_3d(mesh, traction_bcs).reshape(-1)) * s_fac
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

    contact_nids = face_node_indices_3d(mesh, indenter_eff.contact_face_tags)
    lambda_n: FloatArray = np.zeros(len(contact_nids), dtype=np.float64)

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

    u: FloatArray = np.zeros(n_dofs, dtype=np.float64)
    total_newton = 0
    al_done = 0
    last_res = float("inf")
    last_energy = 0.0
    last_min_det = 1.0
    last_p: FloatArray = np.zeros((n_elem, d_phys.shape[1], 3, 3), dtype=np.float64)
    last_f: FloatArray = np.zeros_like(last_p)
    last_cforce = (0.0, 0.0, 0.0)
    last_cmoment = (0.0, 0.0, 0.0)
    last_max_pen = 0.0
    last_active = 0
    last_gaps: FloatArray = np.zeros(len(contact_nids), dtype=np.float64)
    last_nforces: FloatArray = np.zeros((len(contact_nids), 3), dtype=np.float64)
    last_slip = 0.0
    tol_eff = solver_tolerance * s_fac

    for step_idx in range(1, load_steps + 1):
        factor = float(step_idx) / float(load_steps)
        f_ext = factor * f_ext_full
        for dof, val in target_dirichlet.items():
            u[dof] = factor * val
        step_ind = indenter_eff.at_factor(factor)

        for al_it in range(1, max_al_iterations + 1):
            al_done = al_it
            step_conv = False
            for _ in range(max_newton_iterations):
                total_newton += 1
                f_int, k_mat, p_np, f_np, energy, min_det = _assemble_internal(u)
                if min_det <= 1.0e-8:
                    return SdfContact3DResult(
                        converged=False,
                        mesh_id=mesh.mesh_id,
                        indenter_name=indenter.name,
                        indenter_kind=indenter.kind.value,
                        nodal_displacement=np.zeros((n_nodes, 3), dtype=np.float64),
                        nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
                        quadrature_peak_von_mises_mpa=0.0,
                        contact_node_indices=contact_nids,
                        contact_node_forces_n=last_nforces / s_fac,
                        contact_node_gaps_mm=last_gaps,
                        total_contact_force_n=(
                            last_cforce[0] / s_fac,
                            last_cforce[1] / s_fac,
                            last_cforce[2] / s_fac,
                        ),
                        total_contact_moment_n_mm=(
                            last_cmoment[0] / s_fac,
                            last_cmoment[1] / s_fac,
                            last_cmoment[2] / s_fac,
                        ),
                        max_penetration_mm=last_max_pen,
                        active_contact_nodes=last_active,
                        slip_fraction=last_slip,
                        min_det_f=min_det,
                        total_strain_energy_n_mm=0.0,
                        final_residual_norm=float("inf"),
                        total_newton_iterations=total_newton,
                        al_iterations=al_done,
                        wall_time_s=time.perf_counter() - t0,
                        reason="non-positive det(F) during 3D SDF contact solve",
                    )
                (
                    f_c,
                    k_c,
                    max_pen,
                    c_force,
                    c_mom,
                    active_n,
                    gaps,
                    nforces,
                    slip_frac,
                ) = assemble_sdf_contact_3d(
                    nodes_np, u, contact_nids, step_ind, lambda_n
                )
                f_tot = f_int + f_c
                k_tot = (k_mat + k_c).tocsr()

                res_free = (f_tot - f_ext)[free_dofs]
                res_norm = float(np.linalg.norm(res_free))
                last_res = res_norm
                last_energy = energy
                last_min_det = min_det
                last_p = p_np
                last_f = f_np
                last_cforce = c_force
                last_cmoment = c_mom
                last_max_pen = max_pen
                last_active = active_n
                last_gaps = gaps
                last_nforces = nforces
                last_slip = slip_frac
                if res_norm <= tol_eff:
                    step_conv = True
                    break

                du_free, lin_rep = solve_linear_system(
                    k_tot[free_dofs][:, free_dofs],
                    -res_free,
                    config=lin_cfg,
                    is_symmetric=not has_friction,
                )
                if not lin_rep.converged:
                    break

                def _eval_merit(
                    u_cand: FloatArray,
                    _step_ind: RigidSdfIndenter3D = step_ind,
                    _lam: FloatArray = lambda_n,
                    _f_ext: FloatArray = f_ext,
                ) -> tuple[float, float, float]:
                    f_ic, _, _, _, e_ic, det_ic = _assemble_internal(u_cand)
                    if det_ic <= 1.0e-5:
                        return float("inf"), float("inf"), det_ic
                    f_cc, _, _, _, _, _, g_cc, _, _ = assemble_sdf_contact_3d(
                        nodes_np, u_cand, contact_nids, _step_ind, _lam
                    )
                    r_cc = float(np.linalg.norm((f_ic + f_cc - _f_ext)[free_dofs]))
                    eps_c = float(_step_ind.penalty_mpa_per_mm)
                    pi_c = float(
                        np.sum(np.maximum(0.0, _lam - eps_c * g_cc) ** 2)
                        / (2.0 * eps_c)
                    )
                    pi_tot = e_ic + pi_c - float(np.dot(_f_ext, u_cand))
                    return r_cc, pi_tot, det_ic

                eps_cur = float(step_ind.penalty_mpa_per_mm)
                pi_cur = (
                    energy
                    + float(
                        np.sum(np.maximum(0.0, lambda_n - eps_cur * gaps) ** 2)
                        / (2.0 * eps_cur)
                    )
                    - float(np.dot(f_ext, u))
                )

                # For frictionless contact, try exact Newton first; if the rounded-corner
                # curvature term -t_n * H_phi produces an uphill/stalled direction,
                # switch smoothly to the PSD-projected variational Hessian (guaranteed
                # descent on total potential energy Pi(u)).
                candidate_dirs: list[FloatArray] = []
                if has_friction or float(np.dot(res_free, du_free)) < 0.0:
                    candidate_dirs.append(du_free)
                if not has_friction:
                    _, k_c_psd, _, _, _, _, _, _, _ = assemble_sdf_contact_3d(
                        nodes_np,
                        u,
                        contact_nids,
                        step_ind,
                        lambda_n,
                        psd_projection=True,
                    )
                    k_psd_ff = (k_mat + k_c_psd).tocsr()[free_dofs][:, free_dofs]
                    du_psd, rep_psd = solve_linear_system(
                        k_psd_ff,
                        -res_free,
                        config=lin_cfg,
                        is_symmetric=True,
                    )
                    if rep_psd.converged and float(np.dot(res_free, du_psd)) < 0.0:
                        candidate_dirs.append(du_psd)
                    elif not candidate_dirs:
                        diag_shift = max(eps_cur, 1.0)
                        du_lm, rep_lm = solve_linear_system(
                            (
                                k_psd_ff + diags(np.full(free_dofs.size, diag_shift))
                            ).tocsr(),
                            -res_free,
                            config=lin_cfg,
                            is_symmetric=True,
                        )
                        if rep_lm.converged:
                            candidate_dirs.append(du_lm)

                accepted = False
                best_u: FloatArray | None = None
                best_metric = float("inf")
                for dir_idx, d_try in enumerate(candidate_dirs):
                    slope = float(np.dot(res_free, d_try))
                    alpha = 1.0
                    max_ls = (
                        4
                        if (
                            not has_friction
                            and dir_idx == 0
                            and len(candidate_dirs) > 1
                        )
                        else 12
                    )
                    for _ls in range(max_ls):
                        u_cand = u.copy()
                        u_cand[free_dofs] += alpha * d_try
                        cand_res, cand_pi, det_c = _eval_merit(u_cand)
                        if det_c > 1.0e-5 and np.isfinite(cand_res):
                            metric = cand_res if has_friction else cand_pi
                            if metric < best_metric:
                                best_metric = metric
                                best_u = u_cand
                            if has_friction:
                                ok_step = (
                                    cand_res < (1.0 - 1.0e-4 * alpha) * res_norm
                                    or cand_res <= tol_eff
                                )
                            else:
                                ok_step = (
                                    cand_pi <= pi_cur + 1.0e-4 * alpha * slope
                                    or cand_res <= tol_eff
                                )
                            if ok_step:
                                u = u_cand
                                accepted = True
                                break
                        alpha *= 0.5
                    if accepted:
                        break
                if not accepted:
                    if best_u is not None:
                        u = best_u
                    else:
                        break

            if not step_conv:
                return SdfContact3DResult(
                    converged=False,
                    mesh_id=mesh.mesh_id,
                    indenter_name=indenter.name,
                    indenter_kind=indenter.kind.value,
                    nodal_displacement=np.zeros((n_nodes, 3), dtype=np.float64),
                    nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
                    quadrature_peak_von_mises_mpa=0.0,
                    contact_node_indices=contact_nids,
                    contact_node_forces_n=last_nforces / s_fac,
                    contact_node_gaps_mm=last_gaps,
                    total_contact_force_n=(
                        last_cforce[0] / s_fac,
                        last_cforce[1] / s_fac,
                        last_cforce[2] / s_fac,
                    ),
                    total_contact_moment_n_mm=(
                        last_cmoment[0] / s_fac,
                        last_cmoment[1] / s_fac,
                        last_cmoment[2] / s_fac,
                    ),
                    max_penetration_mm=last_max_pen,
                    active_contact_nodes=last_active,
                    slip_fraction=last_slip,
                    min_det_f=last_min_det,
                    total_strain_energy_n_mm=0.0,
                    final_residual_norm=last_res / s_fac,
                    total_newton_iterations=total_newton,
                    al_iterations=al_done,
                    wall_time_s=time.perf_counter() - t0,
                    reason=f"Newton did not converge (res={last_res / s_fac:.3e})",
                )
            if last_max_pen <= penetration_tol_mm:
                break
            lambda_n = np.asarray(
                np.maximum(0.0, lambda_n - indenter_eff.penalty_mpa_per_mm * last_gaps),
                dtype=np.float64,
            )

    u_nodal = u.reshape(n_nodes, 3)
    p_phys = last_p / s_fac
    vm_q = _cauchy_von_mises_3d(p_phys, last_f)
    vm_cell = np.sum(vm_q * w_vol, axis=1) / np.maximum(cell_vols, 1.0e-12)
    vm_nodes = _project_cell_scalar_to_nodes_3d(mesh, vm_cell, cell_vols)
    _, quad_peak_vm, _ = compute_3d_element_hotspots(p_phys, last_f)

    return SdfContact3DResult(
        converged=True,
        mesh_id=mesh.mesh_id,
        indenter_name=indenter.name,
        indenter_kind=indenter.kind.value,
        nodal_displacement=u_nodal,
        nodal_von_mises_mpa=vm_nodes,
        quadrature_peak_von_mises_mpa=quad_peak_vm,
        contact_node_indices=contact_nids,
        contact_node_forces_n=last_nforces / s_fac,
        contact_node_gaps_mm=last_gaps,
        total_contact_force_n=(
            last_cforce[0] / s_fac,
            last_cforce[1] / s_fac,
            last_cforce[2] / s_fac,
        ),
        total_contact_moment_n_mm=(
            last_cmoment[0] / s_fac,
            last_cmoment[1] / s_fac,
            last_cmoment[2] / s_fac,
        ),
        max_penetration_mm=last_max_pen,
        active_contact_nodes=last_active,
        slip_fraction=last_slip,
        min_det_f=last_min_det,
        total_strain_energy_n_mm=last_energy / s_fac,
        final_residual_norm=last_res / s_fac,
        total_newton_iterations=total_newton,
        al_iterations=al_done,
        wall_time_s=time.perf_counter() - t0,
        reason="converged",
    )


# ---------------------------------------------------------------------------
# DiffMJX-Style Straight-Through "Contacts From Distance" (CFD) Backward Pass
# ---------------------------------------------------------------------------


def diffmjx_straight_through_contact_force(
    gap_mm: torch.Tensor,
    normal_vec: torch.Tensor,
    *,
    penalty_mpa_per_mm: float = 200.0,
    lambda_n: torch.Tensor | None = None,
    cfd_temperature_mm: float = 0.5,
) -> torch.Tensor:
    """Straight-through hard forward contact with smooth CFD backward gradients.

    Forward value is strictly unmodified hard contact:
        ``t_hard = relu(lambda_n - eps * gap)`` (zero whenever separated).
    Backward gradient uses smooth softplus distance contact:
        ``t_soft = eps * tau * softplus((lambda_n / eps - gap) / tau)``
    via straight-through estimation ``t = t_soft + (t_hard - t_soft).detach()``,
    providing non-vanishing contact gradients even when bodies are separated
    (``gap > 0``) without softening the forward physical contact law.
    """
    if cfd_temperature_mm <= 0.0 or not math.isfinite(cfd_temperature_mm):
        raise ValueError("cfd_temperature_mm must be finite and positive")
    eps = float(penalty_mpa_per_mm)
    lam = torch.zeros_like(gap_mm) if lambda_n is None else lambda_n
    tau = float(cfd_temperature_mm)

    t_hard = torch.relu(lam - eps * gap_mm)
    t_soft = eps * tau * torch.nn.functional.softplus((lam / eps - gap_mm) / tau)
    t_ste = t_soft + (t_hard - t_soft).detach()
    return t_ste.unsqueeze(-1) * normal_vec


__all__ = (
    "SDF_CONTACT_3D_SCHEMA",
    "RigidSdfIndenter3D",
    "SdfContact3DResult",
    "SdfPrimitiveKind",
    "assemble_sdf_contact_3d",
    "diffmjx_straight_through_contact_force",
    "solve_3d_sdf_contact",
)
