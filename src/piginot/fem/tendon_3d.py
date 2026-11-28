"""3D tendon-routed continuum FEM & finite-rotation Cosserat backbone solver (V6-R1).

Provides two complementary pure-MIT 3D tendon-actuation solvers:

1. **3D Continuum Solid Tendon Routing** (:func:`assemble_tendon_forces_3d`,
   :func:`solve_3d_tendon_solid`): routes cables through ordered 3D eyelet nodes
   on a ``TET4``/``TET10`` finite-strain mesh with exact geometric cable stiffness
   ``K_L = (T / L) * (I - e e^T)`` and optional Capstan friction attenuation
   ``T_{j} = T_{j-1} * exp(-mu * theta)``.
2. **Exact Finite-Rotation 3D Cosserat Rod Tendon Backbone**
   (:func:`solve_3d_tendon_continuum_backbone`): variational ``SE(3)`` Cosserat
   beam + spacer-disk routing solver for high-aspect-ratio continuum robots
   (e.g., Nitinol tubular backbones with 9 disks and 4 tendons), outputting
   exact 6-DOF disk poses, distributed backbone curvature/FBG microstrain,
   tendon strokes, and base 6-axis reaction wrenches.
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
    _cauchy_von_mises_3d,
    _project_cell_scalar_to_nodes_3d,
    assemble_surface_tractions_3d,
    face_node_indices_3d,
)
from piginot.fem.linear_solvers import LinearSolverConfig, solve_linear_system
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D
from piginot.physics import consistent_tangent, energy_density, first_piola

FloatArray = NDArray[np.float64]
TENDON_3D_RESULT_SCHEMA: Final = "formamech.tendon-3d-result-v1"
COSSERAT_TENDON_RESULT_SCHEMA: Final = "formamech.cosserat-tendon-3d-v1"


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


@dataclass(frozen=True, slots=True)
class TendonRoute3D:
    """Routed cable/tendon passing through ordered 3D mesh eyelet nodes."""

    name: str
    eyelet_nodes: tuple[int, ...]
    tension_n: float
    capstan_mu: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("TendonRoute3D name must be non-empty text")
        if len(self.eyelet_nodes) < 2:
            raise ValueError("TendonRoute3D requires at least 2 eyelet nodes")
        if not math.isfinite(self.tension_n) or self.tension_n < 0.0:
            raise ValueError("tension_n must be finite and nonnegative")
        if not math.isfinite(self.capstan_mu) or self.capstan_mu < 0.0:
            raise ValueError("capstan_mu must be finite and nonnegative")

    def at_factor(self, factor: float) -> TendonRoute3D:
        return TendonRoute3D(
            name=self.name,
            eyelet_nodes=self.eyelet_nodes,
            tension_n=self.tension_n * float(factor),
            capstan_mu=self.capstan_mu,
        )


@dataclass(frozen=True, slots=True)
class TendonForceAssembly3D:
    """Assembled 3D tendon nodal force vector and geometric tangent stiffness."""

    external_force: FloatArray  # (3*N,) N
    load_stiffness: csr_matrix  # (3*N, 3*N) N/mm, K_L = -d(f_ext)/du
    tendon_lengths_mm: Mapping[str, float]
    tip_tensions_n: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class TendonSolid3DResult:
    """Converged (or failed) 3D solid continuum tendon-actuated FEM solution."""

    request_id: str
    mesh_id: str
    converged: bool
    nodal_displacement: FloatArray  # (N, 3) mm
    nodal_von_mises_mpa: FloatArray  # (N,) MPa
    tendon_lengths_mm: Mapping[str, float]
    tendon_strokes_mm: Mapping[str, float]
    tip_tensions_n: Mapping[str, float]
    total_strain_energy_n_mm: float
    min_det_f: float
    max_displacement_mm: float
    peak_von_mises_mpa: float
    final_residual_norm: float
    total_newton_iterations: int
    wall_time_s: float
    reason: str = "converged"

    def summary_document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": TENDON_3D_RESULT_SCHEMA,
            "request_id": self.request_id,
            "mesh_id": self.mesh_id,
            "converged": self.converged,
            "tendon_lengths_mm": {
                k: float(v) for k, v in sorted(self.tendon_lengths_mm.items())
            },
            "tendon_strokes_mm": {
                k: float(v) for k, v in sorted(self.tendon_strokes_mm.items())
            },
            "tip_tensions_n": {
                k: float(v) for k, v in sorted(self.tip_tensions_n.items())
            },
            "total_strain_energy_n_mm": float(self.total_strain_energy_n_mm),
            "min_det_f": float(self.min_det_f),
            "max_displacement_mm": float(self.max_displacement_mm),
            "peak_von_mises_mpa": float(self.peak_von_mises_mpa),
            "final_residual_norm": float(self.final_residual_norm),
            "total_newton_iterations": self.total_newton_iterations,
            "wall_time_s": float(self.wall_time_s),
            "reason": self.reason,
        }
        doc["result_hash"] = _sha256_doc(doc)
        return doc


def assemble_tendon_forces_3d(
    nodes_ref: FloatArray,
    u_vec: FloatArray,
    tendons: Sequence[TendonRoute3D],
) -> TendonForceAssembly3D:
    """Assemble exact 3D cable forces & geometric stiffness along tendon routes."""
    n_nodes = int(nodes_ref.shape[0])
    n_dofs = 3 * n_nodes
    x_def = nodes_ref + u_vec.reshape(n_nodes, 3)

    f_ext = np.zeros(n_dofs, dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lengths: dict[str, float] = {}
    tip_tensions: dict[str, float] = {}
    eye3 = np.eye(3, dtype=np.float64)

    for td in tendons:
        t_cur = float(td.tension_n)
        total_len = 0.0
        prev_dir: FloatArray | None = None
        for idx in range(len(td.eyelet_nodes) - 1):
            na = int(td.eyelet_nodes[idx])
            nb = int(td.eyelet_nodes[idx + 1])
            seg = x_def[nb] - x_def[na]
            seg_len = max(float(np.linalg.norm(seg)), 1.0e-12)
            e_dir = seg / seg_len
            total_len += seg_len

            if td.capstan_mu > 0.0 and prev_dir is not None:
                cos_th = max(-1.0, min(1.0, float(np.dot(prev_dir, e_dir))))
                theta = math.acos(cos_th)
                t_cur = t_cur * math.exp(-td.capstan_mu * theta)
            prev_dir = e_dir

            # Segment tension pulls node a toward b (+T e) and node b toward a (-T e)
            f_seg = t_cur * e_dir
            f_ext[3 * na : 3 * na + 3] += f_seg
            f_ext[3 * nb : 3 * nb + 3] -= f_seg

            # K_L = -d(f_ext)/du: diagonal +(T/L)(I - e e^T), off-diag -(T/L)(I - e e^T)
            k_blk = (t_cur / seg_len) * (eye3 - np.outer(e_dir, e_dir))
            for ci in range(3):
                for cj in range(3):
                    kv = float(k_blk[ci, cj])
                    if abs(kv) <= 1.0e-18:
                        continue
                    ra, rb = 3 * na + ci, 3 * nb + ci
                    ca, cb = 3 * na + cj, 3 * nb + cj
                    rows.extend((ra, ra, rb, rb))
                    cols.extend((ca, cb, ca, cb))
                    vals.extend((kv, -kv, -kv, kv))

        lengths[td.name] = float(total_len)
        tip_tensions[td.name] = float(t_cur)

    k_load = coo_matrix((vals, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
    return TendonForceAssembly3D(
        external_force=f_ext,
        load_stiffness=k_load,
        tendon_lengths_mm=lengths,
        tip_tensions_n=tip_tensions,
    )


def solve_3d_tendon_solid(
    mesh: CanonicalMesh3D,
    material: MaterialSpec,
    *,
    dirichlet_bcs: Sequence[DirichletBC3D],
    tendons: Sequence[TendonRoute3D],
    traction_bcs: Sequence[TractionBC3D] = (),
    point_loads_n: Mapping[int, tuple[float, float, float]] | None = None,
    request_id: str = "tendon-solid-3d",
    load_steps: int = 4,
    solver_tolerance: float = 1.0e-8,
    max_newton_iterations: int = 25,
    linear_solver: LinearSolverConfig | None = None,
) -> TendonSolid3DResult:
    """Solve 3D nonlinear continuum ``TET4``/``TET10`` FEM under routed tendon forces."""
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
    tendons_eff = tuple(td.at_factor(s_fac) for td in tendons)

    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes
    nodes_ref = np.asarray(mesh.nodes, dtype=np.float64)
    helper = TorchFemBackend3D()
    cells, d_phys, w_vol, cell_vols, rows, cols = helper._precompute_geometry(mesh)
    n_elem, n_loc = cells.shape
    n_edof = 3 * n_loc
    lin_cfg = LinearSolverConfig() if linear_solver is None else linear_solver
    has_capstan = any(td.capstan_mu > 0.0 for td in tendons_eff)

    ref_lengths = assemble_tendon_forces_3d(
        nodes_ref, np.zeros(n_dofs, dtype=np.float64), tendons_eff
    ).tendon_lengths_mm

    f_dead_full = assemble_surface_tractions_3d(mesh, traction_bcs).reshape(-1) * s_fac
    if point_loads_n:
        for nid, fvec in point_loads_n.items():
            f_dead_full[3 * int(nid) : 3 * int(nid) + 3] += (
                np.asarray(fvec, dtype=np.float64) * s_fac
            )

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
    last_res = float("inf")
    last_energy = 0.0
    last_min_det = 1.0
    last_p: FloatArray = np.zeros((n_elem, d_phys.shape[1], 3, 3), dtype=np.float64)
    last_f: FloatArray = np.zeros_like(last_p)
    last_lengths: Mapping[str, float] = dict(ref_lengths)
    last_tip_tensions: Mapping[str, float] = {}

    tol_eff = solver_tolerance * s_fac
    for step_idx in range(1, load_steps + 1):
        factor = float(step_idx) / float(load_steps)
        step_tendons = tuple(td.at_factor(factor) for td in tendons_eff)
        f_dead = factor * f_dead_full
        for dof, val in target_dirichlet.items():
            u[dof] = factor * val

        step_conv = False
        for _ in range(max_newton_iterations):
            total_newton += 1
            f_int, k_int, p_np, f_np, energy, min_det = _assemble_internal(u)
            if min_det <= 1.0e-8:
                return TendonSolid3DResult(
                    request_id=request_id,
                    mesh_id=mesh.mesh_id,
                    converged=False,
                    nodal_displacement=np.zeros((n_nodes, 3), dtype=np.float64),
                    nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
                    tendon_lengths_mm=last_lengths,
                    tendon_strokes_mm={},
                    tip_tensions_n={k: v / s_fac for k, v in last_tip_tensions.items()},
                    total_strain_energy_n_mm=0.0,
                    min_det_f=min_det,
                    max_displacement_mm=0.0,
                    peak_von_mises_mpa=0.0,
                    final_residual_norm=float("inf"),
                    total_newton_iterations=total_newton,
                    wall_time_s=time.perf_counter() - t0,
                    reason="non-positive det(F) during 3D tendon solve",
                )
            t_asm = assemble_tendon_forces_3d(nodes_ref, u, step_tendons)
            f_ext = f_dead + t_asm.external_force
            k_tot = (k_int + t_asm.load_stiffness).tocsr()

            res_free = (f_int - f_ext)[free_dofs]
            res_norm = float(np.linalg.norm(res_free))
            last_res = res_norm
            last_energy = energy
            last_min_det = min_det
            last_p = p_np
            last_f = f_np
            last_lengths = t_asm.tendon_lengths_mm
            last_tip_tensions = t_asm.tip_tensions_n
            if res_norm <= tol_eff:
                step_conv = True
                break

            du_free, lin_rep = solve_linear_system(
                k_tot[free_dofs][:, free_dofs],
                -res_free,
                config=lin_cfg,
                is_symmetric=not has_capstan,
            )
            if not lin_rep.converged:
                break

            alpha = 1.0
            accepted = False
            best_u: FloatArray | None = None
            best_res = float("inf")
            for _ls in range(10):
                u_cand = u.copy()
                u_cand[free_dofs] += alpha * du_free
                f_int_c, _, _, _, _, det_c = _assemble_internal(u_cand)
                if det_c > 1.0e-5:
                    t_asm_c = assemble_tendon_forces_3d(nodes_ref, u_cand, step_tendons)
                    cand_res = float(
                        np.linalg.norm(
                            (f_int_c - (f_dead + t_asm_c.external_force))[free_dofs]
                        )
                    )
                    if np.isfinite(cand_res):
                        if cand_res < best_res:
                            best_res = cand_res
                            best_u = u_cand
                        if (
                            cand_res < (1.0 - 1.0e-4 * alpha) * res_norm
                            or cand_res <= tol_eff
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
            return TendonSolid3DResult(
                request_id=request_id,
                mesh_id=mesh.mesh_id,
                converged=False,
                nodal_displacement=np.zeros((n_nodes, 3), dtype=np.float64),
                nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
                tendon_lengths_mm=last_lengths,
                tendon_strokes_mm={},
                tip_tensions_n={k: v / s_fac for k, v in last_tip_tensions.items()},
                total_strain_energy_n_mm=0.0,
                min_det_f=last_min_det,
                max_displacement_mm=0.0,
                peak_von_mises_mpa=0.0,
                final_residual_norm=last_res / s_fac,
                total_newton_iterations=total_newton,
                wall_time_s=time.perf_counter() - t0,
                reason=f"Newton did not converge (res={last_res / s_fac:.3e})",
            )

    u_nodal = u.reshape(n_nodes, 3)
    p_phys = last_p / s_fac
    vm_q = _cauchy_von_mises_3d(p_phys, last_f)
    vm_cell = np.sum(vm_q * w_vol, axis=1) / np.maximum(cell_vols, 1.0e-12)
    vm_nodes = _project_cell_scalar_to_nodes_3d(mesh, vm_cell, cell_vols)
    strokes = {k: float(ref_lengths[k] - last_lengths[k]) for k in sorted(ref_lengths)}

    return TendonSolid3DResult(
        request_id=request_id,
        mesh_id=mesh.mesh_id,
        converged=True,
        nodal_displacement=u_nodal,
        nodal_von_mises_mpa=vm_nodes,
        tendon_lengths_mm=last_lengths,
        tendon_strokes_mm=strokes,
        tip_tensions_n={k: v / s_fac for k, v in last_tip_tensions.items()},
        total_strain_energy_n_mm=last_energy / s_fac,
        min_det_f=last_min_det,
        max_displacement_mm=float(np.max(np.linalg.norm(u_nodal, axis=1))),
        peak_von_mises_mpa=float(np.max(vm_nodes)),
        final_residual_norm=last_res / s_fac,
        total_newton_iterations=total_newton,
        wall_time_s=time.perf_counter() - t0,
        reason="converged",
    )


# ---------------------------------------------------------------------------
# Exact Finite-Rotation 3D Cosserat Rod Tendon-Driven Continuum Backbone
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CosseratBackboneSpec:
    """Physical geometry & material specification of a disk-routed continuum robot."""

    length_mm: float
    num_disks: int
    youngs_modulus_mpa: float
    poisson_ratio: float
    outer_radius_mm: float
    inner_radius_mm: float = 0.0
    tendon_offsets_mm: tuple[tuple[float, float], ...] = (
        (37.5, 0.0),
        (0.0, 37.5),
        (-37.5, 0.0),
        (0.0, -37.5),
    )
    linear_mass_kg_per_m: float = 0.0
    disk_mass_kg: float = 0.0
    gravity_m_per_s2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fiber_core_radius_mm: float = 0.05

    def __post_init__(self) -> None:
        if self.length_mm <= 0.0 or not math.isfinite(self.length_mm):
            raise ValueError("length_mm must be finite and positive")
        if type(self.num_disks) is not int or self.num_disks < 2:
            raise ValueError("num_disks must be an integer >= 2")
        if self.outer_radius_mm <= self.inner_radius_mm or self.inner_radius_mm < 0.0:
            raise ValueError("require 0 <= inner_radius_mm < outer_radius_mm")
        if self.youngs_modulus_mpa <= 0.0 or not (0.0 <= self.poisson_ratio < 0.5):
            raise ValueError("invalid elastic constants")

    @property
    def area_mm2(self) -> float:
        return math.pi * (self.outer_radius_mm**2 - self.inner_radius_mm**2)

    @property
    def i_bend_mm4(self) -> float:
        return 0.25 * math.pi * (self.outer_radius_mm**4 - self.inner_radius_mm**4)

    @property
    def j_torsion_mm4(self) -> float:
        return 2.0 * self.i_bend_mm4

    @property
    def shear_modulus_mpa(self) -> float:
        return self.youngs_modulus_mpa / (2.0 * (1.0 + self.poisson_ratio))


@dataclass(frozen=True, slots=True)
class CosseratTendonResult:
    """Converged exact finite-rotation ``SE(3)`` Cosserat tendon backbone state."""

    converged: bool
    disk_positions_mm: FloatArray  # (N_disks+1, 3) including base at index 0
    disk_rotations: FloatArray  # (N_disks+1, 3, 3) SO(3) frames
    segment_curvatures_per_mm: FloatArray  # (N_disks, 3) (kx, ky, kz)
    fbg_microstrain: FloatArray  # (N_disks, 2) bending microstrain along (x, y)
    tendon_strokes_mm: FloatArray  # (M_tendons,) shortening L0 - L
    base_force_n: tuple[float, float, float]
    base_moment_n_mm: tuple[float, float, float]
    total_energy_n_mm: float
    grad_norm: float
    iterations: int
    wall_time_s: float

    def summary_document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": COSSERAT_TENDON_RESULT_SCHEMA,
            "converged": self.converged,
            "tip_position_mm": [float(v) for v in self.disk_positions_mm[-1]],
            "tendon_strokes_mm": [float(v) for v in self.tendon_strokes_mm],
            "base_force_n": [float(v) for v in self.base_force_n],
            "base_moment_n_mm": [float(v) for v in self.base_moment_n_mm],
            "max_fbg_microstrain": float(np.max(np.abs(self.fbg_microstrain))),
            "total_energy_n_mm": float(self.total_energy_n_mm),
            "grad_norm": float(self.grad_norm),
            "iterations": self.iterations,
            "wall_time_s": float(self.wall_time_s),
        }
        doc["result_hash"] = _sha256_doc(doc)
        return doc


def _so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Exact Rodrigues exponential map ``so(3) -> SO(3)`` with Taylor safety near 0."""
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq + 1.0e-24)
    small = theta < 1.0e-4
    a = torch.where(
        small,
        1.0 - theta_sq / 6.0 + (theta_sq * theta_sq) / 120.0,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        small,
        0.5 - theta_sq / 24.0 + (theta_sq * theta_sq) / 720.0,
        (1.0 - torch.cos(theta)) / theta_sq.clamp_min(1.0e-24),
    )
    wx, wy, wz = omega[..., 0], omega[..., 1], omega[..., 2]
    z = torch.zeros_like(wx)
    k_mat = torch.stack(
        (
            torch.stack((z, -wz, wy), dim=-1),
            torch.stack((wz, z, -wx), dim=-1),
            torch.stack((-wy, wx, z), dim=-1),
        ),
        dim=-2,
    )
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(k_mat)
    return eye + a.unsqueeze(-1) * k_mat + b.unsqueeze(-1) * torch.matmul(k_mat, k_mat)


def _forward_cosserat_kinematics(
    spec: CosseratBackboneSpec,
    curvatures: torch.Tensor,  # (N_seg, 3) 1/mm
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate ``SE(3)`` midpoint Cosserat rod kinematics along +z backbone."""
    n_seg = spec.num_disks
    ds = spec.length_mm / float(n_seg)
    dtype, device = curvatures.dtype, curvatures.device
    e_z = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

    pos_list: list[torch.Tensor] = [torch.zeros(3, dtype=dtype, device=device)]
    rot_list: list[torch.Tensor] = [torch.eye(3, dtype=dtype, device=device)]
    r_cur = rot_list[0]
    p_cur = pos_list[0]

    for i in range(n_seg):
        dtheta = ds * curvatures[i]
        r_half = r_cur @ _so3_exp(0.5 * dtheta)
        r_next = r_cur @ _so3_exp(dtheta)
        p_next = p_cur + ds * (r_half @ e_z)
        pos_list.append(p_next)
        rot_list.append(r_next)
        r_cur = r_next
        p_cur = p_next

    return torch.stack(pos_list, dim=0), torch.stack(rot_list, dim=0)


def solve_3d_tendon_continuum_backbone(
    spec: CosseratBackboneSpec,
    tendon_tensions_n: Sequence[float],
    *,
    external_forces_n: Mapping[int, tuple[float, float, float]] | None = None,
    tip_moment_n_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tolerance: float = 1.0e-10,
    max_iterations: int = 80,
) -> CosseratTendonResult:
    """Solve exact finite-rotation 3D tendon-driven Cosserat continuum equilibrium."""
    t0 = time.perf_counter()
    if len(tendon_tensions_n) != len(spec.tendon_offsets_mm):
        raise ValueError("tendon_tensions_n must match len(spec.tendon_offsets_mm)")
    if any(not math.isfinite(float(tv)) or float(tv) < 0.0 for tv in tendon_tensions_n):
        raise ValueError("tendon tensions must be finite and nonnegative")

    n_seg = spec.num_disks
    ds = spec.length_mm / float(n_seg)
    ei = spec.youngs_modulus_mpa * spec.i_bend_mm4  # N*mm^2
    gj = spec.shear_modulus_mpa * spec.j_torsion_mm4  # N*mm^2
    k_diag = torch.tensor([ei, ei, gj], dtype=torch.float64)
    offsets_t = torch.tensor(
        [[ox, oy, 0.0] for ox, oy in spec.tendon_offsets_mm], dtype=torch.float64
    )  # (M, 3)
    tensions_t = torch.tensor(
        [float(v) for v in tendon_tensions_n], dtype=torch.float64
    )
    g_vec = torch.tensor(spec.gravity_m_per_s2, dtype=torch.float64)
    disk_force_grav = (
        spec.disk_mass_kg + spec.linear_mass_kg_per_m * (ds * 1.0e-3)
    ) * g_vec  # N

    ext_f_map: dict[int, torch.Tensor] = {}
    if external_forces_n:
        for d_idx, fv in external_forces_n.items():
            if not (1 <= int(d_idx) <= n_seg):
                raise ValueError(
                    f"external force disk index {d_idx} out of [1, {n_seg}]"
                )
            ext_f_map[int(d_idx)] = torch.tensor(fv, dtype=torch.float64)
    m_tip_t = torch.tensor(tip_moment_n_mm, dtype=torch.float64)

    def _potential(kappa: torch.Tensor) -> torch.Tensor:
        pos, rot = _forward_cosserat_kinematics(spec, kappa)
        # Elastic strain energy (N*mm)
        u_rod = 0.5 * ds * (k_diag.unsqueeze(0) * kappa.square()).sum()
        # Tendon path lengths through all disks
        # q[d, m, :] = pos[d] + rot[d] @ offsets_t[m]
        q_pts = pos.unsqueeze(1) + torch.einsum("dij,mj->dmi", rot, offsets_t)
        seg_vecs = q_pts[1:] - q_pts[:-1]
        t_lengths = torch.sqrt((seg_vecs * seg_vecs).sum(dim=-1) + 1.0e-18).sum(dim=0)
        w_tendon = (tensions_t * t_lengths).sum()

        # External force & gravity potential (-f . p)
        w_ext = torch.zeros((), dtype=torch.float64)
        if float(torch.linalg.norm(disk_force_grav)) > 0.0:
            w_ext = w_ext - (pos[1:] * disk_force_grav.unsqueeze(0)).sum()
        for d_idx, f_t in ext_f_map.items():
            w_ext = w_ext - torch.dot(f_t, pos[d_idx])
        if float(torch.linalg.norm(m_tip_t)) > 0.0:
            # Conservative moment work approximated cleanly via cumulative rotation vector
            theta_tip = ds * kappa.sum(dim=0)
            w_ext = w_ext - torch.dot(m_tip_t, theta_tip)
        return u_rod + w_tendon + w_ext

    kappa = torch.zeros((n_seg, 3), dtype=torch.float64)
    # Exact Newton-Raphson with Levenberg-Marquardt damping + backtracking
    grad_norm = float("inf")
    converged = False
    iters = 0
    for it in range(1, max_iterations + 1):
        iters = it
        k_var = kappa.detach().clone().requires_grad_(True)
        energy_t = _potential(k_var)
        g_t = torch.autograd.grad(energy_t, k_var, create_graph=True)[0]
        g_flat = g_t.reshape(-1)
        grad_norm = float(torch.linalg.norm(g_flat).item())
        if grad_norm <= tolerance:
            converged = True
            break

        # Exact Hessian via autograd over 3*n_seg variables (tiny: ~27-60 DOFs!)
        n_var = int(g_flat.numel())
        h_rows: list[torch.Tensor] = []
        for idx_v in range(n_var):
            row = torch.autograd.grad(
                g_flat[idx_v], k_var, retain_graph=idx_v < n_var - 1
            )[0].reshape(-1)
            h_rows.append(row)
        h_mat = torch.stack(h_rows, dim=0).detach().numpy()
        g_np = g_flat.detach().numpy()
        h_sym = 0.5 * (h_mat + h_mat.T)
        evals = np.linalg.eigvalsh(h_sym)
        min_ev = float(evals[0])
        reg = max(1.0e-10, -min_ev + 1.0e-6) if min_ev <= 1.0e-8 else 1.0e-10
        step = np.linalg.solve(h_sym + reg * np.eye(n_var), -g_np)
        step_t = torch.from_numpy(step.reshape(n_seg, 3))

        e0 = float(energy_t.item())
        alpha = 1.0
        accepted = False
        for _ls in range(16):
            cand = kappa + alpha * step_t
            with torch.no_grad():
                ec = float(_potential(cand).item())
            if math.isfinite(ec) and ec <= e0 + 1.0e-12 * max(1.0, abs(e0)):
                kappa = cand
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            kappa = kappa + 0.1 * step_t

    with torch.no_grad():
        pos_t, rot_t = _forward_cosserat_kinematics(spec, kappa)
        pos_np = pos_t.numpy()
        rot_np = rot_t.numpy()
        kappa_np = kappa.numpy()
        q_pts = pos_t.unsqueeze(1) + torch.einsum("dij,mj->dmi", rot_t, offsets_t)
        t_len = torch.sqrt(((q_pts[1:] - q_pts[:-1]) ** 2).sum(dim=-1) + 1.0e-18).sum(
            dim=0
        )
        strokes_np = (spec.length_mm - t_len).numpy()
        total_e = float(_potential(kappa).item())

    # Bending microstrain at fiber core radius: eps_x = -r_core * kappa_y, eps_y = r_core * kappa_x
    r_f = spec.fiber_core_radius_mm
    fbg_microstrain = np.column_stack(
        (-r_f * kappa_np[:, 1] * 1.0e6, r_f * kappa_np[:, 0] * 1.0e6)
    )

    # Static equilibrium base reaction force & moment:
    # sum of external forces + gravity + tendon exit reactions = 0 at base
    f_base = np.zeros(3, dtype=np.float64)
    m_base = np.zeros(3, dtype=np.float64)
    if float(np.linalg.norm(disk_force_grav.numpy())) > 0.0:
        gf = disk_force_grav.numpy()
        for d in range(1, n_seg + 1):
            f_base -= gf
            m_base -= np.cross(pos_np[d], gf)
    for d_idx, f_t in ext_f_map.items():
        fv_np = f_t.numpy()
        f_base -= fv_np
        m_base -= np.cross(pos_np[d_idx], fv_np)
    # Base moment from first backbone segment internal couple: M(0) = K * kappa[0]
    m_int_base = np.array(
        [ei * kappa_np[0, 0], ei * kappa_np[0, 1], gj * kappa_np[0, 2]],
        dtype=np.float64,
    )
    # Tendons also pull downward on the base plate with -T_m * e_{0->1,m}
    q_np = q_pts.numpy()
    for m_i, tv in enumerate(tendon_tensions_n):
        seg0 = q_np[1, m_i] - q_np[0, m_i]
        e_vec0 = seg0 / max(float(np.linalg.norm(seg0)), 1.0e-12)
        f_base -= float(tv) * e_vec0
    # Total base moment equals internal backbone moment + tendon base anchor couples
    m_total_base = m_int_base.copy()
    for m_i, tv in enumerate(tendon_tensions_n):
        seg0 = q_np[1, m_i] - q_np[0, m_i]
        e_vec0 = seg0 / max(float(np.linalg.norm(seg0)), 1.0e-12)
        m_total_base -= np.cross(q_np[0, m_i], float(tv) * e_vec0)

    return CosseratTendonResult(
        converged=converged,
        disk_positions_mm=pos_np,
        disk_rotations=rot_np,
        segment_curvatures_per_mm=kappa_np,
        fbg_microstrain=fbg_microstrain,
        tendon_strokes_mm=strokes_np,
        base_force_n=(float(f_base[0]), float(f_base[1]), float(f_base[2])),
        base_moment_n_mm=(
            float(m_total_base[0]),
            float(m_total_base[1]),
            float(m_total_base[2]),
        ),
        total_energy_n_mm=total_e,
        grad_norm=grad_norm,
        iterations=iters,
        wall_time_s=time.perf_counter() - t0,
    )


__all__ = (
    "COSSERAT_TENDON_RESULT_SCHEMA",
    "TENDON_3D_RESULT_SCHEMA",
    "CosseratBackboneSpec",
    "CosseratTendonResult",
    "TendonForceAssembly3D",
    "TendonRoute3D",
    "TendonSolid3DResult",
    "assemble_tendon_forces_3d",
    "solve_3d_tendon_continuum_backbone",
    "solve_3d_tendon_solid",
)
