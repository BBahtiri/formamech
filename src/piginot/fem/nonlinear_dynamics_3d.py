"""Large-deformation 3D nonlinear transient structural dynamics (V6-R3).

Solves the finite-strain Total-Lagrangian dynamic equation of motion on 3D
``TET4``/``TET10`` meshes in consistent ``mm-N-s-tonne`` units::

    M u_ddot + C u_dot + f_int(u) = f_ext(t)

using **Nonlinear Implicit Newmark-beta** (``beta = 0.25, gamma = 0.5``) and
**Nonlinear HHT-alpha** (``alpha_hht in [-1/3, 0]``) with exact Newton-Raphson
inner iterations, Rayleigh damping ``C = alpha M + beta K_0``, ``det(F) > 0``
line-search guards, and full kinetic + nonlinear strain energy diagnostics.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
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
    assemble_surface_tractions_3d,
    face_node_indices_3d,
)
from piginot.fem.linear_solvers import LinearSolverConfig, solve_linear_system
from piginot.fem.modal_3d import assemble_3d_mass_matrix
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D
from piginot.physics import consistent_tangent, energy_density, first_piola

FloatArray = NDArray[np.float64]
NONLINEAR_DYNAMICS_3D_SCHEMA: Final = "formamech.nonlinear-dynamics-3d-v1"


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


class NonlinearDynamicsScheme(StrEnum):
    """Implicit finite-strain 3D time-integration schemes."""

    NEWMARK_BETA = "nonlinear-newmark-beta"
    HHT_ALPHA = "nonlinear-hht-alpha"


@dataclass(frozen=True, slots=True)
class NonlinearDynamics3DConfig:
    """Configuration for 3D finite-strain nonlinear transient dynamics."""

    scheme: NonlinearDynamicsScheme = NonlinearDynamicsScheme.NEWMARK_BETA
    dt_s: float = 1.0e-3
    num_steps: int = 20
    density_t_per_mm3: float = 1.1e-9  # ~1100 kg/m^3 elastomer default
    rayleigh_alpha: float = 0.0
    rayleigh_beta: float = 0.0
    newmark_beta: float = 0.25
    newmark_gamma: float = 0.5
    hht_alpha: float = -0.05
    lumped_mass: bool = False
    solver_tolerance: float = 1.0e-7
    max_newton_iterations: int = 15

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if type(self.num_steps) is not int or self.num_steps < 1:
            raise ValueError("num_steps must be an integer >= 1")
        if not math.isfinite(self.density_t_per_mm3) or self.density_t_per_mm3 <= 0.0:
            raise ValueError("density_t_per_mm3 must be positive")
        if self.rayleigh_alpha < 0.0 or self.rayleigh_beta < 0.0:
            raise ValueError("Rayleigh damping coefficients must be nonnegative")
        if not (-1.0 / 3.0 <= self.hht_alpha <= 0.0):
            raise ValueError("hht_alpha must lie in [-1/3, 0]")


@dataclass(frozen=True, slots=True)
class NonlinearDynamics3DResult:
    """Verified large-deformation 3D transient structural dynamics trajectory."""

    converged: bool
    mesh_id: str
    scheme: NonlinearDynamicsScheme
    dt_s: float
    times_s: FloatArray  # (T,)
    displacement_history_mm: FloatArray  # (T, N_nodes, 3)
    velocity_history_mm_s: FloatArray  # (T, N_nodes, 3)
    kinetic_energy_mj: FloatArray  # (T,)
    strain_energy_mj: FloatArray  # (T,)
    total_energy_mj: FloatArray  # (T,)
    min_det_f_history: FloatArray  # (T,)
    energy_relative_drift: float
    total_newton_iterations: int
    wall_time_s: float
    reason: str = "converged"

    def document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": NONLINEAR_DYNAMICS_3D_SCHEMA,
            "converged": self.converged,
            "mesh_id": self.mesh_id,
            "scheme": self.scheme.value,
            "dt_s": float(self.dt_s),
            "num_steps": int(self.times_s.size - 1),
            "max_displacement_mm": float(
                np.max(np.linalg.norm(self.displacement_history_mm, axis=-1))
            ),
            "min_det_f": float(np.min(self.min_det_f_history)),
            "initial_total_energy_mj": float(self.total_energy_mj[0]),
            "final_total_energy_mj": float(self.total_energy_mj[-1]),
            "energy_relative_drift": float(self.energy_relative_drift),
            "total_newton_iterations": self.total_newton_iterations,
            "wall_time_s": float(self.wall_time_s),
            "reason": self.reason,
        }
        doc["result_hash"] = _sha256_doc(doc)
        return doc


def solve_3d_nonlinear_dynamics(
    mesh: CanonicalMesh3D,
    material: MaterialSpec,
    *,
    dirichlet_bcs: Sequence[DirichletBC3D],
    config: NonlinearDynamics3DConfig | None = None,
    traction_bcs: Sequence[TractionBC3D] = (),
    load_multiplier: Callable[[float], float] | None = None,
    initial_displacement_mm: FloatArray | None = None,
    initial_velocity_mm_s: FloatArray | None = None,
    linear_solver: LinearSolverConfig | None = None,
) -> NonlinearDynamics3DResult:
    """Integrate 3D finite-strain nonlinear transient dynamics via implicit Newmark/HHT."""
    t0 = time.perf_counter()
    cfg = NonlinearDynamics3DConfig() if config is None else config
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

    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes
    helper = TorchFemBackend3D()
    cells, d_phys, w_vol, _cell_vols, rows, cols = helper._precompute_geometry(mesh)
    n_elem, n_loc = cells.shape
    n_edof = 3 * n_loc
    lin_cfg = LinearSolverConfig() if linear_solver is None else linear_solver

    m_mat = assemble_3d_mass_matrix(
        mesh, density_t_per_mm3=cfg.density_t_per_mm3 * s_fac, lumped=cfg.lumped_mass
    )
    f_ref = assemble_surface_tractions_3d(mesh, traction_bcs).reshape(-1) * s_fac
    mult_fn: Callable[[float], float] = (
        (lambda _t: 1.0) if load_multiplier is None else load_multiplier
    )

    comp_idx = {"x": 0, "y": 1, "z": 2}
    constrained = np.zeros(n_dofs, dtype=bool)
    u_fixed = np.zeros(n_dofs, dtype=np.float64)
    for bc in dirichlet_bcs:
        nids = face_node_indices_3d(mesh, bc.face_tags)
        for comp, val in zip(bc.components, bc.values_mm, strict=True):
            c = comp_idx[comp]
            for nid in nids:
                dof = 3 * nid + c
                constrained[dof] = True
                u_fixed[dof] = float(val)
    free = np.nonzero(~constrained)[0]

    def _assemble_internal(
        u_vec: FloatArray,
    ) -> tuple[FloatArray, csr_matrix, float, float]:
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
        return f_int.reshape(-1), k_mat, total_e, min_det

    # Reference stiffness K_0 for Rayleigh damping C = alpha*M + beta*K_0
    _, k0_mat, _, _ = _assemble_internal(np.zeros(n_dofs, dtype=np.float64))
    c_mat = (cfg.rayleigh_alpha * m_mat + cfg.rayleigh_beta * k0_mat).tocsr()

    # Initial state
    u_n: FloatArray = (
        np.zeros(n_dofs, dtype=np.float64)
        if initial_displacement_mm is None
        else np.asarray(initial_displacement_mm, dtype=np.float64)
        .reshape(n_dofs)
        .copy()
    )
    u_n[constrained] = u_fixed[constrained]
    v_n: FloatArray = (
        np.zeros(n_dofs, dtype=np.float64)
        if initial_velocity_mm_s is None
        else np.asarray(initial_velocity_mm_s, dtype=np.float64).reshape(n_dofs).copy()
    )
    v_n[constrained] = 0.0

    f_int_n, _, w_n, det_n = _assemble_internal(u_n)
    f_ext_n = float(mult_fn(0.0)) * f_ref
    # Initial acceleration: M_ff a_0 = f_ext_0 - C v_0 - f_int(u_0)
    a_n: FloatArray = np.zeros(n_dofs, dtype=np.float64)
    rhs_a0 = (f_ext_n - (c_mat @ v_n) - f_int_n)[free]
    a_free0, rep0 = solve_linear_system(
        m_mat[free][:, free], rhs_a0, config=lin_cfg, is_symmetric=True
    )
    if rep0.converged:
        a_n[free] = a_free0

    # HHT-alpha parameters (Newmark-beta is the alpha_hht = 0 special case)
    if cfg.scheme is NonlinearDynamicsScheme.HHT_ALPHA:
        alpha_f = float(cfg.hht_alpha)
        beta = 0.25 * ((1.0 - alpha_f) ** 2)
        gamma = 0.5 - alpha_f
    else:
        alpha_f = 0.0
        beta = float(cfg.newmark_beta)
        gamma = float(cfg.newmark_gamma)

    dt = float(cfg.dt_s)
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = 0.5 * dt * (gamma / beta - 2.0)

    n_steps = cfg.num_steps
    times = np.linspace(0.0, n_steps * dt, n_steps + 1, dtype=np.float64)
    u_hist = np.zeros((n_steps + 1, n_nodes, 3), dtype=np.float64)
    v_hist = np.zeros((n_steps + 1, n_nodes, 3), dtype=np.float64)
    ke_hist = np.zeros(n_steps + 1, dtype=np.float64)
    se_hist = np.zeros(n_steps + 1, dtype=np.float64)
    det_hist = np.zeros(n_steps + 1, dtype=np.float64)

    u_hist[0] = u_n.reshape(n_nodes, 3)
    v_hist[0] = v_n.reshape(n_nodes, 3)
    ke_hist[0] = 0.5 * float(v_n @ (m_mat @ v_n)) / s_fac
    se_hist[0] = w_n / s_fac
    det_hist[0] = det_n
    tol_eff = cfg.solver_tolerance * s_fac

    total_newton = 0
    for step in range(1, n_steps + 1):
        t_next = float(times[step])
        f_ext_next = float(mult_fn(t_next)) * f_ref
        f_ext_eff = (1.0 + alpha_f) * f_ext_next - alpha_f * f_ext_n

        # Newmark predictor
        u_pred = u_n + dt * v_n + (0.5 - beta) * dt * dt * a_n
        u_cur = u_pred.copy()
        u_cur[constrained] = u_fixed[constrained]

        step_conv = False
        w_cur = se_hist[step - 1]
        det_cur = det_hist[step - 1]
        f_int_cur: FloatArray = f_int_n
        v_cur: FloatArray = v_n
        a_cur: FloatArray = a_n

        for _ in range(cfg.max_newton_iterations):
            total_newton += 1
            a_cur = a0 * (u_cur - u_n) - a2 * v_n - a3 * a_n
            v_cur = a1 * (u_cur - u_n) - a4 * v_n - a5 * a_n
            a_cur[constrained] = 0.0
            v_cur[constrained] = 0.0

            f_int_cur, k_int, w_cur, det_cur = _assemble_internal(u_cur)
            if det_cur <= 1.0e-8:
                break

            res_full = (
                (m_mat @ a_cur)
                + (1.0 + alpha_f) * (c_mat @ v_cur + f_int_cur)
                - alpha_f * (c_mat @ v_n + f_int_n)
                - f_ext_eff
            )
            res_free = res_full[free]
            res_norm = float(np.linalg.norm(res_free))
            if res_norm <= tol_eff:
                step_conv = True
                break

            k_eff = (
                a0 * m_mat + (1.0 + alpha_f) * a1 * c_mat + (1.0 + alpha_f) * k_int
            ).tocsr()
            du_free, rep = solve_linear_system(
                k_eff[free][:, free], -res_free, config=lin_cfg, is_symmetric=True
            )
            if not rep.converged:
                break

            step_alpha = 1.0
            accepted = False
            for _ls in range(8):
                u_cand = u_cur.copy()
                u_cand[free] += step_alpha * du_free
                _, _, _, det_c = _assemble_internal(u_cand)
                if det_c > 1.0e-5:
                    u_cur = u_cand
                    accepted = True
                    break
                step_alpha *= 0.5
            if not accepted:
                break

        if not step_conv:
            tot_e = ke_hist + se_hist
            return NonlinearDynamics3DResult(
                converged=False,
                mesh_id=mesh.mesh_id,
                scheme=cfg.scheme,
                dt_s=dt,
                times_s=times,
                displacement_history_mm=u_hist,
                velocity_history_mm_s=v_hist,
                kinetic_energy_mj=ke_hist,
                strain_energy_mj=se_hist,
                total_energy_mj=tot_e,
                min_det_f_history=det_hist,
                energy_relative_drift=float("inf"),
                total_newton_iterations=total_newton,
                wall_time_s=time.perf_counter() - t0,
                reason=f"nonlinear dynamics step {step} did not converge",
            )

        u_n = u_cur
        v_n = v_cur
        a_n = a_cur
        f_int_n = f_int_cur
        f_ext_n = f_ext_next

        u_hist[step] = u_n.reshape(n_nodes, 3)
        v_hist[step] = v_n.reshape(n_nodes, 3)
        ke_hist[step] = 0.5 * float(v_n @ (m_mat @ v_n)) / s_fac
        se_hist[step] = w_cur / s_fac
        det_hist[step] = det_cur

    tot_e = ke_hist + se_hist
    e0 = max(float(tot_e[0]), 1.0e-12)
    drift = float(np.max(np.abs(tot_e - tot_e[0])) / e0) if tot_e[0] > 1.0e-12 else 0.0
    return NonlinearDynamics3DResult(
        converged=True,
        mesh_id=mesh.mesh_id,
        scheme=cfg.scheme,
        dt_s=dt,
        times_s=times,
        displacement_history_mm=u_hist,
        velocity_history_mm_s=v_hist,
        kinetic_energy_mj=ke_hist,
        strain_energy_mj=se_hist,
        total_energy_mj=tot_e,
        min_det_f_history=det_hist,
        energy_relative_drift=drift,
        total_newton_iterations=total_newton,
        wall_time_s=time.perf_counter() - t0,
        reason="converged",
    )


__all__ = (
    "NONLINEAR_DYNAMICS_3D_SCHEMA",
    "NonlinearDynamics3DConfig",
    "NonlinearDynamics3DResult",
    "NonlinearDynamicsScheme",
    "solve_3d_nonlinear_dynamics",
)
