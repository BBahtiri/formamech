"""3D transient structural dynamics: Newmark-beta, HHT-alpha & Explicit CFL (V4-3).

Integrates the consistent mm-N-s-tonne structural equation of motion::

    M u_ddot + C u_dot + K u = f(t)

with Rayleigh damping ``C = rayleigh_alpha * M + rayleigh_beta * K`` using:

* **Implicit Newmark-beta** (default average acceleration ``beta=0.25, gamma=0.5``):
  unconditionally stable and strictly energy-conserving for undamped free vibration.
* **Implicit HHT-alpha** (``alpha in [-1/3, 0]``): unconditionally stable with
  controllable high-frequency numerical dissipation.
* **Explicit Central-Difference** with row-sum lumped mass ``M_L`` and automatic
  CFL stability guard (``dt_crit = 2 / omega_max``) that raises
  :class:`CflStabilityError` fail-closed when ``dt_s > cfl_safety * dt_crit``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags  # type: ignore[import-untyped]
from scipy.sparse.linalg import eigsh, splu  # type: ignore[import-untyped]

from piginot.contracts import MaterialSpec
from piginot.fem.backend_3d import (
    DirichletBC3D,
    TorchFemBackend3D,
    TractionBC3D,
    assemble_surface_tractions_3d,
    face_node_indices_3d,
)
from piginot.fem.modal_3d import assemble_3d_mass_matrix
from piginot.geometry.mesh3d import CanonicalMesh3D

FloatArray = NDArray[np.float64]

DYNAMICS_3D_RESULT_SCHEMA: Final = "formamech.dynamics-3d-result-v1"


class DynamicsScheme(StrEnum):
    """Time-integration schemes for 3D transient structural dynamics."""

    NEWMARK_BETA = "newmark-beta"
    HHT_ALPHA = "hht-alpha"
    EXPLICIT_CENTRAL_DIFF = "explicit-central-diff"


class CflStabilityError(ValueError):
    """Fail-closed refusal when explicit ``dt_s`` exceeds the CFL stability limit."""


@dataclass(frozen=True, slots=True)
class TransientDynamicsConfig:
    """Configuration for 3D linear transient structural dynamics."""

    scheme: DynamicsScheme = DynamicsScheme.NEWMARK_BETA
    dt_s: float = 1.0e-4
    num_steps: int = 20
    density_t_per_mm3: float = 7.85e-9
    rayleigh_alpha: float = 0.0
    rayleigh_beta: float = 0.0
    newmark_beta: float = 0.25
    newmark_gamma: float = 0.5
    hht_alpha: float = -0.05
    cfl_safety: float = 0.95
    lumped_mass: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if type(self.num_steps) is not int or self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if not math.isfinite(self.density_t_per_mm3) or self.density_t_per_mm3 <= 0.0:
            raise ValueError("density_t_per_mm3 must be positive")
        if self.rayleigh_alpha < 0.0 or self.rayleigh_beta < 0.0:
            raise ValueError("Rayleigh damping coefficients must be nonnegative")
        if not (-1.0 / 3.0 <= self.hht_alpha <= 0.0):
            raise ValueError("hht_alpha must lie in [-1/3, 0]")
        if not (0.0 < self.cfl_safety <= 1.0):
            raise ValueError("cfl_safety must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class TransientDynamics3DResult:
    """Verified 3D transient structural dynamics time-history output."""

    mesh_id: str
    scheme: DynamicsScheme
    dt_s: float
    dt_crit_s: float | None
    times_s: FloatArray  # (T,)
    kinetic_energy_mj: FloatArray  # (T,)
    strain_energy_mj: FloatArray  # (T,)
    total_energy_mj: FloatArray  # (T,)
    max_displacement_history_mm: FloatArray  # (T,)
    final_displacement_mm: FloatArray  # (N_nodes, 3)
    final_velocity_mm_s: FloatArray  # (N_nodes, 3)
    energy_relative_drift: float
    wall_time_s: float

    def document(self) -> dict[str, Any]:
        return {
            "schema": DYNAMICS_3D_RESULT_SCHEMA,
            "mesh_id": self.mesh_id,
            "scheme": self.scheme.value,
            "dt_s": float(self.dt_s),
            "dt_crit_s": None if self.dt_crit_s is None else float(self.dt_crit_s),
            "num_steps": int(self.times_s.size - 1),
            "max_displacement_mm": float(np.max(self.max_displacement_history_mm)),
            "initial_total_energy_mj": float(self.total_energy_mj[0]),
            "final_total_energy_mj": float(self.total_energy_mj[-1]),
            "energy_relative_drift": float(self.energy_relative_drift),
            "wall_time_s": float(self.wall_time_s),
        }


def estimate_explicit_cfl_dt(k_ff: csr_matrix, m_diag_f: FloatArray) -> float:
    """Compute exact critical time step ``dt_crit = 2 / omega_max`` for lumped mass."""
    if np.any(m_diag_f <= 0.0):
        raise ValueError("lumped mass diagonal must be strictly positive on free DOFs")
    inv_sqrt_m = 1.0 / np.sqrt(m_diag_f)
    # Transform K_tilde = M^{-1/2} K M^{-1/2} so lambda_max(K_tilde) = omega_max^2
    n_f = int(m_diag_f.size)
    if n_f == 1:
        lam_max = float(k_ff[0, 0]) / float(m_diag_f[0])
    elif n_f <= 64:
        k_dense = k_ff.toarray()
        k_tilde = (k_dense * inv_sqrt_m[:, None]) * inv_sqrt_m[None, :]
        eigvals = np.linalg.eigvalsh(0.5 * (k_tilde + k_tilde.T))
        lam_max = float(np.max(eigvals))
    else:
        d_inv = diags(inv_sqrt_m, format="csr")
        k_tilde_sp = (d_inv @ k_ff @ d_inv).tocsr()
        vals = eigsh(k_tilde_sp, k=1, which="LA", return_eigenvectors=False)
        lam_max = float(np.max(vals))
    omega_max = math.sqrt(max(lam_max, 1.0e-20))
    return 2.0 / omega_max


def solve_3d_transient_dynamics(
    mesh: CanonicalMesh3D,
    material: MaterialSpec,
    *,
    dirichlet_bcs: Sequence[DirichletBC3D],
    config: TransientDynamicsConfig | None = None,
    traction_bcs: Sequence[TractionBC3D] = (),
    load_multiplier: Callable[[float], float] | None = None,
    initial_displacement_mm: FloatArray | None = None,
    initial_velocity_mm_s: FloatArray | None = None,
) -> TransientDynamics3DResult:
    """Integrate 3D linear transient elastodynamics via Newmark, HHT-alpha, or Explicit."""
    t0 = time.perf_counter()
    cfg = TransientDynamicsConfig() if config is None else config
    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes

    helper = TorchFemBackend3D()
    cells, d_phys, w_vol, _, rows, cols = helper._precompute_geometry(mesh)
    u_zero = np.zeros(n_dofs, dtype=np.float64)
    _, k_mat, _, _, _, _ = helper._assemble(
        u_zero, cells, d_phys, w_vol, rows, cols, material, n_dofs
    )
    k_sym = (0.5 * (k_mat + k_mat.T)).tocsr()
    use_lumped = cfg.lumped_mass or (cfg.scheme is DynamicsScheme.EXPLICIT_CENTRAL_DIFF)
    m_mat = assemble_3d_mass_matrix(mesh, cfg.density_t_per_mm3, lumped=use_lumped)

    f_base = np.zeros(n_dofs, dtype=np.float64)
    if traction_bcs:
        f_base += assemble_surface_tractions_3d(mesh, traction_bcs).reshape(n_dofs)

    comp_idx = {"x": 0, "y": 1, "z": 2}
    constrained = np.zeros(n_dofs, dtype=bool)
    for bc in dirichlet_bcs:
        nids = face_node_indices_3d(mesh, bc.face_tags)
        for comp in bc.components:
            c = comp_idx[comp]
            for nid in nids:
                constrained[3 * nid + c] = True
    free = np.nonzero(~constrained)[0]
    if free.size == 0:
        raise ValueError("all DOFs are constrained for transient dynamics")

    k_ff = k_sym[free][:, free].tocsr()
    m_ff = m_mat[free][:, free].tocsr()
    c_ff = (cfg.rayleigh_alpha * m_ff + cfg.rayleigh_beta * k_ff).tocsr()
    f_base_f = f_base[free]

    def _f_ext(t_val: float) -> FloatArray:
        if not traction_bcs:
            return np.zeros(free.size, dtype=np.float64)
        scale = 1.0 if load_multiplier is None else float(load_multiplier(t_val))
        return scale * f_base_f

    u: FloatArray = np.zeros(free.size, dtype=np.float64)
    v: FloatArray = np.zeros(free.size, dtype=np.float64)
    if initial_displacement_mm is not None:
        u = (
            np.asarray(initial_displacement_mm, dtype=np.float64)
            .reshape(n_dofs)[free]
            .copy()
        )
    if initial_velocity_mm_s is not None:
        v = (
            np.asarray(initial_velocity_mm_s, dtype=np.float64)
            .reshape(n_dofs)[free]
            .copy()
        )

    dt = float(cfg.dt_s)
    n_steps = cfg.num_steps
    times = np.linspace(0.0, n_steps * dt, n_steps + 1, dtype=np.float64)
    ke_hist = np.zeros(n_steps + 1, dtype=np.float64)
    pe_hist = np.zeros(n_steps + 1, dtype=np.float64)
    max_u_hist = np.zeros(n_steps + 1, dtype=np.float64)

    def _record(step_i: int, u_vec: FloatArray, v_vec: FloatArray) -> None:
        ke_hist[step_i] = 0.5 * float(v_vec @ (m_ff @ v_vec))
        pe_hist[step_i] = 0.5 * float(u_vec @ (k_ff @ u_vec))
        full_u = np.zeros(n_dofs, dtype=np.float64)
        full_u[free] = u_vec
        max_u_hist[step_i] = float(
            np.max(np.linalg.norm(full_u.reshape(n_nodes, 3), axis=1))
        )

    dt_crit: float | None = None
    _record(0, u, v)

    if cfg.scheme is DynamicsScheme.EXPLICIT_CENTRAL_DIFF:
        m_diag = np.asarray(m_ff.diagonal(), dtype=np.float64)
        dt_crit = estimate_explicit_cfl_dt(k_ff, m_diag)
        if dt > cfg.cfl_safety * dt_crit:
            raise CflStabilityError(
                f"Explicit central-difference time step dt={dt:.3e}s exceeds "
                f"safe CFL limit {cfg.cfl_safety * dt_crit:.3e}s "
                f"(dt_crit={dt_crit:.3e}s); reduce dt_s or switch to NEWMARK_BETA/HHT_ALPHA."
            )
        # Initial acceleration: a0 = M^{-1} (f0 - C v0 - K u0)
        a = (_f_ext(0.0) - c_ff @ v - k_ff @ u) / m_diag
        v_half = v + 0.5 * dt * a
        for step in range(1, n_steps + 1):
            u_next = u + dt * v_half
            t_next = float(times[step])
            # Estimate velocity for damping term with lagged half-step velocity
            a_next = (_f_ext(t_next) - c_ff @ v_half - k_ff @ u_next) / m_diag
            v_next = v_half + 0.5 * dt * a_next
            v_half = v_next + 0.5 * dt * a_next
            u = u_next
            v = v_next
            _record(step, u, v)
    else:
        if cfg.scheme is DynamicsScheme.HHT_ALPHA:
            alpha = float(cfg.hht_alpha)
            gamma = 0.5 * (1.0 - 2.0 * alpha)
            beta = 0.25 * ((1.0 - alpha) ** 2)
        else:
            alpha = 0.0
            gamma = float(cfg.newmark_gamma)
            beta = float(cfg.newmark_beta)

        # Solve initial acceleration M a0 = f(0) - C v0 - K u0
        m_lu = splu(m_ff.tocsc())
        a = np.asarray(m_lu.solve(_f_ext(0.0) - c_ff @ v - k_ff @ u), dtype=np.float64)
        one_plus_a = 1.0 + alpha
        k_eff = (
            (1.0 / (beta * dt * dt)) * m_ff
            + (one_plus_a * gamma / (beta * dt)) * c_ff
            + one_plus_a * k_ff
        ).tocsc()
        k_eff_lu = splu(k_eff)

        f_curr = _f_ext(0.0)
        for step in range(1, n_steps + 1):
            t_next = float(times[step])
            f_next = _f_ext(t_next)
            u_tilde = u + dt * v + (0.5 - beta) * (dt * dt) * a
            v_tilde = v + (1.0 - gamma) * dt * a

            rhs = (
                one_plus_a * f_next
                - alpha * f_curr
                + alpha * (c_ff @ v + k_ff @ u)
                + (1.0 / (beta * dt * dt)) * (m_ff @ u_tilde)
                + one_plus_a * (c_ff @ ((gamma / (beta * dt)) * u_tilde - v_tilde))
            )
            u_next = np.asarray(k_eff_lu.solve(rhs), dtype=np.float64)
            a_next = np.asarray((u_next - u_tilde) / (beta * dt * dt), dtype=np.float64)
            v_next = np.asarray(v_tilde + gamma * dt * a_next, dtype=np.float64)

            u, v, a, f_curr = u_next, v_next, a_next, f_next
            _record(step, u, v)

    tot_e = ke_hist + pe_hist
    e0 = float(tot_e[0])
    rel_drift = (
        float(np.max(np.abs(tot_e - e0)) / max(abs(e0), 1.0e-15))
        if abs(e0) > 1.0e-15
        else 0.0
    )

    u_final_full = np.zeros(n_dofs, dtype=np.float64)
    v_final_full = np.zeros(n_dofs, dtype=np.float64)
    u_final_full[free] = u
    v_final_full[free] = v

    return TransientDynamics3DResult(
        mesh_id=mesh.mesh_id,
        scheme=cfg.scheme,
        dt_s=dt,
        dt_crit_s=dt_crit,
        times_s=times,
        kinetic_energy_mj=ke_hist,
        strain_energy_mj=pe_hist,
        total_energy_mj=tot_e,
        max_displacement_history_mm=max_u_hist,
        final_displacement_mm=u_final_full.reshape(n_nodes, 3),
        final_velocity_mm_s=v_final_full.reshape(n_nodes, 3),
        energy_relative_drift=rel_drift,
        wall_time_s=time.perf_counter() - t0,
    )


__all__ = (
    "DYNAMICS_3D_RESULT_SCHEMA",
    "CflStabilityError",
    "DynamicsScheme",
    "TransientDynamics3DResult",
    "TransientDynamicsConfig",
    "estimate_explicit_cfl_dt",
    "solve_3d_transient_dynamics",
)
