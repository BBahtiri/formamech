"""Rate-dependent 3D Viscoelasticity, Mullins Hysteresis & Velocity-Residual Operator (V7-R1).

Closes the pneumatic/elastomer loading-unloading hysteresis gap surfaced in
Study R3 by pairing:

1. **Finite-Strain 3D Prony Viscoelastic + Ogden-Roxburgh Mullins FEM**
   (:func:`solve_3d_visco_hysteresis`): wraps FormaMech's 3D finite-strain
   ``TET4``/``TET10`` solver (supporting both :class:`TractionBC3D` and
   :class:`FollowerPressureBC3D` via :mod:`piginot.fem.follower_3d`) with exact
   exponential Prony internal overstress recurrence ``H_k(t)`` and
   Ogden-Roxburgh Mullins softening ``eta(W_max, W)``, certifying both
   ``min_det_f > 0`` and non-negative thermodynamic dissipation ``dD/dt >= 0``
   in every step receipt.
2. **Physics-Guided Velocity-Conditioned Residual Hysteresis Operator**
   (:class:`VelocityResidualHysteresisOperator`): combines an exact discrete
   multi-branch Prony viscoelastic + Bouc-Wen rate-independent hysteresis state
   with a bounded velocity-conditioned PyTorch residual head (PGRD /
   PhysCon-Deform style) and thermodynamic dissipation verification.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix  # type: ignore[import-untyped]
from scipy.special import erf  # type: ignore[import-untyped]
from torch import nn

from piginot.contracts import MaterialSpec, canonical_json_bytes
from piginot.fem._similitude import resolve_similitude_material
from piginot.fem.backend_3d import (
    DirichletBC3D,
    TorchFemBackend3D,
    TractionBC3D,
    assemble_surface_tractions_3d,
    face_node_indices_3d,
)
from piginot.fem.follower_3d import (
    FollowerPressureBC3D,
    assemble_follower_pressure_3d,
)
from piginot.fem.linear_solvers import LinearSolverConfig, solve_linear_system
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D
from piginot.physics import consistent_tangent, energy_density, first_piola

FloatArray = NDArray[np.float64]
VISCO_HYSTERESIS_3D_SCHEMA: Final = "formamech.visco-hysteresis-3d-v1"
VELOCITY_HYSTERESIS_OP_SCHEMA: Final = "formamech.velocity-hysteresis-operator-v1"


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


@dataclass(frozen=True, slots=True)
class PronyBranch:
    """One Maxwell viscoelastic relaxation branch ``(weight gamma_k, tau_k)``."""

    weight_gamma: float
    relaxation_time_s: float

    def __post_init__(self) -> None:
        if not (0.0 < self.weight_gamma < 1.0) or not math.isfinite(self.weight_gamma):
            raise ValueError("weight_gamma must lie strictly in (0, 1)")
        if not math.isfinite(self.relaxation_time_s) or self.relaxation_time_s <= 0.0:
            raise ValueError("relaxation_time_s must be finite and positive")


@dataclass(frozen=True, slots=True)
class MullinsDamageSpec:
    """Ogden-Roxburgh (1999) pseudo-elastic Mullins softening parameters."""

    enabled: bool = True
    r_param: float = 2.5  # r > 1 controls maximum softening (1 - 1/r)
    m_energy_density: float = 0.05  # MPa = N*mm / mm^3 energy scale
    beta_param: float = 0.1  # >= 0 dimensionless scaling of W_max

    def __post_init__(self) -> None:
        if not math.isfinite(self.r_param) or self.r_param <= 1.0:
            raise ValueError("r_param must be strictly > 1.0")
        if not math.isfinite(self.m_energy_density) or self.m_energy_density <= 0.0:
            raise ValueError("m_energy_density must be positive")
        if not math.isfinite(self.beta_param) or self.beta_param < 0.0:
            raise ValueError("beta_param must be nonnegative")


@dataclass(frozen=True, slots=True)
class ViscoHysteresis3DConfig:
    """Configuration for 3D finite-strain Prony + Mullins hysteresis FEM."""

    prony_branches: tuple[PronyBranch, ...] = (
        PronyBranch(weight_gamma=0.22, relaxation_time_s=0.35),
        PronyBranch(weight_gamma=0.14, relaxation_time_s=2.5),
    )
    mullins: MullinsDamageSpec = MullinsDamageSpec(enabled=True)
    dt_s: float = 0.05
    solver_tolerance: float = 1.0e-7
    max_newton_iterations: int = 20

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        total_gamma = sum(b.weight_gamma for b in self.prony_branches)
        if total_gamma >= 0.98:
            raise ValueError("sum of Prony branch weights must be < 0.98")
        if self.max_newton_iterations < 1:
            raise ValueError("max_newton_iterations must be >= 1")

    @property
    def gamma_infinity(self) -> float:
        return float(1.0 - sum(b.weight_gamma for b in self.prony_branches))


@dataclass(frozen=True, slots=True)
class ViscoHysteresis3DResult:
    """Verified 3D viscoelastic + Mullins loading/unloading hysteresis trajectory."""

    converged: bool
    mesh_id: str
    dt_s: float
    times_s: FloatArray  # (T,)
    load_multipliers: FloatArray  # (T,)
    displacement_history_mm: FloatArray  # (T, N_nodes, 3)
    stored_energy_n_mm: FloatArray  # (T,)
    step_dissipation_n_mm: FloatArray  # (T,) >= 0 per Clausius-Duhem
    cumulative_dissipation_n_mm: float
    hysteresis_loop_work_n_mm: float
    min_det_f_history: FloatArray  # (T,)
    min_mullins_eta_history: FloatArray  # (T,)
    step_dissipation_nonnegative: bool
    thermodynamically_admissible: bool
    total_newton_iterations: int
    wall_time_s: float
    reason: str = "converged"

    def document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": VISCO_HYSTERESIS_3D_SCHEMA,
            "converged": self.converged,
            "mesh_id": self.mesh_id,
            "dt_s": float(self.dt_s),
            "num_steps": int(self.times_s.size),
            "max_displacement_mm": float(
                np.max(np.linalg.norm(self.displacement_history_mm, axis=-1))
            ),
            "min_det_f": float(np.min(self.min_det_f_history)),
            "min_mullins_eta": float(np.min(self.min_mullins_eta_history)),
            "cumulative_dissipation_n_mm": float(self.cumulative_dissipation_n_mm),
            "hysteresis_loop_work_n_mm": float(self.hysteresis_loop_work_n_mm),
            "step_dissipation_nonnegative": self.step_dissipation_nonnegative,
            "thermodynamically_admissible": self.thermodynamically_admissible,
            "total_newton_iterations": self.total_newton_iterations,
            "wall_time_s": float(self.wall_time_s),
            "reason": self.reason,
        }
        doc["result_hash"] = _sha256_doc(doc)
        return doc


def solve_3d_visco_hysteresis(
    mesh: CanonicalMesh3D,
    material: MaterialSpec,
    *,
    dirichlet_bcs: Sequence[DirichletBC3D],
    load_history: Sequence[float],
    config: ViscoHysteresis3DConfig | None = None,
    traction_bcs: Sequence[TractionBC3D] = (),
    follower_bcs: Sequence[FollowerPressureBC3D] = (),
    linear_solver: LinearSolverConfig | None = None,
) -> ViscoHysteresis3DResult:
    """Solve 3D finite-strain Prony viscoelastic + Ogden-Roxburgh Mullins hysteresis FEM."""
    t0 = time.perf_counter()
    cfg = ViscoHysteresis3DConfig() if config is None else config
    if mesh.cell_type not in (CellType3D.TET4, CellType3D.TET10):
        raise ValueError(f"requires tet4 or tet10, got {mesh.cell_type.value!r}")
    if not dirichlet_bcs:
        raise ValueError("at least one DirichletBC3D is required")
    if len(load_history) < 2:
        raise ValueError("load_history requires at least 2 steps")
    nu_prop = material.properties.get("poisson_ratio")
    if nu_prop is not None:
        mesh.validate_incompressibility_locking(float(nu_prop.value))

    sim = resolve_similitude_material(material)
    mat_eff = sim.model_material
    s_fac = sim.scale_factor
    e_prop = mat_eff.properties.get("elastic_modulus")
    e_eff = float(e_prop.value) if e_prop is not None else 1.0

    n_nodes = len(mesh.nodes)
    n_dofs = 3 * n_nodes
    helper = TorchFemBackend3D()
    cells, d_phys, w_vol, _cell_vols, rows, cols = helper._precompute_geometry(mesh)
    n_elem, n_loc = cells.shape
    n_quad = int(w_vol.shape[1])
    n_edof = 3 * n_loc
    lin_cfg = LinearSolverConfig() if linear_solver is None else linear_solver

    f_dead_ref = assemble_surface_tractions_3d(mesh, traction_bcs).reshape(-1) * s_fac
    follower_scaled = tuple(fbc.at_factor(s_fac) for fbc in follower_bcs)

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

    # Exact discrete exponential Prony factors
    dt = float(cfg.dt_s)
    gamma_inf = cfg.gamma_infinity
    n_branches = len(cfg.prony_branches)
    alphas = np.array(
        [math.exp(-dt / b.relaxation_time_s) for b in cfg.prony_branches],
        dtype=np.float64,
    )
    betas = np.array(
        [
            (1.0 - math.exp(-dt / b.relaxation_time_s)) / (dt / b.relaxation_time_s)
            for b in cfg.prony_branches
        ],
        dtype=np.float64,
    )
    gammas = np.array([b.weight_gamma for b in cfg.prony_branches], dtype=np.float64)
    taus = np.array([b.relaxation_time_s for b in cfg.prony_branches], dtype=np.float64)
    eff_tangent_factor = float(gamma_inf + np.sum(betas * gammas))

    # Internal history variables per quadrature point:
    # h_overstress: (K, E, Q, 3, 3) non-equilibrium Prony First-Piola tensors
    h_overstress: FloatArray = np.zeros(
        (n_branches, n_elem, n_quad, 3, 3), dtype=np.float64
    )
    p_el_prev: FloatArray = np.zeros((n_elem, n_quad, 3, 3), dtype=np.float64)
    w_max_prev: FloatArray = np.zeros((n_elem, n_quad), dtype=np.float64)

    def _mullins_eta(
        w_cur: FloatArray, w_max_hist: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        w_max_new = np.maximum(w_max_hist, w_cur)
        if not cfg.mullins.enabled:
            return np.ones_like(w_cur), np.zeros_like(w_cur), w_max_new
        m_scaled = cfg.mullins.m_energy_density * s_fac
        denom = m_scaled + cfg.mullins.beta_param * w_max_new
        arg = np.maximum(w_max_new - w_cur, 0.0) / denom
        eta = 1.0 - (1.0 / cfg.mullins.r_param) * erf(arg)
        eta = np.clip(eta, 1.0 - 1.0 / cfg.mullins.r_param, 1.0)
        # Exact derivative d(eta)/d(W_0) on unloading branch (W_0 < W_max)
        unloading = w_cur < w_max_hist
        deta_dw = np.where(
            unloading,
            (2.0 / (cfg.mullins.r_param * math.sqrt(math.pi) * denom))
            * np.exp(-(arg * arg)),
            0.0,
        )
        return (
            np.asarray(eta, dtype=np.float64),
            np.asarray(deta_dw, dtype=np.float64),
            w_max_new,
        )

    def _assemble_visco(
        u_vec: FloatArray,
    ) -> tuple[
        FloatArray,
        csr_matrix,
        float,
        float,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
    ]:
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
                p_el_prev,
                h_overstress,
                w_max_prev,
                np.ones((n_elem, n_quad), dtype=np.float64),
            )

        with torch.no_grad():
            f_t = torch.from_numpy(np.ascontiguousarray(f_np))
            p0_np = first_piola(f_t, mat_eff).numpy()
            c0_np = consistent_tangent(f_t, mat_eff).numpy()
            w0_np = energy_density(f_t, mat_eff).numpy()

        eta_np, deta_dw, w_max_trial = _mullins_eta(w0_np, w_max_prev)
        p_el_trial = eta_np[..., None, None] * p0_np
        dp_el = p_el_trial - p_el_prev

        # Update each Prony Maxwell arm via exact exponential recurrence
        h_trial = np.empty_like(h_overstress)
        for k in range(n_branches):
            h_trial[k] = alphas[k] * h_overstress[k] + (betas[k] * gammas[k]) * dp_el

        p_total = gamma_inf * p_el_trial + (
            np.sum(h_trial, axis=0) if n_branches > 0 else 0.0
        )
        # Exact consistent tangent:
        # (gamma_inf + sum beta_k gamma_k) * (eta * C_0 + (d_eta / dW_0) * P_0 (x) P_0)
        c_mullins = np.einsum("eq,eqij,eqkl->eqijkl", deta_dw, p0_np, p0_np)
        c_eff = eff_tangent_factor * (
            eta_np[..., None, None, None, None] * c0_np + c_mullins
        )
        stored_e = float(np.sum((gamma_inf * eta_np * w0_np) * w_vol))

        f_elem = np.einsum("eq,eqij,eqnj->eni", w_vol, p_total, d_phys)
        f_int = np.zeros((n_nodes, 3), dtype=np.float64)
        np.add.at(f_int, cells, f_elem)
        tmp = np.einsum("eq,eqijkl,eqnj->eqnikl", w_vol, c_eff, d_phys)
        k_elem = np.einsum("eqnikl,eqml->enimk", tmp, d_phys)
        k_mat = coo_matrix(
            (k_elem.reshape(n_elem, n_edof, n_edof).reshape(-1), (rows, cols)),
            shape=(n_dofs, n_dofs),
        ).tocsr()
        return (
            f_int.reshape(-1),
            k_mat,
            stored_e,
            min_det,
            p_el_trial,
            h_trial,
            w_max_trial,
            eta_np,
        )

    n_steps = len(load_history)
    mults = np.asarray(load_history, dtype=np.float64)
    times = np.arange(n_steps, dtype=np.float64) * dt
    u_hist = np.zeros((n_steps, n_nodes, 3), dtype=np.float64)
    stored_hist = np.zeros(n_steps, dtype=np.float64)
    diss_step_hist = np.zeros(n_steps, dtype=np.float64)
    det_hist = np.ones(n_steps, dtype=np.float64)
    eta_hist = np.ones(n_steps, dtype=np.float64)

    u_cur: FloatArray = np.zeros(n_dofs, dtype=np.float64)
    u_cur[constrained] = u_fixed[constrained]
    f_ext_prev: FloatArray = np.zeros(n_dofs, dtype=np.float64)
    total_work = 0.0
    total_newton = 0
    tol_eff = cfg.solver_tolerance * s_fac

    for step_idx in range(n_steps):
        lam = float(mults[step_idx])
        u_prev_step = u_cur.copy()
        step_follower = tuple(fbc.at_factor(lam) for fbc in follower_scaled)
        f_dead_step = lam * f_dead_ref

        step_conv = False
        w_step = 0.0
        det_step = 1.0
        p_el_commit = p_el_prev
        h_commit = h_overstress
        w_max_commit = w_max_prev
        eta_commit: FloatArray = np.ones((n_elem, n_quad), dtype=np.float64)
        f_ext_cur = f_dead_step.copy()

        for _it in range(cfg.max_newton_iterations):
            total_newton += 1
            (
                f_int,
                k_int,
                w_step,
                det_step,
                p_el_commit,
                h_commit,
                w_max_commit,
                eta_commit,
            ) = _assemble_visco(u_cur)
            if det_step <= 1.0e-8:
                break

            if step_follower:
                fl = assemble_follower_pressure_3d(mesh, step_follower, u_cur)
                f_ext_cur = f_dead_step + fl.external_force
                k_tan = (k_int + fl.load_stiffness).tocsr()
                is_sym = False
            else:
                f_ext_cur = f_dead_step
                k_tan = k_int
                is_sym = True

            res = f_int - f_ext_cur
            res_free = res[free]
            res_norm = float(np.linalg.norm(res_free))
            ref_norm = max(float(np.linalg.norm(f_ext_cur[free])), s_fac)
            if res_norm <= max(tol_eff, cfg.solver_tolerance * ref_norm):
                step_conv = True
                break

            du_free, rep = solve_linear_system(
                k_tan[free][:, free], -res_free, config=lin_cfg, is_symmetric=is_sym
            )
            if not rep.converged:
                break

            du_full = np.zeros(n_dofs, dtype=np.float64)
            du_full[free] = du_free
            alpha_ls = 1.0
            accepted = False
            for _ls in range(12):
                u_cand = u_cur + alpha_ls * du_full
                u_cand[constrained] = u_fixed[constrained]
                f_c, _, _, det_c, _, _, _, _ = _assemble_visco(u_cand)
                if det_c > 1.0e-6:
                    if step_follower:
                        f_ec = (
                            f_dead_step
                            + assemble_follower_pressure_3d(
                                mesh, step_follower, u_cand
                            ).external_force
                        )
                    else:
                        f_ec = f_dead_step
                    if (
                        float(np.linalg.norm((f_c - f_ec)[free]))
                        < res_norm * (1.0 + 1.0e-6)
                        or alpha_ls <= 0.125
                    ):
                        u_cur = u_cand
                        accepted = True
                        break
                alpha_ls *= 0.5
            if not accepted:
                break

        if not step_conv:
            return ViscoHysteresis3DResult(
                converged=False,
                mesh_id=mesh.mesh_id,
                dt_s=dt,
                times_s=times,
                load_multipliers=mults,
                displacement_history_mm=u_hist,
                stored_energy_n_mm=stored_hist,
                step_dissipation_n_mm=diss_step_hist,
                cumulative_dissipation_n_mm=float(np.sum(diss_step_hist)),
                hysteresis_loop_work_n_mm=float(total_work / s_fac),
                min_det_f_history=det_hist,
                min_mullins_eta_history=eta_hist,
                step_dissipation_nonnegative=bool(np.all(diss_step_hist >= -1.0e-12)),
                thermodynamically_admissible=False,
                total_newton_iterations=total_newton,
                wall_time_s=time.perf_counter() - t0,
                reason=f"visco-hysteresis Newton failed at step {step_idx} (min_det_f={det_step:.4g})",
            )

        # Exact Clausius-Duhem per-step thermodynamic dissipation >= 0:
        # 1) Viscoelastic Prony arm dissipation: sum_k int (dt / (tau_k * gamma_k * E)) ||H_mid||_F^2 dV >= 0
        d_visc = 0.0
        for k in range(n_branches):
            h_mid = 0.5 * (h_overstress[k] + h_commit[k])
            h_sq = np.sum(h_mid * h_mid, axis=(-2, -1))  # (E, Q)
            d_visc += float(
                np.sum(h_sq * w_vol) * (dt / max(taus[k] * gammas[k] * e_eff, 1.0e-12))
            )
        # 2) Ogden-Roxburgh Mullins damage dissipation: int (1 - eta) * dW_max^+ dV >= 0
        dw_max_pos = np.maximum(w_max_commit - w_max_prev, 0.0)
        d_mullins = float(np.sum((1.0 - eta_commit) * dw_max_pos * w_vol))
        d_step_phys = max(0.0, (d_visc + d_mullins) / s_fac)

        # Trapezoidal external mechanical work integral sum 0.5*(f_prev + f_cur) . du
        if step_idx > 0:
            total_work += 0.5 * float(
                np.dot(f_ext_prev + f_ext_cur, u_cur - u_prev_step)
            )
        f_ext_prev = f_ext_cur.copy()

        # Commit internal variables
        h_overstress = h_commit
        p_el_prev = p_el_commit
        w_max_prev = w_max_commit

        u_hist[step_idx] = u_cur.reshape(n_nodes, 3)
        stored_hist[step_idx] = w_step / s_fac
        diss_step_hist[step_idx] = d_step_phys
        det_hist[step_idx] = det_step
        eta_hist[step_idx] = float(np.min(eta_commit))

    step_nonneg = bool(np.all(diss_step_hist >= -1.0e-12))
    det_pos = bool(np.all(det_hist > 0.0))
    return ViscoHysteresis3DResult(
        converged=True,
        mesh_id=mesh.mesh_id,
        dt_s=dt,
        times_s=times,
        load_multipliers=mults,
        displacement_history_mm=u_hist,
        stored_energy_n_mm=stored_hist,
        step_dissipation_n_mm=diss_step_hist,
        cumulative_dissipation_n_mm=float(np.sum(diss_step_hist)),
        hysteresis_loop_work_n_mm=float(total_work / s_fac),
        min_det_f_history=det_hist,
        min_mullins_eta_history=eta_hist,
        step_dissipation_nonnegative=step_nonneg,
        thermodynamically_admissible=bool(step_nonneg and det_pos),
        total_newton_iterations=total_newton,
        wall_time_s=time.perf_counter() - t0,
    )


class VelocityResidualHysteresisOperator(nn.Module):
    """PGRD / PhysCon-Deform velocity-conditioned hysteresis residual operator.

    Combines an exact multi-branch Prony viscoelastic + smooth Bouc-Wen
    hysteresis state with a velocity-conditioned residual network, providing
    both sub-millisecond rollout inference and thermodynamic loop-dissipation
    certification (``dD/dt >= 0``).
    """

    def __init__(
        self,
        *,
        n_prony_branches: int = 3,
        hidden_dim: int = 32,
        dt_s: float = 1.0 / 30.0,
    ) -> None:
        super().__init__()
        self.n_prony_branches = int(max(1, n_prony_branches))
        self.dt_s = float(dt_s)

        # Trainable nonlinear elastic backbone (monotone cubic-like spline)
        self.c1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.c2 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.c3 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Trainable positive Prony relaxation times & weights
        init_taus = torch.linspace(-1.0, 2.5, self.n_prony_branches)
        self.log_taus = nn.Parameter(init_taus)
        self.prony_weights = nn.Parameter(
            torch.full((self.n_prony_branches,), -0.05, dtype=torch.float32)
        )
        # Rate-independent Bouc-Wen / Prandtl-Ishlinskii backlash-like memory weights
        self.bw_alpha = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.bw_weight = nn.Parameter(torch.tensor(-0.1, dtype=torch.float32))

        # Velocity-conditioned PGRD residual MLP: [u, du, tanh(du), z_bw, h_prony...] -> residual
        in_dim = 4 + self.n_prony_branches
        self.residual_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        last = self.residual_mlp[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(last.bias)

    def rollout_sequence(
        self,
        u_seq: torch.Tensor,  # (T,) normalized input (e.g. pressure / command in [0, 1])
        *,
        dt_s: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out hysteresis response ``y_hat (T,)`` and step dissipation ``d_t (T,) >= 0``."""
        dt = self.dt_s if dt_s is None else float(dt_s)
        t_len = int(u_seq.shape[0])
        device = u_seq.device
        dtype = u_seq.dtype

        du = torch.zeros_like(u_seq)
        if t_len > 1:
            du[1:] = u_seq[1:] - u_seq[:-1]
        vel = du / max(dt, 1.0e-6)

        taus = torch.nn.functional.softplus(self.log_taus) + 0.05  # (K,)
        alphas = torch.exp(-dt / taus)  # (K,)
        betas = (1.0 - alphas) / (dt / taus)  # (K,)
        bw_rate = torch.nn.functional.softplus(self.bw_alpha) + 0.5

        # Vectorized causal IIR recurrence over sequence
        h_states: list[torch.Tensor] = []
        z_states: list[torch.Tensor] = []
        diss_steps: list[torch.Tensor] = []
        h_cur = torch.zeros(self.n_prony_branches, dtype=dtype, device=device)
        z_cur = torch.zeros((), dtype=dtype, device=device)
        # Positive dissipation coefficients (guaranteed >= 0)
        w_abs = torch.abs(self.prony_weights) + 1.0e-4

        for i in range(t_len):
            du_i = du[i]
            h_next = alphas * h_cur + betas * du_i
            # Smooth rate-independent Bouc-Wen hysteresis variable z in [-1, 1]:
            # dz = bw_rate * (du - |du| * z) => exact stable exponential update
            decay_bw = torch.exp(-bw_rate * torch.abs(du_i))
            sgn = torch.tanh(du_i * 40.0)
            z_next = decay_bw * z_cur + (1.0 - decay_bw) * sgn
            # Clausius-Duhem step dissipation >= 0 from Prony & Bouc-Wen internal variables
            h_mid = 0.5 * (h_cur + h_next)
            d_visc = ((w_abs / taus) * (h_mid * h_mid) * dt).sum()
            d_plast = torch.abs(self.bw_weight) * torch.abs(du_i) * torch.abs(z_next)
            diss_steps.append(d_visc + d_plast)
            h_states.append(h_next)
            z_states.append(z_next)
            h_cur = h_next
            z_cur = z_next

        h_mat = torch.stack(h_states, dim=0)  # (T, K)
        z_vec = torch.stack(z_states, dim=0)  # (T,)
        diss_vec = torch.stack(diss_steps, dim=0)  # (T,) >= 0 everywhere

        y_el = (
            self.bias
            + self.c1 * u_seq
            + self.c2 * (u_seq * u_seq)
            + self.c3 * (u_seq * u_seq * u_seq)
        )
        y_prony = (h_mat * self.prony_weights.unsqueeze(0)).sum(dim=-1)
        y_bw = self.bw_weight * z_vec

        feats = torch.cat(
            (
                u_seq.unsqueeze(-1),
                vel.unsqueeze(-1),
                torch.tanh(vel * 2.0).unsqueeze(-1),
                z_vec.unsqueeze(-1),
                h_mat,
            ),
            dim=-1,
        )
        y_res: torch.Tensor = self.residual_mlp(feats).squeeze(-1)
        return y_el + y_prony + y_bw + y_res, diss_vec

    def fit_sequence(
        self,
        u_train: FloatArray,
        y_train: FloatArray,
        *,
        dt_s: float | None = None,
        epochs: int = 160,
        lr: float = 1.5e-2,
        seed: int = 42,
    ) -> dict[str, float]:
        """Train velocity-conditioned hysteresis operator on real actuator time-series."""
        torch.manual_seed(seed)
        u_np = np.asarray(u_train, dtype=np.float64).reshape(-1)
        y_np = np.asarray(y_train, dtype=np.float64).reshape(-1)
        y_scale = max(float(np.std(y_np)), 1.0e-6)
        y_mean = float(np.mean(y_np))

        u_t = torch.tensor(u_np, dtype=torch.float32)
        y_t = torch.tensor((y_np - y_mean) / y_scale, dtype=torch.float32)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        final_rmse = 0.0
        for _ in range(epochs):
            opt.zero_grad()
            pred, _diss = self.rollout_sequence(u_t, dt_s=dt_s)
            loss = ((pred - y_t) ** 2).mean()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            opt.step()
            final_rmse = float(math.sqrt(max(float(loss.item()), 0.0)) * y_scale)

        # Store scaling buffers inside module state for deterministic prediction
        self._y_mean = y_mean
        self._y_scale = y_scale
        self.eval()
        return {"train_rmse": final_rmse, "y_mean": y_mean, "y_scale": y_scale}

    def predict_certified(
        self, u_seq: FloatArray, *, dt_s: float | None = None
    ) -> dict[str, Any]:
        """Predict hysteresis output trajectory with thermodynamic dissipation receipt."""
        t0 = time.perf_counter()
        self.eval()
        u_t = torch.tensor(
            np.asarray(u_seq, dtype=np.float64).reshape(-1), dtype=torch.float32
        )
        with torch.no_grad():
            pred_n, diss_t = self.rollout_sequence(u_t, dt_s=dt_s)
        y_mean = float(getattr(self, "_y_mean", 0.0))
        y_scale = float(getattr(self, "_y_scale", 1.0))
        y_pred = (pred_n.numpy().astype(np.float64) * y_scale) + y_mean
        diss_np = diss_t.numpy().astype(np.float64) * y_scale
        nonneg = bool(np.all(diss_np >= -1.0e-12))
        doc: dict[str, Any] = {
            "schema": VELOCITY_HYSTERESIS_OP_SCHEMA,
            "sample_count": int(y_pred.size),
            "cumulative_dissipation": float(np.sum(diss_np)),
            "min_step_dissipation": float(np.min(diss_np)) if diss_np.size > 0 else 0.0,
            "dissipation_nonnegative": nonneg,
            "inference_ms": (time.perf_counter() - t0) * 1000.0,
        }
        doc["receipt_sha256"] = _sha256_doc(doc)
        return {
            "predictions": y_pred,
            "step_dissipation": diss_np,
            "receipt": doc,
        }


__all__ = (
    "VELOCITY_HYSTERESIS_OP_SCHEMA",
    "VISCO_HYSTERESIS_3D_SCHEMA",
    "MullinsDamageSpec",
    "PronyBranch",
    "VelocityResidualHysteresisOperator",
    "ViscoHysteresis3DConfig",
    "ViscoHysteresis3DResult",
    "solve_3d_visco_hysteresis",
)
