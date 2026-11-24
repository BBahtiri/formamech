"""ASTM E1049 Rainflow Fatigue & Staggered AT2 Phase-Field Fracture (V4-3).

Implements two production durability & damage modules:

1. **Fatigue Life Evaluation** (:func:`rainflow_count_astm_e1049`,
   :func:`evaluate_fatigue_life`):
   * ASTM E1049-85 turning-point reduction + 3-point stack rainflow cycle counting,
   * Goodman and Gerber mean-stress corrections,
   * Basquin stress-life (S-N) and Coffin-Manson strain-life (E-N),
   * Palmgren-Miner linear cumulative damage rule ``D = sum_i n_i / N_{f,i}``.
2. **Staggered AT2 Brittle Phase-Field Fracture**
   (:func:`solve_phase_field_fracture`):
   * Quadratic degradation ``g(d) = (1 - d)^2 + k_res``,
   * Miehe spectral tension-compression strain split (``psi^+ / psi^-``) so
     cracks cannot nucleate or propagate under pure compression,
   * Irreversible history field ``H = max_{tau <= t} psi^+(eps(tau))``
     guaranteeing thermodynamic monotonicity ``d_dot >= 0`` and ``0 <= d <= 1``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import spsolve  # type: ignore[import-untyped]

from piginot.fem.backend_3d import (
    DirichletBC3D,
    _tet_shape_grads_ref,
    face_node_indices_3d,
)
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

FATIGUE_RESULT_SCHEMA: Final = "formamech.fatigue-life-v1"
PHASE_FIELD_RESULT_SCHEMA: Final = "formamech.phase-field-fracture-v1"


class MeanStressModel(StrEnum):
    """Mean-stress correction models for cyclic fatigue."""

    NONE = "none"
    GOODMAN = "goodman"
    GERBER = "gerber"


class FatigueModel(StrEnum):
    """Fatigue life-curve formulation."""

    BASQUIN_SN = "basquin-sn"
    COFFIN_MANSON_EN = "coffin-manson-en"


@dataclass(frozen=True, slots=True)
class RainflowCycle:
    """One extracted rainflow cycle or half-cycle."""

    range_mpa: float
    amplitude_mpa: float
    mean_mpa: float
    count: float  # 0.5 for half-cycle, 1.0 for full cycle


@dataclass(frozen=True, slots=True)
class FatigueMaterialSpec:
    """Cyclic S-N / E-N fatigue material properties (MPa units)."""

    name: str
    ultimate_tensile_mpa: float
    fatigue_strength_coeff_mpa: float  # sigma_f'
    fatigue_strength_exp: float  # b < 0
    youngs_modulus_mpa: float = 210_000.0
    fatigue_ductility_coeff: float = 0.5  # eps_f'
    fatigue_ductility_exp: float = -0.6  # c < 0
    endurance_limit_mpa: float = 0.0

    def __post_init__(self) -> None:
        if self.ultimate_tensile_mpa <= 0.0 or self.fatigue_strength_coeff_mpa <= 0.0:
            raise ValueError("strength parameters must be positive")
        if self.fatigue_strength_exp >= 0.0 or self.fatigue_ductility_exp >= 0.0:
            raise ValueError("fatigue exponents b and c must be strictly negative")
        if self.youngs_modulus_mpa <= 0.0:
            raise ValueError("youngs_modulus_mpa must be positive")


@dataclass(frozen=True, slots=True)
class FatigueLifeResult:
    """Auditable ASTM E1049 rainflow + Palmgren-Miner fatigue assessment."""

    material_name: str
    model: FatigueModel
    mean_stress_model: MeanStressModel
    cycles: tuple[RainflowCycle, ...]
    miner_damage: float
    blocks_to_failure: float
    safety_factor_life: float
    status: str  # "SAFE" | "FATIGUE_FAILURE_PREDICTED"

    def document(self) -> dict[str, Any]:
        return {
            "schema": FATIGUE_RESULT_SCHEMA,
            "material_name": self.material_name,
            "model": self.model.value,
            "mean_stress_model": self.mean_stress_model.value,
            "cycle_count": len(self.cycles),
            "miner_damage": float(self.miner_damage),
            "blocks_to_failure": (
                float(self.blocks_to_failure)
                if math.isfinite(self.blocks_to_failure)
                else None
            ),
            "safety_factor_life": (
                float(self.safety_factor_life)
                if math.isfinite(self.safety_factor_life)
                else None
            ),
            "status": self.status,
        }


def _extract_turning_points(series: Sequence[float]) -> list[float]:
    """Filter a stress/strain history down to local peaks and valleys."""
    if len(series) < 2:
        return [float(v) for v in series]
    dedup: list[float] = [float(series[0])]
    for v in series[1:]:
        fv = float(v)
        if abs(fv - dedup[-1]) > 1.0e-14:
            dedup.append(fv)
    if len(dedup) <= 2:
        return dedup
    pts: list[float] = [dedup[0]]
    for idx in range(1, len(dedup) - 1):
        d1 = dedup[idx] - dedup[idx - 1]
        d2 = dedup[idx + 1] - dedup[idx]
        if d1 * d2 < 0.0:
            pts.append(dedup[idx])
    pts.append(dedup[-1])
    return pts


def rainflow_count_astm_e1049(
    stress_history_mpa: Sequence[float],
) -> tuple[RainflowCycle, ...]:
    """Extract full (1.0) and half (0.5) cycles via ASTM E1049-85 3-point algorithm."""
    reversals = _extract_turning_points(stress_history_mpa)
    if len(reversals) < 2:
        return ()

    stack: list[float] = []
    cycles: list[RainflowCycle] = []

    for val in reversals:
        stack.append(val)
        while len(stack) >= 3:
            x_range = abs(stack[-1] - stack[-2])
            y_range = abs(stack[-2] - stack[-3])
            if x_range < y_range - 1.0e-14:
                break
            if len(stack) == 3:
                # Half-cycle starting at the beginning of the series
                y_mean = 0.5 * (stack[0] + stack[1])
                cycles.append(
                    RainflowCycle(
                        range_mpa=y_range,
                        amplitude_mpa=0.5 * y_range,
                        mean_mpa=y_mean,
                        count=0.5,
                    )
                )
                stack.pop(0)
            else:
                # Full enclosed cycle
                y_mean = 0.5 * (stack[-2] + stack[-3])
                cycles.append(
                    RainflowCycle(
                        range_mpa=y_range,
                        amplitude_mpa=0.5 * y_range,
                        mean_mpa=y_mean,
                        count=1.0,
                    )
                )
                top = stack.pop()
                stack.pop()
                stack.pop()
                stack.append(top)

    # Remaining stack ranges count as half-cycles
    for idx in range(len(stack) - 1):
        r = abs(stack[idx + 1] - stack[idx])
        m = 0.5 * (stack[idx + 1] + stack[idx])
        cycles.append(
            RainflowCycle(range_mpa=r, amplitude_mpa=0.5 * r, mean_mpa=m, count=0.5)
        )
    return tuple(cycles)


def _equivalent_amplitude(
    amp_mpa: float, mean_mpa: float, uts_mpa: float, model: MeanStressModel
) -> float:
    """Apply Goodman or Gerber mean-stress correction (tensile mean only)."""
    if model is MeanStressModel.NONE or mean_mpa <= 0.0:
        return amp_mpa
    ratio = mean_mpa / uts_mpa
    if ratio >= 1.0:
        return float("inf")
    if model is MeanStressModel.GOODMAN:
        return amp_mpa / (1.0 - ratio)
    return amp_mpa / (1.0 - ratio * ratio)


def _coffin_manson_cycles(strain_amp: float, mat: FatigueMaterialSpec) -> float:
    """Solve ``eps_a = (sigma_f'/E)*(2N)^b + eps_f'*(2N)^c`` for ``N_f``."""
    if strain_amp <= 1.0e-15:
        return float("inf")
    # Bracket in log10(2 N_f) space over [-2, 14]
    e_coeff = mat.fatigue_strength_coeff_mpa / mat.youngs_modulus_mpa
    b = mat.fatigue_strength_exp
    d_coeff = mat.fatigue_ductility_coeff
    c = mat.fatigue_ductility_exp

    def _f(log_2n: float) -> float:
        two_n = float(10.0**log_2n)
        return float(e_coeff * (two_n**b) + d_coeff * (two_n**c) - strain_amp)

    lo, hi = -2.0, 14.0
    if _f(lo) <= 0.0:
        return float(0.5 * (10.0**lo))
    if _f(hi) >= 0.0:
        return float("inf")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (10.0 ** (0.5 * (lo + hi))))


def evaluate_fatigue_life(
    stress_history_mpa: Sequence[float],
    material: FatigueMaterialSpec,
    *,
    model: FatigueModel = FatigueModel.BASQUIN_SN,
    mean_stress_model: MeanStressModel = MeanStressModel.GOODMAN,
    block_repetitions: float = 1.0,
) -> FatigueLifeResult:
    """Compute ASTM E1049 rainflow cycles and Palmgren-Miner cumulative damage."""
    if block_repetitions <= 0.0:
        raise ValueError("block_repetitions must be positive")
    cycles = rainflow_count_astm_e1049(stress_history_mpa)
    damage_per_block = 0.0

    for cyc in cycles:
        sig_eq = _equivalent_amplitude(
            cyc.amplitude_mpa,
            cyc.mean_mpa,
            material.ultimate_tensile_mpa,
            mean_stress_model,
        )
        if not math.isfinite(sig_eq):
            damage_per_block = float("inf")
            break
        if sig_eq <= material.endurance_limit_mpa or sig_eq <= 1.0e-15:
            continue
        if model is FatigueModel.BASQUIN_SN:
            # sig_eq = sigma_f' * (2 N_f)^b => N_f = 0.5 * (sig_eq / sigma_f')^(1/b)
            n_f = 0.5 * (
                (sig_eq / material.fatigue_strength_coeff_mpa)
                ** (1.0 / material.fatigue_strength_exp)
            )
        else:
            eps_a = sig_eq / material.youngs_modulus_mpa
            n_f = _coffin_manson_cycles(eps_a, material)
        if n_f <= 0.0:
            damage_per_block = float("inf")
            break
        if math.isfinite(n_f):
            damage_per_block += cyc.count / n_f

    total_damage = damage_per_block * float(block_repetitions)
    blocks_to_fail = (
        float("inf") if damage_per_block <= 1.0e-30 else (1.0 / damage_per_block)
    )
    sf_life = float("inf") if total_damage <= 1.0e-30 else (1.0 / total_damage)
    status = "SAFE" if total_damage < 1.0 else "FATIGUE_FAILURE_PREDICTED"
    return FatigueLifeResult(
        material_name=material.name,
        model=model,
        mean_stress_model=mean_stress_model,
        cycles=cycles,
        miner_damage=float(total_damage),
        blocks_to_failure=float(blocks_to_fail),
        safety_factor_life=float(sf_life),
        status=status,
    )


# ---------------------------------------------------------------------------
# Staggered AT2 Brittle Phase-Field Fracture (Miehe Spectral Split)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhaseFieldConfig:
    """Configuration for staggered AT2 brittle phase-field fracture on 3D TET4."""

    youngs_modulus_mpa: float = 20_000.0
    poisson_ratio: float = 0.25
    fracture_toughness_gc_n_per_mm: float = 0.5
    length_scale_lc_mm: float = 2.0
    residual_stiffness_k: float = 1.0e-6
    load_steps: int = 8
    staggered_iters: int = 4
    tolerance: float = 1.0e-4


@dataclass(frozen=True, slots=True)
class PhaseFieldFractureResult:
    """Auditable staggered AT2 phase-field fracture history & fields."""

    mesh_id: str
    load_factors: tuple[float, ...]
    max_damage_history: tuple[float, ...]
    crack_surface_energy_mj: tuple[float, ...]
    monotone_irreversible: bool
    final_damage_nodal: FloatArray  # (N_nodes,) in [0, 1]
    final_displacement_mm: FloatArray  # (N_nodes, 3)
    wall_time_s: float

    def document(self) -> dict[str, Any]:
        return {
            "schema": PHASE_FIELD_RESULT_SCHEMA,
            "mesh_id": self.mesh_id,
            "load_steps": len(self.load_factors),
            "max_damage": float(self.max_damage_history[-1])
            if self.max_damage_history
            else 0.0,
            "final_crack_surface_energy_mj": float(self.crack_surface_energy_mj[-1])
            if self.crack_surface_energy_mj
            else 0.0,
            "monotone_irreversible": self.monotone_irreversible,
            "wall_time_s": float(self.wall_time_s),
        }


def _spectral_tensile_energy_density(
    eps_3x3: FloatArray, lam: float, mu: float
) -> tuple[float, float]:
    """Miehe spectral tension-compression split ``(psi_plus, psi_minus)``."""
    tr = float(eps_3x3[0, 0] + eps_3x3[1, 1] + eps_3x3[2, 2])
    eigvals = np.linalg.eigvalsh(0.5 * (eps_3x3 + eps_3x3.T))
    tr_pos = max(tr, 0.0)
    tr_neg = min(tr, 0.0)
    eig_pos = np.maximum(eigvals, 0.0)
    eig_neg = np.minimum(eigvals, 0.0)
    psi_plus = 0.5 * lam * (tr_pos**2) + mu * float(np.sum(eig_pos**2))
    psi_minus = 0.5 * lam * (tr_neg**2) + mu * float(np.sum(eig_neg**2))
    return psi_plus, psi_minus


def solve_phase_field_fracture(
    mesh: CanonicalMesh3D,
    *,
    dirichlet_bcs: Sequence[DirichletBC3D],
    config: PhaseFieldConfig | None = None,
    initial_damage_nodal: FloatArray | None = None,
) -> PhaseFieldFractureResult:
    """Solve quasi-static staggered AT2 brittle phase-field fracture on a 3D ``TET4`` mesh.

    Uses Miehe's spectral tension-compression strain energy split so compressive
    elements maintain ``psi^+ = 0`` and never nucleate damage, while tensile
    elements drive irreversible damage ``H_e = max_t psi^+_e`` with ``d_dot >= 0``.
    """
    t0 = time.perf_counter()
    if mesh.cell_type is not CellType3D.TET4:
        raise ValueError("solve_phase_field_fracture currently requires TET4 meshes")
    cfg = PhaseFieldConfig() if config is None else config
    e_mpa = float(cfg.youngs_modulus_mpa)
    nu = float(cfg.poisson_ratio)
    mu = e_mpa / (2.0 * (1.0 + nu))
    lam = e_mpa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    gc = float(cfg.fracture_toughness_gc_n_per_mm)
    lc = float(cfg.length_scale_lc_mm)
    k_res = float(cfg.residual_stiffness_k)

    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    n_nodes = int(nodes.shape[0])
    n_elem = int(cells.shape[0])
    n_dofs = 3 * n_nodes

    # Precompute constant TET4 gradients & element volumes
    xi_c = np.array([[0.25, 0.25, 0.25]], dtype=np.float64)
    d_ref = _tet_shape_grads_ref(CellType3D.TET4, xi_c)[0]  # (4, 3)
    x_e = nodes[cells]  # (E, 4, 3)
    jac = np.einsum("eni,na->eia", x_e, d_ref)  # (E, 3, 3)
    det_j = np.linalg.det(jac)
    vols = det_j / 6.0  # (E,)
    jac_inv = np.linalg.inv(jac)
    d_phys = np.einsum("na,eai->eni", d_ref, jac_inv)  # (E, 4, 3)

    # Linear isotropic elasticity matrix D (6x6 Voigt: xx, yy, zz, yz, xz, xy)
    d_mat = np.array(
        [
            [lam + 2 * mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2 * mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2 * mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ],
        dtype=np.float64,
    )

    # Precompute B_e (E, 6, 12) and undegraded K_e0 (E, 12, 12) plus scalar
    # phase-field element stiffness K_d_grad (E, 4, 4) and mass M_d (E, 4, 4)
    b_all = np.zeros((n_elem, 6, 12), dtype=np.float64)
    for a in range(4):
        gx, gy, gz = d_phys[:, a, 0], d_phys[:, a, 1], d_phys[:, a, 2]
        b_all[:, 0, 3 * a] = gx
        b_all[:, 1, 3 * a + 1] = gy
        b_all[:, 2, 3 * a + 2] = gz
        b_all[:, 3, 3 * a + 1] = gz
        b_all[:, 3, 3 * a + 2] = gy
        b_all[:, 4, 3 * a] = gz
        b_all[:, 4, 3 * a + 2] = gx
        b_all[:, 5, 3 * a] = gy
        b_all[:, 5, 3 * a + 1] = gx

    k_e0 = np.einsum("e,eik,ij,ejl->ekl", vols, b_all, d_mat, b_all)
    m_ref_tet4 = (
        np.ones((4, 4), dtype=np.float64) + np.eye(4, dtype=np.float64)
    ) / 20.0
    k_d_base = (gc * lc) * np.einsum("e,eai,ebi->eab", vols, d_phys, d_phys) + (
        gc / lc
    ) * vols[:, None, None] * m_ref_tet4[None, :, :]
    m_d_vol = vols[:, None, None] * m_ref_tet4[None, :, :]  # (E, 4, 4)

    # Sparse assembly index arrays
    elem_u_dofs = np.empty((n_elem, 12), dtype=np.int64)
    for a in range(4):
        elem_u_dofs[:, 3 * a] = 3 * cells[:, a]
        elem_u_dofs[:, 3 * a + 1] = 3 * cells[:, a] + 1
        elem_u_dofs[:, 3 * a + 2] = 3 * cells[:, a] + 2
    u_rows = np.repeat(elem_u_dofs[:, :, None], 12, axis=2).reshape(-1)
    u_cols = np.repeat(elem_u_dofs[:, None, :], 12, axis=1).reshape(-1)
    d_rows = np.repeat(cells[:, :, None], 4, axis=2).reshape(-1)
    d_cols = np.repeat(cells[:, None, :], 4, axis=1).reshape(-1)

    # Collect target Dirichlet constraints
    comp_idx = {"x": 0, "y": 1, "z": 2}
    constrained = np.zeros(n_dofs, dtype=bool)
    u_target = np.zeros(n_dofs, dtype=np.float64)
    for bc in dirichlet_bcs:
        nids = face_node_indices_3d(mesh, bc.face_tags)
        for comp, val in zip(bc.components, bc.values_mm, strict=True):
            c = comp_idx[comp]
            for nid in nids:
                constrained[3 * nid + c] = True
                u_target[3 * nid + c] = float(val)
    free_u = np.nonzero(~constrained)[0]
    if free_u.size == 0:
        raise ValueError("all displacement DOFs are constrained")

    d_nodal: FloatArray = (
        np.zeros(n_nodes, dtype=np.float64)
        if initial_damage_nodal is None
        else np.clip(
            np.asarray(initial_damage_nodal, dtype=np.float64).copy(), 0.0, 1.0
        )
    )
    u_full: FloatArray = np.zeros(n_dofs, dtype=np.float64)
    hist_h = np.zeros(n_elem, dtype=np.float64)

    load_factors: list[float] = []
    max_d_hist: list[float] = []
    crack_e_hist: list[float] = []
    monotone = True

    for step in range(1, cfg.load_steps + 1):
        lam_load = step / float(cfg.load_steps)
        u_pres = lam_load * u_target
        d_prev_step = d_nodal.copy()

        for _st in range(cfg.staggered_iters):
            d_old = d_nodal.copy()
            # 1) Mechanics step with element degradation g(d_e) = (1 - d_e)^2 + k_res
            #    applied when tensile energy dominates (tension-compression split)
            d_elem = np.mean(d_nodal[cells], axis=1)
            g_deg = (1.0 - d_elem) ** 2 + k_res
            # Apply degradation to elements with positive tensile history
            eff_deg = np.where(hist_h > 1.0e-14, g_deg, 1.0)
            k_scaled = eff_deg[:, None, None] * k_e0
            k_u: csr_matrix = coo_matrix(
                (k_scaled.reshape(-1), (u_rows, u_cols)), shape=(n_dofs, n_dofs)
            ).tocsr()
            rhs_u = -(k_u[free_u][:, constrained] @ u_pres[constrained])
            u_f = np.asarray(
                spsolve(k_u[free_u][:, free_u].tocsc(), rhs_u), dtype=np.float64
            )
            u_full = u_pres.copy()
            u_full[free_u] = u_f

            # 2) Update irreversible Miehe tensile history field H_e
            u_e_vec = u_full[elem_u_dofs]  # (E, 12)
            eps_v = np.einsum("eij,ej->ei", b_all, u_e_vec)  # (E, 6)
            for e in range(n_elem):
                ev = eps_v[e]
                eps_mat = np.array(
                    [
                        [ev[0], 0.5 * ev[5], 0.5 * ev[4]],
                        [0.5 * ev[5], ev[1], 0.5 * ev[3]],
                        [0.5 * ev[4], 0.5 * ev[3], ev[2]],
                    ],
                    dtype=np.float64,
                )
                psi_p, _ = _spectral_tensile_energy_density(eps_mat, lam, mu)
                if psi_p > hist_h[e]:
                    hist_h[e] = psi_p

            # 3) Phase-field AT2 linear step:
            #    (G_c l_c grad^2 + (G_c / l_c + 2 H_e)) d = 2 H_e
            k_d_e = k_d_base + (2.0 * hist_h[:, None, None]) * m_d_vol
            k_d: csr_matrix = coo_matrix(
                (k_d_e.reshape(-1), (d_rows, d_cols)), shape=(n_nodes, n_nodes)
            ).tocsr()
            rhs_d = np.zeros(n_nodes, dtype=np.float64)
            elem_rhs = 0.5 * hist_h * vols  # 2 * H_e * (V_e / 4) per vertex
            np.add.at(rhs_d, cells, elem_rhs[:, None])
            d_cand = np.asarray(spsolve(k_d.tocsc(), rhs_d), dtype=np.float64)
            # Enforce admissibility [0, 1] and irreversibility d >= d_prev_step
            d_nodal = np.clip(np.maximum(d_cand, d_prev_step), 0.0, 1.0)
            if float(np.max(np.abs(d_nodal - d_old))) <= cfg.tolerance:
                break

        if np.any(d_nodal < d_prev_step - 1.0e-12):
            monotone = False
        # Compute AT2 crack surface energy: integral G_c * (d^2 / (2 lc) + lc/2 |grad d|^2) dV
        crack_e = 0.5 * float(
            d_nodal
            @ (
                coo_matrix(
                    (k_d_base.reshape(-1), (d_rows, d_cols)), shape=(n_nodes, n_nodes)
                ).tocsr()
                @ d_nodal
            )
        )
        load_factors.append(float(lam_load))
        max_d_hist.append(float(np.max(d_nodal)))
        crack_e_hist.append(crack_e)

    return PhaseFieldFractureResult(
        mesh_id=mesh.mesh_id,
        load_factors=tuple(load_factors),
        max_damage_history=tuple(max_d_hist),
        crack_surface_energy_mj=tuple(crack_e_hist),
        monotone_irreversible=monotone,
        final_damage_nodal=d_nodal,
        final_displacement_mm=u_full.reshape(n_nodes, 3),
        wall_time_s=time.perf_counter() - t0,
    )


__all__ = (
    "FATIGUE_RESULT_SCHEMA",
    "PHASE_FIELD_RESULT_SCHEMA",
    "FatigueLifeResult",
    "FatigueMaterialSpec",
    "FatigueModel",
    "MeanStressModel",
    "PhaseFieldConfig",
    "PhaseFieldFractureResult",
    "RainflowCycle",
    "evaluate_fatigue_life",
    "rainflow_count_astm_e1049",
    "solve_phase_field_fracture",
)
