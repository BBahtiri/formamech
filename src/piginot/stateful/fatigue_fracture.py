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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix, diags  # type: ignore[import-untyped]
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
        for name, val in (
            ("ultimate_tensile_mpa", self.ultimate_tensile_mpa),
            ("fatigue_strength_coeff_mpa", self.fatigue_strength_coeff_mpa),
            ("fatigue_strength_exp", self.fatigue_strength_exp),
            ("youngs_modulus_mpa", self.youngs_modulus_mpa),
            ("fatigue_ductility_coeff", self.fatigue_ductility_coeff),
            ("fatigue_ductility_exp", self.fatigue_ductility_exp),
            ("endurance_limit_mpa", self.endurance_limit_mpa),
        ):
            if not math.isfinite(float(val)):
                raise ValueError(f"FatigueMaterialSpec.{name} must be finite")
        if self.ultimate_tensile_mpa <= 0.0 or self.fatigue_strength_coeff_mpa <= 0.0:
            raise ValueError("strength parameters must be positive")
        if self.fatigue_strength_exp >= 0.0 or self.fatigue_ductility_exp >= 0.0:
            raise ValueError("fatigue exponents b and c must be strictly negative")
        if self.fatigue_ductility_coeff <= 0.0:
            raise ValueError("fatigue_ductility_coeff must be positive")
        if self.youngs_modulus_mpa <= 0.0:
            raise ValueError("youngs_modulus_mpa must be positive")
        if self.endurance_limit_mpa < 0.0:
            raise ValueError("endurance_limit_mpa must be nonnegative")


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
            "miner_damage": (
                float(self.miner_damage) if math.isfinite(self.miner_damage) else None
            ),
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
    vals = [float(v) for v in series]
    if any(not math.isfinite(v) for v in vals):
        raise ValueError("stress_history_mpa must contain only finite numbers")
    if len(vals) < 2:
        return vals
    dedup: list[float] = [vals[0]]
    for fv in vals[1:]:
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


def _rearrange_repeating_block(reversals: list[float]) -> list[float]:
    """Rearrange a periodic block to start & end at its global maximum (ASTM E1049 §5.4.4)."""
    pts = list(reversals)
    if len(pts) < 3:
        return pts
    if abs(pts[-1] - pts[0]) <= 1.0e-14:
        pts.pop()
    # Prune circular non-turning points across the periodic wrap-around boundary
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        n = len(pts)
        for i in range(n):
            p_prev = pts[i - 1]
            p_curr = pts[i]
            p_next = pts[(i + 1) % n]
            if (p_curr - p_prev) * (p_next - p_curr) >= -1.0e-28:
                pts.pop(i)
                changed = True
                break
    if len(pts) < 2:
        return pts
    i_max = max(range(len(pts)), key=lambda i: pts[i])
    return pts[i_max:] + pts[:i_max] + [pts[i_max]]


def rainflow_count_astm_e1049(
    stress_history_mpa: Sequence[float],
    *,
    repeating_block: bool = False,
) -> tuple[RainflowCycle, ...]:
    """Extract full (1.0) and half (0.5) cycles via ASTM E1049-85 3-point algorithm."""
    reversals = _extract_turning_points(stress_history_mpa)
    if repeating_block:
        reversals = _rearrange_repeating_block(reversals)
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
                y_mean = 0.5 * (stack[0] + stack[1])
                if repeating_block:
                    # Rearranged periodic block starts/ends at global extremum:
                    # touching stack[0] closes a full cycle (ASTM E1049 §5.4.4).
                    cycles.append(
                        RainflowCycle(
                            range_mpa=y_range,
                            amplitude_mpa=0.5 * y_range,
                            mean_mpa=y_mean,
                            count=1.0,
                        )
                    )
                    stack.pop()
                    stack.pop()
                else:
                    # Half-cycle starting at the beginning of an open-ended series
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
    """Apply Goodman or Gerber mean-stress correction with static UTS rupture guard."""
    if mean_mpa + amp_mpa >= uts_mpa:
        return float("inf")
    if model is MeanStressModel.NONE or mean_mpa <= 0.0:
        return amp_mpa
    ratio = mean_mpa / uts_mpa
    if ratio >= 1.0:
        return float("inf")
    if model is MeanStressModel.GOODMAN:
        return amp_mpa / (1.0 - ratio)
    return amp_mpa / (1.0 - ratio * ratio)


def _neuber_elastoplastic_strain_amp(
    sig_eq_mpa: float, mat: FatigueMaterialSpec
) -> float:
    """Convert elastic nominal stress amplitude to local elastoplastic strain via Neuber + Ramberg-Osgood.

    Compatibility of Basquin and Coffin-Manson defines the cyclic Ramberg-Osgood curve::

        eps_a(sigma_a) = sigma_a / E + eps_f' * (sigma_a / sigma_f')^(c / b)

    Neuber's rule ``sigma_a * eps_a(sigma_a) = sig_eq^2 / E`` has a unique positive
    root ``sigma_a in (0, sig_eq]``, capturing cyclic plasticity in LCF while
    recovering ``eps_a -> sig_eq / E`` smoothly in HCF.
    """
    if sig_eq_mpa <= 1.0e-15:
        return 0.0
    e_mpa = mat.youngs_modulus_mpa
    target_energy = (sig_eq_mpa * sig_eq_mpa) / e_mpa
    exp_ratio = mat.fatigue_ductility_exp / mat.fatigue_strength_exp  # c / b > 0

    def _neuber_res(sig_a: float) -> float:
        if sig_a <= 0.0:
            return -target_energy
        eps_tot = sig_a / e_mpa + mat.fatigue_ductility_coeff * (
            (sig_a / mat.fatigue_strength_coeff_mpa) ** exp_ratio
        )
        return float(sig_a * eps_tot - target_energy)

    lo = 0.0
    hi = sig_eq_mpa
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _neuber_res(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    sig_local = max(0.5 * (lo + hi), 1.0e-30)
    return float(target_energy / sig_local)


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
    repeating_block: bool | None = None,
) -> FatigueLifeResult:
    """Compute ASTM E1049 rainflow cycles and Palmgren-Miner cumulative damage."""
    if not math.isfinite(block_repetitions) or block_repetitions <= 0.0:
        raise ValueError("block_repetitions must be finite and positive")
    use_repeating = (
        (block_repetitions > 1.0) if repeating_block is None else bool(repeating_block)
    )
    cycles = rainflow_count_astm_e1049(
        stress_history_mpa, repeating_block=use_repeating
    )
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
            eps_a = _neuber_elastoplastic_strain_amp(sig_eq, material)
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

    def __post_init__(self) -> None:
        if not math.isfinite(self.youngs_modulus_mpa) or self.youngs_modulus_mpa <= 0.0:
            raise ValueError("youngs_modulus_mpa must be finite and positive")
        if not math.isfinite(self.poisson_ratio) or not (
            -1.0 < self.poisson_ratio < 0.5
        ):
            raise ValueError("poisson_ratio must lie in (-1, 0.5)")
        if (
            not math.isfinite(self.fracture_toughness_gc_n_per_mm)
            or self.fracture_toughness_gc_n_per_mm <= 0.0
        ):
            raise ValueError(
                "fracture_toughness_gc_n_per_mm must be finite and positive"
            )
        if not math.isfinite(self.length_scale_lc_mm) or self.length_scale_lc_mm <= 0.0:
            raise ValueError("length_scale_lc_mm must be finite and positive")
        if not math.isfinite(self.residual_stiffness_k) or not (
            0.0 < self.residual_stiffness_k < 1.0
        ):
            raise ValueError("residual_stiffness_k must lie in (0, 1)")
        if type(self.load_steps) is not int or self.load_steps < 1:
            raise ValueError("load_steps must be an integer >= 1")
        if type(self.staggered_iters) is not int or self.staggered_iters < 1:
            raise ValueError("staggered_iters must be an integer >= 1")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")


PhaseFieldWarmStartPredictor = Callable[
    [float, FloatArray, FloatArray],
    tuple[FloatArray | None, FloatArray | None],
]


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
    total_newton_iterations: int = 0
    total_staggered_iterations: int = 0
    per_step_newton_iterations: tuple[int, ...] = ()
    per_step_staggered_iterations: tuple[int, ...] = ()
    reaction_force_history_n: tuple[float, ...] = ()
    damage_snapshots: tuple[FloatArray, ...] = ()
    displacement_snapshots: tuple[FloatArray, ...] = ()

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
            "total_newton_iterations": int(self.total_newton_iterations),
            "total_staggered_iterations": int(self.total_staggered_iterations),
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


def _miehe_spectral_stress_tangent_batch(
    eps_v: FloatArray,
    g_deg: FloatArray,
    lam: float,
    mu: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Exact vectorized Miehe (2010) spectral split: ``(sigma_v, d_tan, psi_plus, psi_tot)``.

    Uses Voigt order ``[xx, yy, zz, yz, xz, xy]`` with engineering shear strains.
    Degrades ONLY the tensile volumetric/principal parts ``sigma = g sigma^+ + sigma^-``
    so compressed elements retain 100% compressive stiffness and pre-cracks act on step 0.
    """
    n_elem = int(eps_v.shape[0])
    eps_mat = np.empty((n_elem, 3, 3), dtype=np.float64)
    eps_mat[:, 0, 0] = eps_v[:, 0]
    eps_mat[:, 1, 1] = eps_v[:, 1]
    eps_mat[:, 2, 2] = eps_v[:, 2]
    eps_mat[:, 1, 2] = eps_mat[:, 2, 1] = 0.5 * eps_v[:, 3]
    eps_mat[:, 0, 2] = eps_mat[:, 2, 0] = 0.5 * eps_v[:, 4]
    eps_mat[:, 0, 1] = eps_mat[:, 1, 0] = 0.5 * eps_v[:, 5]

    eigvals, eigvecs = np.linalg.eigh(eps_mat)  # (E, 3), (E, 3, 3) cols=n_a
    tr = eps_v[:, 0] + eps_v[:, 1] + eps_v[:, 2]
    tr_pos = np.maximum(tr, 0.0)
    tr_neg = np.minimum(tr, 0.0)
    h_tr_pos = (tr > 0.0).astype(np.float64)
    h_tr_neg = 1.0 - h_tr_pos

    ep_pos = np.maximum(eigvals, 0.0)
    ep_neg = np.minimum(eigvals, 0.0)
    psi_plus = 0.5 * lam * (tr_pos**2) + mu * np.sum(ep_pos**2, axis=1)
    psi_minus = 0.5 * lam * (tr_neg**2) + mu * np.sum(ep_neg**2, axis=1)
    psi_tot = g_deg * psi_plus + psi_minus

    g_col = g_deg[:, None]
    phi_e = g_col * ep_pos + ep_neg
    dphi_e = np.where(eigvals > 0.0, g_col, 1.0)

    sig_iso = lam * (g_deg * tr_pos + tr_neg)
    sig_mat = 2.0 * mu * np.einsum("ea,eia,eja->eij", phi_e, eigvecs, eigvecs)
    sig_mat[:, 0, 0] += sig_iso
    sig_mat[:, 1, 1] += sig_iso
    sig_mat[:, 2, 2] += sig_iso

    sig_v = np.empty((n_elem, 6), dtype=np.float64)
    sig_v[:, 0] = sig_mat[:, 0, 0]
    sig_v[:, 1] = sig_mat[:, 1, 1]
    sig_v[:, 2] = sig_mat[:, 2, 2]
    sig_v[:, 3] = sig_mat[:, 1, 2]
    sig_v[:, 4] = sig_mat[:, 0, 2]
    sig_v[:, 5] = sig_mat[:, 0, 1]

    v_eye = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    vol_coeff = lam * (g_deg * h_tr_pos + h_tr_neg)
    d_tan = vol_coeff[:, None, None] * np.outer(v_eye, v_eye)[None, :, :]

    n0 = eigvecs[:, 0, :]
    n1 = eigvecs[:, 1, :]
    n2 = eigvecs[:, 2, :]
    v_aa = np.stack(
        (n0 * n0, n1 * n1, n2 * n2, n1 * n2, n0 * n2, n0 * n1), axis=-1
    )  # (E, 3, 6)
    d_tan += 2.0 * mu * np.einsum("ea,eai,eaj->eij", dphi_e, v_aa, v_aa)

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        diff = eigvals[:, a] - eigvals[:, b]
        scale = np.maximum(
            np.maximum(np.abs(eigvals[:, a]), np.abs(eigvals[:, b])), 1.0
        )
        distinct = np.abs(diff) > 1.0e-12 * scale
        safe_diff = np.where(distinct, diff, 1.0)
        theta_ab = np.where(
            distinct,
            (phi_e[:, a] - phi_e[:, b]) / safe_diff,
            0.5 * (dphi_e[:, a] + dphi_e[:, b]),
        )
        na = eigvecs[:, :, a]
        nb = eigvecs[:, :, b]
        v_ab = inv_sqrt2 * np.stack(
            (
                2.0 * na[:, 0] * nb[:, 0],
                2.0 * na[:, 1] * nb[:, 1],
                2.0 * na[:, 2] * nb[:, 2],
                na[:, 1] * nb[:, 2] + na[:, 2] * nb[:, 1],
                na[:, 0] * nb[:, 2] + na[:, 2] * nb[:, 0],
                na[:, 0] * nb[:, 1] + na[:, 1] * nb[:, 0],
            ),
            axis=-1,
        )
        d_tan += 2.0 * mu * np.einsum("e,ei,ej->eij", theta_ab, v_ab, v_ab)

    return sig_v, d_tan, psi_plus, psi_tot


def solve_phase_field_fracture(
    mesh: CanonicalMesh3D,
    *,
    dirichlet_bcs: Sequence[DirichletBC3D],
    config: PhaseFieldConfig | None = None,
    initial_damage_nodal: FloatArray | None = None,
    warm_start_predictor: PhaseFieldWarmStartPredictor | None = None,
    record_history: bool = False,
    cold_start_each_step: bool = False,
) -> PhaseFieldFractureResult:
    """Solve quasi-static staggered AT2 brittle phase-field fracture on a 3D ``TET4`` mesh.

    Uses exact Miehe (2010) spectral tension-compression stress/tangent splitting
    so compressed regions retain full elastic stiffness and pre-cracks degrade
    tensile resistance immediately, coupled to a lumped-mass AT2 phase-field
    problem with primal-dual active-set irreversibility ``d >= d_prev``.
    """
    t0 = time.perf_counter()
    if mesh.cell_type is not CellType3D.TET4:
        raise ValueError("solve_phase_field_fracture currently requires TET4 meshes")
    cfg = PhaseFieldConfig() if config is None else config
    mesh.validate_incompressibility_locking(cfg.poisson_ratio)
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
    if np.any(~np.isfinite(det_j)) or np.any(det_j <= 0.0):
        raise ValueError(
            "mesh contains degenerate or inverted TET4 elements (det_j <= 0)"
        )
    vols = det_j / 6.0  # (E,)
    jac_inv = np.linalg.inv(jac)
    d_phys = np.einsum("na,eai->eni", d_ref, jac_inv)  # (E, 4, 3)

    # Precompute B_e (E, 6, 12), gradient stiffness K_d_grad, and lumped nodal volumes
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

    k_d_grad = (gc * lc) * np.einsum("e,eai,ebi->eab", vols, d_phys, d_phys)
    m_d_lump = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(m_d_lump, cells, (vols / 4.0)[:, None])

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
    k_d_grad_mat: csr_matrix = coo_matrix(
        (k_d_grad.reshape(-1), (d_rows, d_cols)), shape=(n_nodes, n_nodes)
    ).tocsr()

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

    if initial_damage_nodal is None:
        d_nodal: FloatArray = np.zeros(n_nodes, dtype=np.float64)
    else:
        d_init = np.asarray(initial_damage_nodal, dtype=np.float64).reshape(-1)
        if d_init.shape != (n_nodes,) or not np.all(np.isfinite(d_init)):
            raise ValueError(
                f"initial_damage_nodal must be finite with length {n_nodes}"
            )
        d_nodal = np.clip(d_init.copy(), 0.0, 1.0)

    u_full: FloatArray = np.zeros(n_dofs, dtype=np.float64)
    hist_h: FloatArray = np.zeros(n_elem, dtype=np.float64)

    load_factors: list[float] = []
    max_d_hist: list[float] = []
    crack_e_hist: list[float] = []
    reaction_hist: list[float] = []
    per_step_newton: list[int] = []
    per_step_staggered: list[int] = []
    damage_snaps: list[FloatArray] = []
    disp_snaps: list[FloatArray] = []
    monotone = True

    loaded_dofs = np.nonzero(np.abs(u_target) > 1.0e-15)[0]
    loaded_signs = np.sign(u_target[loaded_dofs])

    def _assemble_mechanics(
        u_vec: FloatArray, g_deg: FloatArray
    ) -> tuple[FloatArray, csr_matrix, FloatArray, float]:
        u_e_vec = u_vec[elem_u_dofs]
        eps_v = np.einsum("eij,ej->ei", b_all, u_e_vec)
        sig_v, d_tan, psi_p, psi_tot = _miehe_spectral_stress_tangent_batch(
            eps_v, g_deg, lam, mu
        )
        f_elem = np.einsum("e,eij,ei->ej", vols, b_all, sig_v)
        f_int = np.zeros(n_dofs, dtype=np.float64)
        np.add.at(f_int, elem_u_dofs, f_elem)
        k_e = np.einsum("e,eik,eij,ejl->ekl", vols, b_all, d_tan, b_all)
        k_u: csr_matrix = coo_matrix(
            (k_e.reshape(-1), (u_rows, u_cols)), shape=(n_dofs, n_dofs)
        ).tocsr()
        total_psi = float(np.sum(vols * psi_tot))
        return f_int, k_u, psi_p, total_psi

    for step in range(1, cfg.load_steps + 1):
        lam_load = step / float(cfg.load_steps)
        u_pres = lam_load * u_target
        d_prev_step = d_nodal.copy()
        hist_h_prev = hist_h.copy()
        hist_h_trial = hist_h_prev.copy()
        u_prev_3d = u_full.reshape(n_nodes, 3).copy()

        if cold_start_each_step:
            u_full[:] = 0.0

        # Optional neural-operator / physics warm start for free DOFs and trial damage.
        # Irreversible history (d_prev_step, hist_h_prev) stays strictly frozen until
        # exact FEM equilibrium and active-set phase-field convergence complete.
        if warm_start_predictor is not None:
            u_guess, d_guess = warm_start_predictor(lam_load, u_prev_3d, d_prev_step)
            if u_guess is not None:
                u_arr = np.asarray(u_guess, dtype=np.float64).reshape(-1)
                if u_arr.shape != (n_dofs,) or not np.all(np.isfinite(u_arr)):
                    raise ValueError(
                        f"warm-start displacement must be finite with {n_dofs} entries"
                    )
                u_full = u_arr.copy()
            if d_guess is not None:
                d_arr = np.asarray(d_guess, dtype=np.float64).reshape(-1)
                if d_arr.shape != (n_nodes,) or not np.all(np.isfinite(d_arr)):
                    raise ValueError(
                        f"warm-start damage must be finite with {n_nodes} entries"
                    )
                d_nodal = np.clip(np.maximum(d_arr, d_prev_step), 0.0, 1.0)

        # Apply exact boundary displacements
        u_full[constrained] = u_pres[constrained]

        step_staggered = 0
        step_newton = 0
        for _st in range(cfg.staggered_iters):
            step_staggered += 1
            d_old = d_nodal.copy()
            # 1) Nonlinear mechanics equilibrium with exact Miehe spectral stress & tangent
            d_elem = np.mean(d_nodal[cells], axis=1)
            g_deg = (1.0 - d_elem) ** 2 + k_res
            psi_p: FloatArray = np.zeros(n_elem, dtype=np.float64)

            for _n_it in range(15):
                f_int, k_u, psi_p, total_psi = _assemble_mechanics(u_full, g_deg)
                res_free = f_int[free_u]
                res_norm = float(np.linalg.norm(res_free))
                f_scale = max(float(np.linalg.norm(f_int)), 1.0)
                if res_norm <= 1.0e-10 * f_scale:
                    break
                step_newton += 1
                du_f = np.asarray(
                    spsolve(k_u[free_u][:, free_u].tocsc(), -res_free),
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(du_f)):
                    raise RuntimeError(
                        "non-finite Newton step in phase-field mechanics"
                    )
                # Convex potential-energy backtracking guarantees global convergence
                alpha = 1.0
                for _ls in range(10):
                    u_cand = u_full.copy()
                    u_cand[free_u] += alpha * du_f
                    eps_c = np.einsum("eij,ej->ei", b_all, u_cand[elem_u_dofs])
                    _, _, _, psi_c = _miehe_spectral_stress_tangent_batch(
                        eps_c, g_deg, lam, mu
                    )
                    if float(np.sum(vols * psi_c)) <= total_psi + 1.0e-10 * max(
                        abs(total_psi), 1.0
                    ):
                        u_full = u_cand
                        break
                    alpha *= 0.5
                else:
                    u_full[free_u] += alpha * du_f

            # Re-evaluate tensile energy at converged u_full for history trial
            eps_v_fin = np.einsum("eij,ej->ei", b_all, u_full[elem_u_dofs])
            _, _, psi_p, _ = _miehe_spectral_stress_tangent_batch(
                eps_v_fin, g_deg, lam, mu
            )
            # Keep hist_h_prev frozen across staggered iterations (commit only after step)
            hist_h_trial = np.maximum(hist_h_prev, psi_p)

            # 2) Lumped-mass AT2 phase-field step with primal-dual active-set irreversibility
            h_lump = np.zeros(n_nodes, dtype=np.float64)
            np.add.at(h_lump, cells, (hist_h_trial * vols / 4.0)[:, None])
            diag_d = (gc / lc) * m_d_lump + 2.0 * h_lump
            k_d: csr_matrix = (k_d_grad_mat + diags(diag_d, format="csr")).tocsr()
            rhs_d = 2.0 * h_lump

            active = np.zeros(n_nodes, dtype=bool)
            d_cand = d_prev_step.copy()
            for _as in range(6):
                free_d = np.nonzero(~active)[0]
                if free_d.size == 0:
                    d_cand = d_prev_step.copy()
                    break
                rhs_free = rhs_d[free_d] - (
                    k_d[free_d][:, active] @ d_prev_step[active]
                )
                d_sol = np.asarray(
                    spsolve(k_d[free_d][:, free_d].tocsc(), rhs_free),
                    dtype=np.float64,
                )
                d_next = d_prev_step.copy()
                d_next[free_d] = d_sol
                res_dual = k_d @ d_next - rhs_d
                new_active = (d_next < d_prev_step - 1.0e-12) | (
                    active & (res_dual >= -1.0e-12)
                )
                d_cand = d_next
                if np.array_equal(new_active, active):
                    break
                active = new_active

            if np.any(d_cand < d_prev_step - 1.0e-9):
                monotone = False
            d_nodal = np.clip(np.maximum(d_cand, d_prev_step), 0.0, 1.0)
            if float(np.max(np.abs(d_nodal - d_old))) <= cfg.tolerance:
                break

        # Commit irreversible history ONLY after load step completes
        hist_h = hist_h_trial
        # Compute exact reaction force at converged state along prescribed non-zero BCs
        d_elem_fin = np.mean(d_nodal[cells], axis=1)
        g_deg_fin = (1.0 - d_elem_fin) ** 2 + k_res
        eps_v_post = np.einsum("eij,ej->ei", b_all, u_full[elem_u_dofs])
        sig_v_post, _, _, _ = _miehe_spectral_stress_tangent_batch(
            eps_v_post, g_deg_fin, lam, mu
        )
        f_elem_post = np.einsum("e,eij,ei->ej", vols, b_all, sig_v_post)
        f_int_post = np.zeros(n_dofs, dtype=np.float64)
        np.add.at(f_int_post, elem_u_dofs, f_elem_post)
        react_n = (
            float(np.sum(f_int_post[loaded_dofs] * loaded_signs))
            if loaded_dofs.size > 0
            else 0.0
        )

        # Exact AT2 crack surface energy: integral G_c * (d^2 / (2 lc) + lc/2 |grad d|^2) dV
        crack_e = 0.5 * (
            float(d_nodal @ (k_d_grad_mat @ d_nodal))
            + (gc / lc) * float(np.sum(m_d_lump * d_nodal * d_nodal))
        )
        load_factors.append(float(lam_load))
        max_d_hist.append(float(np.max(d_nodal)))
        crack_e_hist.append(crack_e)
        reaction_hist.append(react_n)
        per_step_newton.append(step_newton)
        per_step_staggered.append(step_staggered)
        if record_history:
            damage_snaps.append(d_nodal.copy())
            disp_snaps.append(u_full.reshape(n_nodes, 3).copy())

    return PhaseFieldFractureResult(
        mesh_id=mesh.mesh_id,
        load_factors=tuple(load_factors),
        max_damage_history=tuple(max_d_hist),
        crack_surface_energy_mj=tuple(crack_e_hist),
        monotone_irreversible=monotone,
        final_damage_nodal=d_nodal,
        final_displacement_mm=u_full.reshape(n_nodes, 3),
        wall_time_s=time.perf_counter() - t0,
        total_newton_iterations=sum(per_step_newton),
        total_staggered_iterations=sum(per_step_staggered),
        per_step_newton_iterations=tuple(per_step_newton),
        per_step_staggered_iterations=tuple(per_step_staggered),
        reaction_force_history_n=tuple(reaction_hist),
        damage_snapshots=tuple(damage_snaps),
        displacement_snapshots=tuple(disp_snaps),
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
    "PhaseFieldWarmStartPredictor",
    "RainflowCycle",
    "evaluate_fatigue_life",
    "rainflow_count_astm_e1049",
    "solve_phase_field_fracture",
)
