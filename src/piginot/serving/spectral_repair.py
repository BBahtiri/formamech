"""Spectrally safe warm-start repair for nonlinear Newton-Raphson FEM (Tier 2).

Implements the spectrally safe warm-start pattern motivated by Oh, Lee, Darbon,
and Karniadakis (arXiv:2606.21828): low L2 error does not guarantee that a
neural displacement prediction lies inside the positive-definite Newton basin
of attraction because spectral bias can leave localized ``det(F)`` defects that
inject spurious negative eigenvalues into the discrete tangent stiffness matrix.

Given an initial neural displacement ``u_0``, :func:`spectrally_safe_repair`:

1. Checks admissibility (``min det(F_2D) > min_det_f``) and coercivity
   (``lambda_min(K_sym) > 0`` via :func:`coercive_inertia`). If already safe,
   returns immediately with zero repair steps.
2. If localized ``det(F)`` defects render ``u_0`` inadmissible, performs bounded
   homotopy relaxation on free DOFs (preserving exact Dirichlet values) toward
   the zero-free-DOF lifted state until every quadrature point is admissible.
3. If the discrete tangent is indefinite, runs a bounded label-free discrete
   potential-energy Levenberg-Marquardt descent pass ``(K_sym + mu I) d = -r``
   with backtracking admissibility/energy line search until
   ``lambda_min(K_sym) > 0`` is restored before handing off to warm Newton.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import eye as sparse_eye  # type: ignore[import-untyped]
from scipy.sparse.linalg import spsolve  # type: ignore[import-untyped]

from piginot.certify.guards import coercive_inertia
from piginot.contracts.state_machine import (
    StateMachineSpec,
    TransitionSpec,
    register_state_machine,
)
from piginot.fem.models import FemSolveFailure
from piginot.fem.protocol import DiscreteLinearization

FloatArray = NDArray[np.float64]


class SpectralRepairStatus(StrEnum):
    """Explicit outcome states of Tier 2 spectrally safe warm-start repair."""

    CHECKING = "CHECKING"
    ALREADY_SAFE = "ALREADY_SAFE"
    REPAIRED_HOMOTOPY = "REPAIRED_HOMOTOPY"
    REPAIRED_LM = "REPAIRED_LM"
    FAILED_ADMISSIBILITY = "FAILED_ADMISSIBILITY"
    FAILED_LM_BUDGET = "FAILED_LM_BUDGET"


SPECTRAL_REPAIR_MACHINE: Final[StateMachineSpec[SpectralRepairStatus]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.serving.spectral_repair",
            description=(
                "Tier 2 warm-start spectral repair: admissibility homotopy followed "
                "by shifted Levenberg-Marquardt SPD basin restoration."
            ),
            states=tuple(SpectralRepairStatus),
            initial_states=(SpectralRepairStatus.CHECKING,),
            terminal_states=(
                SpectralRepairStatus.ALREADY_SAFE,
                SpectralRepairStatus.REPAIRED_HOMOTOPY,
                SpectralRepairStatus.REPAIRED_LM,
                SpectralRepairStatus.FAILED_ADMISSIBILITY,
                SpectralRepairStatus.FAILED_LM_BUDGET,
            ),
            transitions=(
                TransitionSpec(
                    SpectralRepairStatus.CHECKING,
                    SpectralRepairStatus.ALREADY_SAFE,
                    "initial_spd_ok",
                    "Initial guess is already admissible with SPD symmetric tangent.",
                ),
                TransitionSpec(
                    SpectralRepairStatus.CHECKING,
                    SpectralRepairStatus.REPAIRED_HOMOTOPY,
                    "homotopy_restored_spd",
                    "Stage 1 admissibility homotopy restored det(F) and SPD tangent.",
                ),
                TransitionSpec(
                    SpectralRepairStatus.CHECKING,
                    SpectralRepairStatus.REPAIRED_LM,
                    "lm_restored_spd",
                    "Stage 2 shifted Levenberg-Marquardt descent restored SPD tangent.",
                ),
                TransitionSpec(
                    SpectralRepairStatus.CHECKING,
                    SpectralRepairStatus.FAILED_ADMISSIBILITY,
                    "admissibility_unrecoverable",
                    "Stage 1 homotopy could not reach det(F) >= min_det_f.",
                ),
                TransitionSpec(
                    SpectralRepairStatus.CHECKING,
                    SpectralRepairStatus.FAILED_LM_BUDGET,
                    "lm_budget_exhausted",
                    "Stage 2 LM could not restore SPD tangent within max_steps.",
                ),
            ),
        )
    )
)


@dataclass(frozen=True, slots=True)
class SpectralRepairConfig:
    """Bounded budget and safety floors for label-free discrete spectral repair."""

    max_steps: int = 40
    min_det_f: float = 0.05
    initial_damping: float = 1.0e-2
    max_damping: float = 1.0e8
    max_backtracks: int = 12
    admissibility_relax_steps: int = 16

    def __post_init__(self) -> None:
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not math.isfinite(self.min_det_f) or self.min_det_f <= 0.0:
            raise ValueError("min_det_f must be finite and strictly positive")
        if not math.isfinite(self.initial_damping) or self.initial_damping <= 0.0:
            raise ValueError("initial_damping must be finite and positive")
        if self.max_damping < self.initial_damping:
            raise ValueError("max_damping must be >= initial_damping")


@dataclass(frozen=True, slots=True)
class SpectralRepairReport:
    """Auditable diagnostics of one spectrally safe warm-start repair pass."""

    needed_repair: bool
    succeeded: bool
    steps_taken: int
    initial_admissible: bool
    initial_positive_definite: bool | None
    final_positive_definite: bool | None
    initial_min_det_f: float | None
    final_min_det_f: float | None
    initial_residual_norm: float | None
    final_residual_norm: float | None
    wall_seconds: float
    reason: str

    def __post_init__(self) -> None:
        if self.steps_taken < 0:
            raise ValueError("steps_taken must be non-negative")
        if not self.needed_repair and not self.succeeded:
            raise ValueError("needed_repair=False requires succeeded=True")
        if self.succeeded and self.final_positive_definite is False:
            raise ValueError(
                "succeeded=True conflicts with final_positive_definite=False"
            )

    @property
    def status(self) -> SpectralRepairStatus:
        """Explicit finite-state outcome derived from repair diagnostics."""
        if not self.needed_repair:
            return SpectralRepairStatus.ALREADY_SAFE
        if self.succeeded:
            if not self.initial_admissible and self.steps_taken == 0:
                return SpectralRepairStatus.REPAIRED_HOMOTOPY
            return SpectralRepairStatus.REPAIRED_LM
        if self.final_positive_definite is None:
            return SpectralRepairStatus.FAILED_ADMISSIBILITY
        return SpectralRepairStatus.FAILED_LM_BUDGET

    def document(self) -> dict[str, Any]:
        return {
            "needed_repair": self.needed_repair,
            "succeeded": self.succeeded,
            "steps_taken": self.steps_taken,
            "initial_admissible": self.initial_admissible,
            "initial_positive_definite": self.initial_positive_definite,
            "final_positive_definite": self.final_positive_definite,
            "initial_min_det_f": self.initial_min_det_f,
            "final_min_det_f": self.final_min_det_f,
            "initial_residual_norm": self.initial_residual_norm,
            "final_residual_norm": self.final_residual_norm,
            "wall_seconds": self.wall_seconds,
            "reason": self.reason,
        }


def _min_det_2d(state: Any) -> float:
    f3d = np.asarray(state.assembly.deformation_gradient, dtype=np.float64)
    f2d = f3d[..., :2, :2]
    det2d = f2d[..., 0, 0] * f2d[..., 1, 1] - f2d[..., 0, 1] * f2d[..., 1, 0]
    return float(np.min(det2d))


def _residual_norm(linearization: DiscreteLinearization, state: Any) -> float:
    free = linearization.constraints.free
    return float(np.linalg.norm(state.residual[free]))


def _try_state(
    linearization: DiscreteLinearization, vector: FloatArray, min_det_f: float
) -> tuple[Any | None, float | None]:
    try:
        st = linearization.state(vector)
    except FemSolveFailure:
        return None, None
    mdet = _min_det_2d(st)
    if not math.isfinite(mdet) or mdet < min_det_f:
        return None, mdet
    return st, mdet


def spectrally_safe_repair(
    linearization: DiscreteLinearization,
    initial_vector: FloatArray,
    *,
    config: SpectralRepairConfig | None = None,
) -> tuple[FloatArray, SpectralRepairReport]:
    """Repair ``initial_vector`` until admissible and strictly coercive/SPD."""
    if config is None:
        config = SpectralRepairConfig()
    t0 = time.perf_counter()
    vec = linearization.with_exact_dirichlet(
        np.asarray(initial_vector, dtype=np.float64)
    )
    free = linearization.constraints.free

    state, init_det = _try_state(linearization, vec, config.min_det_f)
    init_admissible = state is not None
    init_res: float | None = (
        None if state is None else _residual_norm(linearization, state)
    )
    init_pd: bool | None = None
    if state is not None:
        init_pd = coercive_inertia(linearization.free_tangent(state)).positive_definite
        if init_pd is True:
            return vec, SpectralRepairReport(
                needed_repair=False,
                succeeded=True,
                steps_taken=0,
                initial_admissible=True,
                initial_positive_definite=True,
                final_positive_definite=True,
                initial_min_det_f=init_det,
                final_min_det_f=init_det,
                initial_residual_norm=init_res,
                final_residual_norm=init_res,
                wall_seconds=time.perf_counter() - t0,
                reason="initial state already admissible and positive definite",
            )

    steps_taken = 0
    # Stage 1: restore admissibility if localized det(F) defects broke assembly.
    if state is None:
        base = linearization.with_exact_dirichlet(np.zeros_like(vec, dtype=np.float64))
        # Try shrinking free-DOF perturbation smoothly toward the boundary-lifted base.
        alpha = 1.0
        for _ in range(config.admissibility_relax_steps):
            alpha *= 0.5
            steps_taken += 1
            cand = base.copy()
            cand[free] = base[free] + alpha * (vec[free] - base[free])
            cand = linearization.with_exact_dirichlet(cand)
            cand_state, cand_det = _try_state(linearization, cand, config.min_det_f)
            if cand_state is not None:
                vec = cand
                state = cand_state
                break
        if state is None:
            cand_state, cand_det = _try_state(linearization, base, config.min_det_f)
            if cand_state is not None:
                vec = base
                state = cand_state
            else:
                return vec, SpectralRepairReport(
                    needed_repair=True,
                    succeeded=False,
                    steps_taken=steps_taken,
                    initial_admissible=init_admissible,
                    initial_positive_definite=init_pd,
                    final_positive_definite=None,
                    initial_min_det_f=init_det,
                    final_min_det_f=cand_det,
                    initial_residual_norm=init_res,
                    final_residual_norm=None,
                    wall_seconds=time.perf_counter() - t0,
                    reason="could not restore positive Jacobian admissibility",
                )

    # Stage 2: Levenberg-Marquardt shifted SPD descent on discrete potential energy.
    n_free = int(free.size)
    identity = sparse_eye(n_free, format="csc", dtype=np.float64)
    current_pd = coercive_inertia(linearization.free_tangent(state)).positive_definite
    if current_pd is True:
        return vec, SpectralRepairReport(
            needed_repair=True,
            succeeded=True,
            steps_taken=steps_taken,
            initial_admissible=init_admissible,
            initial_positive_definite=init_pd,
            final_positive_definite=True,
            initial_min_det_f=init_det,
            final_min_det_f=_min_det_2d(state),
            initial_residual_norm=init_res,
            final_residual_norm=_residual_norm(linearization, state),
            wall_seconds=time.perf_counter() - t0,
            reason="admissibility relaxation restored positive-definite tangent",
        )

    for _ in range(config.max_steps):
        steps_taken += 1
        k_free = linearization.free_tangent(state).tocsc()
        k_sym = 0.5 * (k_free + k_free.T)
        r_free = np.asarray(state.residual[free], dtype=np.float64)
        diag_scale = max(float(np.max(np.abs(k_sym.diagonal()))), 1.0)
        mu = config.initial_damping * diag_scale
        shifted_ok = False
        while mu <= config.max_damping * diag_scale:
            shifted = (k_sym + mu * identity).tocsc()
            if coercive_inertia(shifted).positive_definite is True:
                shifted_ok = True
                break
            mu *= 4.0
        if not shifted_ok:
            break
        direction = np.asarray(spsolve(shifted, -r_free), dtype=np.float64)
        if not bool(np.isfinite(direction).all()):
            break
        slope = float(r_free @ direction)
        if slope >= 0.0:
            direction = -r_free
            slope = -float(r_free @ r_free)

        accepted = False
        step_alpha = 1.0
        cur_energy = float(state.potential_energy)
        cur_res_norm = _residual_norm(linearization, state)
        for _ in range(config.max_backtracks):
            trial = vec.copy()
            trial[free] += step_alpha * direction
            trial = linearization.with_exact_dirichlet(trial)
            trial_state, _ = _try_state(linearization, trial, config.min_det_f)
            if trial_state is not None and (
                float(trial_state.potential_energy) < cur_energy
                or _residual_norm(linearization, trial_state) < cur_res_norm
            ):
                vec = trial
                state = trial_state
                accepted = True
                break
            step_alpha *= 0.5

        if not accepted:
            break

        current_pd = coercive_inertia(
            linearization.free_tangent(state)
        ).positive_definite
        if current_pd is True:
            return vec, SpectralRepairReport(
                needed_repair=True,
                succeeded=True,
                steps_taken=steps_taken,
                initial_admissible=init_admissible,
                initial_positive_definite=init_pd,
                final_positive_definite=True,
                initial_min_det_f=init_det,
                final_min_det_f=_min_det_2d(state),
                initial_residual_norm=init_res,
                final_residual_norm=_residual_norm(linearization, state),
                wall_seconds=time.perf_counter() - t0,
                reason="discrete Levenberg-Marquardt repair restored positive definiteness",
            )

    return vec, SpectralRepairReport(
        needed_repair=True,
        succeeded=False,
        steps_taken=steps_taken,
        initial_admissible=init_admissible,
        initial_positive_definite=init_pd,
        final_positive_definite=current_pd,
        initial_min_det_f=init_det,
        final_min_det_f=_min_det_2d(state),
        initial_residual_norm=init_res,
        final_residual_norm=_residual_norm(linearization, state),
        wall_seconds=time.perf_counter() - t0,
        reason="repair budget exhausted before positive definiteness was restored",
    )


__all__ = (
    "SpectralRepairConfig",
    "SpectralRepairReport",
    "spectrally_safe_repair",
)
