"""Tier 0 instantaneous zero-FEM physics-admissibility gate.

Evaluates hard physical admissibility criteria on neural operator predictions
in milliseconds using only runtime dependencies (NumPy/PyTorch, zero FEM cost):

1. All served fields finite (displacement, deformation gradient, stress).
2. Local deformation Jacobian positivity (``min det(F_2D) > floor``,
   ``min det(F_3D) > floor``).
3. Optional exact Dirichlet compliance, equilibrium residual RMS, free-boundary
   traction RMS, and axial/section-force balance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _positive_float(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    num = float(value)
    if not math.isfinite(num) or num <= 0.0:
        raise ValueError(f"{label} must be finite and strictly positive")
    return num


def _optional_nonnegative(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number or None")
    num = float(value)
    if not math.isfinite(num) or num < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return num


@dataclass(frozen=True, slots=True)
class AdmissibilityThresholds:
    """Hard admissibility thresholds for the Tier 0 zero-FEM gate."""

    min_det_f_2d: float = 1.0e-4
    min_det_f_3d: float = 1.0e-4
    max_dirichlet_deviation: float = 1.0e-6
    max_equilibrium_residual_rms: float | None = None
    max_free_traction_rms: float | None = None
    max_section_force_imbalance: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "min_det_f_2d", _positive_float(self.min_det_f_2d, "min_det_f_2d")
        )
        object.__setattr__(
            self, "min_det_f_3d", _positive_float(self.min_det_f_3d, "min_det_f_3d")
        )
        object.__setattr__(
            self,
            "max_dirichlet_deviation",
            _optional_nonnegative(
                self.max_dirichlet_deviation, "max_dirichlet_deviation"
            ),
        )
        object.__setattr__(
            self,
            "max_equilibrium_residual_rms",
            _optional_nonnegative(
                self.max_equilibrium_residual_rms, "max_equilibrium_residual_rms"
            ),
        )
        object.__setattr__(
            self,
            "max_free_traction_rms",
            _optional_nonnegative(self.max_free_traction_rms, "max_free_traction_rms"),
        )
        object.__setattr__(
            self,
            "max_section_force_imbalance",
            _optional_nonnegative(
                self.max_section_force_imbalance, "max_section_force_imbalance"
            ),
        )


@dataclass(frozen=True, slots=True)
class Tier0AdmissibilityResult:
    """Canonical record of one Tier 0 zero-FEM admissibility check."""

    passed: bool
    all_finite: bool
    min_det_f_2d: float | None
    min_det_f_3d: float | None
    dirichlet_deviation: float | None
    equilibrium_residual_rms: float | None
    free_traction_rms: float | None
    section_force_imbalance: float | None
    failed_checks: tuple[str, ...]

    def __post_init__(self) -> None:
        expected_passed = bool(self.all_finite) and len(self.failed_checks) == 0
        if self.passed is not expected_passed:
            raise ValueError(
                f"Tier0AdmissibilityResult invariant violated: passed={self.passed} "
                f"with all_finite={self.all_finite} and "
                f"failed_checks={self.failed_checks!r}"
            )

    def document(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "all_finite": self.all_finite,
            "min_det_f_2d": self.min_det_f_2d,
            "min_det_f_3d": self.min_det_f_3d,
            "dirichlet_deviation": self.dirichlet_deviation,
            "equilibrium_residual_rms": self.equilibrium_residual_rms,
            "free_traction_rms": self.free_traction_rms,
            "section_force_imbalance": self.section_force_imbalance,
            "failed_checks": list(self.failed_checks),
        }


def _to_array(value: Any) -> NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64)


def _rms(values: Any) -> float:
    arr = _to_array(values).reshape(-1)
    if arr.size == 0:
        return 0.0
    if not bool(np.isfinite(arr).all()):
        return float("inf")
    return float(np.sqrt(np.mean(arr * arr)))


def evaluate_tier0_admissibility(
    fields: Mapping[str, Any],
    *,
    thresholds: AdmissibilityThresholds = AdmissibilityThresholds(),
    dirichlet_deviation: float | None = None,
    equilibrium_residual: Any | None = None,
    free_traction_residual: Any | None = None,
    section_forces: Sequence[float] | None = None,
) -> Tier0AdmissibilityResult:
    """Evaluate Tier 0 physical admissibility on served field arrays."""
    failed: list[str] = []
    all_finite = True

    for key in ("u", "F_2D", "F33", "J_3D", "P_served", "sigma"):
        if key in fields and fields[key] is not None:
            arr = _to_array(fields[key])
            if not bool(np.isfinite(arr).all()):
                all_finite = False
                failed.append(f"non_finite:{key}")

    min_det_2d: float | None = None
    if "F_2D" in fields and fields["F_2D"] is not None:
        f2d = _to_array(fields["F_2D"])
        if f2d.ndim >= 2 and f2d.shape[-2:] == (2, 2):
            det2d = f2d[..., 0, 0] * f2d[..., 1, 1] - f2d[..., 0, 1] * f2d[..., 1, 0]
            if bool(np.isfinite(det2d).all()) and det2d.size > 0:
                min_det_2d = float(np.min(det2d))
                if min_det_2d <= thresholds.min_det_f_2d:
                    failed.append("det_f_2d_below_floor")
            else:
                all_finite = False
                failed.append("det_f_2d_non_finite")

    min_det_3d: float | None = None
    if "J_3D" in fields and fields["J_3D"] is not None:
        j3d = _to_array(fields["J_3D"])
        if bool(np.isfinite(j3d).all()) and j3d.size > 0:
            min_det_3d = float(np.min(j3d))
            if min_det_3d <= thresholds.min_det_f_3d:
                failed.append("det_f_3d_below_floor")
        else:
            all_finite = False
            failed.append("det_f_3d_non_finite")

    dir_dev: float | None = None
    if dirichlet_deviation is not None:
        dir_dev = float(dirichlet_deviation)
        if not math.isfinite(dir_dev) or dir_dev > thresholds.max_dirichlet_deviation:
            failed.append("dirichlet_deviation_exceeded")

    eq_rms: float | None = None
    if equilibrium_residual is not None:
        eq_rms = _rms(equilibrium_residual)
        if not math.isfinite(eq_rms):
            all_finite = False
            failed.append("equilibrium_residual_non_finite")
        elif (
            thresholds.max_equilibrium_residual_rms is not None
            and eq_rms > thresholds.max_equilibrium_residual_rms
        ):
            failed.append("equilibrium_residual_exceeded")

    trac_rms: float | None = None
    if free_traction_residual is not None:
        trac_rms = _rms(free_traction_residual)
        if not math.isfinite(trac_rms):
            all_finite = False
            failed.append("free_traction_residual_non_finite")
        elif (
            thresholds.max_free_traction_rms is not None
            and trac_rms > thresholds.max_free_traction_rms
        ):
            failed.append("free_traction_residual_exceeded")

    sec_imb: float | None = None
    if section_forces is not None and len(section_forces) >= 2:
        sf = np.asarray(section_forces, dtype=np.float64)
        if not bool(np.isfinite(sf).all()):
            all_finite = False
            failed.append("section_force_non_finite")
        else:
            scale = max(float(np.mean(np.abs(sf))), 1.0e-12)
            sec_imb = float((np.max(sf) - np.min(sf)) / scale)
            if (
                thresholds.max_section_force_imbalance is not None
                and sec_imb > thresholds.max_section_force_imbalance
            ):
                failed.append("section_force_imbalance_exceeded")

    return Tier0AdmissibilityResult(
        passed=all_finite and not failed,
        all_finite=all_finite,
        min_det_f_2d=min_det_2d
        if (min_det_2d is not None and math.isfinite(min_det_2d))
        else None,
        min_det_f_3d=min_det_3d
        if (min_det_3d is not None and math.isfinite(min_det_3d))
        else None,
        dirichlet_deviation=dir_dev
        if (dir_dev is not None and math.isfinite(dir_dev))
        else None,
        equilibrium_residual_rms=eq_rms
        if (eq_rms is not None and math.isfinite(eq_rms))
        else None,
        free_traction_rms=trac_rms
        if (trac_rms is not None and math.isfinite(trac_rms))
        else None,
        section_force_imbalance=sec_imb
        if (sec_imb is not None and math.isfinite(sec_imb))
        else None,
        failed_checks=tuple(failed),
    )


__all__ = (
    "AdmissibilityThresholds",
    "Tier0AdmissibilityResult",
    "evaluate_tier0_admissibility",
)
