"""Versioned CAD-level numerical trust report and QoI error accounting (Feature D).

Separates four distinct error sources without collapsing deterministic numerical
differences and statistical coverage statements:

1. **Geometry representation** (``GeometryErrorRecord``): exact CAD vs. polygon.
2. **FEM discretization** (``DiscretizationErrorRecord``): mesh/solver resolution.
3. **Surrogate/interpolation** (``SurrogateErrorRecord``): neural prediction vs.
   discrete FEM solution on the same mesh (Newton / DWR indicators).
4. **Statistical calibration** (``StatisticalCalibrationRecord``): marginal
   split-conformal coverage over exchangeable in-envelope populations.

This module uses pure Python standard library + canonical hashing contracts and
requires no FEM, CAD, or ML libraries, preserving ``.[runtime]`` isolation.
"""

from __future__ import annotations

import enum
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from piginot.contracts import canonical_json_bytes

CAD_TRUST_REPORT_SCHEMA: Final = "piginot.cad-trust-report-v1"
SUPPORTED_QOI_KINDS: Final = frozenset(
    ("reaction_force", "regional_average_stress", "peak_stress")
)
ALLOWED_CLAIM_LEVELS: Final = (
    "NOT_VERIFIED",
    "DEVELOPMENT_VERIFIED",
    "HOLDOUT_VERIFIED",
)


def _finite_or_none(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number or None")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{label} must be finite")
    return num


class TrustStatus(enum.StrEnum):
    """Fail-closed per-QoI and report-level numerical trust status."""

    QUALIFIED_INDICATORS = "QUALIFIED_INDICATORS"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    UNCERTIFIED = "UNCERTIFIED"
    OUT_OF_ENVELOPE = "OUT_OF_ENVELOPE"
    REFERENCE_NOT_CONVERGED = "REFERENCE_NOT_CONVERGED"
    INDEFINITE = "INDEFINITE"
    INADMISSIBLE = "INADMISSIBLE"
    SOLVE_FAILED = "SOLVE_FAILED"


@dataclass(frozen=True, slots=True)
class GeometryErrorRecord:
    """CAD-to-polygon approximation diagnostics (never a continuum proof by default)."""

    cad_source_hash: str | None
    declared_curve_tolerance_mm: float | None
    achieved_max_chord_error_mm: float | None
    remesh_status: str = "NOT_EVALUATED"
    remesh_qoi_delta: float | None = None
    excluded_weight_fraction: float | None = None
    is_rigorous_bound: bool = False

    def __post_init__(self) -> None:
        _finite_or_none(self.declared_curve_tolerance_mm, "declared_curve_tolerance_mm")
        _finite_or_none(self.achieved_max_chord_error_mm, "achieved_max_chord_error_mm")
        _finite_or_none(self.remesh_qoi_delta, "remesh_qoi_delta")
        _finite_or_none(self.excluded_weight_fraction, "excluded_weight_fraction")

    def document(self) -> dict[str, Any]:
        return {
            "cad_source_hash": self.cad_source_hash,
            "declared_curve_tolerance_mm": self.declared_curve_tolerance_mm,
            "achieved_max_chord_error_mm": self.achieved_max_chord_error_mm,
            "remesh_status": self.remesh_status,
            "remesh_qoi_delta": self.remesh_qoi_delta,
            "excluded_weight_fraction": self.excluded_weight_fraction,
            "is_rigorous_bound": self.is_rigorous_bound,
        }


@dataclass(frozen=True, slots=True)
class DiscretizationErrorRecord:
    """FEM discretization convergence diagnostics on the chosen polygonal mesh."""

    reference_status: str
    element_order: int
    dof_count: int
    m1_to_m2_qoi_change: float | None = None
    m1_to_m2_u_rel_l2: float | None = None
    m1_to_m2_p_rel_l2: float | None = None
    is_rigorous_bound: bool = False

    def __post_init__(self) -> None:
        if type(self.element_order) is not int or self.element_order not in (1, 2):
            raise ValueError("element_order must be 1 or 2")
        if type(self.dof_count) is not int or self.dof_count < 0:
            raise ValueError("dof_count must be a nonnegative integer")
        _finite_or_none(self.m1_to_m2_qoi_change, "m1_to_m2_qoi_change")
        _finite_or_none(self.m1_to_m2_u_rel_l2, "m1_to_m2_u_rel_l2")
        _finite_or_none(self.m1_to_m2_p_rel_l2, "m1_to_m2_p_rel_l2")

    def document(self) -> dict[str, Any]:
        return {
            "reference_status": self.reference_status,
            "element_order": self.element_order,
            "dof_count": self.dof_count,
            "m1_to_m2_qoi_change": self.m1_to_m2_qoi_change,
            "m1_to_m2_u_rel_l2": self.m1_to_m2_u_rel_l2,
            "m1_to_m2_p_rel_l2": self.m1_to_m2_p_rel_l2,
            "is_rigorous_bound": self.is_rigorous_bound,
        }


@dataclass(frozen=True, slots=True)
class SurrogateErrorRecord:
    """Discrete surrogate-vs-FEM error indicators kept as distinct fields."""

    verdict: str
    eta_energy: float | None
    dwr_estimate: float | None
    newton_qoi_change: float | None
    interpolation_gap: float | None = None
    dirichlet_deviation: float = 0.0
    guards: Mapping[str, bool | None] = ()  # type: ignore[assignment]
    is_rigorous_bound: bool = False

    def __post_init__(self) -> None:
        _finite_or_none(self.eta_energy, "eta_energy")
        _finite_or_none(self.dwr_estimate, "dwr_estimate")
        _finite_or_none(self.newton_qoi_change, "newton_qoi_change")
        _finite_or_none(self.interpolation_gap, "interpolation_gap")
        _finite_or_none(self.dirichlet_deviation, "dirichlet_deviation")
        if not isinstance(self.guards, Mapping):
            object.__setattr__(self, "guards", {})
        else:
            object.__setattr__(self, "guards", dict(sorted(self.guards.items())))

    def document(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "eta_energy": self.eta_energy,
            "dwr_estimate": self.dwr_estimate,
            "newton_qoi_change": self.newton_qoi_change,
            "interpolation_gap": self.interpolation_gap,
            "dirichlet_deviation": self.dirichlet_deviation,
            "guards": dict(self.guards),
            "is_rigorous_bound": self.is_rigorous_bound,
        }


@dataclass(frozen=True, slots=True)
class StatisticalCalibrationRecord:
    """Marginal split-conformal coverage record, never summed into continuum proofs."""

    nominal_coverage: float
    calibration_sample_count: int
    conformal_quantile: float | None
    half_width: float | None
    exchangeability_scope: str
    in_calibration_envelope: bool
    refused: bool
    refusal_reason: str = ""

    def __post_init__(self) -> None:
        if not (0.0 < float(self.nominal_coverage) < 1.0):
            raise ValueError("nominal_coverage must lie strictly in (0, 1)")
        if (
            type(self.calibration_sample_count) is not int
            or self.calibration_sample_count < 0
        ):
            raise ValueError("calibration_sample_count must be nonnegative")
        _finite_or_none(self.conformal_quantile, "conformal_quantile")
        _finite_or_none(self.half_width, "half_width")

    def document(self) -> dict[str, Any]:
        return {
            "nominal_coverage": self.nominal_coverage,
            "calibration_sample_count": self.calibration_sample_count,
            "conformal_quantile": self.conformal_quantile,
            "half_width": self.half_width,
            "exchangeability_scope": self.exchangeability_scope,
            "in_calibration_envelope": self.in_calibration_envelope,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "is_deterministic_error_term": False,
        }


def reduce_qoi_trust_status(
    *,
    in_parameter_envelope: bool,
    in_geometry_envelope: bool,
    solve_failed: bool,
    surrogate: SurrogateErrorRecord,
    discretization: DiscretizationErrorRecord,
    singular_or_reentrant: bool,
    indicators_qualified: bool = False,
) -> tuple[TrustStatus, str]:
    """Fail-closed status reducer enforcing Feature D refusal rules."""
    if not in_parameter_envelope:
        return (
            TrustStatus.OUT_OF_ENVELOPE,
            "query parameters lie outside the declared input envelope",
        )
    if not in_geometry_envelope:
        return (
            TrustStatus.OUT_OF_ENVELOPE,
            "query geometry lies outside supported CAD geometry envelope or topology",
        )
    if solve_failed:
        return (
            TrustStatus.SOLVE_FAILED,
            "discrete reference or escalation solver failed to converge",
        )
    if (
        surrogate.verdict == "INADMISSIBLE"
        or surrogate.guards.get("admissible") is False
    ):
        return (
            TrustStatus.INADMISSIBLE,
            "predicted deformation failed admissibility or Jacobian positivity guard",
        )
    if (
        surrogate.verdict == "INDEFINITE"
        or surrogate.guards.get("positive_definite") is False
    ):
        return (
            TrustStatus.INDEFINITE,
            "linearized tangent is indefinite or singular at surrogate state",
        )
    if discretization.reference_status == "REFERENCE_NOT_CONVERGED":
        return (
            TrustStatus.REFERENCE_NOT_CONVERGED,
            "FEM mesh reference failed its convergence floor; accuracy withheld",
        )
    if singular_or_reentrant:
        return (
            TrustStatus.UNCERTIFIED,
            "singular or re-entrant peak QoI is explicitly withheld from certification",
        )
    if indicators_qualified and discretization.reference_status == "CONVERGED":
        return (
            TrustStatus.QUALIFIED_INDICATORS,
            "discrete indicators passed guards on a converged reference mesh",
        )
    return (
        TrustStatus.DIAGNOSTIC_ONLY,
        "uncalibrated diagnostic indicators only; no certified continuum bound",
    )


@dataclass(frozen=True, slots=True)
class QoITrustResult:
    """Per-QoI trust record separating geometry, FEM, surrogate, and statistical terms."""

    qoi_name: str
    qoi_kind: str
    unit: str
    predicted_value: float | None
    corrected_value: float | None
    status: TrustStatus
    reason: str
    geometry: GeometryErrorRecord
    discretization: DiscretizationErrorRecord
    surrogate: SurrogateErrorRecord
    statistical: StatisticalCalibrationRecord | None = None
    singular_or_reentrant: bool = False
    rigorous_continuum_bound: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.qoi_name, str) or not self.qoi_name.strip():
            raise ValueError("qoi_name must be non-empty text")
        if self.qoi_kind not in SUPPORTED_QOI_KINDS:
            raise ValueError(
                f"unsupported qoi_kind {self.qoi_kind!r}; expected one of "
                f"{sorted(SUPPORTED_QOI_KINDS)}"
            )
        _finite_or_none(self.predicted_value, "predicted_value")
        _finite_or_none(self.corrected_value, "corrected_value")
        _finite_or_none(self.rigorous_continuum_bound, "rigorous_continuum_bound")
        if self.rigorous_continuum_bound is not None and not (
            self.geometry.is_rigorous_bound
            and self.discretization.is_rigorous_bound
            and self.surrogate.is_rigorous_bound
        ):
            raise ValueError(
                "rigorous_continuum_bound requires every deterministic error source "
                "to carry a documented rigorous bound"
            )

    def document(self) -> dict[str, Any]:
        return {
            "qoi_name": self.qoi_name,
            "qoi_kind": self.qoi_kind,
            "unit": self.unit,
            "predicted_value": self.predicted_value,
            "corrected_value": self.corrected_value,
            "status": self.status.value,
            "reason": self.reason,
            "singular_or_reentrant": self.singular_or_reentrant,
            "rigorous_continuum_bound": self.rigorous_continuum_bound,
            "error_components": {
                "geometry_representation": self.geometry.document(),
                "fem_discretization": self.discretization.document(),
                "surrogate_discrete": self.surrogate.document(),
                "statistical_calibration": (
                    None if self.statistical is None else self.statistical.document()
                ),
            },
        }


_STATUS_SEVERITY: Final[dict[TrustStatus, int]] = {
    TrustStatus.SOLVE_FAILED: 7,
    TrustStatus.INADMISSIBLE: 6,
    TrustStatus.OUT_OF_ENVELOPE: 5,
    TrustStatus.INDEFINITE: 4,
    TrustStatus.REFERENCE_NOT_CONVERGED: 3,
    TrustStatus.UNCERTIFIED: 2,
    TrustStatus.DIAGNOSTIC_ONLY: 1,
    TrustStatus.QUALIFIED_INDICATORS: 0,
}


@dataclass(frozen=True, slots=True)
class CADTrustReport:
    """Versioned canonical trust report binding CAD, mesh, problem, and QoI results."""

    problem_hash: str
    mesh_id: str
    cad_source_hash: str | None
    bundle_digest: str | None
    claim_level: str
    qoi_results: tuple[QoITrustResult, ...]
    exact_mode_semantics: str = (
        "converged-discrete-fem-under-declared-mesh-and-solver-tolerances"
    )

    def __post_init__(self) -> None:
        raw = self.claim_level
        value = raw.value if isinstance(raw, enum.Enum) else raw
        if type(value) is not str or value not in ALLOWED_CLAIM_LEVELS:
            raise ValueError(
                f"claim_level must be one of {ALLOWED_CLAIM_LEVELS}, got {raw!r}"
            )
        object.__setattr__(self, "claim_level", value)
        if not self.qoi_results:
            raise ValueError("CADTrustReport requires at least one QoITrustResult")
        names = [item.qoi_name for item in self.qoi_results]
        if len(set(names)) != len(names):
            raise ValueError("qoi_results must have unique qoi_name entries")

    @property
    def overall_status(self) -> TrustStatus:
        return max(
            (item.status for item in self.qoi_results),
            key=lambda s: _STATUS_SEVERITY[s],
        )

    def document(self) -> dict[str, Any]:
        return {
            "schema": CAD_TRUST_REPORT_SCHEMA,
            "problem_hash": self.problem_hash,
            "mesh_id": self.mesh_id,
            "cad_source_hash": self.cad_source_hash,
            "bundle_digest": self.bundle_digest,
            "claim_level": self.claim_level,
            "overall_status": self.overall_status.value,
            "exact_mode_semantics": self.exact_mode_semantics,
            "qoi_results": [item.document() for item in self.qoi_results],
        }

    @property
    def report_hash(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()


def build_cad_trust_report(
    *,
    problem_hash: str,
    mesh_id: str,
    cad_source_hash: str | None,
    bundle_digest: str | None,
    claim_level: str | enum.Enum,
    qoi_results: Sequence[QoITrustResult],
) -> CADTrustReport:
    value = claim_level.value if isinstance(claim_level, enum.Enum) else claim_level
    return CADTrustReport(
        problem_hash=problem_hash,
        mesh_id=mesh_id,
        cad_source_hash=cad_source_hash,
        bundle_digest=bundle_digest,
        claim_level=str(value),
        qoi_results=tuple(qoi_results),
    )


__all__ = (
    "CAD_TRUST_REPORT_SCHEMA",
    "SUPPORTED_QOI_KINDS",
    "CADTrustReport",
    "DiscretizationErrorRecord",
    "GeometryErrorRecord",
    "QoITrustResult",
    "StatisticalCalibrationRecord",
    "SurrogateErrorRecord",
    "TrustStatus",
    "build_cad_trust_report",
    "reduce_qoi_trust_status",
)
