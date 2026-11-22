"""Declarative numerical qualification policies (pure contract, zero ML/FEM imports).

A :class:`QualificationPolicy` declares the exact numerical acceptance
thresholds, conformal calibration requirements, DWR effectivity bounds, and
target :class:`~piginot.verification.status.ClaimLevel` used to qualify a
trained FormaMech bundle before deployment.

Qualification establishes *numerical verification against a declared FEM
reference*, never experimental validation or regulatory approval.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from piginot.contracts.canonical import canonical_json_bytes
from piginot.verification.status import ClaimLevel

QUALIFICATION_POLICY_SCHEMA: Final = "formamech.qualification-policy-v1"
_VALID_TARGETS: Final = (
    ClaimLevel.DEVELOPMENT_VERIFIED,
    ClaimLevel.HOLDOUT_VERIFIED,
)


def _finite_number(
    value: object,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{label} must be finite")
    if positive and num <= 0.0:
        raise ValueError(f"{label} must be strictly positive")
    if nonnegative and num < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return num


def _quantile(value: object, label: str) -> float:
    q = _finite_number(value, label, positive=True)
    if q > 1.0:
        raise ValueError(f"{label} must lie in (0, 1]")
    return q


def _int_at_least(value: object, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class QoiTolerance:
    """Acceptance tolerance for one scalar engineering quantity of interest."""

    max_rel_error: float | None = None
    max_abs_error: float | None = None
    quantile: float = 0.90
    rel_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.max_rel_error is None and self.max_abs_error is None:
            raise ValueError(
                "QoiTolerance requires at least one of max_rel_error or max_abs_error"
            )
        if self.max_rel_error is not None:
            object.__setattr__(
                self,
                "max_rel_error",
                _finite_number(self.max_rel_error, "max_rel_error", positive=True),
            )
        if self.max_abs_error is not None:
            object.__setattr__(
                self,
                "max_abs_error",
                _finite_number(self.max_abs_error, "max_abs_error", positive=True),
            )
        object.__setattr__(self, "quantile", _quantile(self.quantile, "quantile"))
        object.__setattr__(
            self,
            "rel_floor",
            _finite_number(self.rel_floor, "rel_floor", positive=True),
        )

    def document(self) -> dict[str, Any]:
        return {
            "max_rel_error": self.max_rel_error,
            "max_abs_error": self.max_abs_error,
            "quantile": self.quantile,
            "rel_floor": self.rel_floor,
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> QoiTolerance:
        return cls(
            max_rel_error=doc.get("max_rel_error"),
            max_abs_error=doc.get("max_abs_error"),
            quantile=float(doc.get("quantile", 0.90)),
            rel_floor=float(doc.get("rel_floor", 1.0e-12)),
        )


@dataclass(frozen=True, slots=True)
class FieldTolerance:
    """Field-level accuracy and physical admissibility thresholds."""

    disp_rel_l2_max: float = 0.05
    stress_rel_l2_max: float = 0.10
    quantile: float = 0.90
    min_det_f_floor: float = 1.0e-6
    max_dirichlet_residual: float = 1.0e-6

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disp_rel_l2_max",
            _finite_number(self.disp_rel_l2_max, "disp_rel_l2_max", positive=True),
        )
        object.__setattr__(
            self,
            "stress_rel_l2_max",
            _finite_number(self.stress_rel_l2_max, "stress_rel_l2_max", positive=True),
        )
        object.__setattr__(self, "quantile", _quantile(self.quantile, "quantile"))
        object.__setattr__(
            self,
            "min_det_f_floor",
            _finite_number(self.min_det_f_floor, "min_det_f_floor", positive=True),
        )
        object.__setattr__(
            self,
            "max_dirichlet_residual",
            _finite_number(
                self.max_dirichlet_residual, "max_dirichlet_residual", nonnegative=True
            ),
        )

    def document(self) -> dict[str, Any]:
        return {
            "disp_rel_l2_max": self.disp_rel_l2_max,
            "stress_rel_l2_max": self.stress_rel_l2_max,
            "quantile": self.quantile,
            "min_det_f_floor": self.min_det_f_floor,
            "max_dirichlet_residual": self.max_dirichlet_residual,
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> FieldTolerance:
        return cls(
            disp_rel_l2_max=float(doc.get("disp_rel_l2_max", 0.05)),
            stress_rel_l2_max=float(doc.get("stress_rel_l2_max", 0.10)),
            quantile=float(doc.get("quantile", 0.90)),
            min_det_f_floor=float(doc.get("min_det_f_floor", 1.0e-6)),
            max_dirichlet_residual=float(doc.get("max_dirichlet_residual", 1.0e-6)),
        )


@dataclass(frozen=True, slots=True)
class CalibrationPolicy:
    """Split-conformal QoI and functional-band calibration requirements."""

    enabled: bool = True
    alpha: float = 0.05
    min_calibration_cases: int = 19
    require_functional_band: bool = True
    min_holdout_coverage: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean")
        if not isinstance(self.require_functional_band, bool):
            raise TypeError("require_functional_band must be boolean")
        alpha = _finite_number(self.alpha, "alpha", positive=True)
        if alpha >= 1.0:
            raise ValueError("alpha must lie strictly in (0, 1)")
        object.__setattr__(self, "alpha", alpha)
        min_n = _int_at_least(self.min_calibration_cases, "min_calibration_cases", 1)
        if self.enabled:
            required_rank = math.ceil((min_n + 1) * (1.0 - alpha))
            if required_rank > min_n:
                min_feasible = math.ceil((1.0 - alpha) / alpha)
                raise ValueError(
                    f"min_calibration_cases={min_n} is too small for alpha={alpha}; "
                    f"split-conformal requires at least {min_feasible} cases"
                )
        object.__setattr__(self, "min_calibration_cases", min_n)
        if self.min_holdout_coverage is not None:
            cov = _quantile(self.min_holdout_coverage, "min_holdout_coverage")
            object.__setattr__(self, "min_holdout_coverage", cov)

    def document(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "alpha": self.alpha,
            "min_calibration_cases": self.min_calibration_cases,
            "require_functional_band": self.require_functional_band,
            "min_holdout_coverage": self.min_holdout_coverage,
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> CalibrationPolicy:
        raw_cov = doc.get("min_holdout_coverage")
        return cls(
            enabled=bool(doc.get("enabled", True)),
            alpha=float(doc.get("alpha", 0.05)),
            min_calibration_cases=int(doc.get("min_calibration_cases", 19)),
            require_functional_band=bool(doc.get("require_functional_band", True)),
            min_holdout_coverage=None if raw_cov is None else float(raw_cov),
        )


@dataclass(frozen=True, slots=True)
class DwrPolicy:
    """Goal-oriented DWR adjoint effectivity requirements."""

    enabled: bool = True
    effectivity_min: float = 0.1
    effectivity_max: float = 10.0
    min_valid_fraction: float = 0.75
    error_floor: float = 1.0e-8

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean")
        emin = _finite_number(self.effectivity_min, "effectivity_min", positive=True)
        emax = _finite_number(self.effectivity_max, "effectivity_max", positive=True)
        if emax <= emin:
            raise ValueError("effectivity_max must strictly exceed effectivity_min")
        object.__setattr__(self, "effectivity_min", emin)
        object.__setattr__(self, "effectivity_max", emax)
        object.__setattr__(
            self,
            "min_valid_fraction",
            _quantile(self.min_valid_fraction, "min_valid_fraction"),
        )
        object.__setattr__(
            self,
            "error_floor",
            _finite_number(self.error_floor, "error_floor", positive=True),
        )

    def document(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "effectivity_min": self.effectivity_min,
            "effectivity_max": self.effectivity_max,
            "min_valid_fraction": self.min_valid_fraction,
            "error_floor": self.error_floor,
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> DwrPolicy:
        return cls(
            enabled=bool(doc.get("enabled", True)),
            effectivity_min=float(doc.get("effectivity_min", 0.1)),
            effectivity_max=float(doc.get("effectivity_max", 10.0)),
            min_valid_fraction=float(doc.get("min_valid_fraction", 0.75)),
            error_floor=float(doc.get("error_floor", 1.0e-8)),
        )


@dataclass(frozen=True, slots=True)
class QualificationPolicy:
    """Immutable, content-addressed numerical qualification policy."""

    policy_id: str
    target_claim_level: ClaimLevel = ClaimLevel.DEVELOPMENT_VERIFIED
    field_tolerances: FieldTolerance = field(default_factory=FieldTolerance)
    qoi_tolerances: Mapping[str, QoiTolerance] = field(default_factory=dict)
    calibration: CalibrationPolicy = field(default_factory=CalibrationPolicy)
    dwr: DwrPolicy = field(default_factory=DwrPolicy)
    require_zero_fem_failures: bool = True
    require_partition_disjointness: bool = True
    require_zero_inadmissible: bool = True
    report_breakeven: bool = True
    training_wall_seconds: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("policy_id must be non-empty text")
        target = (
            self.target_claim_level
            if isinstance(self.target_claim_level, ClaimLevel)
            else ClaimLevel(str(self.target_claim_level))
        )
        if target not in _VALID_TARGETS:
            raise ValueError(
                f"target_claim_level must be one of {[c.value for c in _VALID_TARGETS]}"
            )
        object.__setattr__(self, "target_claim_level", target)
        if not isinstance(self.field_tolerances, FieldTolerance):
            raise TypeError("field_tolerances must be a FieldTolerance")
        if not isinstance(self.calibration, CalibrationPolicy):
            raise TypeError("calibration must be a CalibrationPolicy")
        if not isinstance(self.dwr, DwrPolicy):
            raise TypeError("dwr must be a DwrPolicy")
        cleaned_qois: dict[str, QoiTolerance] = {}
        for name, tol in sorted(self.qoi_tolerances.items()):
            if not isinstance(name, str) or not name.strip():
                raise ValueError("QoI tolerance names must be non-empty text")
            if not isinstance(tol, QoiTolerance):
                raise TypeError(f"qoi_tolerances[{name!r}] must be a QoiTolerance")
            cleaned_qois[name] = tol
        object.__setattr__(self, "qoi_tolerances", cleaned_qois)
        for flag in (
            "require_zero_fem_failures",
            "require_partition_disjointness",
            "require_zero_inadmissible",
            "report_breakeven",
        ):
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be boolean")
        if self.training_wall_seconds is not None:
            object.__setattr__(
                self,
                "training_wall_seconds",
                _finite_number(
                    self.training_wall_seconds,
                    "training_wall_seconds",
                    nonnegative=True,
                ),
            )
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")

    def document(self) -> dict[str, Any]:
        return {
            "schema": QUALIFICATION_POLICY_SCHEMA,
            "policy_id": self.policy_id,
            "target_claim_level": self.target_claim_level.value,
            "field_tolerances": self.field_tolerances.document(),
            "qoi_tolerances": {
                name: tol.document()
                for name, tol in sorted(self.qoi_tolerances.items())
            },
            "calibration": self.calibration.document(),
            "dwr": self.dwr.document(),
            "require_zero_fem_failures": self.require_zero_fem_failures,
            "require_partition_disjointness": self.require_partition_disjointness,
            "require_zero_inadmissible": self.require_zero_inadmissible,
            "report_breakeven": self.report_breakeven,
            "training_wall_seconds": self.training_wall_seconds,
            "notes": self.notes,
        }

    @property
    def policy_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()

    def save(self, path: Path) -> None:
        payload = {**self.document(), "policy_sha256": self.policy_sha256}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> QualificationPolicy:
        if doc.get("schema") != QUALIFICATION_POLICY_SCHEMA:
            raise ValueError(f"document is not a {QUALIFICATION_POLICY_SCHEMA} policy")
        raw_qois = doc.get("qoi_tolerances", {})
        if not isinstance(raw_qois, Mapping):
            raise ValueError("qoi_tolerances must be a mapping")
        qoi_tols = {str(k): QoiTolerance.from_document(v) for k, v in raw_qois.items()}
        raw_train = doc.get("training_wall_seconds")
        policy = cls(
            policy_id=str(doc["policy_id"]),
            target_claim_level=ClaimLevel(str(doc["target_claim_level"])),
            field_tolerances=FieldTolerance.from_document(
                doc.get("field_tolerances", {})
            ),
            qoi_tolerances=qoi_tols,
            calibration=CalibrationPolicy.from_document(doc.get("calibration", {})),
            dwr=DwrPolicy.from_document(doc.get("dwr", {})),
            require_zero_fem_failures=bool(doc.get("require_zero_fem_failures", True)),
            require_partition_disjointness=bool(
                doc.get("require_partition_disjointness", True)
            ),
            require_zero_inadmissible=bool(doc.get("require_zero_inadmissible", True)),
            report_breakeven=bool(doc.get("report_breakeven", True)),
            training_wall_seconds=None if raw_train is None else float(raw_train),
            notes=str(doc.get("notes", "")),
        )
        recorded_sha = doc.get("policy_sha256")
        if recorded_sha is not None and recorded_sha != policy.policy_sha256:
            raise ValueError("qualification policy SHA-256 digest mismatch")
        return policy

    @classmethod
    def load(cls, path: Path) -> QualificationPolicy:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("qualification policy must be a JSON object")
        return cls.from_document(raw)


def default_development_policy(
    *,
    policy_id: str = "formamech-dev-standard-v1",
    disp_rel_l2_max: float = 0.05,
    stress_rel_l2_max: float = 0.10,
    alpha: float = 0.10,
    min_calibration_cases: int = 9,
) -> QualificationPolicy:
    """Standard policy for ``DEVELOPMENT_VERIFIED`` qualification."""
    return QualificationPolicy(
        policy_id=policy_id,
        target_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
        field_tolerances=FieldTolerance(
            disp_rel_l2_max=disp_rel_l2_max,
            stress_rel_l2_max=stress_rel_l2_max,
            quantile=0.90,
        ),
        qoi_tolerances={
            "avg_p11": QoiTolerance(max_rel_error=stress_rel_l2_max, quantile=0.90),
            "total_strain_energy": QoiTolerance(
                max_rel_error=stress_rel_l2_max, quantile=0.90
            ),
        },
        calibration=CalibrationPolicy(
            enabled=True,
            alpha=alpha,
            min_calibration_cases=min_calibration_cases,
            require_functional_band=True,
        ),
        dwr=DwrPolicy(enabled=True),
    )


def default_holdout_policy(
    *,
    policy_id: str = "formamech-holdout-strict-v1",
    disp_rel_l2_max: float = 0.03,
    stress_rel_l2_max: float = 0.06,
    alpha: float = 0.05,
    min_calibration_cases: int = 19,
    min_holdout_coverage: float = 0.85,
) -> QualificationPolicy:
    """Strict policy for ``HOLDOUT_VERIFIED`` qualification with conformal coverage."""
    return QualificationPolicy(
        policy_id=policy_id,
        target_claim_level=ClaimLevel.HOLDOUT_VERIFIED,
        field_tolerances=FieldTolerance(
            disp_rel_l2_max=disp_rel_l2_max,
            stress_rel_l2_max=stress_rel_l2_max,
            quantile=0.90,
        ),
        qoi_tolerances={
            "avg_p11": QoiTolerance(max_rel_error=stress_rel_l2_max, quantile=0.90),
            "total_strain_energy": QoiTolerance(
                max_rel_error=stress_rel_l2_max, quantile=0.90
            ),
        },
        calibration=CalibrationPolicy(
            enabled=True,
            alpha=alpha,
            min_calibration_cases=min_calibration_cases,
            require_functional_band=True,
            min_holdout_coverage=min_holdout_coverage,
        ),
        dwr=DwrPolicy(enabled=True),
    )


__all__ = (
    "QUALIFICATION_POLICY_SCHEMA",
    "CalibrationPolicy",
    "DwrPolicy",
    "FieldTolerance",
    "QoiTolerance",
    "QualificationPolicy",
    "default_development_policy",
    "default_holdout_policy",
)
