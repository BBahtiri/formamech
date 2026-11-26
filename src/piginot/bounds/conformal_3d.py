"""Negligible-overhead (<1%) 3D split-conformal UQ & AS6983 conservative shifts (V5-1).

Provides single-forward-pass, NumPy-only calibrated uncertainty quantification for
3D neural operators with zero FEM overhead during ``fast`` serving:

* **Two-sided split-conformal intervals** scaled by a fast latent-distance
  difficulty proxy ``sigma_hat(z)``, fitted strictly on ``DEV`` cases.
* **AS6983 conservative one-sided shifted bounds** protecting against
  underpredicting safety-critical peaks (von Mises hotspot stress, displacement,
  temperature) with exact finite-sample exchangeability guarantee
  ``P(y_true > y_shifted) <= alpha``.
* **Physical-unit error summaries** mandating mean, p90, and worst-case (max)
  held-out error reporting in physical units (``mm``, ``MPa``, ``K``).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from piginot.bounds.conformal import acceptance_count, split_conformal_bound
from piginot.contracts.canonical import canonical_json_bytes

FloatArray = NDArray[np.float64]

CONFORMAL_3D_SCHEMA: Final = "formamech.conformal-3d-v1"
AS6983_ASSESSMENT_SCHEMA: Final = "formamech.as6983-assessment-v1"


def _sha256_doc(doc: object) -> str:
    return hashlib.sha256(canonical_json_bytes(doc)).hexdigest()


def _finite_float(
    val: float, label: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(f"{label} must be a real number")
    num = float(val)
    if not math.isfinite(num):
        raise ValueError(f"{label} must be finite, got {val!r}")
    if positive and num <= 0.0:
        raise ValueError(f"{label} must be strictly positive, got {num}")
    if nonnegative and num < 0.0:
        raise ValueError(f"{label} must be nonnegative, got {num}")
    return num


@dataclass(frozen=True, slots=True)
class PhysicalUnitErrorSummary:
    """Mandatory mean, p90, and worst-case error reporting in physical units."""

    qoi_name: str
    unit: str
    sample_count: int
    mean_abs_error: float
    p90_abs_error: float
    max_abs_error: float
    mean_rel_error: float
    max_rel_error: float

    def document(self) -> dict[str, Any]:
        return {
            "qoi_name": self.qoi_name,
            "unit": self.unit,
            "sample_count": self.sample_count,
            "mean_abs_error": float(self.mean_abs_error),
            "p90_abs_error": float(self.p90_abs_error),
            "max_abs_error": float(self.max_abs_error),
            "mean_rel_error": float(self.mean_rel_error),
            "max_rel_error": float(self.max_rel_error),
        }


def summarize_physical_errors(
    qoi_name: str,
    unit: str,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    rel_floor: float = 1.0e-12,
) -> PhysicalUnitErrorSummary:
    """Compute mean, p90, and worst-case absolute and relative physical errors."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal non-empty length")
    t_arr = np.asarray(y_true, dtype=np.float64)
    p_arr = np.asarray(y_pred, dtype=np.float64)
    if not (np.all(np.isfinite(t_arr)) and np.all(np.isfinite(p_arr))):
        raise ValueError("y_true and y_pred must contain only finite values")
    abs_err = np.abs(t_arr - p_arr)
    rel_err = abs_err / np.maximum(np.abs(t_arr), float(rel_floor))
    return PhysicalUnitErrorSummary(
        qoi_name=qoi_name,
        unit=unit,
        sample_count=int(t_arr.size),
        mean_abs_error=float(np.mean(abs_err)),
        p90_abs_error=float(np.quantile(abs_err, 0.90)),
        max_abs_error=float(np.max(abs_err)),
        mean_rel_error=float(np.mean(rel_err)),
        max_rel_error=float(np.max(rel_err)),
    )


@dataclass(frozen=True, slots=True)
class Conformal3DDevSample:
    """One held-out ``DEV`` calibration sample for 3D split-conformal & AS6983 fitting."""

    case_id: str
    predicted_qois: Mapping[str, float]
    true_qois: Mapping[str, float]
    latent_vector: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be non-empty text")
        if not self.predicted_qois:
            raise ValueError("predicted_qois must be non-empty")
        for k, v in self.predicted_qois.items():
            _finite_float(v, f"predicted_qois[{k!r}]")
        for k, v in self.true_qois.items():
            _finite_float(v, f"true_qois[{k!r}]")
        for i, v in enumerate(self.latent_vector):
            _finite_float(v, f"latent_vector[{i}]")


@dataclass(frozen=True, slots=True)
class Conformal3DQoiCalibration:
    """DEV-calibrated two-sided conformal + AS6983 one-sided shift parameters for one QoI."""

    qoi_name: str
    unit: str
    alpha: float
    calibration_n: int
    two_sided_quantile: float
    as6983_upper_quantile: float
    base_scale: float
    dev_error_summary: PhysicalUnitErrorSummary

    def document(self) -> dict[str, Any]:
        return {
            "qoi_name": self.qoi_name,
            "unit": self.unit,
            "alpha": float(self.alpha),
            "calibration_n": self.calibration_n,
            "two_sided_quantile": float(self.two_sided_quantile),
            "as6983_upper_quantile": float(self.as6983_upper_quantile),
            "base_scale": float(self.base_scale),
            "dev_error_summary": self.dev_error_summary.document(),
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> Conformal3DQoiCalibration:
        err_doc = doc["dev_error_summary"]
        return cls(
            qoi_name=str(doc["qoi_name"]),
            unit=str(doc["unit"]),
            alpha=float(doc["alpha"]),
            calibration_n=int(doc["calibration_n"]),
            two_sided_quantile=float(doc["two_sided_quantile"]),
            as6983_upper_quantile=float(doc["as6983_upper_quantile"]),
            base_scale=float(doc["base_scale"]),
            dev_error_summary=PhysicalUnitErrorSummary(
                qoi_name=str(err_doc["qoi_name"]),
                unit=str(err_doc["unit"]),
                sample_count=int(err_doc["sample_count"]),
                mean_abs_error=float(err_doc["mean_abs_error"]),
                p90_abs_error=float(err_doc["p90_abs_error"]),
                max_abs_error=float(err_doc["max_abs_error"]),
                mean_rel_error=float(err_doc["mean_rel_error"]),
                max_rel_error=float(err_doc["max_rel_error"]),
            ),
        )


@dataclass(frozen=True, slots=True)
class As6983ConservativeBound:
    """One-sided AS6983 conservative shifted bound preventing peak underprediction."""

    qoi_name: str
    unit: str
    alpha: float
    predicted_value: float
    lower_conformal_bound: float
    upper_conformal_bound: float
    conservative_shifted_value: float
    shift_margin: float
    difficulty_scale: float
    calibration_n: int

    def document(self) -> dict[str, Any]:
        return {
            "qoi_name": self.qoi_name,
            "unit": self.unit,
            "alpha": float(self.alpha),
            "predicted_value": float(self.predicted_value),
            "lower_conformal_bound": float(self.lower_conformal_bound),
            "upper_conformal_bound": float(self.upper_conformal_bound),
            "conservative_shifted_value": float(self.conservative_shifted_value),
            "shift_margin": float(self.shift_margin),
            "difficulty_scale": float(self.difficulty_scale),
            "calibration_n": self.calibration_n,
        }


@dataclass(frozen=True, slots=True)
class Conformal3DAssessment:
    """Negligible-overhead per-inference UQ & AS6983 conservative shift verdict."""

    calibration_sha256: str
    alpha: float
    difficulty_scale: float
    latent_distance: float
    bounds: Mapping[str, As6983ConservativeBound]
    refused: bool = False
    reason: str = "calibrated"

    def document(self) -> dict[str, Any]:
        return {
            "schema": AS6983_ASSESSMENT_SCHEMA,
            "calibration_sha256": self.calibration_sha256,
            "alpha": float(self.alpha),
            "difficulty_scale": float(self.difficulty_scale),
            "latent_distance": float(self.latent_distance),
            "refused": self.refused,
            "reason": self.reason,
            "bounds": {k: v.document() for k, v in sorted(self.bounds.items())},
        }


@dataclass(frozen=True, slots=True)
class Conformal3DCalibrationRecord:
    """Content-addressed DEV-fitted 3D conformal & AS6983 calibration artifact."""

    bundle_hash: str
    alpha: float
    dev_case_ids: tuple[str, ...]
    latent_mean: tuple[float, ...]
    latent_std: tuple[float, ...]
    latent_slope: float
    qoi_calibrations: Mapping[str, Conformal3DQoiCalibration]

    @property
    def calibration_sha256(self) -> str:
        return _sha256_doc(self.document())

    def difficulty_scale(
        self, latent_vector: Sequence[float] | None = None
    ) -> tuple[float, float]:
        """Compute O(D) latent-distance difficulty multiplier ``>= 1.0``."""
        if not self.latent_mean or latent_vector is None or len(latent_vector) == 0:
            return 1.0, 0.0
        if len(latent_vector) != len(self.latent_mean):
            raise ValueError(
                f"latent_vector dim {len(latent_vector)} != calibrated dim {len(self.latent_mean)}"
            )
        z = np.asarray(latent_vector, dtype=np.float64)
        mu = np.asarray(self.latent_mean, dtype=np.float64)
        sd = np.asarray(self.latent_std, dtype=np.float64)
        dist = float(np.sqrt(np.mean(np.square((z - mu) / sd))))
        scale = 1.0 + self.latent_slope * dist
        return scale, dist

    def evaluate(
        self,
        predicted_qois: Mapping[str, float],
        *,
        latent_vector: Sequence[float] | None = None,
    ) -> Conformal3DAssessment:
        """Evaluate two-sided conformal & AS6983 one-sided bounds in microseconds."""
        scale, dist = self.difficulty_scale(latent_vector)
        out: dict[str, As6983ConservativeBound] = {}
        for qname, cal in self.qoi_calibrations.items():
            if qname not in predicted_qois:
                continue
            pred = _finite_float(predicted_qois[qname], f"predicted_qois[{qname!r}]")
            half_width = cal.two_sided_quantile * scale
            # AS6983 one-sided safety shift never reduces a peak prediction below pred
            shift_margin = max(cal.as6983_upper_quantile, 0.0) * scale
            out[qname] = As6983ConservativeBound(
                qoi_name=qname,
                unit=cal.unit,
                alpha=cal.alpha,
                predicted_value=pred,
                lower_conformal_bound=pred - half_width,
                upper_conformal_bound=pred + half_width,
                conservative_shifted_value=pred + shift_margin,
                shift_margin=shift_margin,
                difficulty_scale=scale,
                calibration_n=cal.calibration_n,
            )
        return Conformal3DAssessment(
            calibration_sha256=self.calibration_sha256,
            alpha=self.alpha,
            difficulty_scale=scale,
            latent_distance=dist,
            bounds=out,
            refused=False,
            reason="calibrated",
        )

    def document(self) -> dict[str, Any]:
        return {
            "schema": CONFORMAL_3D_SCHEMA,
            "bundle_hash": self.bundle_hash,
            "alpha": float(self.alpha),
            "dev_case_ids": list(self.dev_case_ids),
            "latent_mean": [float(v) for v in self.latent_mean],
            "latent_std": [float(v) for v in self.latent_std],
            "latent_slope": float(self.latent_slope),
            "qoi_calibrations": {
                k: v.document() for k, v in sorted(self.qoi_calibrations.items())
            },
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> Conformal3DCalibrationRecord:
        if doc.get("schema") != CONFORMAL_3D_SCHEMA:
            raise ValueError(
                f"unexpected conformal 3D schema {doc.get('schema')!r}; expected {CONFORMAL_3D_SCHEMA!r}"
            )
        qois = {
            str(k): Conformal3DQoiCalibration.from_document(v)
            for k, v in dict(doc["qoi_calibrations"]).items()
        }
        return cls(
            bundle_hash=str(doc["bundle_hash"]),
            alpha=float(doc["alpha"]),
            dev_case_ids=tuple(str(cid) for cid in doc["dev_case_ids"]),
            latent_mean=tuple(float(v) for v in doc["latent_mean"]),
            latent_std=tuple(float(v) for v in doc["latent_std"]),
            latent_slope=float(doc["latent_slope"]),
            qoi_calibrations=qois,
        )


class Conformal3DCalibrator:
    """Fits ``DEV``-only 3D split-conformal & AS6983 conservative shift records."""

    @staticmethod
    def fit(
        bundle_hash: str,
        dev_samples: Sequence[Conformal3DDevSample],
        *,
        qoi_units: Mapping[str, str],
        alpha: float = 0.05,
        latent_slope: float = 0.15,
    ) -> Conformal3DCalibrationRecord:
        """Fit exact split-conformal and AS6983 upper quantiles on ``DEV`` cases."""
        _finite_float(alpha, "alpha", positive=True)
        if alpha >= 1.0:
            raise ValueError("alpha must lie strictly in (0, 1)")
        _finite_float(latent_slope, "latent_slope", nonnegative=True)
        if not dev_samples:
            raise ValueError("dev_samples must be non-empty")

        ids = [s.case_id for s in dev_samples]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate case_id detected in DEV calibration samples")

        has_latents = all(len(s.latent_vector) > 0 for s in dev_samples)
        if has_latents:
            dim = len(dev_samples[0].latent_vector)
            if any(len(s.latent_vector) != dim for s in dev_samples):
                raise ValueError("all DEV latent_vectors must share identical length")
            mat = np.asarray([s.latent_vector for s in dev_samples], dtype=np.float64)
            mu = mat.mean(axis=0)
            sd = np.maximum(mat.std(axis=0), 1.0e-3)
            dists = np.sqrt(
                np.mean(np.square((mat - mu[None, :]) / sd[None, :]), axis=1)
            )
            scales = 1.0 + latent_slope * dists
            latent_mean = tuple(float(v) for v in mu)
            latent_std = tuple(float(v) for v in sd)
        else:
            scales = np.ones(len(dev_samples), dtype=np.float64)
            latent_mean = ()
            latent_std = ()

        qoi_names = sorted(
            set.intersection(
                *(set(s.predicted_qois) & set(s.true_qois) for s in dev_samples)
            )
        )
        if not qoi_names:
            raise ValueError("no shared predicted/true QoI keys across DEV samples")

        calibrations: dict[str, Conformal3DQoiCalibration] = {}
        for qname in qoi_names:
            y_pred = [float(s.predicted_qois[qname]) for s in dev_samples]
            y_true = [float(s.true_qois[qname]) for s in dev_samples]
            abs_scores = [
                abs(yt - yp) / float(sc)
                for yt, yp, sc in zip(y_true, y_pred, scales, strict=True)
            ]
            # Signed upper underprediction score: positive when y_true > y_pred
            upper_scores = [
                (yt - yp) / float(sc)
                for yt, yp, sc in zip(y_true, y_pred, scales, strict=True)
            ]
            two_sided = split_conformal_bound(abs_scores, alpha)
            as6983 = split_conformal_bound(upper_scores, alpha)
            unit = str(qoi_units.get(qname, "unitless"))
            err_sum = summarize_physical_errors(qname, unit, y_true, y_pred)
            calibrations[qname] = Conformal3DQoiCalibration(
                qoi_name=qname,
                unit=unit,
                alpha=alpha,
                calibration_n=len(dev_samples),
                two_sided_quantile=two_sided.bound,
                as6983_upper_quantile=as6983.bound,
                base_scale=float(np.mean(scales)),
                dev_error_summary=err_sum,
            )

        return Conformal3DCalibrationRecord(
            bundle_hash=bundle_hash,
            alpha=alpha,
            dev_case_ids=tuple(sorted(ids)),
            latent_mean=latent_mean,
            latent_std=latent_std,
            latent_slope=latent_slope,
            qoi_calibrations=calibrations,
        )


@dataclass(frozen=True, slots=True)
class As6983HoldoutCoverageReport:
    """Beta-Binomial verification of AS6983 non-underprediction coverage on HOLDOUT."""

    qoi_name: str
    unit: str
    alpha: float
    calibration_n: int
    holdout_m: int
    covered_two_sided_count: int
    non_underpredicted_count: int
    required_acceptance_count: int
    two_sided_passed: bool
    as6983_passed: bool
    holdout_error_summary: PhysicalUnitErrorSummary

    @property
    def empirical_non_underprediction_rate(self) -> float:
        return float(self.non_underpredicted_count) / float(max(self.holdout_m, 1))

    @property
    def empirical_two_sided_coverage(self) -> float:
        return float(self.covered_two_sided_count) / float(max(self.holdout_m, 1))

    def document(self) -> dict[str, Any]:
        return {
            "qoi_name": self.qoi_name,
            "unit": self.unit,
            "alpha": float(self.alpha),
            "calibration_n": self.calibration_n,
            "holdout_m": self.holdout_m,
            "covered_two_sided_count": self.covered_two_sided_count,
            "non_underpredicted_count": self.non_underpredicted_count,
            "required_acceptance_count": self.required_acceptance_count,
            "empirical_two_sided_coverage": self.empirical_two_sided_coverage,
            "empirical_non_underprediction_rate": self.empirical_non_underprediction_rate,
            "two_sided_passed": self.two_sided_passed,
            "as6983_passed": self.as6983_passed,
            "holdout_error_summary": self.holdout_error_summary.document(),
        }


def verify_as6983_holdout_coverage(
    record: Conformal3DCalibrationRecord,
    holdout_samples: Sequence[Conformal3DDevSample],
    *,
    test_level: float = 0.05,
) -> dict[str, As6983HoldoutCoverageReport]:
    """Verify two-sided conformal and AS6983 shifted coverage on blind HOLDOUT."""
    if not holdout_samples:
        raise ValueError("holdout_samples must be non-empty")
    dev_id_set = set(record.dev_case_ids)
    for s in holdout_samples:
        if s.case_id in dev_id_set:
            raise ValueError(
                f"split leakage refused: HOLDOUT case {s.case_id!r} was used in DEV calibration"
            )

    reports: dict[str, As6983HoldoutCoverageReport] = {}
    m = len(holdout_samples)
    for qname, cal in record.qoi_calibrations.items():
        if not all(
            qname in s.predicted_qois and qname in s.true_qois for s in holdout_samples
        ):
            continue
        two_ok = 0
        as6983_ok = 0
        y_true: list[float] = []
        y_pred: list[float] = []
        for s in holdout_samples:
            assess = record.evaluate(
                s.predicted_qois,
                latent_vector=s.latent_vector if s.latent_vector else None,
            )
            bnd = assess.bounds[qname]
            yt = float(s.true_qois[qname])
            yp = float(s.predicted_qois[qname])
            y_true.append(yt)
            y_pred.append(yp)
            if bnd.lower_conformal_bound <= yt <= bnd.upper_conformal_bound:
                two_ok += 1
            if yt <= bnd.conservative_shifted_value:
                as6983_ok += 1

        req = acceptance_count(cal.calibration_n, cal.alpha, m, test_level)
        reports[qname] = As6983HoldoutCoverageReport(
            qoi_name=qname,
            unit=cal.unit,
            alpha=cal.alpha,
            calibration_n=cal.calibration_n,
            holdout_m=m,
            covered_two_sided_count=two_ok,
            non_underpredicted_count=as6983_ok,
            required_acceptance_count=req,
            two_sided_passed=two_ok >= req,
            as6983_passed=as6983_ok >= req,
            holdout_error_summary=summarize_physical_errors(
                qname, cal.unit, y_true, y_pred
            ),
        )
    return reports


__all__ = (
    "AS6983_ASSESSMENT_SCHEMA",
    "CONFORMAL_3D_SCHEMA",
    "As6983ConservativeBound",
    "As6983HoldoutCoverageReport",
    "Conformal3DAssessment",
    "Conformal3DCalibrationRecord",
    "Conformal3DCalibrator",
    "Conformal3DDevSample",
    "Conformal3DQoiCalibration",
    "PhysicalUnitErrorSummary",
    "summarize_physical_errors",
    "verify_as6983_holdout_coverage",
)
