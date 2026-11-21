"""Per-CAD split-conformal QoI and functional-band calibration sidecars (Tier 1a).

Extends S23 split-conformal prediction beyond the two historical specimen
bundles to arbitrary user-compiled CAD bundles. Supports:

* **Scalar QoI split-conformal bounds** (``|Q_FEM - Q_pred| <= q_hat``)
  requiring ``ceil((n + 1)(1 - alpha)) <= n`` (for example ``n >= 19`` for
  nominal 95% coverage at ``alpha = 0.05``).
* **Functional-band calibration** (pointwise simultaneous field bands via
  supremum nonconformity ``max_j |y(x_j) - y_hat(x_j)| / s(x_j)``, mirroring
  PhysicsNeMo's ``FunctionalBandCalibrator``).

Every calibration artifact is bound by RFC 8785 JCS SHA-256 to the target
bundle's ``bundle_manifest_sha256`` and an explicit in-envelope gate so out-of-
distribution queries refuse cleanly instead of extrapolating bounds.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from piginot.bounds.conformal import split_conformal_bound
from piginot.certify.trust_report import StatisticalCalibrationRecord
from piginot.contracts.canonical import canonical_json_bytes

CAD_CALIBRATION_SCHEMA: Final = "formamech.cad-conformal-calibration-v1"


@dataclass(frozen=True, slots=True)
class FunctionalBandCalibration:
    """Distribution-free functional-band conformal calibration over field points."""

    field_name: str
    unit: str
    alpha: float
    sample_count: int
    rank_k: int
    supremum_quantile: float
    scale_floor: float = 1.0e-12

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - self.alpha

    def band_half_widths(
        self, local_scale: Sequence[float] | None = None
    ) -> list[float] | float:
        if local_scale is None:
            return self.supremum_quantile
        return [
            self.supremum_quantile * max(abs(float(s)), self.scale_floor)
            for s in local_scale
        ]

    def document(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "unit": self.unit,
            "alpha": self.alpha,
            "nominal_coverage": self.nominal_coverage,
            "sample_count": self.sample_count,
            "rank_k": self.rank_k,
            "supremum_quantile": self.supremum_quantile,
            "scale_floor": self.scale_floor,
        }


def calibrate_functional_band(
    *,
    field_name: str,
    unit: str,
    residuals: Sequence[Sequence[float]],
    alpha: float = 0.05,
    scales: Sequence[Sequence[float]] | None = None,
    scale_floor: float = 1.0e-12,
) -> FunctionalBandCalibration:
    """Calibrate a finite-sample functional band from exchangeable field residuals."""
    if not residuals:
        raise ValueError("residuals must contain at least one calibration sample")
    if scale_floor <= 0.0 or not math.isfinite(scale_floor):
        raise ValueError("scale_floor must be finite and positive")
    sup_scores: list[float] = []
    for idx, sample in enumerate(residuals):
        arr = np.asarray(sample, dtype=np.float64).reshape(-1)
        if arr.size == 0 or not bool(np.isfinite(arr).all()):
            raise ValueError(f"calibration residual {idx} must be non-empty and finite")
        if scales is not None:
            sc = np.asarray(scales[idx], dtype=np.float64).reshape(-1)
            if sc.shape != arr.shape or not bool(np.isfinite(sc).all()):
                raise ValueError(f"scale {idx} must align with residual {idx}")
            normed = np.abs(arr) / np.maximum(np.abs(sc), scale_floor)
        else:
            normed = np.abs(arr)
        sup_scores.append(float(np.max(normed)))

    bound = split_conformal_bound(sup_scores, alpha)
    return FunctionalBandCalibration(
        field_name=field_name,
        unit=unit,
        alpha=bound.alpha,
        sample_count=bound.n,
        rank_k=bound.k,
        supremum_quantile=bound.bound,
        scale_floor=float(scale_floor),
    )


@dataclass(frozen=True, slots=True)
class ScalarQoiCalibration:
    """Split-conformal calibration for one scalar engineering QoI."""

    qoi_name: str
    unit: str
    alpha: float
    sample_count: int
    rank_k: int
    quantile: float

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - self.alpha

    def document(self) -> dict[str, Any]:
        return {
            "qoi_name": self.qoi_name,
            "unit": self.unit,
            "alpha": self.alpha,
            "nominal_coverage": self.nominal_coverage,
            "sample_count": self.sample_count,
            "rank_k": self.rank_k,
            "quantile": self.quantile,
        }


@dataclass(frozen=True, slots=True)
class CadCalibrationRecord:
    """Bundle-bound split-conformal + functional-band sidecar for arbitrary CAD."""

    bundle_name: str
    bundle_manifest_sha256: str
    problem_hash: str | None
    exchangeability_scope: str
    alpha: float
    qoi_calibrations: Mapping[str, ScalarQoiCalibration]
    functional_bands: Mapping[str, FunctionalBandCalibration] = ()  # type: ignore[assignment]
    parameter_bounds: Mapping[str, tuple[float, float]] = ()  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.bundle_name, str) or not self.bundle_name.strip():
            raise ValueError("bundle_name must be non-empty text")
        if (
            not isinstance(self.bundle_manifest_sha256, str)
            or len(self.bundle_manifest_sha256) != 64
        ):
            raise ValueError("bundle_manifest_sha256 must be a 64-char hex digest")
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must lie strictly in (0, 1)")
        object.__setattr__(
            self, "qoi_calibrations", dict(sorted(self.qoi_calibrations.items()))
        )
        fb = (
            dict(sorted(self.functional_bands.items()))
            if isinstance(self.functional_bands, Mapping)
            else {}
        )
        object.__setattr__(self, "functional_bands", fb)
        pb = (
            {
                k: (float(v[0]), float(v[1]))
                for k, v in sorted(self.parameter_bounds.items())
            }
            if isinstance(self.parameter_bounds, Mapping)
            else {}
        )
        object.__setattr__(self, "parameter_bounds", pb)

    def in_envelope(self, inputs: Mapping[str, Any]) -> bool:
        for name, (lo, hi) in self.parameter_bounds.items():
            if name not in inputs:
                return False
            raw = inputs[name]
            val = float(getattr(raw, "value", raw))
            if not math.isfinite(val) or val < lo - 1e-12 or val > hi + 1e-12:
                return False
        return True

    def statistical_record(
        self, qoi_name: str, *, in_envelope: bool
    ) -> StatisticalCalibrationRecord | None:
        cal = self.qoi_calibrations.get(qoi_name)
        if cal is None:
            return None
        if not in_envelope:
            return StatisticalCalibrationRecord(
                nominal_coverage=cal.nominal_coverage,
                calibration_sample_count=cal.sample_count,
                conformal_quantile=None,
                half_width=None,
                exchangeability_scope=self.exchangeability_scope,
                in_calibration_envelope=False,
                refused=True,
                refusal_reason="query lies outside calibration parameter envelope",
            )
        return StatisticalCalibrationRecord(
            nominal_coverage=cal.nominal_coverage,
            calibration_sample_count=cal.sample_count,
            conformal_quantile=cal.quantile,
            half_width=cal.quantile,
            exchangeability_scope=self.exchangeability_scope,
            in_calibration_envelope=True,
            refused=False,
            refusal_reason="",
        )

    def evaluate(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        ok = self.in_envelope(inputs)
        if not ok:
            return {
                "schema": CAD_CALIBRATION_SCHEMA,
                "bundle_manifest_sha256": self.bundle_manifest_sha256,
                "nominal_coverage": 1.0 - self.alpha,
                "in_calibration_envelope": False,
                "refused": True,
                "refusal_reason": "query lies outside calibration parameter envelope",
                "qoi_bounds": {},
                "functional_bands": {},
            }
        return {
            "schema": CAD_CALIBRATION_SCHEMA,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "nominal_coverage": 1.0 - self.alpha,
            "in_calibration_envelope": True,
            "refused": False,
            "refusal_reason": "",
            "qoi_bounds": {
                name: cal.document() for name, cal in self.qoi_calibrations.items()
            },
            "functional_bands": {
                name: band.document() for name, band in self.functional_bands.items()
            },
        }

    def document(self) -> dict[str, Any]:
        return {
            "schema": CAD_CALIBRATION_SCHEMA,
            "bundle_name": self.bundle_name,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "problem_hash": self.problem_hash,
            "exchangeability_scope": self.exchangeability_scope,
            "alpha": self.alpha,
            "nominal_coverage": 1.0 - self.alpha,
            "parameter_bounds": {
                k: [v[0], v[1]] for k, v in self.parameter_bounds.items()
            },
            "qoi_calibrations": {
                k: v.document() for k, v in self.qoi_calibrations.items()
            },
            "functional_bands": {
                k: v.document() for k, v in self.functional_bands.items()
            },
        }

    @property
    def calibration_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()

    def save(self, path: Path) -> None:
        payload = {**self.document(), "calibration_sha256": self.calibration_sha256}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")

    @classmethod
    def load(
        cls, path: Path, *, expected_manifest_sha256: str | None = None
    ) -> CadCalibrationRecord:
        raw = json.loads(path.read_text("utf-8"))
        if raw.get("schema") != CAD_CALIBRATION_SCHEMA:
            raise ValueError(f"{path} is not a {CAD_CALIBRATION_SCHEMA} artifact")
        if (
            expected_manifest_sha256 is not None
            and raw.get("bundle_manifest_sha256") != expected_manifest_sha256
        ):
            raise ValueError("CAD calibration sidecar belongs to a different bundle")
        qois = {
            name: ScalarQoiCalibration(
                qoi_name=entry["qoi_name"],
                unit=entry["unit"],
                alpha=float(entry["alpha"]),
                sample_count=int(entry["sample_count"]),
                rank_k=int(entry["rank_k"]),
                quantile=float(entry["quantile"]),
            )
            for name, entry in raw.get("qoi_calibrations", {}).items()
        }
        bands = {
            name: FunctionalBandCalibration(
                field_name=entry["field_name"],
                unit=entry["unit"],
                alpha=float(entry["alpha"]),
                sample_count=int(entry["sample_count"]),
                rank_k=int(entry["rank_k"]),
                supremum_quantile=float(entry["supremum_quantile"]),
                scale_floor=float(entry.get("scale_floor", 1.0e-12)),
            )
            for name, entry in raw.get("functional_bands", {}).items()
        }
        pbounds = {
            name: (float(pair[0]), float(pair[1]))
            for name, pair in raw.get("parameter_bounds", {}).items()
        }
        record = cls(
            bundle_name=str(raw["bundle_name"]),
            bundle_manifest_sha256=str(raw["bundle_manifest_sha256"]),
            problem_hash=raw.get("problem_hash"),
            exchangeability_scope=str(raw["exchangeability_scope"]),
            alpha=float(raw["alpha"]),
            qoi_calibrations=qois,
            functional_bands=bands,
            parameter_bounds=pbounds,
        )
        recorded_sha = raw.get("calibration_sha256")
        if recorded_sha is not None and recorded_sha != record.calibration_sha256:
            raise ValueError("CAD calibration sidecar SHA-256 digest mismatch")
        return record


def build_cad_calibration(
    *,
    bundle_name: str,
    bundle_manifest_sha256: str,
    problem_hash: str | None,
    qoi_scores: Mapping[str, tuple[str, Sequence[float]]],
    alpha: float = 0.05,
    exchangeability_scope: str = "in-envelope-exchangeable-cad-calibration",
    functional_residuals: Mapping[str, tuple[str, Sequence[Sequence[float]]]]
    | None = None,
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> CadCalibrationRecord:
    """Calibrate scalar QoIs and optional functional bands for a CAD bundle."""
    qoi_cals: dict[str, ScalarQoiCalibration] = {}
    for name, (unit, scores) in qoi_scores.items():
        bound = split_conformal_bound(scores, alpha)
        qoi_cals[name] = ScalarQoiCalibration(
            qoi_name=name,
            unit=unit,
            alpha=bound.alpha,
            sample_count=bound.n,
            rank_k=bound.k,
            quantile=bound.bound,
        )
    bands: dict[str, FunctionalBandCalibration] = {}
    if functional_residuals is not None:
        for name, (unit, res) in functional_residuals.items():
            bands[name] = calibrate_functional_band(
                field_name=name,
                unit=unit,
                residuals=res,
                alpha=alpha,
            )
    return CadCalibrationRecord(
        bundle_name=bundle_name,
        bundle_manifest_sha256=bundle_manifest_sha256,
        problem_hash=problem_hash,
        exchangeability_scope=exchangeability_scope,
        alpha=alpha,
        qoi_calibrations=qoi_cals,
        functional_bands=bands,
        parameter_bounds={} if parameter_bounds is None else parameter_bounds,
    )


__all__ = (
    "CAD_CALIBRATION_SCHEMA",
    "CadCalibrationRecord",
    "FunctionalBandCalibration",
    "ScalarQoiCalibration",
    "build_cad_calibration",
    "calibrate_functional_band",
)
