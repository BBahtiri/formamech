"""The calibration envelope gate: Mahalanobis distance to TRAIN (S23, ADR 0104).

The distance is taken to the TRAIN mean under the TRAIN covariance, on the
primitive axes that vary across TRAIN (as the S20 post-mortem's D7). A request
is inside the calibration envelope if its distance is at most the largest
distance among the calibration cases. Every number is stored, so the gate
evaluates identically after a canonical-JSON round trip.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

_FIXED_AXIS_RELATIVE_SPREAD = 1e-9
_MAX_CONDITION = 1e12


class GateError(ValueError):
    """A malformed gate, or a request that lacks a gated input."""


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise GateError(f"{label} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class MahalanobisGate:
    """TRAIN statistics and the largest calibration distance."""

    axes: tuple[str, ...]
    mean: tuple[float, ...]
    precision: tuple[tuple[float, ...], ...]
    max_distance: float

    def __post_init__(self) -> None:
        if not self.axes or len(set(self.axes)) != len(self.axes):
            raise GateError("axes must be non-empty and unique")
        size = len(self.axes)
        if len(self.mean) != size or len(self.precision) != size:
            raise GateError("mean and precision must match the axes")
        if any(len(row) != size for row in self.precision):
            raise GateError("precision must be square")
        for value in self.mean:
            _finite(value, "mean")
        for row in self.precision:
            for value in row:
                _finite(value, "precision")
        if _finite(self.max_distance, "max_distance") < 0.0:
            raise GateError("max_distance must not be negative")

    @classmethod
    def fit(
        cls,
        axes: Sequence[str],
        train: Sequence[Sequence[float]],
        calibration: Sequence[Sequence[float]],
    ) -> MahalanobisGate:
        """Fit on TRAIN points; the threshold is the largest calibration distance.

        Points are rows of values in the order of ``axes``. Axes that do not
        vary across TRAIN (relative spread below 1e-9) are dropped.
        """

        names = tuple(axes)
        points = np.asarray(train, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != len(names):
            raise GateError("train must be a (cases, axes) array")
        if not np.isfinite(points).all():
            raise GateError("train must be finite")
        spread = points.std(axis=0)
        keep = spread > _FIXED_AXIS_RELATIVE_SPREAD * np.maximum(
            1.0, np.abs(points.mean(axis=0))
        )
        if not keep.any():
            raise GateError("no axis varies across TRAIN")
        kept = points[:, keep]
        if kept.shape[0] <= kept.shape[1]:
            raise GateError("TRAIN needs more cases than varying axes")
        covariance = np.atleast_2d(np.cov(kept, rowvar=False))
        if not np.linalg.cond(covariance) < _MAX_CONDITION:
            raise GateError("the TRAIN covariance is singular on the varying axes")
        precision = np.linalg.inv(covariance)
        precision = 0.5 * (precision + precision.T)
        partial = cls(
            tuple(name for name, k in zip(names, keep, strict=True) if k),
            tuple(float(v) for v in kept.mean(axis=0)),
            tuple(tuple(float(v) for v in row) for row in precision),
            0.0,
        )
        rows = np.asarray(calibration, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(names) or rows.shape[0] == 0:
            raise GateError("calibration must be a non-empty (cases, axes) array")
        distances = [
            partial.distance(dict(zip(names, row, strict=True))) for row in rows
        ]
        return cls(partial.axes, partial.mean, partial.precision, max(distances))

    def distance(self, values: Mapping[str, Any]) -> float:
        """Mahalanobis distance of one request's values to the TRAIN mean."""

        try:
            point = np.array(
                [_finite(_value_of(values[axis]), axis) for axis in self.axes],
                dtype=np.float64,
            )
        except KeyError as exc:
            raise GateError(f"the request has no input {exc.args[0]!r}") from exc
        centred = point - np.asarray(self.mean, dtype=np.float64)
        squared = float(
            centred @ np.asarray(self.precision, dtype=np.float64) @ centred
        )
        return math.sqrt(max(squared, 0.0))

    def admits(self, values: Mapping[str, Any]) -> bool:
        return self.distance(values) <= self.max_distance

    def document(self) -> dict[str, Any]:
        return {
            "axes": list(self.axes),
            "mean": list(self.mean),
            "precision": [list(row) for row in self.precision],
            "max_distance": self.max_distance,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> MahalanobisGate:
        try:
            return cls(
                tuple(document["axes"]),
                tuple(float(v) for v in document["mean"]),
                tuple(tuple(float(v) for v in row) for row in document["precision"]),
                float(document["max_distance"]),
            )
        except (KeyError, TypeError) as exc:
            raise GateError(f"malformed gate document: {exc}") from exc


def _value_of(value: Any) -> Any:
    """A canonical quantity's value, or a plain number."""

    return getattr(value, "value", value)


__all__ = ("GateError", "MahalanobisGate")
