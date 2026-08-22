"""Versioned, recipe-addressable body-force providers for S7."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from piginot.contracts import hash_spec
from piginot.contracts.specs import SpecContract

FloatArray = NDArray[np.float64]


@runtime_checkable
class BodyForceProvider(Protocol):
    """Evaluate canonical reference-volume force density in N/mm^3."""

    @property
    def provider_id(self) -> str: ...

    def evaluate(self, reference_points: FloatArray) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class ConstantBodyForce(SpecContract):
    """A constant body force whose content determines its provider identity."""

    vector_n_per_mm3: tuple[float, float]

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.vector_n_per_mm3)
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("body-force vector must contain two finite values")
        object.__setattr__(self, "vector_n_per_mm3", values)

    @property
    def provider_id(self) -> str:
        return hash_spec(self, "recipe")

    def evaluate(self, reference_points: FloatArray) -> FloatArray:
        if reference_points.ndim != 3 or reference_points.shape[0] != 2:
            raise ValueError("reference points must have shape (2, elements, points)")
        vector = np.asarray(self.vector_n_per_mm3, dtype=np.float64)[:, None, None]
        return np.broadcast_to(vector, reference_points.shape).copy()


__all__ = ("BodyForceProvider", "ConstantBodyForce", "FloatArray")
