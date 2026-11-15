"""Identity geometry mapping for a model tied to one imported mesh."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from piginot.contracts import CanonicalQuantity, hash_spec
from piginot.morphing.models import Matrix2, MorphingError, MorphReasonCode, Point


def _point(value: object) -> Point:
    if not isinstance(value, tuple) or len(value) != 2:
        raise MorphingError(
            MorphReasonCode.INVALID_MAP_CONFIGURATION,
            "evaluation point must be a two-coordinate tuple",
        )
    if any(
        isinstance(coordinate, bool) or not isinstance(coordinate, (int, float))
        for coordinate in value
    ):
        raise MorphingError(
            MorphReasonCode.INVALID_MAP_CONFIGURATION,
            "evaluation coordinates must be real numbers",
        )
    coordinates = (float(value[0]), float(value[1]))
    if not all(math.isfinite(coordinate) for coordinate in coordinates):
        raise MorphingError(
            MorphReasonCode.INVALID_MAP_CONFIGURATION,
            "evaluation coordinates must be finite",
        )
    return coordinates


@dataclass(frozen=True, slots=True)
class FixedGeometryMap:
    """A parameter-free recipe; load and material inputs do not deform geometry."""

    __canonical_namespace__: ClassVar[str] = "piginot.morphing"
    kind: ClassVar[str] = "fixed-geometry-v1"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ()

    def bind(
        self, theta: Mapping[str, CanonicalQuantity]
    ) -> BoundFixedGeometryDeformation:
        if not isinstance(theta, Mapping):
            raise MorphingError(
                MorphReasonCode.INVALID_PARAMETER_VALUE,
                "theta must be an empty mapping of geometry parameters",
            )
        if theta:
            raise MorphingError(
                MorphReasonCode.UNKNOWN_PARAMETER,
                "a fixed geometry map accepts no geometry parameters",
            )
        return BoundFixedGeometryDeformation()


@dataclass(frozen=True, slots=True)
class BoundFixedGeometryDeformation:
    """Exact identity and inverse, with a stable empty-parameter binding."""

    kind: ClassVar[str] = FixedGeometryMap.kind

    @property
    def parameter_hash(self) -> str:
        return hash_spec({}, "content")

    def map_point(self, point: Point) -> Point:
        return _point(point)

    def inverse_point(self, point: Point) -> Point:
        return _point(point)

    def jacobian(self, point: Point) -> Matrix2:
        _point(point)
        return ((1.0, 0.0), (0.0, 1.0))
