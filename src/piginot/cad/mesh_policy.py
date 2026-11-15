"""Explicit, immutable sizing and acceptance limits for prepared CAD meshes.

Cell and P2-vector-DOF budgets apply to the largest prepared reference mesh,
including its level-4 midpoint refinement. These are input settings, not
evidence that a particular geometry or finite-element solution is accurate.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from piginot.contracts import canonical_json_bytes, strict_json_loads

SCHEMA = "piginot.cad-mesh-policy-v1"
_BOUNDARY = re.compile(r"edge-[0-9]{6}\Z")
_SIZE_FIELDS = ("size_mm", "distance_min_mm", "distance_max_mm")
_POLICY_FIELDS = (
    "schema",
    "global_size_mm",
    "minimum_size_mm",
    "curve_tolerance_mm",
    "curvature_points_per_circle",
    "feature_sizes",
    "point_sizes",
    "minimum_angle_deg",
    "maximum_aspect_ratio",
    "maximum_triangles",
    "maximum_p2_dofs",
)


def _number(value: object, name: str, *, positive: bool = True) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (positive and value <= 0):
        raise ValueError(f"{name} must be a positive finite number")


def _integer(value: object, name: str, minimum: int = 1) -> None:
    if type(value) is not int or not minimum <= value <= 2**53 - 1:
        raise ValueError(f"{name} must be an integer between {minimum} and 2**53-1")


def _transition(size: float, inner: float, outer: float) -> None:
    _number(size, "size_mm")
    _number(inner, "distance_min_mm", positive=False)
    _number(outer, "distance_max_mm")
    if inner < 0:
        raise ValueError("distance_min_mm must be nonnegative")
    if outer <= inner:
        raise ValueError("distance_max_mm must exceed distance_min_mm")


def _object(value: object, fields: tuple[str, ...], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{name} has invalid fields")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


@dataclass(frozen=True, slots=True)
class FeatureSize:
    boundary: str
    size_mm: float
    distance_min_mm: float
    distance_max_mm: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.boundary, str)
            or _BOUNDARY.fullmatch(self.boundary) is None
        ):
            raise ValueError("boundary must use an imported edge-NNNNNN identifier")
        _transition(self.size_mm, self.distance_min_mm, self.distance_max_mm)

    def document(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "size_mm": self.size_mm,
            "distance_min_mm": self.distance_min_mm,
            "distance_max_mm": self.distance_max_mm,
        }


@dataclass(frozen=True, slots=True)
class PointSize:
    center_mm: tuple[float, float]
    size_mm: float
    distance_min_mm: float
    distance_max_mm: float

    def __post_init__(self) -> None:
        if type(self.center_mm) is not tuple or len(self.center_mm) != 2:
            raise ValueError("center_mm must be an immutable pair of coordinates")
        for value in self.center_mm:
            _number(value, "center_mm", positive=False)
        _transition(self.size_mm, self.distance_min_mm, self.distance_max_mm)

    def document(self) -> dict[str, Any]:
        return {
            "center_mm": list(self.center_mm),
            "size_mm": self.size_mm,
            "distance_min_mm": self.distance_min_mm,
            "distance_max_mm": self.distance_max_mm,
        }


@dataclass(frozen=True, slots=True)
class MeshPolicy:
    global_size_mm: float
    minimum_size_mm: float
    curve_tolerance_mm: float
    curvature_points_per_circle: int
    feature_sizes: tuple[FeatureSize, ...]
    point_sizes: tuple[PointSize, ...]
    minimum_angle_deg: float
    maximum_aspect_ratio: float
    maximum_triangles: int
    maximum_p2_dofs: int

    def __post_init__(self) -> None:
        for name in (
            "global_size_mm",
            "minimum_size_mm",
            "curve_tolerance_mm",
            "minimum_angle_deg",
            "maximum_aspect_ratio",
        ):
            _number(getattr(self, name), name)
        _integer(self.curvature_points_per_circle, "curvature_points_per_circle", 8)
        _integer(self.maximum_triangles, "maximum_triangles")
        _integer(self.maximum_p2_dofs, "maximum_p2_dofs")
        if self.minimum_size_mm > self.global_size_mm:
            raise ValueError("minimum_size_mm must not exceed global_size_mm")
        if self.minimum_angle_deg > 60:
            raise ValueError("minimum_angle_deg must not exceed 60 degrees")
        for name, kind in (("feature_sizes", FeatureSize), ("point_sizes", PointSize)):
            targets = getattr(self, name)
            if type(targets) is not tuple or any(
                type(item) is not kind for item in targets
            ):
                raise ValueError(
                    f"{name} must be an immutable tuple of {kind.__name__}"
                )
            if any(
                not self.minimum_size_mm <= item.size_mm <= self.global_size_mm
                for item in targets
            ):
                raise ValueError(
                    f"{name} size_mm must lie within the global size limits"
                )
        boundaries = [item.boundary for item in self.feature_sizes]
        centers = [item.center_mm for item in self.point_sizes]
        if len(set(boundaries)) != len(boundaries) or len(set(centers)) != len(centers):
            raise ValueError("duplicate mesh target selectors are not allowed")

    def document(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "global_size_mm": self.global_size_mm,
            "minimum_size_mm": self.minimum_size_mm,
            "curve_tolerance_mm": self.curve_tolerance_mm,
            "curvature_points_per_circle": self.curvature_points_per_circle,
            "feature_sizes": [item.document() for item in self.feature_sizes],
            "point_sizes": [item.document() for item in self.point_sizes],
            "minimum_angle_deg": self.minimum_angle_deg,
            "maximum_aspect_ratio": self.maximum_aspect_ratio,
            "maximum_triangles": self.maximum_triangles,
            "maximum_p2_dofs": self.maximum_p2_dofs,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()

    @classmethod
    def from_document(cls, value: object) -> MeshPolicy:
        document = _object(value, _POLICY_FIELDS, "mesh policy")
        if document["schema"] != SCHEMA:
            raise ValueError("unsupported mesh policy schema")
        features = tuple(
            FeatureSize(**_object(item, ("boundary", *_SIZE_FIELDS), "feature size"))
            for item in _array(document["feature_sizes"], "feature_sizes")
        )
        points = []
        for item in _array(document["point_sizes"], "point_sizes"):
            point = _object(item, ("center_mm", *_SIZE_FIELDS), "point size")
            center = _array(point["center_mm"], "center_mm")
            if len(center) != 2:
                raise ValueError("center_mm must contain exactly two coordinates")
            points.append(
                PointSize(
                    (center[0], center[1]),
                    point["size_mm"],
                    point["distance_min_mm"],
                    point["distance_max_mm"],
                )
            )
        return cls(
            **{
                **{
                    key: value
                    for key, value in document.items()
                    if key
                    not in (
                        "schema",
                        "feature_sizes",
                        "point_sizes",
                    )
                },
                "feature_sizes": features,
                "point_sizes": tuple(points),
            }
        )

    @classmethod
    def load(cls, path: Path) -> MeshPolicy:
        path = Path(path)
        if not path.is_file() or any(
            p.is_symlink() for p in (path, *path.absolute().parents)
        ):
            raise ValueError(
                "mesh policy input must be a regular file without symlinks"
            )
        return cls.from_document(strict_json_loads(path.read_bytes()))
