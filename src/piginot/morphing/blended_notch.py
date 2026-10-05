"""Exact fixed-connectivity morphing for a C2-blended edge notch (S17).

The quarter domain ``[0, a] x [0, h(x)]`` has a free upper profile

``h(x) = W/2 - d (1 - (x/b)^2)^3`` for ``x < b`` and ``h(x) = W/2`` otherwise,

with ``b = sqrt(6 d r)``. The notch root lies on ``x = 0``, its depth is ``d``
and its root radius of curvature is exactly ``r``; the profile and its first
two derivatives are continuous at ``x = b``. The half length ``a`` is fixed by
the reference; the width, notch radius and notch depth are the parameters.

The map is the same construction as the circular transition: ``x`` is mapped
piecewise linearly with its knot at the blend length, and ``y`` is scaled by
the ratio of the target and reference profile heights. Polynomials are
evaluated with plain multiplication so every platform gives the same bits.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from piginot.contracts import CanonicalMesh, CanonicalQuantity, Unit, hash_spec
from piginot.geometry.canonicalize import canonicalize_mesh
from piginot.geometry.models import CellType, MeshPayload
from piginot.morphing.circular_transition import _finite, _positive_integer, _tag
from piginot.morphing.models import Matrix2, MorphingError, MorphReasonCode, Point

_PROFILE_TOLERANCE = 0.01


def _invalid(message: str) -> MorphingError:
    return MorphingError(MorphReasonCode.INVALID_MAP_CONFIGURATION, message)


@dataclass(frozen=True, slots=True)
class BlendedNotchParameters:
    """Half length, width, notch radius and notch depth of one notched strip."""

    __canonical_namespace__: ClassVar[str] = "piginot.morphing"

    half_length_mm: float
    width_mm: float
    notch_radius_mm: float
    notch_depth_mm: float

    def __post_init__(self) -> None:
        for field_name in (
            "half_length_mm",
            "width_mm",
            "notch_radius_mm",
            "notch_depth_mm",
        ):
            value = _finite(getattr(self, field_name), field_name)
            if value <= 0.0:
                raise _invalid(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)
        if self.notch_depth_mm >= self.half_width_mm:
            raise _invalid("notch depth leaves no ligament")
        if self.blend_length_mm >= self.half_length_mm:
            raise _invalid("the notch blend does not fit inside the half length")

    @property
    def half_width_mm(self) -> float:
        return self.width_mm / 2.0

    @property
    def blend_length_mm(self) -> float:
        return math.sqrt(6.0 * self.notch_depth_mm * self.notch_radius_mm)

    def profile_height_mm(self, x_value: float) -> float:
        x_coordinate = self._bounded_x(x_value)
        blend = self.blend_length_mm
        if x_coordinate >= blend:
            return self.half_width_mm
        s = x_coordinate / blend
        t = 1.0 - s * s
        return self.half_width_mm - self.notch_depth_mm * (t * t * t)

    def profile_slope(self, x_value: float) -> float:
        x_coordinate = self._bounded_x(x_value)
        blend = self.blend_length_mm
        if x_coordinate >= blend:
            return 0.0
        s = x_coordinate / blend
        t = 1.0 - s * s
        return 6.0 * self.notch_depth_mm * s * (t * t) / blend

    def _bounded_x(self, x_value: float) -> float:
        x_coordinate = _finite(x_value, "profile coordinate")
        tolerance = 1.0e-12 * max(1.0, self.half_length_mm)
        if -tolerance <= x_coordinate <= 0.0:
            return 0.0
        if self.half_length_mm <= x_coordinate <= self.half_length_mm + tolerance:
            return self.half_length_mm
        if 0.0 < x_coordinate < self.half_length_mm:
            return x_coordinate
        raise _invalid("profile coordinate lies outside the quarter domain")


@dataclass(frozen=True, slots=True)
class BlendedNotchMeshSpec:
    """Deterministic structured reference-mesh and semantic-tag recipe."""

    __canonical_namespace__: ClassVar[str] = "piginot.morphing"

    notch_columns: int
    outer_columns: int
    transverse_rows: int
    lower_tag: str
    loaded_end_tag: str
    notch_profile_tag: str
    flat_profile_tag: str
    left_end_tag: str

    def __post_init__(self) -> None:
        for field_name in ("notch_columns", "outer_columns", "transverse_rows"):
            object.__setattr__(
                self,
                field_name,
                _positive_integer(getattr(self, field_name), field_name),
            )
        names = []
        for field_name in (
            "lower_tag",
            "loaded_end_tag",
            "notch_profile_tag",
            "flat_profile_tag",
            "left_end_tag",
        ):
            value = _tag(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)
            names.append(value)
        if len(names) != len(set(names)):
            raise _invalid("reference-mesh semantic tags must be unique")


@dataclass(frozen=True, slots=True)
class BlendedNotchMap:
    """Piecewise analytic map between two blended-notch domains."""

    __canonical_namespace__: ClassVar[str] = "piginot.morphing"
    kind: ClassVar[str] = "blended-notch-profile-v1"

    reference: BlendedNotchParameters
    width_parameter: str
    notch_radius_parameter: str
    notch_depth_parameter: str

    def __post_init__(self) -> None:
        if not isinstance(self.reference, BlendedNotchParameters):
            raise _invalid("reference must be BlendedNotchParameters")
        names = self.parameter_names
        if any(not name.isidentifier() for name in names) or len(set(names)) != 3:
            raise _invalid("map parameter names must be three unique identifiers")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            self.width_parameter,
            self.notch_radius_parameter,
            self.notch_depth_parameter,
        )

    def bind(
        self, theta: Mapping[str, CanonicalQuantity]
    ) -> BoundBlendedNotchDeformation:
        if not isinstance(theta, Mapping):
            raise MorphingError(
                MorphReasonCode.INVALID_PARAMETER_VALUE,
                "theta must be a mapping of canonical quantities",
            )
        submitted = dict(theta)
        expected = frozenset(self.parameter_names)
        if frozenset(submitted) != expected:
            reason = (
                MorphReasonCode.MISSING_PARAMETER
                if expected - frozenset(submitted)
                else MorphReasonCode.UNKNOWN_PARAMETER
            )
            raise MorphingError(
                reason, "theta does not match the map parameter vocabulary"
            )
        values: dict[str, float] = {}
        for name in self.parameter_names:
            quantity = submitted[name]
            if not isinstance(quantity, CanonicalQuantity):
                raise MorphingError(
                    MorphReasonCode.INVALID_PARAMETER_VALUE,
                    f"parameter {name!r} must be a CanonicalQuantity",
                )
            if quantity.unit is not Unit.MM:
                raise MorphingError(
                    MorphReasonCode.PARAMETER_UNIT_MISMATCH,
                    f"parameter {name!r} must use canonical millimetres",
                )
            values[name] = quantity.value
        try:
            target = BlendedNotchParameters(
                half_length_mm=self.reference.half_length_mm,
                width_mm=values[self.width_parameter],
                notch_radius_mm=values[self.notch_radius_parameter],
                notch_depth_mm=values[self.notch_depth_parameter],
            )
        except MorphingError as exc:
            raise MorphingError(
                MorphReasonCode.INVALID_PARAMETER_VALUE, str(exc)
            ) from exc
        canonical_theta = {name: submitted[name] for name in self.parameter_names}
        return BoundBlendedNotchDeformation(
            reference=self.reference,
            target=target,
            parameter_hash=hash_spec(canonical_theta, "content"),
        )


@dataclass(frozen=True, slots=True)
class BoundBlendedNotchDeformation:
    """One parameter-bound profile map with analytic Jacobian."""

    kind: ClassVar[str] = BlendedNotchMap.kind

    reference: BlendedNotchParameters
    target: BlendedNotchParameters
    parameter_hash: str

    def map_point(self, point: Point) -> Point:
        x_value, y_value = self._point(point)
        mapped_x, _ = self._longitudinal_map(x_value)
        scale = self.target.profile_height_mm(
            mapped_x
        ) / self.reference.profile_height_mm(x_value)
        return mapped_x, y_value * scale

    def jacobian(self, point: Point) -> Matrix2:
        x_value, y_value = self._point(point)
        mapped_x, longitudinal_slope = self._longitudinal_map(x_value)
        reference_height = self.reference.profile_height_mm(x_value)
        target_height = self.target.profile_height_mm(mapped_x)
        reference_slope = self.reference.profile_slope(x_value)
        target_slope = self.target.profile_slope(mapped_x)
        scale = target_height / reference_height
        scale_slope = (
            target_slope * longitudinal_slope * reference_height
            - target_height * reference_slope
        ) / (reference_height * reference_height)
        return ((longitudinal_slope, 0.0), (y_value * scale_slope, scale))

    def _point(self, point: Point) -> Point:
        if not isinstance(point, tuple) or len(point) != 2:
            raise _invalid("evaluation point must be a two-coordinate tuple")
        x_value = _finite(point[0], "evaluation x coordinate")
        y_value = _finite(point[1], "evaluation y coordinate")
        half_length = self.reference.half_length_mm
        tolerance = 1.0e-12 * max(1.0, half_length)
        if not -tolerance <= x_value <= half_length + tolerance:
            raise _invalid("evaluation point lies outside the reference length")
        x_value = min(half_length, max(0.0, x_value))
        height = self.reference.profile_height_mm(x_value)
        # Straight cell edges lie slightly off the curved profile; accept the
        # quadrature points of boundary cells, never a point far outside.
        if not -tolerance <= y_value <= height * (1.0 + _PROFILE_TOLERANCE):
            raise _invalid("evaluation point lies outside the reference profile")
        return x_value, max(0.0, y_value)

    def _longitudinal_map(self, x_value: float) -> tuple[float, float]:
        reference_blend = self.reference.blend_length_mm
        target_blend = self.target.blend_length_mm
        half_length = self.reference.half_length_mm
        if x_value <= reference_blend:
            slope = target_blend / reference_blend
            return x_value * slope, slope
        slope = (half_length - target_blend) / (half_length - reference_blend)
        mapped = target_blend + (x_value - reference_blend) * slope
        if x_value == half_length:
            mapped = half_length
        return mapped, slope


def build_blended_notch_reference_mesh(
    parameters: BlendedNotchParameters,
    spec: BlendedNotchMeshSpec,
    *,
    source_provenance: Mapping[str, str],
) -> CanonicalMesh:
    """Build one deterministic structured triangle mesh aligned to the notch."""

    if not isinstance(parameters, BlendedNotchParameters):
        raise TypeError("parameters must be BlendedNotchParameters")
    if not isinstance(spec, BlendedNotchMeshSpec):
        raise TypeError("spec must be BlendedNotchMeshSpec")
    if not isinstance(source_provenance, Mapping) or any(
        not isinstance(key, str) or not key or not isinstance(value, str)
        for key, value in source_provenance.items()
    ):
        raise ValueError("source_provenance must map non-empty strings to strings")

    blend = parameters.blend_length_mm
    outer = parameters.half_length_mm - blend
    x_coordinates = tuple(
        blend * index / spec.notch_columns for index in range(spec.notch_columns + 1)
    ) + tuple(
        blend + outer * index / spec.outer_columns
        for index in range(1, spec.outer_columns + 1)
    )
    x_coordinates = (*x_coordinates[:-1], parameters.half_length_mm)
    row_count = spec.transverse_rows + 1
    nodes = tuple(
        (x_value, parameters.profile_height_mm(x_value) * row / spec.transverse_rows)
        for x_value in x_coordinates
        for row in range(row_count)
    )

    def node(column: int, row: int) -> int:
        return column * row_count + row

    column_count = len(x_coordinates) - 1
    cells = tuple(
        cell
        for column in range(column_count)
        for row in range(spec.transverse_rows)
        for cell in (
            (node(column, row), node(column + 1, row), node(column + 1, row + 1)),
            (node(column, row), node(column + 1, row + 1), node(column, row + 1)),
        )
    )

    boundary_facets: list[tuple[int, int]] = []
    boundary_tags: dict[str, list[int]] = {
        spec.lower_tag: [],
        spec.loaded_end_tag: [],
        spec.notch_profile_tag: [],
        spec.flat_profile_tag: [],
        spec.left_end_tag: [],
    }

    def add_facet(tag_name: str, facet: tuple[int, int]) -> None:
        boundary_tags[tag_name].append(len(boundary_facets))
        boundary_facets.append(facet)

    for column in range(column_count):
        add_facet(spec.lower_tag, (node(column, 0), node(column + 1, 0)))
    for row in range(spec.transverse_rows):
        add_facet(
            spec.loaded_end_tag,
            (node(column_count, row), node(column_count, row + 1)),
        )
    for column in reversed(range(column_count)):
        tag_name = (
            spec.notch_profile_tag
            if column < spec.notch_columns
            else spec.flat_profile_tag
        )
        add_facet(
            tag_name,
            (
                node(column + 1, spec.transverse_rows),
                node(column, spec.transverse_rows),
            ),
        )
    for row in reversed(range(spec.transverse_rows)):
        add_facet(spec.left_end_tag, (node(0, row + 1), node(0, row)))

    payload = MeshPayload(
        nodes=nodes,
        cells=cells,
        cell_type=CellType.TRIANGLE,
        boundary_facets=tuple(boundary_facets),
        boundary_tags={name: tuple(indices) for name, indices in boundary_tags.items()},
        region_ids=tuple(1 for _ in cells),
        unit=Unit.MM,
        source_provenance=dict(source_provenance),
    )
    return canonicalize_mesh(payload)


__all__ = (
    "BlendedNotchMap",
    "BlendedNotchMeshSpec",
    "BlendedNotchParameters",
    "BoundBlendedNotchDeformation",
    "build_blended_notch_reference_mesh",
)
