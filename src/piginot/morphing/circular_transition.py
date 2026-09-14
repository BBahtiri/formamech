"""Exact fixed-connectivity morphing for a circular width transition."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from piginot.contracts import (
    CanonicalMesh,
    CanonicalQuantity,
    Unit,
    hash_spec,
)
from piginot.geometry.canonicalize import canonicalize_mesh
from piginot.geometry.models import CellType, MeshPayload
from piginot.morphing.models import Matrix2, MorphingError, MorphReasonCode, Point


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MorphingError(
            MorphReasonCode.INVALID_MAP_CONFIGURATION,
            f"{label} must be a real number",
        )
    number = float(value)
    if not math.isfinite(number):
        raise MorphingError(
            MorphReasonCode.INVALID_MAP_CONFIGURATION,
            f"{label} must be finite",
        )
    return number


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MorphingError(
            MorphReasonCode.INVALID_MAP_CONFIGURATION,
            f"{label} must be a positive integer",
        )
    return value


def _tag(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MorphingError(
            MorphReasonCode.INVALID_MAP_CONFIGURATION,
            f"{label} must be non-empty text",
        )
    return value


@dataclass(frozen=True, slots=True)
class CircularTransitionParameters:
    """Four physical dimensions defining one symmetric circular transition."""

    __canonical_namespace__: ClassVar[str] = "piginot.morphing"

    total_length_mm: float
    outer_width_mm: float
    inner_width_mm: float
    transition_radius_mm: float

    def __post_init__(self) -> None:
        for field_name in (
            "total_length_mm",
            "outer_width_mm",
            "inner_width_mm",
            "transition_radius_mm",
        ):
            value = _finite(getattr(self, field_name), field_name)
            if value <= 0.0:
                raise MorphingError(
                    MorphReasonCode.INVALID_MAP_CONFIGURATION,
                    f"{field_name} must be positive",
                )
            object.__setattr__(self, field_name, value)
        if self.outer_width_mm <= self.inner_width_mm:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "outer width must be greater than inner width",
            )
        if self.transition_radius_mm <= self.half_width_difference_mm:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "transition radius must exceed the half-width difference",
            )
        if self.transition_start_x_mm <= 0.0:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "circular transition leaves no positive inner section",
            )

    @property
    def half_length_mm(self) -> float:
        return self.total_length_mm / 2.0

    @property
    def outer_half_width_mm(self) -> float:
        return self.outer_width_mm / 2.0

    @property
    def inner_half_width_mm(self) -> float:
        return self.inner_width_mm / 2.0

    @property
    def half_width_difference_mm(self) -> float:
        return self.outer_half_width_mm - self.inner_half_width_mm

    @property
    def transition_span_mm(self) -> float:
        difference = self.half_width_difference_mm
        return math.sqrt(difference * (2.0 * self.transition_radius_mm - difference))

    @property
    def transition_start_x_mm(self) -> float:
        return self.half_length_mm - self.transition_span_mm

    def profile_height_mm(self, x_value: float) -> float:
        x_coordinate = self._bounded_x(x_value)
        if x_coordinate is None:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "profile coordinate lies outside the quarter domain",
            )
        if x_coordinate <= self.transition_start_x_mm:
            return self.inner_half_width_mm
        offset = x_coordinate - self.transition_start_x_mm
        radicand = self.transition_radius_mm**2 - offset**2
        if radicand < -1.0e-12:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "circular transition radicand is negative",
            )
        return (
            self.inner_half_width_mm
            + self.transition_radius_mm
            - math.sqrt(max(0.0, radicand))
        )

    def profile_slope(self, x_value: float) -> float:
        x_coordinate = self._bounded_x(x_value)
        if x_coordinate is None:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "profile coordinate lies outside the quarter domain",
            )
        if x_coordinate <= self.transition_start_x_mm:
            return 0.0
        offset = x_coordinate - self.transition_start_x_mm
        denominator = math.sqrt(max(0.0, self.transition_radius_mm**2 - offset**2))
        if denominator == 0.0:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "circular transition has an unbounded endpoint slope",
            )
        return offset / denominator

    def _bounded_x(self, x_value: float) -> float | None:
        x_coordinate = _finite(x_value, "profile coordinate")
        tolerance = 1.0e-12 * max(1.0, self.half_length_mm)
        if -tolerance <= x_coordinate <= 0.0:
            return 0.0
        if self.half_length_mm <= x_coordinate <= self.half_length_mm + tolerance:
            return self.half_length_mm
        if 0.0 < x_coordinate < self.half_length_mm:
            return x_coordinate
        return None


@dataclass(frozen=True, slots=True)
class CircularTransitionMeshSpec:
    """Deterministic structured reference-mesh and semantic-tag recipe."""

    __canonical_namespace__: ClassVar[str] = "piginot.morphing"

    inner_columns: int
    transition_columns: int
    transverse_rows: int
    lower_tag: str
    loaded_end_tag: str
    transition_profile_tag: str
    inner_profile_tag: str
    left_end_tag: str

    def __post_init__(self) -> None:
        for field_name in (
            "inner_columns",
            "transition_columns",
            "transverse_rows",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_integer(getattr(self, field_name), field_name),
            )
        tag_names = []
        for field_name in (
            "lower_tag",
            "loaded_end_tag",
            "transition_profile_tag",
            "inner_profile_tag",
            "left_end_tag",
        ):
            value = _tag(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)
            tag_names.append(value)
        if len(tag_names) != len(set(tag_names)):
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "reference-mesh semantic tags must be unique",
            )


@dataclass(frozen=True, slots=True)
class CircularTransitionMap:
    """Piecewise analytic map between two circular-transition domains."""

    __canonical_namespace__: ClassVar[str] = "piginot.morphing"
    kind: ClassVar[str] = "circular-transition-profile-v1"

    reference: CircularTransitionParameters
    total_length_parameter: str
    outer_width_parameter: str
    inner_width_parameter: str
    transition_radius_parameter: str
    minimum_inner_length_mm: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.reference, CircularTransitionParameters):
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "reference must be CircularTransitionParameters",
            )
        names = self.parameter_names
        if any(not name.isidentifier() for name in names) or len(set(names)) != 4:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "map parameter names must be four unique identifiers",
            )
        minimum = _finite(self.minimum_inner_length_mm, "minimum_inner_length_mm")
        if minimum < 0.0:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "minimum inner length must be non-negative",
            )
        if self.reference.transition_start_x_mm < minimum:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "reference violates the minimum inner length",
            )
        object.__setattr__(self, "minimum_inner_length_mm", minimum)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            self.total_length_parameter,
            self.outer_width_parameter,
            self.inner_width_parameter,
            self.transition_radius_parameter,
        )

    def bind(
        self, theta: Mapping[str, CanonicalQuantity]
    ) -> BoundCircularTransitionDeformation:
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
            target = CircularTransitionParameters(
                total_length_mm=values[self.total_length_parameter],
                outer_width_mm=values[self.outer_width_parameter],
                inner_width_mm=values[self.inner_width_parameter],
                transition_radius_mm=values[self.transition_radius_parameter],
            )
        except MorphingError as exc:
            raise MorphingError(
                MorphReasonCode.INVALID_PARAMETER_VALUE,
                str(exc),
            ) from exc
        if target.transition_start_x_mm < self.minimum_inner_length_mm:
            raise MorphingError(
                MorphReasonCode.INVALID_PARAMETER_VALUE,
                "target violates the minimum inner length",
            )
        canonical_theta = {name: submitted[name] for name in self.parameter_names}
        return BoundCircularTransitionDeformation(
            reference=self.reference,
            target=target,
            parameter_hash=hash_spec(canonical_theta, "content"),
        )


@dataclass(frozen=True, slots=True)
class BoundCircularTransitionDeformation:
    """One parameter-bound exact profile map with analytic Jacobian."""

    kind: ClassVar[str] = CircularTransitionMap.kind

    reference: CircularTransitionParameters
    target: CircularTransitionParameters
    parameter_hash: str

    def map_point(self, point: Point) -> Point:
        x_value, y_value = self._point(point)
        mapped_x, _ = self._longitudinal_map(x_value)
        scale = self.target.profile_height_mm(
            mapped_x
        ) / self.reference.profile_height_mm(x_value)
        return mapped_x, y_value * scale

    def inverse_point(self, point: Point) -> Point:
        """Return the exact inverse on the target quarter domain."""

        target_x, target_y = self._target_point(point)
        if target_x <= self.target.transition_start_x_mm:
            slope = (
                self.reference.transition_start_x_mm / self.target.transition_start_x_mm
            )
            reference_x = target_x * slope
        else:
            target_span = self.target.half_length_mm - self.target.transition_start_x_mm
            reference_span = (
                self.reference.half_length_mm - self.reference.transition_start_x_mm
            )
            slope = reference_span / target_span
            reference_x = (
                self.reference.transition_start_x_mm
                + (target_x - self.target.transition_start_x_mm) * slope
            )
            if target_x == self.target.half_length_mm:
                reference_x = self.reference.half_length_mm
        scale = self.reference.profile_height_mm(
            reference_x
        ) / self.target.profile_height_mm(target_x)
        return reference_x, target_y * scale

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
        return (
            (longitudinal_slope, 0.0),
            (y_value * scale_slope, scale),
        )

    def _point(self, point: Point) -> Point:
        if not isinstance(point, tuple) or len(point) != 2:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "evaluation point must be a two-coordinate tuple",
            )
        x_value = _finite(point[0], "evaluation x coordinate")
        y_value = _finite(point[1], "evaluation y coordinate")
        if not 0.0 <= x_value <= self.reference.half_length_mm:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "evaluation point lies outside the reference length",
            )
        reference_height = self.reference.profile_height_mm(x_value)
        if not 0.0 <= y_value <= reference_height + 1.0e-12:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "evaluation point lies outside the reference profile",
            )
        return x_value, y_value

    def _target_point(self, point: Point) -> Point:
        if not isinstance(point, tuple) or len(point) != 2:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "inverse point must be a two-coordinate tuple",
            )
        x_value = _finite(point[0], "inverse x coordinate")
        y_value = _finite(point[1], "inverse y coordinate")
        bounded_x = self.target._bounded_x(x_value)
        if bounded_x is None:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "inverse point lies outside the target length",
            )
        target_height = self.target.profile_height_mm(bounded_x)
        tolerance = 1.0e-12 * max(1.0, target_height)
        if not -tolerance <= y_value <= target_height + tolerance:
            raise MorphingError(
                MorphReasonCode.INVALID_MAP_CONFIGURATION,
                "inverse point lies outside the target profile",
            )
        bounded_y = min(target_height, max(0.0, y_value))
        return bounded_x, bounded_y

    def _longitudinal_map(self, x_value: float) -> tuple[float, float]:
        reference_start = self.reference.transition_start_x_mm
        target_start = self.target.transition_start_x_mm
        if x_value <= reference_start:
            slope = target_start / reference_start
            return x_value * slope, slope
        reference_span = self.reference.half_length_mm - reference_start
        target_span = self.target.half_length_mm - target_start
        slope = target_span / reference_span
        mapped = target_start + (x_value - reference_start) * slope
        if x_value == self.reference.half_length_mm:
            mapped = self.target.half_length_mm
        return mapped, slope


def build_circular_transition_reference_mesh(
    parameters: CircularTransitionParameters,
    spec: CircularTransitionMeshSpec,
    *,
    source_provenance: Mapping[str, str],
) -> CanonicalMesh:
    """Build one deterministic structured triangle mesh aligned to the transition."""

    if not isinstance(parameters, CircularTransitionParameters):
        raise TypeError("parameters must be CircularTransitionParameters")
    if not isinstance(spec, CircularTransitionMeshSpec):
        raise TypeError("spec must be CircularTransitionMeshSpec")
    if not isinstance(source_provenance, Mapping) or any(
        not isinstance(key, str) or not key or not isinstance(value, str)
        for key, value in source_provenance.items()
    ):
        raise ValueError("source_provenance must map non-empty strings to strings")

    transition_start = parameters.transition_start_x_mm
    x_coordinates = tuple(
        transition_start * index / spec.inner_columns
        for index in range(spec.inner_columns + 1)
    ) + tuple(
        transition_start
        + parameters.transition_span_mm * index / spec.transition_columns
        for index in range(1, spec.transition_columns + 1)
    )
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
        spec.transition_profile_tag: [],
        spec.inner_profile_tag: [],
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
            spec.transition_profile_tag
            if column >= spec.inner_columns
            else spec.inner_profile_tag
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
    "BoundCircularTransitionDeformation",
    "CircularTransitionMap",
    "CircularTransitionMeshSpec",
    "CircularTransitionParameters",
    "build_circular_transition_reference_mesh",
)
