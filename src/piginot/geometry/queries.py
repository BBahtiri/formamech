"""Point-containment queries for canonical triangle meshes."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from numbers import Real

from piginot.contracts import CanonicalMesh
from piginot.geometry.errors import (
    GeometryReasonCode,
    QueryContainmentError,
)
from piginot.geometry.models import PointLocation

_ROUNDING_FACTOR = 64.0


def locate_points(
    mesh: CanonicalMesh, points: Sequence[object]
) -> tuple[PointLocation, ...]:
    """Classify finite 2D query points relative to a canonical mesh domain."""

    parsed: list[tuple[float, float] | None] = []
    invalid_ids: list[int] = []
    for query_id, point in enumerate(points):
        parsed_point = _parse_query_point(point)
        parsed.append(parsed_point)
        if parsed_point is None:
            invalid_ids.append(query_id)

    if invalid_ids:
        raise QueryContainmentError(
            GeometryReasonCode.INVALID_QUERY,
            "query points must be finite numeric 2D pairs",
            tuple(invalid_ids),
        )

    boundary_segments = tuple(
        (mesh.nodes[left], mesh.nodes[right]) for left, right in mesh.boundary_facets
    )
    triangles = tuple(
        (mesh.nodes[cell[0]], mesh.nodes[cell[1]], mesh.nodes[cell[2]])
        for cell in mesh.cells
    )

    locations: list[PointLocation] = []
    for point in parsed:
        assert point is not None  # Invalid values were rejected above.
        if any(
            _point_on_segment(point, left, right) for left, right in boundary_segments
        ):
            locations.append(PointLocation.BOUNDARY)
        elif any(_point_in_triangle(point, triangle) for triangle in triangles):
            # Edges shared by two cells are part of the domain interior. Only an
            # explicitly declared boundary facet receives BOUNDARY classification.
            locations.append(PointLocation.INTERIOR)
        else:
            locations.append(PointLocation.EXTERIOR)
    return tuple(locations)


def require_points_contained(mesh: CanonicalMesh, points: Sequence[object]) -> None:
    """Require every query point to lie in the mesh closure."""

    locations = locate_points(mesh, points)
    exterior_ids = tuple(
        query_id
        for query_id, location in enumerate(locations)
        if location is PointLocation.EXTERIOR
    )
    if exterior_ids:
        raise QueryContainmentError(
            GeometryReasonCode.QUERY_OUTSIDE_DOMAIN,
            "query points lie outside the mesh domain",
            exterior_ids,
        )


def _parse_query_point(point: object) -> tuple[float, float] | None:
    if (
        not isinstance(point, Sequence)
        or isinstance(point, (str, bytes, bytearray))
        or len(point) != 2
    ):
        return None

    first, second = point
    if not _is_finite_real(first) or not _is_finite_real(second):
        return None
    return float(first), float(second)


def _is_finite_real(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
    )


def _point_on_segment(
    point: tuple[float, float],
    left: tuple[float, float],
    right: tuple[float, float],
) -> bool:
    edge_x = right[0] - left[0]
    edge_y = right[1] - left[1]
    query_x = point[0] - left[0]
    query_y = point[1] - left[1]
    edge_length_squared = edge_x * edge_x + edge_y * edge_y

    determinant = edge_x * query_y - edge_y * query_x
    determinant_tolerance = _rounding_tolerance(
        abs(edge_x * query_y) + abs(edge_y * query_x),
        edge_length_squared,
    )
    if abs(determinant) > determinant_tolerance:
        return False

    projection = query_x * edge_x + query_y * edge_y
    projection_tolerance = _rounding_tolerance(
        abs(query_x * edge_x) + abs(query_y * edge_y),
        edge_length_squared,
    )
    return (
        projection >= -projection_tolerance
        and projection <= edge_length_squared + projection_tolerance
    )


def _point_in_triangle(
    point: tuple[float, float],
    triangle: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> bool:
    first, second, third = triangle
    area, area_tolerance = _oriented_edge(first, second, third)
    if abs(area) <= area_tolerance:
        return False

    edge_results = (
        _oriented_edge(first, second, point),
        _oriented_edge(second, third, point),
        _oriented_edge(third, first, point),
    )
    if area > 0.0:
        return all(value >= -tolerance for value, tolerance in edge_results)
    return all(value <= tolerance for value, tolerance in edge_results)


def _oriented_edge(
    left: tuple[float, float],
    right: tuple[float, float],
    point: tuple[float, float],
) -> tuple[float, float]:
    edge_x = right[0] - left[0]
    edge_y = right[1] - left[1]
    query_x = point[0] - left[0]
    query_y = point[1] - left[1]
    determinant = edge_x * query_y - edge_y * query_x
    tolerance = _rounding_tolerance(
        abs(edge_x * query_y) + abs(edge_y * query_x),
        edge_x * edge_x + edge_y * edge_y,
    )
    return determinant, tolerance


def _rounding_tolerance(product_magnitude: float, edge_scale: float) -> float:
    scale = max(product_magnitude, edge_scale, sys.float_info.min)
    return _ROUNDING_FACTOR * sys.float_info.epsilon * scale
