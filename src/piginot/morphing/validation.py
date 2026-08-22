"""Fail-closed validation for bound S6 deformation maps."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

from piginot.contracts import CanonicalMesh, hash_spec
from piginot.geometry import canonicalize_mesh
from piginot.geometry.errors import GeometryReasonCode, GeometryValidationError
from piginot.geometry.models import CellType, MeshPayload
from piginot.morphing.models import (
    Matrix2,
    MorphingError,
    MorphReasonCode,
    MorphValidityCriteria,
    MorphValidityReport,
    Point,
)

_QUADRATURE_RULE = "triangle-degree-2-v1"
_TRIANGLE_DEGREE_2 = (
    (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
    (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
    (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
)


class _BoundDeformation(Protocol):
    @property
    def kind(self) -> str: ...

    @property
    def parameter_hash(self) -> str: ...

    def map_point(self, point: Point) -> Point: ...

    def jacobian(self, point: Point) -> Matrix2: ...


def validate_deformation(
    reference: CanonicalMesh,
    deformation: _BoundDeformation,
    criteria: MorphValidityCriteria,
) -> tuple[CanonicalMesh | None, MorphValidityReport]:
    """Map and validate one reference mesh without healing or remeshing it."""

    if not isinstance(reference, CanonicalMesh):
        raise TypeError("reference must be a CanonicalMesh")
    if not isinstance(criteria, MorphValidityCriteria):
        raise TypeError("criteria must be MorphValidityCriteria")

    policy_hash = hash_spec(criteria, domain="content")
    reasons: list[MorphReasonCode] = []
    determinants, quadrature_count, evaluation_reason = _jacobian_determinants(
        reference, deformation
    )
    minimum_determinant = min(determinants, default=None)
    if evaluation_reason is not None:
        _add_reason(reasons, evaluation_reason)
    elif minimum_determinant is not None and not (
        minimum_determinant > criteria.jacobian_margin
    ):
        _add_reason(reasons, MorphReasonCode.NONINJECTIVE_MORPH)

    mapped_nodes, mapping_reason = _map_nodes(reference, deformation)
    if mapping_reason is not None:
        _add_reason(reasons, mapping_reason)
        return None, _report(
            reference,
            policy_hash,
            quadrature_count,
            minimum_determinant,
            None,
            None,
            reasons,
        )

    morphed, geometry_reason = _canonical_morphed_mesh(
        reference, deformation, mapped_nodes
    )
    if geometry_reason is not None:
        _add_reason(reasons, geometry_reason)
        return None, _report(
            reference,
            policy_hash,
            quadrature_count,
            minimum_determinant,
            None,
            None,
            reasons,
        )

    assert morphed is not None
    if morphed.topology_signature != reference.topology_signature:
        _add_reason(reasons, MorphReasonCode.TOPOLOGY_CHANGED)
    tags_preserved = _tags_are_preserved(reference, mapped_nodes, morphed)
    if not tags_preserved:
        _add_reason(reasons, MorphReasonCode.TAG_IDENTITY_CHANGED)

    feature_size = _minimum_boundary_feature_size(morphed)
    if feature_size < criteria.minimum_feature_size_mm:
        _add_reason(reasons, MorphReasonCode.FEATURE_SIZE_BELOW_MARGIN)

    minimum_angle = morphed.quality_metrics["minimum_angle_degrees"]
    maximum_aspect = morphed.quality_metrics["maximum_aspect_ratio"]
    if (
        minimum_angle < criteria.minimum_angle_degrees
        or maximum_aspect > criteria.maximum_aspect_ratio
    ):
        _add_reason(reasons, MorphReasonCode.MESH_QUALITY_OUT_OF_BOUNDS)

    report = _report(
        reference,
        policy_hash,
        quadrature_count,
        minimum_determinant,
        morphed,
        feature_size,
        reasons,
        tags_preserved=tags_preserved,
    )
    return (morphed if not reasons else None), report


def _jacobian_determinants(
    reference: CanonicalMesh, deformation: _BoundDeformation
) -> tuple[tuple[float, ...], int, MorphReasonCode | None]:
    determinants: list[float] = []
    checked = 0
    try:
        for cell in reference.cells:
            vertices = tuple(reference.nodes[node_id] for node_id in cell)
            for barycentric in _TRIANGLE_DEGREE_2:
                point = (
                    math.fsum(
                        weight * vertex[0]
                        for weight, vertex in zip(barycentric, vertices, strict=True)
                    ),
                    math.fsum(
                        weight * vertex[1]
                        for weight, vertex in zip(barycentric, vertices, strict=True)
                    ),
                )
                matrix = _matrix2(deformation.jacobian(point))
                determinant = math.fsum(
                    (matrix[0][0] * matrix[1][1], -matrix[0][1] * matrix[1][0])
                )
                if not math.isfinite(determinant):
                    raise ValueError("Jacobian determinant must be finite")
                determinants.append(determinant)
                checked += 1
    except MorphingError as exc:
        return tuple(determinants), checked, exc.reason_code
    except (ArithmeticError, TypeError, ValueError):
        return tuple(determinants), checked, MorphReasonCode.INVALID_MORPHED_MESH
    return tuple(determinants), checked, None


def _map_nodes(
    reference: CanonicalMesh, deformation: _BoundDeformation
) -> tuple[tuple[Point, ...], MorphReasonCode | None]:
    mapped: list[Point] = []
    try:
        for point in reference.nodes:
            mapped.append(_point2(deformation.map_point(point)))
    except MorphingError as exc:
        return (), exc.reason_code
    except (ArithmeticError, TypeError, ValueError):
        return (), MorphReasonCode.INVALID_MORPHED_MESH
    return tuple(mapped), None


def _canonical_morphed_mesh(
    reference: CanonicalMesh,
    deformation: _BoundDeformation,
    mapped_nodes: tuple[Point, ...],
) -> tuple[CanonicalMesh | None, MorphReasonCode | None]:
    try:
        kind = deformation.kind
        parameter_hash = deformation.parameter_hash
        if not isinstance(kind, str) or not kind:
            raise TypeError("deformation kind must be non-empty text")
        if not isinstance(parameter_hash, str) or not parameter_hash:
            raise TypeError("deformation parameter hash must be non-empty text")
        provenance = {
            **reference.source_provenance,
            "morph_kind": kind,
            "morph_parameter_hash": parameter_hash,
            "morph_reference_mesh_id": reference.mesh_id,
        }
        payload = MeshPayload(
            nodes=mapped_nodes,
            cells=reference.cells,
            cell_type=CellType.TRIANGLE,
            boundary_facets=reference.boundary_facets,
            boundary_tags=reference.semantic_tags,
            region_ids=reference.region_ids,
            unit=reference.unit,
            source_provenance=provenance,
        )
        return canonicalize_mesh(payload), None
    except GeometryValidationError as exc:
        return None, _geometry_reason(exc.reason_code)
    except (AttributeError, TypeError, ValueError):
        return None, MorphReasonCode.INVALID_MAP_CONFIGURATION


def _geometry_reason(reason: GeometryReasonCode) -> MorphReasonCode:
    if reason in {
        GeometryReasonCode.DEGENERATE_ELEMENT,
        GeometryReasonCode.DUPLICATE_NODE,
        GeometryReasonCode.INVERTED_ELEMENT,
    }:
        return MorphReasonCode.NONINJECTIVE_MORPH
    return MorphReasonCode.INVALID_MORPHED_MESH


def _tags_are_preserved(
    reference: CanonicalMesh,
    mapped_nodes: tuple[Point, ...],
    morphed: CanonicalMesh,
) -> bool:
    expected = _tagged_geometric_edges(
        mapped_nodes, reference.boundary_facets, reference.semantic_tags
    )
    actual = _tagged_geometric_edges(
        morphed.nodes, morphed.boundary_facets, morphed.semantic_tags
    )
    return expected == actual


def _tagged_geometric_edges(
    nodes: Sequence[Point],
    facets: Sequence[tuple[int, int]],
    tags: Mapping[str, tuple[int, ...]],
) -> dict[str, tuple[tuple[Point, Point], ...]]:
    result: dict[str, tuple[tuple[Point, Point], ...]] = {}
    for name, facet_ids in tags.items():
        edges: list[tuple[Point, Point]] = []
        for facet_id in facet_ids:
            left_id, right_id = facets[facet_id]
            left = nodes[left_id]
            right = nodes[right_id]
            edges.append((left, right) if left <= right else (right, left))
        result[name] = tuple(sorted(edges))
    return result


def _minimum_boundary_feature_size(mesh: CanonicalMesh) -> float:
    segments = tuple(
        (facet, mesh.nodes[facet[0]], mesh.nodes[facet[1]])
        for facet in mesh.boundary_facets
    )
    minimum = min(_distance(left, right) for _, left, right in segments)
    for left_id, (left_nodes, left_start, left_end) in enumerate(segments):
        for right_nodes, right_start, right_end in segments[left_id + 1 :]:
            if set(left_nodes).isdisjoint(right_nodes):
                minimum = min(
                    minimum,
                    _segment_distance(left_start, left_end, right_start, right_end),
                )
    return minimum


def _segment_distance(
    left_start: Point, left_end: Point, right_start: Point, right_end: Point
) -> float:
    if _segments_intersect(left_start, left_end, right_start, right_end):
        return 0.0
    return min(
        _point_segment_distance(left_start, right_start, right_end),
        _point_segment_distance(left_end, right_start, right_end),
        _point_segment_distance(right_start, left_start, left_end),
        _point_segment_distance(right_end, left_start, left_end),
    )


def _segments_intersect(
    left_start: Point, left_end: Point, right_start: Point, right_end: Point
) -> bool:
    orientations = (
        _cross(left_start, left_end, right_start),
        _cross(left_start, left_end, right_end),
        _cross(right_start, right_end, left_start),
        _cross(right_start, right_end, left_end),
    )
    first, second, third, fourth = orientations
    if ((first > 0.0 and second < 0.0) or (first < 0.0 and second > 0.0)) and (
        (third > 0.0 and fourth < 0.0) or (third < 0.0 and fourth > 0.0)
    ):
        return True
    return (
        (first == 0.0 and _within(left_start, right_start, left_end))
        or (second == 0.0 and _within(left_start, right_end, left_end))
        or (third == 0.0 and _within(right_start, left_start, right_end))
        or (fourth == 0.0 and _within(right_start, left_end, right_end))
    )


def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared == 0.0:
        return _distance(point, start)
    projection = (
        (point[0] - start[0]) * delta_x + (point[1] - start[1]) * delta_y
    ) / length_squared
    parameter = max(0.0, min(1.0, projection))
    closest = (start[0] + parameter * delta_x, start[1] + parameter * delta_y)
    return _distance(point, closest)


def _cross(origin: Point, left: Point, right: Point) -> float:
    return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (
        right[0] - origin[0]
    )


def _within(start: Point, point: Point, end: Point) -> bool:
    return min(start[0], end[0]) <= point[0] <= max(start[0], end[0]) and min(
        start[1], end[1]
    ) <= point[1] <= max(start[1], end[1])


def _distance(left: Point, right: Point) -> float:
    return math.hypot(right[0] - left[0], right[1] - left[1])


def _matrix2(value: object) -> Matrix2:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("Jacobian must contain two rows")
    if len(value) != 2:
        raise ValueError("Jacobian must contain two rows")
    return _point2(value[0]), _point2(value[1])


def _point2(value: object) -> Point:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("point must contain two coordinates")
    if len(value) != 2:
        raise ValueError("point must contain two coordinates")
    coordinates = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError("coordinates must be real numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("coordinates must be finite")
        coordinates.append(0.0 if number == 0.0 else number)
    return coordinates[0], coordinates[1]


def _report(
    reference: CanonicalMesh,
    policy_hash: str,
    quadrature_count: int,
    minimum_determinant: float | None,
    morphed: CanonicalMesh | None,
    feature_size: float | None,
    reasons: Sequence[MorphReasonCode],
    *,
    tags_preserved: bool = False,
) -> MorphValidityReport:
    quality = {} if morphed is None else morphed.quality_metrics
    tag_counts = (
        {}
        if morphed is None
        else {name: len(facet_ids) for name, facet_ids in morphed.semantic_tags.items()}
    )
    return MorphValidityReport(
        reference_mesh_id=reference.mesh_id,
        validity_policy_hash=policy_hash,
        quadrature_rule=_QUADRATURE_RULE,
        quadrature_points_checked=quadrature_count,
        minimum_jacobian_determinant=minimum_determinant,
        topology_signature=None if morphed is None else morphed.topology_signature,
        semantic_tag_counts=tag_counts,
        tags_preserved=tags_preserved,
        minimum_feature_size_mm=feature_size,
        minimum_angle_degrees=quality.get("minimum_angle_degrees"),
        maximum_aspect_ratio=quality.get("maximum_aspect_ratio"),
        reason_codes=tuple(reasons),
    )


def _add_reason(reasons: list[MorphReasonCode], reason: MorphReasonCode) -> None:
    if reason not in reasons:
        reasons.append(reason)
