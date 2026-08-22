"""Deterministic validation and canonicalization of typed triangle meshes."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal
from fractions import Fraction

from piginot.contracts import CanonicalMesh, Unit, hash_spec
from piginot.geometry.errors import (
    BoundaryTagError,
    GeometryReasonCode,
    GeometryValidationError,
)
from piginot.geometry.models import CellType, MeshPayload
from piginot.geometry.topology import summarize_cells

_Node = tuple[float, float]
_Cell = tuple[int, int, int]
_Facet = tuple[int, int]


def canonicalize_mesh(payload: MeshPayload) -> CanonicalMesh:
    """Validate and canonicalize one explicitly-unitized triangle mesh.

    The operation only reindexes a valid mesh. It never flips an inverted cell,
    merges coordinates, manufactures tags, or otherwise heals invalid input.
    """

    _require_payload(payload)
    scale = _coordinate_scale(payload.unit)
    source_nodes = _scaled_nodes(payload.nodes, scale)
    node_order, canonical_nodes = _canonical_nodes(source_nodes)
    source_to_canonical = {
        source_id: canonical_id for canonical_id, source_id in enumerate(node_order)
    }
    _reject_duplicate_nodes(canonical_nodes)

    cells_with_regions = _canonical_cells(
        payload.cells,
        payload.region_ids,
        source_to_canonical,
        canonical_nodes,
    )
    canonical_cells = tuple(cell for cell, _ in cells_with_regions)
    region_ids = tuple(region for _, region in cells_with_regions)
    _reject_unused_nodes(canonical_nodes, canonical_cells)
    directed_boundary = _validate_edge_incidence(canonical_cells)
    canonical_facets = tuple(sorted(directed_boundary))
    semantic_tags = _canonical_boundary_tags(
        payload.boundary_facets,
        payload.boundary_tags,
        source_to_canonical,
        canonical_facets,
    )

    topology = summarize_cells(canonical_cells)
    quality_metrics = _quality_metrics(canonical_nodes, canonical_cells)
    identity = {
        "schema": "canonical-triangle-mesh-v1",
        "nodes": canonical_nodes,
        "cells": canonical_cells,
        "cell_type": CellType.TRIANGLE,
        "boundary_facets": canonical_facets,
        "semantic_tags": semantic_tags,
        "region_ids": region_ids,
        "unit": Unit.MM,
        "orientation": "counterclockwise",
    }
    mesh_id = hash_spec(identity, domain="content")
    return CanonicalMesh(
        mesh_id=mesh_id,
        nodes=canonical_nodes,
        cells=canonical_cells,
        cell_type=CellType.TRIANGLE,
        boundary_facets=canonical_facets,
        semantic_tags=semantic_tags,
        region_ids=region_ids,
        unit=Unit.MM,
        orientation="counterclockwise",
        topology_signature=topology.signature,
        quality_metrics=quality_metrics,
        source_provenance=dict(payload.source_provenance),
    )


def _require_payload(payload: object) -> None:
    if not isinstance(payload, MeshPayload):
        raise TypeError("payload must be a MeshPayload")
    if payload.cell_type is not CellType.TRIANGLE:
        raise GeometryValidationError(
            GeometryReasonCode.UNSUPPORTED_CELL_TYPE,
            "S5 supports linear triangle cells only",
        )


def _coordinate_scale(unit: object) -> float:
    if not isinstance(unit, Unit) or unit.canonical_unit is not Unit.MM:
        raise GeometryValidationError(
            GeometryReasonCode.UNIT_NOT_CANONICAL,
            "coordinate unit must be an explicit supported length unit",
        )
    return unit.canonical_factor


def _scaled_nodes(nodes: Sequence[Sequence[object]], scale: float) -> tuple[_Node, ...]:
    if not nodes:
        raise GeometryValidationError(
            GeometryReasonCode.INVALID_POINTS, "nodes must be non-empty 2D pairs"
        )
    result: list[_Node] = []
    for node_id, node in enumerate(nodes):
        if len(node) != 2:
            raise GeometryValidationError(
                GeometryReasonCode.INVALID_POINTS,
                "nodes must be 2D coordinate pairs",
                (node_id,),
            )
        values: list[float] = []
        for value in node:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise GeometryValidationError(
                    GeometryReasonCode.INVALID_POINTS,
                    "node coordinates must be real numbers",
                    (node_id,),
                )
            source_value = float(value)
            if not math.isfinite(source_value):
                raise GeometryValidationError(
                    GeometryReasonCode.NON_FINITE_POINT,
                    "node coordinates must be finite",
                    (node_id,),
                )
            scaled = float(Decimal(str(source_value)) * Decimal(str(scale)))
            if not math.isfinite(scaled):
                raise GeometryValidationError(
                    GeometryReasonCode.NON_FINITE_POINT,
                    "node coordinates must be finite after unit conversion",
                    (node_id,),
                )
            values.append(_positive_zero(scaled))
        result.append((values[0], values[1]))
    return tuple(result)


def _canonical_nodes(
    nodes: tuple[_Node, ...],
) -> tuple[tuple[int, ...], tuple[_Node, ...]]:
    order = tuple(sorted(range(len(nodes)), key=lambda index: nodes[index]))
    return order, tuple(nodes[index] for index in order)


def _reject_duplicate_nodes(nodes: tuple[_Node, ...]) -> None:
    duplicates: list[int] = []
    for node_id in range(1, len(nodes)):
        if nodes[node_id] == nodes[node_id - 1]:
            if not duplicates or duplicates[-1] != node_id - 1:
                duplicates.append(node_id - 1)
            duplicates.append(node_id)
    if duplicates:
        raise GeometryValidationError(
            GeometryReasonCode.DUPLICATE_NODE,
            "node coordinates must be exactly unique",
            tuple(duplicates),
        )


def _canonical_cells(
    cells: Sequence[Sequence[object]],
    regions: Sequence[object],
    source_to_canonical: Mapping[int, int],
    nodes: tuple[_Node, ...],
) -> tuple[tuple[_Cell, int], ...]:
    if not cells:
        raise GeometryValidationError(
            GeometryReasonCode.UNSUPPORTED_CELL_TYPE,
            "at least one triangle cell is required",
        )
    if len(regions) != len(cells):
        raise GeometryValidationError(
            GeometryReasonCode.MISSING_REGION_TAG,
            "one integer region ID is required per cell",
        )

    candidates: list[tuple[_Cell, int]] = []
    for source_cell_id, (cell, region) in enumerate(zip(cells, regions, strict=True)):
        if len(cell) != 3:
            raise GeometryValidationError(
                GeometryReasonCode.UNSUPPORTED_CELL_TYPE,
                "S5 supports three-node triangles only",
                (source_cell_id,),
            )
        mapped = _mapped_indices(
            cell,
            source_to_canonical,
            GeometryReasonCode.INVALID_CELL_INDEX,
            source_cell_id,
        )
        canonical = _cyclic_minimum((mapped[0], mapped[1], mapped[2]))
        if isinstance(region, bool) or not isinstance(region, int):
            raise GeometryValidationError(
                GeometryReasonCode.MISSING_REGION_TAG,
                "region IDs must be integers",
                (source_cell_id,),
            )
        if region <= 0:
            raise GeometryValidationError(
                GeometryReasonCode.MISSING_REGION_TAG,
                "region IDs must be positive integers",
                (source_cell_id,),
            )
        candidates.append((canonical, region))

    ordered = tuple(sorted(candidates, key=lambda item: (item[0], item[1])))
    duplicate_ids = _duplicate_cell_ids(tuple(cell for cell, _ in ordered))
    if duplicate_ids:
        raise GeometryValidationError(
            GeometryReasonCode.DUPLICATE_CELL,
            "cyclically equivalent cells are duplicated",
            duplicate_ids,
        )
    for cell_id, (cell, _) in enumerate(ordered):
        orientation = _orientation(nodes[cell[0]], nodes[cell[1]], nodes[cell[2]])
        if orientation == 0:
            raise GeometryValidationError(
                GeometryReasonCode.DEGENERATE_ELEMENT,
                "triangle has exactly zero signed area",
                (cell_id,),
            )
        if orientation < 0:
            raise GeometryValidationError(
                GeometryReasonCode.INVERTED_ELEMENT,
                "triangle orientation must be counterclockwise",
                (cell_id,),
            )
    return ordered


def _mapped_indices(
    indices: Sequence[object],
    source_to_canonical: Mapping[int, int],
    reason_code: GeometryReasonCode,
    entity_id: int,
) -> tuple[int, ...]:
    result: list[int] = []
    for index in indices:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in source_to_canonical
        ):
            raise GeometryValidationError(
                reason_code,
                "connectivity contains an invalid node index",
                (entity_id,),
            )
        result.append(source_to_canonical[index])
    return tuple(result)


def _duplicate_cell_ids(cells: tuple[_Cell, ...]) -> tuple[int, ...]:
    duplicates: list[int] = []
    for cell_id in range(1, len(cells)):
        if cells[cell_id] == cells[cell_id - 1]:
            if not duplicates or duplicates[-1] != cell_id - 1:
                duplicates.append(cell_id - 1)
            duplicates.append(cell_id)
    return tuple(duplicates)


def _reject_unused_nodes(nodes: tuple[_Node, ...], cells: tuple[_Cell, ...]) -> None:
    used = {node_id for cell in cells for node_id in cell}
    unused = tuple(node_id for node_id in range(len(nodes)) if node_id not in used)
    if unused:
        raise GeometryValidationError(
            GeometryReasonCode.UNUSED_NODE,
            "every node must belong to at least one cell",
            unused,
        )


def _validate_edge_incidence(cells: tuple[_Cell, ...]) -> tuple[_Facet, ...]:
    directed_by_edge: dict[_Facet, list[_Facet]] = defaultdict(list)
    for cell in cells:
        for directed in _cell_edges(cell):
            directed_by_edge[_undirected(directed)].append(directed)

    ordered_edges = tuple(sorted(directed_by_edge))
    non_manifold = tuple(
        edge_id
        for edge_id, edge in enumerate(ordered_edges)
        if len(directed_by_edge[edge]) > 2
    )
    if non_manifold:
        raise GeometryValidationError(
            GeometryReasonCode.NON_MANIFOLD_EDGE,
            "an edge belongs to more than two cells",
            non_manifold,
        )
    inconsistent = tuple(
        edge_id
        for edge_id, edge in enumerate(ordered_edges)
        if len(directed_by_edge[edge]) == 2
        and directed_by_edge[edge][0] == directed_by_edge[edge][1]
    )
    if inconsistent:
        raise GeometryValidationError(
            GeometryReasonCode.INCONSISTENT_EDGE_ORIENTATION,
            "adjacent cells traverse a shared edge in the same direction",
            inconsistent,
        )
    return tuple(
        directed_by_edge[edge][0]
        for edge in ordered_edges
        if len(directed_by_edge[edge]) == 1
    )


def _canonical_boundary_tags(
    source_facets: Sequence[Sequence[object]],
    source_tags: Mapping[str, Sequence[object]],
    source_to_canonical: Mapping[int, int],
    canonical_facets: tuple[_Facet, ...],
) -> dict[str, tuple[int, ...]]:
    canonical_id_by_edge = {
        _undirected(facet): facet_id for facet_id, facet in enumerate(canonical_facets)
    }
    mapped_source_edges: list[_Facet] = []
    for source_facet_id, facet in enumerate(source_facets):
        if len(facet) != 2:
            raise GeometryValidationError(
                GeometryReasonCode.INVALID_FACET_INDEX,
                "boundary facets must contain two node indices",
                (source_facet_id,),
            )
        mapped = _mapped_indices(
            facet,
            source_to_canonical,
            GeometryReasonCode.INVALID_FACET_INDEX,
            source_facet_id,
        )
        if mapped[0] == mapped[1]:
            raise GeometryValidationError(
                GeometryReasonCode.BOUNDARY_MISMATCH,
                "boundary facet has identical endpoints",
                (source_facet_id,),
            )
        mapped_source_edges.append(_undirected((mapped[0], mapped[1])))

    expected_edges = set(canonical_id_by_edge)
    provided_counts = Counter(mapped_source_edges)
    provided_edges = set(provided_counts)
    if provided_edges != expected_edges or any(
        count != 1 for count in provided_counts.values()
    ):
        edge_universe = tuple(sorted(expected_edges | provided_edges))
        bad = tuple(
            edge_id
            for edge_id, edge in enumerate(edge_universe)
            if edge not in expected_edges or provided_counts.get(edge, 0) != 1
        )
        raise GeometryValidationError(
            GeometryReasonCode.BOUNDARY_MISMATCH,
            "provided facets must equal the cell-derived boundary exactly",
            bad,
        )

    owners: list[list[str]] = [[] for _ in canonical_facets]
    for tag_name, source_facet_ids in source_tags.items():
        if not isinstance(tag_name, str) or not tag_name:
            raise BoundaryTagError(
                GeometryReasonCode.AMBIGUOUS_BOUNDARY_TAG,
                "boundary tag names must be non-empty text",
            )
        for tagged_source_facet_id in source_facet_ids:
            if (
                isinstance(tagged_source_facet_id, bool)
                or not isinstance(tagged_source_facet_id, int)
                or tagged_source_facet_id < 0
                or tagged_source_facet_id >= len(mapped_source_edges)
            ):
                raise BoundaryTagError(
                    GeometryReasonCode.INVALID_FACET_INDEX,
                    "boundary tag references an invalid source facet",
                )
            canonical_facet_id = canonical_id_by_edge[
                mapped_source_edges[tagged_source_facet_id]
            ]
            owners[canonical_facet_id].append(tag_name)

    missing = tuple(index for index, names in enumerate(owners) if not names)
    if missing:
        raise BoundaryTagError(
            GeometryReasonCode.UNCLASSIFIED_BOUNDARY_FACET,
            "every boundary facet requires exactly one semantic tag",
            missing,
        )
    ambiguous = tuple(index for index, names in enumerate(owners) if len(names) != 1)
    if ambiguous:
        raise BoundaryTagError(
            GeometryReasonCode.AMBIGUOUS_BOUNDARY_TAG,
            "boundary facets cannot belong to multiple semantic tags",
            ambiguous,
        )

    result: dict[str, list[int]] = defaultdict(list)
    for facet_id, names in enumerate(owners):
        result[names[0]].append(facet_id)
    return {name: tuple(facet_ids) for name, facet_ids in sorted(result.items())}


def _quality_metrics(
    nodes: tuple[_Node, ...], cells: tuple[_Cell, ...]
) -> dict[str, float]:
    minimum_angle = 180.0
    maximum_angle = 0.0
    minimum_edge = math.inf
    maximum_edge = 0.0
    minimum_area = math.inf
    maximum_aspect_ratio = 0.0
    for cell in cells:
        points = tuple(nodes[node_id] for node_id in cell)
        edge_lengths = tuple(
            math.hypot(
                points[(index + 1) % 3][0] - points[index][0],
                points[(index + 1) % 3][1] - points[index][1],
            )
            for index in range(3)
        )
        minimum_edge = min(minimum_edge, *edge_lengths)
        maximum_edge = max(maximum_edge, *edge_lengths)
        area = abs(
            (
                (points[1][0] - points[0][0]) * (points[2][1] - points[0][1])
                - (points[1][1] - points[0][1]) * (points[2][0] - points[0][0])
            )
            / 2.0
        )
        minimum_area = min(minimum_area, area)
        maximum_aspect_ratio = max(
            maximum_aspect_ratio,
            max(edge_lengths) * math.fsum(edge_lengths) / (4.0 * math.sqrt(3.0) * area),
        )
        for vertex in range(3):
            origin = points[vertex]
            left = points[(vertex + 1) % 3]
            right = points[(vertex + 2) % 3]
            left_vector = (left[0] - origin[0], left[1] - origin[1])
            right_vector = (right[0] - origin[0], right[1] - origin[1])
            denominator = math.hypot(*left_vector) * math.hypot(*right_vector)
            cosine = (
                math.fsum(
                    (left_vector[0] * right_vector[0], left_vector[1] * right_vector[1])
                )
                / denominator
            )
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            minimum_angle = min(minimum_angle, angle)
            maximum_angle = max(maximum_angle, angle)
    return {
        "maximum_angle_degrees": maximum_angle,
        "maximum_aspect_ratio": maximum_aspect_ratio,
        "maximum_edge_length_mm": maximum_edge,
        "minimum_angle_degrees": minimum_angle,
        "minimum_area_mm2": minimum_area,
        "minimum_edge_length_mm": minimum_edge,
    }


def _orientation(left: _Node, middle: _Node, right: _Node) -> int:
    """Return the exact sign of orient2d for the represented binary floats."""

    left_x, left_y = (Fraction(value) for value in left)
    middle_x, middle_y = (Fraction(value) for value in middle)
    right_x, right_y = (Fraction(value) for value in right)
    determinant = (middle_x - left_x) * (right_y - left_y) - (middle_y - left_y) * (
        right_x - left_x
    )
    return (determinant > 0) - (determinant < 0)


def _cyclic_minimum(cell: _Cell) -> _Cell:
    rotations = (cell, (cell[1], cell[2], cell[0]), (cell[2], cell[0], cell[1]))
    return min(rotations)


def _cell_edges(cell: _Cell) -> tuple[_Facet, _Facet, _Facet]:
    return ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0]))


def _undirected(edge: _Facet) -> _Facet:
    return edge if edge[0] < edge[1] else (edge[1], edge[0])


def _positive_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value
