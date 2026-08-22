from __future__ import annotations

from collections import Counter
from typing import Any, cast

import pytest

from piginot.contracts import CanonicalMesh, Unit
from piginot.geometry import (
    GeometryReasonCode,
    PointLocation,
    QueryContainmentError,
    locate_points,
    require_points_contained,
)


def _annulus_mesh() -> CanonicalMesh:
    active_squares = {(x, y) for x in range(3) for y in range(3) if (x, y) != (1, 1)}
    coordinate_cells: list[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    ] = []
    for x, y in sorted(active_squares):
        bottom_left = (float(x), float(y))
        bottom_right = (float(x + 1), float(y))
        top_right = (float(x + 1), float(y + 1))
        top_left = (float(x), float(y + 1))
        coordinate_cells.extend(
            [
                (bottom_left, bottom_right, top_right),
                (bottom_left, top_right, top_left),
            ]
        )

    nodes = tuple(sorted({point for cell in coordinate_cells for point in cell}))
    node_ids = {point: index for index, point in enumerate(nodes)}
    cells = tuple(tuple(node_ids[point] for point in cell) for cell in coordinate_cells)
    incidence: Counter[tuple[int, int]] = Counter()
    directed_edges: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b, c in cells:
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            incidence[key] += 1
            directed_edges[key] = edge
    boundary_facets = tuple(
        sorted(directed_edges[key] for key, count in incidence.items() if count == 1)
    )

    return CanonicalMesh(
        mesh_id="annulus",
        nodes=nodes,
        cells=cells,
        cell_type="triangle",
        boundary_facets=boundary_facets,
        semantic_tags={"boundary": tuple(range(len(boundary_facets)))},
        region_ids=(1,) * len(cells),
        unit=Unit.MM,
        orientation="counterclockwise",
        topology_signature="planar-cell-complex-v1:components=1;holes=1",
        quality_metrics={},
        source_provenance={"provider": "typed-fixture"},
    )


def test_locate_points_distinguishes_domain_boundary_and_exterior() -> None:
    mesh = _annulus_mesh()

    locations = locate_points(
        mesh,
        [
            (0.25, 0.25),
            (0.5, 0.5),
            (0.0, 1.5),
            (1.0, 1.5),
            (1.5, 1.5),
            (-0.01, 1.5),
        ],
    )

    assert locations == (
        PointLocation.INTERIOR,
        PointLocation.INTERIOR,
        PointLocation.BOUNDARY,
        PointLocation.BOUNDARY,
        PointLocation.EXTERIOR,
        PointLocation.EXTERIOR,
    )


def test_require_points_contained_includes_outer_hole_and_shared_boundaries() -> None:
    mesh = _annulus_mesh()

    result = require_points_contained(
        mesh,
        [(0.25, 0.25), (0.5, 0.5), (0.0, 1.5), (1.0, 1.5)],
    )

    assert result is None


def test_require_points_contained_reports_all_exterior_query_ordinals() -> None:
    mesh = _annulus_mesh()

    with pytest.raises(QueryContainmentError) as caught:
        require_points_contained(
            mesh,
            [(0.25, 0.25), (1.5, 1.5), (2.5, 2.5), (4.0, 4.0)],
        )

    assert caught.value.reason_code is GeometryReasonCode.QUERY_OUTSIDE_DOMAIN
    assert caught.value.query_ids == (1, 3)


def test_locate_points_rejects_all_malformed_and_nonfinite_queries() -> None:
    mesh = _annulus_mesh()
    queries = cast(
        list[tuple[float, float]],
        cast(
            Any,
            [(0.25, 0.25), (float("nan"), 0.0), (1.0,), (float("inf"), 1.0)],
        ),
    )

    with pytest.raises(QueryContainmentError) as caught:
        locate_points(mesh, queries)

    assert caught.value.reason_code is GeometryReasonCode.INVALID_QUERY
    assert caught.value.query_ids == (1, 2, 3)
