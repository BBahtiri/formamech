from __future__ import annotations

from collections import Counter

import pytest

from piginot.contracts import CanonicalMesh, Unit
from piginot.geometry import TopologySummary, topology_summary


def _cell_complex(
    active_squares: set[tuple[int, int]], *, mesh_id: str
) -> CanonicalMesh:
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
        mesh_id=mesh_id,
        nodes=nodes,
        cells=cells,
        cell_type="triangle",
        boundary_facets=boundary_facets,
        semantic_tags={"boundary": tuple(range(len(boundary_facets)))},
        region_ids=(1,) * len(cells),
        unit=Unit.MM,
        orientation="counterclockwise",
        topology_signature="fixture-unverified",
        quality_metrics={},
        source_provenance={"provider": "typed-fixture"},
    )


@pytest.mark.parametrize(
    ("active_squares", "component_count", "hole_count"),
    [
        ({(0, 0)}, 1, 0),
        (
            {(x, y) for x in range(3) for y in range(3)} - {(1, 1)},
            1,
            1,
        ),
        (
            {(x, y) for x in range(7) for y in range(3)} - {(1, 1), (3, 1), (5, 1)},
            1,
            3,
        ),
        ({(0, 0), (3, 0)}, 2, 0),
        ({(0, 0), (1, 1)}, 2, 0),
    ],
    ids=(
        "zero-holes",
        "one-hole",
        "three-holes",
        "two-components",
        "corner-touch-components",
    ),
)
def test_topology_summary_is_euler_consistent(
    active_squares: set[tuple[int, int]],
    component_count: int,
    hole_count: int,
) -> None:
    mesh = _cell_complex(active_squares, mesh_id=f"mesh-{component_count}-{hole_count}")

    summary = topology_summary(mesh)

    assert isinstance(summary, TopologySummary)
    assert summary.component_count == component_count
    assert summary.hole_count == hole_count
    assert summary.signature == (
        f"planar-cell-complex-v1:components={component_count};holes={hole_count}"
    )


def test_topology_signature_does_not_encode_refinement() -> None:
    coarse = _cell_complex({(0, 0)}, mesh_id="coarse")
    refined = _cell_complex(
        {(x, y) for x in range(2) for y in range(2)}, mesh_id="refined"
    )

    assert topology_summary(coarse).signature == topology_summary(refined).signature


def test_topology_components_do_not_merge_at_a_single_corner() -> None:
    mesh = CanonicalMesh(
        mesh_id="corner-touching-components",
        nodes=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)),
        cells=((0, 1, 2), (0, 3, 4)),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)),
        semantic_tags={"boundary": (0, 1, 2, 3, 4, 5)},
        region_ids=(1, 1),
        unit=Unit.MM,
        orientation="counterclockwise",
        topology_signature="fixture-unverified",
        quality_metrics={},
        source_provenance={"provider": "typed-fixture"},
    )

    summary = topology_summary(mesh)

    assert summary.component_count == 2
    assert summary.hole_count == 0
    assert summary.signature == "planar-cell-complex-v1:components=2;holes=0"
