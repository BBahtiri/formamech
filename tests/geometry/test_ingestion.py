from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from piginot.contracts import GeometryFamilySpec, ParameterSpec, Quantity, Unit
from piginot.geometry import (
    GeometryValidationError,
    canonicalize_meshio,
    classify_geometry_family,
    contains_points,
    from_gmsh,
    to_gmsh,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "geometry"


def _meshio():
    import meshio

    return meshio


def _length(value: float):
    return Quantity(value, Unit.MM).to_canonical()


def _plate_mesh(*, ambiguous: bool = False, missing_boundary: bool = False):
    meshio = _meshio()
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    cells = [
        ("line", np.array([[0, 2], [2, 3], [3, 1], [1, 0]], dtype=int)),
        ("triangle", np.array([[0, 2, 3], [0, 3, 1]], dtype=int)),
    ]
    line_tags = np.array([11, 12, 13, 14], dtype=int)
    if missing_boundary:
        line_tags = np.array([11, 0, 13, 14], dtype=int)
    triangle_tags = np.array([21, 21], dtype=int)
    field_data = {
        "fixed": np.array([11, 1], dtype=int),
        "loaded": np.array([12, 1], dtype=int),
        "free": np.array([13, 1], dtype=int),
        "symmetry": np.array([14, 1], dtype=int),
        "solid": np.array([21, 2], dtype=int),
    }
    if ambiguous:
        field_data["also_loaded"] = np.array([12, 1], dtype=int)
    return meshio.Mesh(
        points,
        cells,
        cell_data={
            "gmsh:physical": [line_tags, triangle_tags],
            "gmsh:geometrical": [
                np.array([101, 102, 103, 104], dtype=int),
                np.array([201, 201], dtype=int),
            ],
        },
        field_data=field_data,
        point_data={"gmsh:dim_tags": np.zeros((4, 2), dtype=int)},
    )


def _annulus_mesh():
    mesh = _plate_mesh()
    meshio = _meshio()
    mesh.points = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
            [1.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.0, 3.0, 0.0],
            [1.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    mesh.cells = [
        meshio.CellBlock(
            "line",
            np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [7, 4],
                ],
                dtype=int,
            ),
        ),
        meshio.CellBlock(
            "triangle",
            np.array(
                [
                    [0, 1, 5],
                    [0, 5, 4],
                    [1, 2, 6],
                    [1, 6, 5],
                    [2, 3, 7],
                    [2, 7, 6],
                    [3, 0, 4],
                    [3, 4, 7],
                ],
                dtype=int,
            ),
        ),
    ]
    mesh.cell_data["gmsh:physical"] = [
        np.array([11, 12, 13, 14, 31, 31, 31, 31], dtype=int),
        np.full(8, 21, dtype=int),
    ]
    mesh.cell_data["gmsh:geometrical"] = [
        np.arange(101, 109, dtype=int),
        np.full(8, 201, dtype=int),
    ]
    mesh.field_data["hole"] = np.array([31, 1], dtype=int)
    return mesh


def _three_hole_mesh():
    meshio = _meshio()
    active_squares = {(x, y) for x in range(7) for y in range(3)} - {
        (1, 1),
        (3, 1),
        (5, 1),
    }
    coordinates = sorted(
        {
            point
            for x, y in active_squares
            for point in (
                (float(x), float(y)),
                (float(x + 1), float(y)),
                (float(x + 1), float(y + 1)),
                (float(x), float(y + 1)),
            )
        }
    )
    node_ids = {point: index for index, point in enumerate(coordinates)}
    triangles = []
    for x, y in sorted(active_squares):
        bottom_left = node_ids[(float(x), float(y))]
        bottom_right = node_ids[(float(x + 1), float(y))]
        top_right = node_ids[(float(x + 1), float(y + 1))]
        top_left = node_ids[(float(x), float(y + 1))]
        triangles.extend(
            [
                [bottom_left, bottom_right, top_right],
                [bottom_left, top_right, top_left],
            ]
        )
    edge_counts: Counter[tuple[int, int]] = Counter()
    directed_edges = {}
    for left, middle, right in triangles:
        for edge in ((left, middle), (middle, right), (right, left)):
            key = tuple(sorted(edge))
            edge_counts[key] += 1
            directed_edges[key] = edge
    lines = [
        directed_edges[key] for key, count in sorted(edge_counts.items()) if count == 1
    ]
    return meshio.Mesh(
        np.column_stack(
            [
                np.asarray(coordinates, dtype=float),
                np.zeros(len(coordinates), dtype=float),
            ]
        ),
        [
            ("line", np.array(lines, dtype=int)),
            ("triangle", np.array(triangles, dtype=int)),
        ],
        cell_data={
            "gmsh:physical": [
                np.full(len(lines), 31, dtype=int),
                np.full(len(triangles), 21, dtype=int),
            ],
            "gmsh:geometrical": [
                np.arange(101, 101 + len(lines), dtype=int),
                np.full(len(triangles), 201, dtype=int),
            ],
        },
        field_data={
            "boundary": np.array([31, 1], dtype=int),
            "solid": np.array([21, 2], dtype=int),
        },
    )


def test_canonical_gmsh_roundtrip_preserves_nodes_cells_and_tags_exactly() -> None:
    source = _plate_mesh()
    canonical = canonicalize_meshio(source, unit=Unit.MM)
    exported = to_gmsh(canonical)
    roundtripped = canonicalize_meshio(exported, unit=Unit.MM)

    assert roundtripped.nodes == canonical.nodes
    assert roundtripped.cells == canonical.cells
    assert roundtripped.boundary_facets == canonical.boundary_facets
    assert roundtripped.semantic_tags == canonical.semantic_tags
    assert roundtripped.region_ids == canonical.region_ids
    assert np.array_equal(exported.points, source.points)
    assert np.array_equal(
        exported.get_cells_type("line"),
        np.array(canonical.boundary_facets, dtype=int),
    )
    assert np.array_equal(
        exported.get_cells_type("triangle"),
        np.array(canonical.cells, dtype=int),
    )


def test_from_gmsh_reads_tagged_v41_file() -> None:
    mesh = from_gmsh(FIXTURES / "tagged_square_v41.msh", unit=Unit.MM)

    assert mesh.nodes == ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
    assert mesh.cells == ((0, 2, 3), (0, 3, 1))
    assert mesh.semantic_tags["fixed"] == (1,)
    assert mesh.semantic_tags["loaded"] == (2,)
    assert mesh.region_ids == (21, 21)


def test_compatibility_adapter_uses_the_versioned_topology_signature() -> None:
    assert canonicalize_meshio(_plate_mesh(), unit=Unit.MM).topology_signature == (
        "planar-cell-complex-v1:components=1;holes=0"
    )
    assert canonicalize_meshio(_annulus_mesh(), unit=Unit.MM).topology_signature == (
        "planar-cell-complex-v1:components=1;holes=1"
    )


def test_equivalent_node_orderings_canonicalize_identically() -> None:
    source = _plate_mesh()
    permuted = _plate_mesh()
    order = np.array([2, 0, 3, 1], dtype=int)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    permuted.points = source.points[order]
    for block in permuted.cells:
        block.data = inverse[block.data]

    left = canonicalize_meshio(source, unit=Unit.MM)
    right = canonicalize_meshio(permuted, unit=Unit.MM)

    assert left.nodes == right.nodes
    assert left.cells == right.cells
    assert left.boundary_facets == right.boundary_facets
    assert left.semantic_tags == right.semantic_tags
    assert left.region_ids == right.region_ids


def test_missing_and_ambiguous_tags_fail_with_facet_ids_and_reason_codes() -> None:
    with pytest.raises(GeometryValidationError) as missing:
        canonicalize_meshio(_plate_mesh(missing_boundary=True), unit=Unit.MM)

    assert missing.value.reason_code == "UNCLASSIFIED_BOUNDARY_FACET"
    assert missing.value.offending_ids == (1,)

    with pytest.raises(GeometryValidationError) as ambiguous:
        canonicalize_meshio(_plate_mesh(ambiguous=True), unit=Unit.MM)

    assert ambiguous.value.reason_code == "AMBIGUOUS_BOUNDARY_TAG"
    assert ambiguous.value.offending_ids == (1,)


def test_defects_duplicate_inverted_and_non_manifold_are_detected() -> None:
    duplicate = _plate_mesh()
    duplicate.points[1] = duplicate.points[0]
    with pytest.raises(GeometryValidationError) as duplicate_error:
        canonicalize_meshio(duplicate, unit=Unit.MM)
    assert duplicate_error.value.reason_code == "DUPLICATE_NODE"

    inverted = _plate_mesh()
    inverted.cells[1].data[0] = np.array([0, 3, 2], dtype=int)
    with pytest.raises(GeometryValidationError) as inverted_error:
        canonicalize_meshio(inverted, unit=Unit.MM)
    assert inverted_error.value.reason_code == "INVERTED_ELEMENT"

    non_manifold = _plate_mesh()
    non_manifold.points = np.vstack(
        [non_manifold.points, np.array([[0.5, -1.0, 0.0]], dtype=float)]
    )
    non_manifold.cells[1].data = np.array([[0, 2, 3], [0, 2, 1], [0, 4, 2]], dtype=int)
    non_manifold.cell_data["gmsh:physical"][1] = np.full(3, 21, dtype=int)
    non_manifold.cell_data["gmsh:geometrical"][1] = np.full(3, 201, dtype=int)
    with pytest.raises(GeometryValidationError) as manifold_error:
        canonicalize_meshio(non_manifold, unit=Unit.MM)
    assert manifold_error.value.reason_code == "NON_MANIFOLD_EDGE"


def test_query_containment_and_fixed_family_classification() -> None:
    mesh = canonicalize_meshio(_plate_mesh(), unit=Unit.MM)

    assert contains_points(mesh, [(0.25, 0.25), (1.0, 0.5)]) == (True, True)
    assert contains_points(mesh, [(-0.01, 0.5), (1.1, 1.1)]) == (False, False)

    fixed_family = GeometryFamilySpec(name="fixed_plate")
    variable_family = GeometryFamilySpec(
        name="variable_plate",
        parameters=(ParameterSpec("width", _length(1.0), _length(2.0)),),
    )

    fixed = classify_geometry_family(fixed_family, mesh)
    variable = classify_geometry_family(variable_family, mesh)

    assert fixed.kind == "degenerate_fixed_mesh"
    assert fixed.can_be_used_as_variable is False
    assert variable.kind == "variable_family"
    assert variable.can_be_used_as_variable is True
