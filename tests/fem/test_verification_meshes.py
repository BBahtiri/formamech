"""Deterministic mesh-family tests for the independent S8 verification lane."""

from __future__ import annotations

import math

import pytest

from piginot.contracts import CanonicalMesh, canonical_json_bytes
from piginot.fem.verification_meshes import (
    VERIFICATION_LEVELS,
    VerificationMeshKind,
    unit_square_verification_mesh,
    verification_mesh_metadata,
)


def _twice_signed_area(mesh: CanonicalMesh, cell: tuple[int, ...]) -> float:
    left, middle, right = (mesh.nodes[node_id] for node_id in cell)
    return (middle[0] - left[0]) * (right[1] - left[1]) - (middle[1] - left[1]) * (
        right[0] - left[0]
    )


def _facet_midpoint(mesh: CanonicalMesh, facet_id: int) -> tuple[float, float]:
    left_id, right_id = mesh.boundary_facets[facet_id]
    left = mesh.nodes[left_id]
    right = mesh.nodes[right_id]
    return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)


@pytest.mark.parametrize("level", VERIFICATION_LEVELS)
@pytest.mark.parametrize("kind", tuple(VerificationMeshKind))
def test_unit_square_meshes_have_valid_oriented_topology(
    level: int, kind: VerificationMeshKind
) -> None:
    mesh = unit_square_verification_mesh(level, kind=kind)
    subdivisions = 2**level

    assert isinstance(mesh, CanonicalMesh)
    assert len(mesh.nodes) == (subdivisions + 1) ** 2
    assert len(mesh.cells) == 2 * subdivisions**2
    assert len(mesh.boundary_facets) == 4 * subdivisions
    assert mesh.orientation == "counterclockwise"
    assert mesh.topology_signature == "planar-cell-complex-v1:components=1;holes=0"
    assert all(_twice_signed_area(mesh, cell) > 0.0 for cell in mesh.cells)
    assert math.fsum(
        _twice_signed_area(mesh, cell) / 2.0 for cell in mesh.cells
    ) == pytest.approx(1.0, rel=0.0, abs=1.0e-14)


@pytest.mark.parametrize("kind", tuple(VerificationMeshKind))
def test_boundary_is_exact_and_has_side_and_all_tags(
    kind: VerificationMeshKind,
) -> None:
    mesh = unit_square_verification_mesh(3, kind=kind)

    assert set(mesh.semantic_tags) == {"all", "bottom", "left", "right", "top"}
    assert mesh.semantic_tags["all"] == tuple(range(len(mesh.boundary_facets)))
    assert set().union(
        *(set(mesh.semantic_tags[tag]) for tag in ("bottom", "left", "right", "top"))
    ) == set(mesh.semantic_tags["all"])
    assert sum(
        len(mesh.semantic_tags[tag]) for tag in ("bottom", "left", "right", "top")
    ) == len(mesh.boundary_facets)

    expected_coordinate = {
        "bottom": (1, 0.0),
        "left": (0, 0.0),
        "right": (0, 1.0),
        "top": (1, 1.0),
    }
    for tag, (coordinate, expected) in expected_coordinate.items():
        assert len(mesh.semantic_tags[tag]) == 2**3
        assert all(
            _facet_midpoint(mesh, facet_id)[coordinate] == expected
            for facet_id in mesh.semantic_tags[tag]
        )


def test_unstructured_mesh_is_distorted_and_not_structured_connectivity() -> None:
    structured = unit_square_verification_mesh(3, kind="structured")
    unstructured = unit_square_verification_mesh(3, kind="unstructured")
    spacing = 1.0 / 2**3

    unstructured_interior = (
        node
        for node in unstructured.nodes
        if 0.0 < node[0] < 1.0 and 0.0 < node[1] < 1.0
    )
    assert any(
        not math.isclose(node[0] / spacing, round(node[0] / spacing), abs_tol=1.0e-12)
        or not math.isclose(
            node[1] / spacing, round(node[1] / spacing), abs_tol=1.0e-12
        )
        for node in unstructured_interior
    )
    assert unstructured.nodes != structured.nodes
    assert unstructured.cells != structured.cells
    assert (
        len(
            {
                round(_twice_signed_area(unstructured, cell), 12)
                for cell in unstructured.cells
            }
        )
        > 4
    )


@pytest.mark.parametrize("kind", tuple(VerificationMeshKind))
def test_identical_inputs_have_identical_canonical_bytes_and_hash(
    kind: VerificationMeshKind,
) -> None:
    first = unit_square_verification_mesh(2, kind=kind, seed=1729)
    second = unit_square_verification_mesh(2, kind=kind, seed=1729)

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first.mesh_id == second.mesh_id


def test_kind_level_and_unstructured_seed_have_distinct_identities() -> None:
    meshes = (
        unit_square_verification_mesh(1, kind="structured"),
        unit_square_verification_mesh(2, kind="structured"),
        unit_square_verification_mesh(1, kind="unstructured", seed=1729),
        unit_square_verification_mesh(1, kind="unstructured", seed=1730),
    )

    assert len({mesh.mesh_id for mesh in meshes}) == len(meshes)


@pytest.mark.parametrize("kind", tuple(VerificationMeshKind))
def test_metadata_reports_geometric_h_and_refinement(
    kind: VerificationMeshKind,
) -> None:
    coarse = verification_mesh_metadata(unit_square_verification_mesh(1, kind=kind))
    fine = verification_mesh_metadata(unit_square_verification_mesh(4, kind=kind))

    assert coarse.kind is kind
    assert coarse.level == 1
    assert coarse.subdivisions == 2
    assert coarse.node_count == 9
    assert coarse.cell_count == 8
    assert 0.0 < fine.h_min_mm <= fine.h_max_mm < coarse.h_max_mm
    assert fine.level == 4
    assert fine.subdivisions == 16


@pytest.mark.parametrize("invalid_level", (True, 0, -1, 1.5, "2"))
def test_invalid_levels_are_rejected(invalid_level: object) -> None:
    with pytest.raises(ValueError, match="level"):
        unit_square_verification_mesh(invalid_level)  # type: ignore[arg-type]


def test_invalid_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="kind"):
        unit_square_verification_mesh(1, kind="random")


@pytest.mark.parametrize("invalid_seed", (True, -1, 1.5, "1729"))
def test_invalid_seed_is_rejected(invalid_seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        unit_square_verification_mesh(1, seed=invalid_seed)  # type: ignore[arg-type]


def test_metadata_rejects_meshes_outside_the_verification_family() -> None:
    mesh = unit_square_verification_mesh(1)
    unrelated = CanonicalMesh(
        mesh_id=mesh.mesh_id,
        nodes=mesh.nodes,
        cells=mesh.cells,
        cell_type=mesh.cell_type,
        boundary_facets=mesh.boundary_facets,
        semantic_tags=mesh.semantic_tags,
        region_ids=mesh.region_ids,
        unit=mesh.unit,
        orientation=mesh.orientation,
        topology_signature=mesh.topology_signature,
        quality_metrics=mesh.quality_metrics,
        source_provenance={"fixture": "not-the-s8-generator"},
    )

    with pytest.raises(ValueError, match="verification mesh"):
        verification_mesh_metadata(unrelated)
