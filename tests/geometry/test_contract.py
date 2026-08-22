from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from piginot.contracts import CanonicalMesh, Unit
from piginot.geometry import (
    BoundaryTagError,
    CellType,
    GeometryReasonCode,
    GeometryValidationError,
    MeshPayload,
    canonicalize_mesh,
)
from tests.geometry.helpers import canonical_content, payload_with, square_payload


def test_geometry_contract_types_are_public_and_closed_to_triangles() -> None:
    assert tuple(CellType) == (CellType.TRIANGLE,)
    assert CellType.TRIANGLE.value == "triangle"
    assert issubclass(BoundaryTagError, GeometryValidationError)

    required_codes = {
        "DUPLICATE_NODE",
        "DEGENERATE_ELEMENT",
        "INVERTED_ELEMENT",
        "UNUSED_NODE",
        "DUPLICATE_CELL",
        "NON_MANIFOLD_EDGE",
        "INCONSISTENT_EDGE_ORIENTATION",
        "BOUNDARY_MISMATCH",
        "UNCLASSIFIED_BOUNDARY_FACET",
        "AMBIGUOUS_BOUNDARY_TAG",
        "UNSUPPORTED_CELL_TYPE",
    }
    assert required_codes <= GeometryReasonCode.__members__.keys()
    assert all(code.value == code.name for code in GeometryReasonCode)


def test_mesh_payload_is_an_immutable_defensive_snapshot() -> None:
    mutable_tags = {
        "bottom": [0],
        "right": [1],
        "top": [2],
        "left": [3],
    }
    mutable_provenance = {"adapter": "fixture", "source": "mutable.msh"}
    payload = MeshPayload(
        nodes=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        cells=[[0, 1, 2], [0, 2, 3]],
        cell_type=CellType.TRIANGLE,
        boundary_facets=[[0, 1], [1, 2], [2, 3], [3, 0]],
        boundary_tags=mutable_tags,
        region_ids=[7, 7],
        unit=Unit.MM,
        source_provenance=mutable_provenance,
    )

    mutable_tags["bottom"][0] = 3
    mutable_provenance["source"] = "changed-after-construction.msh"

    assert payload.nodes == (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )
    assert payload.boundary_tags["bottom"] == (0,)
    assert payload.source_provenance["source"] == "mutable.msh"
    with pytest.raises(FrozenInstanceError):
        payload.nodes = ()  # type: ignore[misc]


def test_canonicalize_returns_the_canonical_contract_and_stable_semantic_ids() -> None:
    mesh = canonicalize_mesh(square_payload())

    assert isinstance(mesh, CanonicalMesh)
    assert mesh.cell_type is CellType.TRIANGLE
    assert mesh.unit is Unit.MM
    assert mesh.orientation == "counterclockwise"
    assert dict(mesh.semantic_tags) == {
        "bottom": (0,),
        "left": (1,),
        "right": (2,),
        "top": (3,),
    }
    assert mesh.source_provenance["source"] == "fixture-a.msh"


def test_mesh_identity_excludes_provenance_but_includes_semantic_tags() -> None:
    first_payload = square_payload(provenance_source="one/path/plate.msh")
    second_payload = square_payload(provenance_source="another/path/plate.msh")
    first = canonicalize_mesh(first_payload)
    second = canonicalize_mesh(second_payload)

    assert canonical_content(first) == canonical_content(second)
    assert first.source_provenance != second.source_provenance

    renamed_tags = {
        "bottom": (0,),
        "right": (1,),
        "top": (2,),
        "clamped": (3,),
    }
    renamed = canonicalize_mesh(payload_with(first_payload, boundary_tags=renamed_tags))
    assert renamed.mesh_id != first.mesh_id
