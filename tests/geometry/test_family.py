from __future__ import annotations

from dataclasses import replace

import pytest

from piginot.contracts import CanonicalMesh, Unit
from piginot.geometry import (
    DegenerateFamilyError,
    GeometryReasonCode,
    MeshFamilyKind,
    classify_mesh_family,
    require_variable_family,
)


def _triangle_mesh(mesh_id: str) -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id=mesh_id,
        nodes=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        cells=((0, 1, 2),),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 0)),
        semantic_tags={"boundary": (0, 1, 2)},
        region_ids=(1,),
        unit=Unit.MM,
        orientation="counterclockwise",
        topology_signature="planar-cell-complex-v1:components=1;holes=0",
        quality_metrics={},
        source_provenance={"provider": "typed-fixture"},
    )


def test_family_kind_is_based_exactly_on_unique_mesh_ids() -> None:
    first = _triangle_mesh("mesh-a")
    same_id = replace(first, source_provenance={"provider": "second-fixture"})
    different_id = replace(first, mesh_id="mesh-b")

    assert classify_mesh_family([first]) is MeshFamilyKind.FIXED_DEGENERATE
    assert classify_mesh_family([first, same_id]) is MeshFamilyKind.FIXED_DEGENERATE
    assert classify_mesh_family([first, different_id]) is MeshFamilyKind.VARIABLE


@pytest.mark.parametrize("count", [1, 3], ids=("single", "repeated-same-id"))
def test_degenerate_fixed_family_cannot_be_required_as_variable(count: int) -> None:
    mesh = _triangle_mesh("mesh-a")

    with pytest.raises(DegenerateFamilyError) as caught:
        require_variable_family([mesh] * count)

    assert caught.value.reason_code is GeometryReasonCode.DEGENERATE_VARIABLE_FAMILY


def test_distinct_mesh_ids_form_a_variable_family() -> None:
    meshes = [_triangle_mesh("mesh-a"), _triangle_mesh("mesh-b")]

    assert require_variable_family(meshes) is None
