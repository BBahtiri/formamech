"""A fixed CAD polygon is refined without moving its boundary or changing tags."""

from dataclasses import replace

import pytest

from piginot.cad.refinement import refine_fixed_mesh
from piginot.contracts import Unit
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh


def rectangle():
    return canonicalize_mesh(
        MeshPayload(
            nodes=((2.0, 3.0), (6.0, 3.0), (6.0, 5.0), (2.0, 5.0)),
            cells=((0, 1, 2), (0, 2, 3)),
            cell_type=CellType.TRIANGLE,
            boundary_facets=((0, 1), (1, 2), (2, 3), (3, 0)),
            boundary_tags={"bottom": (0,), "right": (1,), "top": (2,), "left": (3,)},
            region_ids=(1, 1),
            unit=Unit.MM,
            source_provenance={"fixture": "offset-rectangle"},
        )
    )


@pytest.mark.parametrize("level", [1, 2, 4])
def test_nested_refinement_preserves_physical_domain_and_boundary_names(level):
    base = rectangle()
    mesh = refine_fixed_mesh(base, level)
    assert mesh == refine_fixed_mesh(base, level)
    assert set(base.nodes) <= set(mesh.nodes)
    assert len(mesh.cells) == 2 * level * level
    assert set(mesh.semantic_tags) == set(base.semantic_tags)
    assert {len(indices) for indices in mesh.semantic_tags.values()} == {level}
    assert mesh.topology_signature == base.topology_signature
    assert set(mesh.region_ids) == {1}
    areas = []
    for a, b, c in mesh.cells:
        x, y, z = (mesh.nodes[index] for index in (a, b, c))
        areas.append(
            ((y[0] - x[0]) * (z[1] - x[1]) - (y[1] - x[1]) * (z[0] - x[0])) / 2
        )
    assert all(area > 0 for area in areas)
    assert sum(areas) == pytest.approx(8.0, abs=1e-12)
    for name, indices in mesh.semantic_tags.items():
        axis, expected = {
            "left": (0, 2),
            "right": (0, 6),
            "bottom": (1, 3),
            "top": (1, 5),
        }[name]
        assert all(
            mesh.nodes[node][axis] == expected
            for index in indices
            for node in mesh.boundary_facets[index]
        )


def test_levels_are_nested_and_have_distinct_content_identities():
    meshes = [refine_fixed_mesh(rectangle(), level) for level in (1, 2, 4)]
    assert set(meshes[0].nodes) < set(meshes[1].nodes) < set(meshes[2].nodes)
    assert len({mesh.mesh_id for mesh in meshes}) == 3


@pytest.mark.parametrize("level", [True, False, 0, -1, 3, 2.0, "2", None])
def test_undeclared_refinement_levels_are_refused(level):
    with pytest.raises(ValueError, match="1, 2, 4"):
        refine_fixed_mesh(rectangle(), level)


def test_a_structural_mesh_identity_mismatch_is_refused():
    with pytest.raises(ValueError, match="identity"):
        refine_fixed_mesh(replace(rectangle(), mesh_id="0" * 64), 1)


def test_regions_remain_attached_to_each_parent_triangle():
    original = rectangle()
    base = canonicalize_mesh(
        MeshPayload(
            original.nodes,
            original.cells,
            CellType.TRIANGLE,
            original.boundary_facets,
            original.semantic_tags,
            (1, 2),
            Unit.MM,
            {},
        )
    )
    mesh = refine_fixed_mesh(base, 2)
    assert mesh.region_ids.count(1) == mesh.region_ids.count(2) == 4
