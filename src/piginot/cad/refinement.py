"""Nested reference meshes for a fixed polygonal approximation of a CAD profile."""

from __future__ import annotations

from piginot.contracts import CanonicalMesh, Unit
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh


def _payload(mesh: CanonicalMesh) -> MeshPayload:
    return MeshPayload(
        mesh.nodes,
        mesh.cells,
        CellType(mesh.cell_type),
        mesh.boundary_facets,
        mesh.semantic_tags,
        mesh.region_ids,
        Unit.MM,
        mesh.source_provenance,
    )


def _subdivide(mesh: CanonicalMesh, base_id: str, level: int) -> CanonicalMesh:
    edges = tuple(
        sorted(
            {
                tuple(sorted((a, b)))
                for cell in mesh.cells
                for a, b in zip(cell, (*cell[1:], cell[0]), strict=True)
            }
        )
    )
    midpoint = {edge: len(mesh.nodes) + index for index, edge in enumerate(edges)}
    nodes = mesh.nodes + tuple(
        (
            (mesh.nodes[a][0] + mesh.nodes[b][0]) / 2,
            (mesh.nodes[a][1] + mesh.nodes[b][1]) / 2,
        )
        for a, b in edges
    )

    def middle(a: int, b: int) -> int:
        return midpoint[tuple(sorted((a, b)))]

    cells = tuple(
        child
        for a, b, c in mesh.cells
        for ab, bc, ca in [(middle(a, b), middle(b, c), middle(c, a))]
        for child in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
    )
    facets = tuple(
        child
        for a, b in mesh.boundary_facets
        for child in ((a, middle(a, b)), (middle(a, b), b))
    )
    tags = {
        name: tuple(child for index in indices for child in (2 * index, 2 * index + 1))
        for name, indices in mesh.semantic_tags.items()
    }
    return canonicalize_mesh(
        MeshPayload(
            nodes,
            cells,
            CellType.TRIANGLE,
            facets,
            tags,
            tuple(region for region in mesh.region_ids for _ in range(4)),
            Unit.MM,
            {
                **mesh.source_provenance,
                "refinement": "fixed-polygon-midpoint-v1",
                "base_mesh_id": base_id,
                "level": str(level),
            },
        )
    )


def refine_fixed_mesh(mesh: CanonicalMesh, level: int) -> CanonicalMesh:
    """Refine the same linearized domain; never project nodes onto curved CAD."""

    if type(level) is not int or level not in (1, 2, 4):
        raise ValueError("fixed CAD refinement level must be one of 1, 2, 4")
    validated = canonicalize_mesh(_payload(mesh))
    if validated.mesh_id != mesh.mesh_id:
        raise ValueError("fixed CAD mesh identity differs from its canonical content")
    result = validated
    for current in (2, 4):
        if current <= level:
            result = _subdivide(result, mesh.mesh_id, current)
    return result
