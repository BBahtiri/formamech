"""Small, explicit mesh fixtures shared by the S5 geometry tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from piginot.contracts import CanonicalMesh, Unit
from piginot.geometry import CellType, MeshPayload


def square_payload(
    *,
    provenance_source: str = "fixture-a.msh",
    boundary_tags: Mapping[str, tuple[int, ...]] | None = None,
) -> MeshPayload:
    """Return a two-triangle unit square with four named boundary facets."""

    tags = boundary_tags or {
        "bottom": (0,),
        "right": (1,),
        "top": (2,),
        "left": (3,),
    }
    return MeshPayload(
        nodes=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        cells=((0, 1, 2), (0, 2, 3)),
        cell_type=CellType.TRIANGLE,
        boundary_facets=((0, 1), (1, 2), (2, 3), (3, 0)),
        boundary_tags=tags,
        region_ids=(7, 7),
        unit=Unit.MM,
        source_provenance={"adapter": "fixture", "source": provenance_source},
    )


def payload_with(payload: MeshPayload, **changes: Any) -> MeshPayload:
    """Copy a payload while replacing only explicitly supplied fields."""

    values: dict[str, Any] = {
        "nodes": payload.nodes,
        "cells": payload.cells,
        "cell_type": payload.cell_type,
        "boundary_facets": payload.boundary_facets,
        "boundary_tags": payload.boundary_tags,
        "region_ids": payload.region_ids,
        "unit": payload.unit,
        "source_provenance": payload.source_provenance,
    }
    values.update(changes)
    return MeshPayload(**values)


def permute_payload(
    payload: MeshPayload,
    *,
    node_order: Sequence[int] | None = None,
    cell_order: Sequence[int] | None = None,
    facet_order: Sequence[int] | None = None,
    cell_rotations: Sequence[int] | None = None,
    reverse_facets: bool = False,
    provenance_source: str = "fixture-permuted.msh",
) -> MeshPayload:
    """Reindex and reorder a payload without changing the represented mesh."""

    old_node_order = tuple(node_order or range(len(payload.nodes)))
    old_cell_order = tuple(cell_order or range(len(payload.cells)))
    old_facet_order = tuple(facet_order or range(len(payload.boundary_facets)))
    rotations = tuple(cell_rotations or (0,) * len(old_cell_order))
    if len(rotations) != len(old_cell_order):
        raise ValueError("one cyclic rotation is required per reordered cell")

    new_index_by_old = {
        old_index: new_index for new_index, old_index in enumerate(old_node_order)
    }
    nodes = tuple(payload.nodes[index] for index in old_node_order)

    cells = []
    for old_cell_index, rotation in zip(old_cell_order, rotations, strict=True):
        remapped = tuple(
            new_index_by_old[index] for index in payload.cells[old_cell_index]
        )
        offset = rotation % len(remapped)
        cells.append(remapped[offset:] + remapped[:offset])

    facets = []
    for old_facet_index in old_facet_order:
        remapped = tuple(
            new_index_by_old[index]
            for index in payload.boundary_facets[old_facet_index]
        )
        facets.append(tuple(reversed(remapped)) if reverse_facets else remapped)

    new_facet_index_by_old = {
        old_index: new_index for new_index, old_index in enumerate(old_facet_order)
    }
    boundary_tags = {
        name: tuple(new_facet_index_by_old[index] for index in source_indices)
        for name, source_indices in payload.boundary_tags.items()
    }

    return MeshPayload(
        nodes=nodes,
        cells=tuple(cells),
        cell_type=payload.cell_type,
        boundary_facets=tuple(facets),
        boundary_tags=boundary_tags,
        region_ids=tuple(payload.region_ids[index] for index in old_cell_order),
        unit=payload.unit,
        source_provenance={
            **dict(payload.source_provenance),
            "source": provenance_source,
        },
    )


def canonical_content(mesh: CanonicalMesh) -> tuple[object, ...]:
    """Return every content field; provenance is deliberately not content identity."""

    return (
        mesh.mesh_id,
        mesh.nodes,
        mesh.cells,
        mesh.cell_type,
        mesh.boundary_facets,
        mesh.semantic_tags,
        mesh.region_ids,
        mesh.unit,
        mesh.orientation,
        mesh.topology_signature,
        mesh.quality_metrics,
    )
