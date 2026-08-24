"""Deterministic unit-square meshes for independent S8 FEM verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.spatial import Delaunay  # type: ignore[import-untyped]

from piginot.contracts import CanonicalMesh, Unit, hash_spec
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh

VERIFICATION_LEVELS = (1, 2, 3, 4)
"""The four refinement levels used by the S8 convergence studies."""

_GENERATOR_ID = "s8-unit-square-verification-mesh-v1"
_DEFAULT_SEED = 1729
_JITTER_FRACTION = 0.22


class VerificationMeshKind(StrEnum):
    """Supported topology families for discretization cross-checks."""

    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"


@dataclass(frozen=True, slots=True)
class VerificationMeshMetadata:
    """Geometric resolution metadata derived without changing the mesh."""

    mesh_id: str
    kind: VerificationMeshKind
    level: int
    subdivisions: int
    node_count: int
    cell_count: int
    h_min_mm: float
    h_max_mm: float


def unit_square_verification_mesh(
    level: int,
    *,
    kind: VerificationMeshKind | str = VerificationMeshKind.STRUCTURED,
    seed: int = _DEFAULT_SEED,
) -> CanonicalMesh:
    """Return one deterministic, conforming unit-square triangle mesh.

    ``level`` is one-based and uses ``2**level`` subdivisions per side.  Boundary
    vertices are never perturbed.  The unstructured family jitters only interior
    vertices before applying a deterministic Delaunay triangulation.
    """

    valid_level = _validate_level(level)
    valid_kind = _validate_kind(kind)
    valid_seed = _validate_seed(seed)
    subdivisions = 2**valid_level
    nodes = _nodes(subdivisions, kind=valid_kind, level=valid_level, seed=valid_seed)
    cells = (
        _structured_cells(subdivisions)
        if valid_kind is VerificationMeshKind.STRUCTURED
        else _delaunay_cells(nodes)
    )
    facets, tags = _boundary_partition(nodes, cells)
    provenance = {
        "generator": _GENERATOR_ID,
        "kind": valid_kind.value,
        "level": str(valid_level),
        "subdivisions": str(subdivisions),
    }
    if valid_kind is VerificationMeshKind.UNSTRUCTURED:
        provenance = {**provenance, "seed": str(valid_seed)}

    canonical = canonicalize_mesh(
        MeshPayload(
            nodes=nodes,
            cells=cells,
            cell_type=CellType.TRIANGLE,
            boundary_facets=facets,
            boundary_tags=tags,
            region_ids=(1,) * len(cells),
            unit=Unit.MM,
            source_provenance=provenance,
        )
    )
    return _with_aggregate_boundary_tag(canonical)


def verification_mesh_metadata(mesh: CanonicalMesh) -> VerificationMeshMetadata:
    """Return immutable resolution metadata for an S8-generated mesh."""

    if not isinstance(mesh, CanonicalMesh):
        raise TypeError("mesh must be a CanonicalMesh")
    provenance = mesh.source_provenance
    if provenance.get("generator") != _GENERATOR_ID:
        raise ValueError("mesh is not an S8 verification mesh")
    try:
        kind = VerificationMeshKind(provenance["kind"])
        level = int(provenance["level"])
        subdivisions = int(provenance["subdivisions"])
        h_min = float(mesh.quality_metrics["minimum_edge_length_mm"])
        h_max = float(mesh.quality_metrics["maximum_edge_length_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("verification mesh metadata is incomplete") from exc
    if subdivisions != 2**level:
        raise ValueError("verification mesh metadata is inconsistent")
    return VerificationMeshMetadata(
        mesh_id=mesh.mesh_id,
        kind=kind,
        level=level,
        subdivisions=subdivisions,
        node_count=len(mesh.nodes),
        cell_count=len(mesh.cells),
        h_min_mm=h_min,
        h_max_mm=h_max,
    )


def _validate_level(level: object) -> int:
    if isinstance(level, bool) or not isinstance(level, int) or level < 1:
        raise ValueError("level must be a positive integer")
    return level


def _validate_kind(kind: object) -> VerificationMeshKind:
    try:
        return VerificationMeshKind(kind)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        choices = ", ".join(member.value for member in VerificationMeshKind)
        raise ValueError(f"kind must be one of: {choices}") from exc


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _nodes(
    subdivisions: int,
    *,
    kind: VerificationMeshKind,
    level: int,
    seed: int,
) -> tuple[tuple[float, float], ...]:
    spacing = 1.0 / subdivisions
    result: list[tuple[float, float]] = []
    for row in range(subdivisions + 1):
        for column in range(subdivisions + 1):
            x_coordinate = column * spacing
            y_coordinate = row * spacing
            is_interior = 0 < row < subdivisions and 0 < column < subdivisions
            if kind is VerificationMeshKind.UNSTRUCTURED and is_interior:
                x_jitter, y_jitter = _interior_jitter(seed, level, row, column)
                x_coordinate += x_jitter * spacing
                y_coordinate += y_jitter * spacing
            result.append((x_coordinate, y_coordinate))
    return tuple(result)


def _interior_jitter(
    seed: int, level: int, row: int, column: int
) -> tuple[float, float]:
    digest = hashlib.blake2b(
        f"{_GENERATOR_ID}:{seed}:{level}:{row}:{column}".encode(),
        digest_size=16,
    ).digest()
    scale = float(2**64)
    x_unit = (int.from_bytes(digest[:8], "big") + 0.5) / scale
    y_unit = (int.from_bytes(digest[8:], "big") + 0.5) / scale
    return (
        _JITTER_FRACTION * (2.0 * x_unit - 1.0),
        _JITTER_FRACTION * (2.0 * y_unit - 1.0),
    )


def _structured_cells(subdivisions: int) -> tuple[tuple[int, int, int], ...]:
    width = subdivisions + 1
    cells: list[tuple[int, int, int]] = []
    for row in range(subdivisions):
        for column in range(subdivisions):
            lower_left = row * width + column
            lower_right = lower_left + 1
            upper_left = lower_left + width
            upper_right = upper_left + 1
            cells.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    return tuple(cells)


def _delaunay_cells(
    nodes: tuple[tuple[float, float], ...],
) -> tuple[tuple[int, int, int], ...]:
    points = np.asarray(nodes, dtype=np.float64)
    triangulation = Delaunay(points, qhull_options="Qbb Qc Qz Q12")
    cells = []
    for simplex in np.asarray(triangulation.simplices, dtype=np.int64):
        cell = (int(simplex[0]), int(simplex[1]), int(simplex[2]))
        if _twice_signed_area(nodes, cell) < 0.0:
            cell = (cell[0], cell[2], cell[1])
        cells.append(cell)
    return tuple(cells)


def _boundary_partition(
    nodes: tuple[tuple[float, float], ...],
    cells: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[tuple[int, int], ...], dict[str, tuple[int, ...]]]:
    edge_counts: dict[tuple[int, int], int] = {}
    for cell in cells:
        for left, right in (
            (cell[0], cell[1]),
            (cell[1], cell[2]),
            (cell[2], cell[0]),
        ):
            edge = (left, right) if left < right else (right, left)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    facets = tuple(sorted(edge for edge, count in edge_counts.items() if count == 1))
    mutable_tags: dict[str, list[int]] = {
        "bottom": [],
        "left": [],
        "right": [],
        "top": [],
    }
    for facet_id, facet in enumerate(facets):
        tag = _boundary_side(nodes[facet[0]], nodes[facet[1]])
        mutable_tags[tag].append(facet_id)
    tags = {name: tuple(indices) for name, indices in mutable_tags.items()}
    return facets, tags


def _boundary_side(left: tuple[float, float], right: tuple[float, float]) -> str:
    if left[1] == 0.0 and right[1] == 0.0:
        return "bottom"
    if left[0] == 0.0 and right[0] == 0.0:
        return "left"
    if left[0] == 1.0 and right[0] == 1.0:
        return "right"
    if left[1] == 1.0 and right[1] == 1.0:
        return "top"
    raise RuntimeError("Delaunay triangulation produced a non-square boundary edge")


def _twice_signed_area(
    nodes: tuple[tuple[float, float], ...], cell: tuple[int, int, int]
) -> float:
    left, middle, right = (nodes[node_id] for node_id in cell)
    return (middle[0] - left[0]) * (right[1] - left[1]) - (middle[1] - left[1]) * (
        right[0] - left[0]
    )


def _with_aggregate_boundary_tag(mesh: CanonicalMesh) -> CanonicalMesh:
    tags = {**dict(mesh.semantic_tags), "all": tuple(range(len(mesh.boundary_facets)))}
    identity = {
        "schema": "canonical-triangle-mesh-v1",
        "nodes": mesh.nodes,
        "cells": mesh.cells,
        "cell_type": mesh.cell_type,
        "boundary_facets": mesh.boundary_facets,
        "semantic_tags": tags,
        "region_ids": mesh.region_ids,
        "unit": mesh.unit,
        "orientation": mesh.orientation,
    }
    return CanonicalMesh(
        mesh_id=hash_spec(identity, domain="content"),
        nodes=mesh.nodes,
        cells=mesh.cells,
        cell_type=mesh.cell_type,
        boundary_facets=mesh.boundary_facets,
        semantic_tags=tags,
        region_ids=mesh.region_ids,
        unit=mesh.unit,
        orientation=mesh.orientation,
        topology_signature=mesh.topology_signature,
        quality_metrics=mesh.quality_metrics,
        source_provenance=mesh.source_provenance,
    )
