"""Strict Gmsh v4.1 import and deterministic export."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from piginot.contracts import CanonicalMesh, Unit
from piginot.geometry.canonicalize import canonicalize_mesh
from piginot.geometry.errors import (
    BoundaryTagError,
    GeometryFormatError,
    GeometryReasonCode,
    GeometryValidationError,
)
from piginot.geometry.models import CellType, MeshPayload

_VOLUME_PREFIXES = ("tetra", "hexahedron", "wedge", "pyramid", "voxel")
_SURFACE_PREFIXES = ("triangle", "quad", "polygon")


def read_gmsh(path: str | Path, *, unit: Unit) -> CanonicalMesh:
    """Read a tagged, planar Gmsh v4.1 mesh into canonical form."""

    source = Path(path)
    _require_gmsh_v41(source)

    import meshio  # type: ignore[import-untyped]

    raw_mesh = meshio.read(source, file_format="gmsh")
    return canonicalize_meshio(raw_mesh, unit=unit, source_format="gmsh-v4.1")


def canonicalize_meshio(
    raw_mesh: Any,
    *,
    unit: Unit,
    source_path: Path | None = None,
    source_format: str | None = None,
) -> CanonicalMesh:
    """Canonicalize an in-memory meshio mesh using the strict Gmsh tag contract.

    ``source_path`` remains accepted for compatibility but is excluded from
    canonical content identity.
    """

    points = _planar_points(raw_mesh.points)
    line_blocks, triangle_blocks = _supported_blocks(raw_mesh)
    field_names = _physical_names(raw_mesh)

    facets = _connectivity(raw_mesh, line_blocks, width=2)
    cells = _connectivity(raw_mesh, triangle_blocks, width=3)
    line_tags = _physical_tags(raw_mesh, line_blocks)
    triangle_tags = _physical_tags(raw_mesh, triangle_blocks)
    line_memberships = _cell_set_memberships(
        raw_mesh, line_blocks, names=_names_for_dimension(field_names, 1)
    )
    triangle_memberships = _cell_set_memberships(
        raw_mesh, triangle_blocks, names=_names_for_dimension(field_names, 2)
    )

    boundary_tags = _boundary_tags(
        line_tags, line_memberships, field_names, len(facets)
    )
    region_ids = _region_ids(
        triangle_tags, triangle_memberships, field_names, len(cells)
    )
    payload = MeshPayload(
        nodes=tuple((float(point[0]), float(point[1])) for point in points),
        cells=tuple(tuple(int(value) for value in cell) for cell in cells),
        cell_type=CellType.TRIANGLE,
        boundary_facets=tuple((int(facet[0]), int(facet[1])) for facet in facets),
        boundary_tags=boundary_tags,
        region_ids=region_ids,
        unit=unit,
        source_provenance={
            "adapter": "meshio",
            "format": source_format or ("gmsh-v4.1" if source_path else "meshio"),
        },
    )
    return canonicalize_mesh(payload)


def write_gmsh(mesh: CanonicalMesh, path: str | Path) -> None:
    """Write canonical triangles as deterministic ASCII Gmsh v4.1."""

    if str(mesh.cell_type) != CellType.TRIANGLE.value:
        raise GeometryValidationError(
            GeometryReasonCode.UNSUPPORTED_CELL_TYPE,
            "Gmsh export supports linear triangles only",
        )

    names = tuple(sorted(mesh.semantic_tags))
    physical_ids = {name: index for index, name in enumerate(names, start=1)}
    facet_physical_ids = _facet_physical_ids(mesh, physical_ids)
    region_ids = _positive_region_ids(mesh)
    region_names = {region_id: f"region_{region_id}" for region_id in region_ids}
    collisions = tuple(sorted(set(names).intersection(region_names.values())))
    if collisions:
        raise GeometryValidationError(
            GeometryReasonCode.INVALID_FIELD_DATA,
            "boundary names must not collide with generated region names: "
            + ", ".join(collisions),
        )

    lines = ["$MeshFormat", "4.1 0 8", "$EndMeshFormat", "$PhysicalNames"]
    lines.append(str(len(names) + len(region_names)))
    lines.extend(f"1 {physical_ids[name]} {_quoted_ascii_name(name)}" for name in names)
    lines.extend(
        f"2 {region_id} {_quoted_ascii_name(region_names[region_id])}"
        for region_id in sorted(region_names)
    )
    lines.extend(("$EndPhysicalNames", "$Entities"))
    lines.extend(_entity_lines(mesh, facet_physical_ids, region_ids))
    lines.extend(("$EndEntities", "$Nodes"))
    lines.extend(_node_lines(mesh))
    lines.extend(("$EndNodes", "$Elements"))
    lines.extend(_element_lines(mesh))
    lines.extend(("$EndElements", ""))

    Path(path).write_text("\n".join(lines), encoding="ascii", newline="\n")


def to_meshio(mesh: CanonicalMesh) -> Any:
    """Convert a canonical mesh to a deterministic in-memory meshio object."""

    import meshio

    if str(mesh.cell_type) != CellType.TRIANGLE.value:
        raise GeometryValidationError(
            GeometryReasonCode.UNSUPPORTED_CELL_TYPE,
            "meshio export supports linear triangles only",
        )

    names = tuple(sorted(mesh.semantic_tags))
    physical_ids = {name: index for index, name in enumerate(names, start=1)}
    line_tags = np.asarray(_facet_physical_ids(mesh, physical_ids), dtype=np.int64)
    region_ids = np.asarray(_positive_region_ids(mesh), dtype=np.int64)
    field_data = {
        name: np.asarray([physical_id, 1], dtype=np.int64)
        for name, physical_id in physical_ids.items()
    }
    for region_id in sorted({int(value) for value in region_ids}):
        field_data[f"region_{region_id}"] = np.asarray([region_id, 2], dtype=np.int64)

    return meshio.Mesh(
        np.column_stack(
            [
                np.asarray(mesh.nodes, dtype=float),
                np.zeros(len(mesh.nodes), dtype=float),
            ]
        ),
        [
            ("line", np.asarray(mesh.boundary_facets, dtype=np.int64)),
            ("triangle", np.asarray(mesh.cells, dtype=np.int64)),
        ],
        cell_data={
            "gmsh:physical": [line_tags, region_ids],
            "gmsh:geometrical": [
                np.arange(1, len(mesh.boundary_facets) + 1, dtype=np.int64),
                np.arange(1, len(mesh.cells) + 1, dtype=np.int64),
            ],
        },
        field_data=field_data,
        point_data={"gmsh:dim_tags": np.zeros((len(mesh.nodes), 2), dtype=np.int64)},
    )


def _require_gmsh_v41(path: Path) -> None:
    with path.open("rb") as handle:
        marker = handle.readline().strip()
        header = handle.readline().split()
    if marker != b"$MeshFormat" or not header or header[0] != b"4.1":
        raise GeometryFormatError(
            GeometryReasonCode.UNSUPPORTED_GMSH_VERSION,
            "only Gmsh v4.1 is supported",
        )


def _planar_points(raw_points: object) -> NDArray[np.float64]:
    try:
        points = np.asarray(raw_points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GeometryValidationError(
            GeometryReasonCode.INVALID_POINTS,
            "Gmsh points must be a numeric matrix",
        ) from exc
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise GeometryValidationError(
            GeometryReasonCode.UNSUPPORTED_MESH_DIMENSION,
            "Gmsh geometry must have two coordinate dimensions",
        )
    if points.shape[1] == 3 and np.any(points[:, 2] != 0.0):
        raise GeometryValidationError(
            GeometryReasonCode.UNSUPPORTED_MESH_DIMENSION,
            "Gmsh geometry must lie exactly in the z=0 plane",
        )
    return np.ascontiguousarray(points[:, :2], dtype=np.float64)


def _supported_blocks(raw_mesh: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    line_blocks: list[int] = []
    triangle_blocks: list[int] = []
    other_boundary_types: set[str] = set()
    other_surface_types: set[str] = set()
    unknown_types: set[str] = set()
    volume_types: set[str] = set()

    for block_id, block in enumerate(raw_mesh.cells):
        if len(block.data) == 0:
            continue
        cell_type = str(block.type)
        if cell_type == "vertex":
            continue
        if cell_type == "line":
            line_blocks.append(block_id)
        elif cell_type.startswith("line"):
            other_boundary_types.add(cell_type)
        elif cell_type == "triangle":
            triangle_blocks.append(block_id)
        elif cell_type.startswith(_SURFACE_PREFIXES):
            other_surface_types.add(cell_type)
        elif cell_type.startswith(_VOLUME_PREFIXES) or cell_type == "polyhedron":
            volume_types.add(cell_type)
        else:
            unknown_types.add(cell_type)

    if volume_types:
        raise GeometryValidationError(
            GeometryReasonCode.UNSUPPORTED_MESH_DIMENSION,
            "volume cells are not supported: " + ", ".join(sorted(volume_types)),
        )
    if unknown_types:
        raise GeometryValidationError(
            GeometryReasonCode.UNSUPPORTED_CELL_TYPE,
            "unsupported element types: " + ", ".join(sorted(unknown_types)),
        )
    if other_surface_types:
        reason = (
            GeometryReasonCode.MIXED_CELL_TYPES
            if triangle_blocks
            else GeometryReasonCode.UNSUPPORTED_CELL_TYPE
        )
        raise GeometryValidationError(
            reason,
            "S5 supports linear triangle area cells only: "
            + ", ".join(sorted(other_surface_types)),
        )
    if not triangle_blocks:
        raise GeometryValidationError(
            GeometryReasonCode.UNSUPPORTED_CELL_TYPE,
            "at least one linear triangle block is required",
        )
    if other_boundary_types or not line_blocks:
        detail = ", ".join(sorted(other_boundary_types)) or "none"
        raise GeometryValidationError(
            GeometryReasonCode.UNSUPPORTED_BOUNDARY_TYPE,
            "linear line boundary cells are required; found " + detail,
        )
    return tuple(line_blocks), tuple(triangle_blocks)


def _connectivity(
    raw_mesh: Any, block_ids: tuple[int, ...], *, width: int
) -> NDArray[np.int64]:
    blocks: list[NDArray[np.int64]] = []
    for block_id in block_ids:
        values = np.asarray(raw_mesh.cells[block_id].data)
        if (
            values.ndim != 2
            or values.shape[1] != width
            or not np.issubdtype(values.dtype, np.integer)
        ):
            reason = (
                GeometryReasonCode.UNSUPPORTED_BOUNDARY_TYPE
                if width == 2
                else GeometryReasonCode.UNSUPPORTED_CELL_TYPE
            )
            raise GeometryValidationError(reason, "invalid linear connectivity")
        blocks.append(np.asarray(values, dtype=np.int64))
    return np.ascontiguousarray(np.concatenate(blocks), dtype=np.int64)


def _physical_names(raw_mesh: Any) -> dict[int, dict[int, tuple[str, ...]]]:
    collected: dict[int, dict[int, list[str]]] = {
        1: defaultdict(list),
        2: defaultdict(list),
    }
    for name, raw_value in raw_mesh.field_data.items():
        values = np.asarray(raw_value)
        if (
            not isinstance(name, str)
            or not name
            or values.shape != (2,)
            or not np.issubdtype(values.dtype, np.integer)
        ):
            raise GeometryValidationError(
                GeometryReasonCode.INVALID_FIELD_DATA,
                "physical names must map to integer [tag, dimension] pairs",
            )
        tag, dimension = int(values[0]), int(values[1])
        if dimension in collected:
            collected[dimension][tag].append(name)
    return {
        dimension: {tag: tuple(sorted(names)) for tag, names in by_tag.items()}
        for dimension, by_tag in collected.items()
    }


def _names_for_dimension(
    field_names: dict[int, dict[int, tuple[str, ...]]], dimension: int
) -> frozenset[str]:
    return frozenset(
        name for names in field_names[dimension].values() for name in names
    )


def _physical_tags(raw_mesh: Any, block_ids: tuple[int, ...]) -> tuple[int, ...]:
    raw_blocks = raw_mesh.cell_data.get("gmsh:physical")
    if raw_blocks is None or len(raw_blocks) != len(raw_mesh.cells):
        raise GeometryValidationError(
            GeometryReasonCode.MISSING_PHYSICAL_TAGS,
            "one Gmsh physical-tag array per element block is required",
        )
    tags: list[int] = []
    for block_id in block_ids:
        values = np.asarray(raw_blocks[block_id])
        count = len(raw_mesh.cells[block_id].data)
        if values.shape != (count,) or not np.issubdtype(values.dtype, np.integer):
            raise GeometryValidationError(
                GeometryReasonCode.MISSING_PHYSICAL_TAGS,
                "one integer physical tag per element is required",
            )
        tags.extend(int(value) for value in values)
    return tuple(tags)


def _cell_set_memberships(
    raw_mesh: Any, block_ids: tuple[int, ...], *, names: frozenset[str]
) -> tuple[frozenset[str], ...]:
    offsets: dict[int, int] = {}
    total = 0
    for block_id in block_ids:
        offsets[block_id] = total
        total += len(raw_mesh.cells[block_id].data)
    memberships: list[set[str]] = [set() for _ in range(total)]

    for name in sorted(names):
        raw_blocks = raw_mesh.cell_sets.get(name)
        if raw_blocks is None:
            continue
        if len(raw_blocks) != len(raw_mesh.cells):
            raise GeometryValidationError(
                GeometryReasonCode.INVALID_FIELD_DATA,
                f"cell set {name!r} does not align with element blocks",
            )
        for block_id in block_ids:
            raw_indices = raw_blocks[block_id]
            if raw_indices is None:
                continue
            indices = np.asarray(raw_indices)
            count = len(raw_mesh.cells[block_id].data)
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise GeometryValidationError(
                    GeometryReasonCode.INVALID_FIELD_DATA,
                    f"cell set {name!r} must contain integer local indices",
                )
            if np.any(indices < 0) or np.any(indices >= count):
                raise GeometryValidationError(
                    GeometryReasonCode.INVALID_FIELD_DATA,
                    f"cell set {name!r} contains an invalid local index",
                )
            for local_index in indices:
                memberships[offsets[block_id] + int(local_index)].add(name)
    return tuple(frozenset(values) for values in memberships)


def _boundary_tags(
    physical_tags: tuple[int, ...],
    memberships: tuple[frozenset[str], ...],
    field_names: dict[int, dict[int, tuple[str, ...]]],
    facet_count: int,
) -> dict[str, tuple[int, ...]]:
    if len(physical_tags) != facet_count or len(memberships) != facet_count:
        raise GeometryValidationError(
            GeometryReasonCode.MISSING_PHYSICAL_TAGS,
            "one physical boundary tag per facet is required",
        )
    result: dict[str, list[int]] = defaultdict(list)
    for facet_id, (tag, cell_set_names) in enumerate(
        zip(physical_tags, memberships, strict=True)
    ):
        tag_names = frozenset(field_names[1].get(tag, ()))
        if tag <= 0 or not tag_names:
            raise BoundaryTagError(
                GeometryReasonCode.UNCLASSIFIED_BOUNDARY_FACET,
                "boundary facet has no named physical group",
                (facet_id,),
            )
        if (
            len(tag_names) != 1
            or len(cell_set_names) > 1
            or (cell_set_names and cell_set_names != tag_names)
        ):
            raise BoundaryTagError(
                GeometryReasonCode.AMBIGUOUS_BOUNDARY_TAG,
                "boundary facet belongs to multiple named physical groups",
                (facet_id,),
            )
        result[next(iter(tag_names))].append(facet_id)
    return {name: tuple(indices) for name, indices in sorted(result.items())}


def _region_ids(
    physical_tags: tuple[int, ...],
    memberships: tuple[frozenset[str], ...],
    field_names: dict[int, dict[int, tuple[str, ...]]],
    cell_count: int,
) -> tuple[int, ...]:
    if len(physical_tags) != cell_count or len(memberships) != cell_count:
        raise GeometryValidationError(
            GeometryReasonCode.MISSING_REGION_TAG,
            "one physical region tag per cell is required",
        )
    result: list[int] = []
    for cell_id, (tag, cell_set_names) in enumerate(
        zip(physical_tags, memberships, strict=True)
    ):
        tag_names = frozenset(field_names[2].get(tag, ()))
        if tag <= 0 or not tag_names:
            raise GeometryValidationError(
                GeometryReasonCode.MISSING_REGION_TAG,
                "cell has no positive named physical region",
                (cell_id,),
            )
        if (
            len(tag_names) != 1
            or len(cell_set_names) > 1
            or (cell_set_names and cell_set_names != tag_names)
        ):
            raise GeometryValidationError(
                GeometryReasonCode.AMBIGUOUS_REGION_TAG,
                "cell belongs to multiple named physical regions",
                (cell_id,),
            )
        result.append(tag)
    return tuple(result)


def _facet_physical_ids(
    mesh: CanonicalMesh, physical_ids: dict[str, int]
) -> tuple[int, ...]:
    owners: list[list[str]] = [[] for _ in mesh.boundary_facets]
    for name in sorted(mesh.semantic_tags):
        for facet_id in mesh.semantic_tags[name]:
            owners[facet_id].append(name)
    missing = tuple(index for index, names in enumerate(owners) if not names)
    if missing:
        raise BoundaryTagError(
            GeometryReasonCode.UNCLASSIFIED_BOUNDARY_FACET,
            "every exported boundary facet requires one semantic tag",
            missing,
        )
    ambiguous = tuple(index for index, names in enumerate(owners) if len(names) > 1)
    if ambiguous:
        raise BoundaryTagError(
            GeometryReasonCode.AMBIGUOUS_BOUNDARY_TAG,
            "every exported boundary facet requires exactly one semantic tag",
            ambiguous,
        )
    return tuple(physical_ids[names[0]] for names in owners)


def _positive_region_ids(mesh: CanonicalMesh) -> tuple[int, ...]:
    bad = tuple(index for index, value in enumerate(mesh.region_ids) if value <= 0)
    if bad:
        raise GeometryValidationError(
            GeometryReasonCode.MISSING_REGION_TAG,
            "exported region IDs must be positive",
            bad,
        )
    return tuple(int(value) for value in mesh.region_ids)


def _quoted_ascii_name(name: str) -> str:
    if not name.isascii() or any(character in "\r\n\x00" for character in name):
        raise GeometryValidationError(
            GeometryReasonCode.INVALID_FIELD_DATA,
            "ASCII Gmsh physical names must contain only printable ASCII text",
        )
    return json.dumps(name, ensure_ascii=True)


def _entity_lines(
    mesh: CanonicalMesh,
    facet_physical_ids: tuple[int, ...],
    region_ids: tuple[int, ...],
) -> list[str]:
    lines = [f"{len(mesh.nodes)} {len(mesh.boundary_facets)} {len(mesh.cells)} 0"]
    for node_id, (x, y) in enumerate(mesh.nodes, start=1):
        lines.append(f"{node_id} {_number(x)} {_number(y)} 0 0")
    for entity_id, ((left, right), physical_id) in enumerate(
        zip(mesh.boundary_facets, facet_physical_ids, strict=True), start=1
    ):
        x0, y0 = mesh.nodes[left]
        x1, y1 = mesh.nodes[right]
        lines.append(
            f"{entity_id} {_number(min(x0, x1))} {_number(min(y0, y1))} 0 "
            f"{_number(max(x0, x1))} {_number(max(y0, y1))} 0 "
            f"1 {physical_id} 2 {left + 1} -{right + 1}"
        )
    for entity_id, (cell, region_id) in enumerate(
        zip(mesh.cells, region_ids, strict=True), start=1
    ):
        coordinates = [mesh.nodes[node_id] for node_id in cell]
        xs = [point[0] for point in coordinates]
        ys = [point[1] for point in coordinates]
        lines.append(
            f"{entity_id} {_number(min(xs))} {_number(min(ys))} 0 "
            f"{_number(max(xs))} {_number(max(ys))} 0 1 {region_id} 0"
        )
    return lines


def _node_lines(mesh: CanonicalMesh) -> list[str]:
    lines = [f"{len(mesh.nodes)} {len(mesh.nodes)} 1 {len(mesh.nodes)}"]
    for node_id, (x, y) in enumerate(mesh.nodes, start=1):
        lines.extend((f"0 {node_id} 0 1", str(node_id), f"{_number(x)} {_number(y)} 0"))
    return lines


def _element_lines(mesh: CanonicalMesh) -> list[str]:
    total = len(mesh.boundary_facets) + len(mesh.cells)
    lines = [f"{total} {total} 1 {total}"]
    element_id = 1
    for entity_id, (left, right) in enumerate(mesh.boundary_facets, start=1):
        lines.extend((f"1 {entity_id} 1 1", f"{element_id} {left + 1} {right + 1}"))
        element_id += 1
    for entity_id, cell in enumerate(mesh.cells, start=1):
        lines.extend(
            (
                f"2 {entity_id} 2 1",
                f"{element_id} " + " ".join(str(node_id + 1) for node_id in cell),
            )
        )
        element_id += 1
    return lines


def _number(value: float) -> str:
    return format(float(value), ".17g")
