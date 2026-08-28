"""Compatibility wrappers for the S5 geometry ingestion API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from piginot.contracts import CanonicalMesh, GeometryFamilySpec, Unit
from piginot.geometry.canonicalize import canonicalize_mesh
from piginot.geometry.family import classify_mesh_family, require_variable_family
from piginot.geometry.models import CellType, MeshFamilyKind, MeshPayload, PointLocation
from piginot.geometry.queries import locate_points, require_points_contained
from piginot.geometry.topology import TopologySummary, topology_summary


@dataclass(frozen=True, slots=True)
class GeometryFamilyClassification:
    """Legacy S5 classification shape retained for early integration tests."""

    kind: str
    can_be_used_as_variable: bool
    topology_signature: str


def read_gmsh(path: str | Path, *, unit: Unit) -> CanonicalMesh:
    """Read a tagged, planar Gmsh v4.1 mesh into canonical form."""

    from piginot.geometry.gmsh import read_gmsh as _read_gmsh

    return _read_gmsh(path, unit=unit)


def canonicalize_meshio(
    raw_mesh: Any,
    *,
    unit: Unit,
    source_path: Path | None = None,
    source_format: str | None = None,
) -> CanonicalMesh:
    """Canonicalize an in-memory meshio mesh using the strict Gmsh tag contract."""

    from piginot.geometry.gmsh import canonicalize_meshio as _canonicalize_meshio

    return _canonicalize_meshio(
        raw_mesh,
        unit=unit,
        source_path=source_path,
        source_format=source_format,
    )


def write_gmsh(mesh: CanonicalMesh, path: str | Path) -> None:
    """Write canonical triangles as deterministic ASCII Gmsh v4.1."""

    from piginot.geometry.gmsh import write_gmsh as _write_gmsh

    _write_gmsh(mesh, path)


def to_meshio(mesh: CanonicalMesh) -> Any:
    """Convert a canonical mesh to a deterministic in-memory meshio object."""

    from piginot.geometry.gmsh import to_meshio as _to_meshio

    return _to_meshio(mesh)


def from_gmsh(path: str | Path, *, unit: Unit) -> CanonicalMesh:
    """Compatibility alias for :func:`read_gmsh`."""

    return read_gmsh(path, unit=unit)


def to_gmsh(mesh: CanonicalMesh) -> Any:
    """Return a meshio mesh for callers that still use the in-memory adapter."""

    return to_meshio(mesh)


def contains_points(mesh: CanonicalMesh, points: Sequence[object]) -> tuple[bool, ...]:
    """Return whether each point lies in the mesh closure."""

    return tuple(
        location is not PointLocation.EXTERIOR
        for location in locate_points(mesh, points)
    )


def classify_geometry_family(
    family: GeometryFamilySpec, mesh: CanonicalMesh
) -> GeometryFamilyClassification:
    """Classify the S2 family declaration against one canonical mesh.

    S5 can only distinguish fixed declarations from parameterized declarations.
    Cross-sample variability remains the job of ``classify_mesh_family``.
    """

    if len(family.parameters) == 0 and len(family.derived_quantities) == 0:
        return GeometryFamilyClassification(
            kind="degenerate_fixed_mesh",
            can_be_used_as_variable=False,
            topology_signature=mesh.topology_signature,
        )
    return GeometryFamilyClassification(
        kind="variable_family",
        can_be_used_as_variable=True,
        topology_signature=mesh.topology_signature,
    )


__all__ = (
    "CellType",
    "GeometryFamilyClassification",
    "MeshFamilyKind",
    "MeshPayload",
    "PointLocation",
    "TopologySummary",
    "canonicalize_mesh",
    "canonicalize_meshio",
    "classify_geometry_family",
    "classify_mesh_family",
    "contains_points",
    "from_gmsh",
    "locate_points",
    "read_gmsh",
    "require_points_contained",
    "require_variable_family",
    "topology_summary",
    "to_gmsh",
    "write_gmsh",
)
