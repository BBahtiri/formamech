"""Geometry ingestion and validation for canonical 2D meshes."""

from piginot.geometry.canonicalize import canonicalize_mesh
from piginot.geometry.errors import (
    BoundaryTagError,
    DegenerateFamilyError,
    GeometryFormatError,
    GeometryReasonCode,
    GeometryValidationError,
    QueryContainmentError,
)
from piginot.geometry.family import classify_mesh_family, require_variable_family
from piginot.geometry.gmsh import canonicalize_meshio, read_gmsh, to_meshio, write_gmsh
from piginot.geometry.ingest import (
    GeometryFamilyClassification,
    classify_geometry_family,
    contains_points,
    from_gmsh,
    to_gmsh,
)
from piginot.geometry.models import CellType, MeshFamilyKind, MeshPayload, PointLocation
from piginot.geometry.queries import locate_points, require_points_contained
from piginot.geometry.topology import TopologySummary, topology_summary

__all__ = (
    "BoundaryTagError",
    "CellType",
    "DegenerateFamilyError",
    "GeometryFamilyClassification",
    "GeometryFormatError",
    "GeometryReasonCode",
    "GeometryValidationError",
    "MeshFamilyKind",
    "MeshPayload",
    "PointLocation",
    "QueryContainmentError",
    "TopologySummary",
    "canonicalize_meshio",
    "canonicalize_mesh",
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
    "to_meshio",
    "write_gmsh",
)
