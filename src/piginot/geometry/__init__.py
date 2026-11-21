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
from piginot.geometry.ingest import (
    GeometryFamilyClassification,
    canonicalize_meshio,
    classify_geometry_family,
    contains_points,
    from_gmsh,
    read_gmsh,
    to_gmsh,
    to_meshio,
    write_gmsh,
)
from piginot.geometry.mesh3d import (
    MESH3D_SCHEMA,
    TET4_LOCKING_POISSON_LIMIT,
    CanonicalMesh3D,
    CellType3D,
    VolumetricLockingError,
    make_brick_hex8_mesh,
    make_brick_tet4_mesh,
    make_brick_tet10_mesh,
)
from piginot.geometry.models import CellType, MeshFamilyKind, MeshPayload, PointLocation
from piginot.geometry.queries import locate_points, require_points_contained
from piginot.geometry.topology import TopologySummary, topology_summary

__all__ = (
    "MESH3D_SCHEMA",
    "TET4_LOCKING_POISSON_LIMIT",
    "BoundaryTagError",
    "CanonicalMesh3D",
    "CellType",
    "CellType3D",
    "DegenerateFamilyError",
    "VolumetricLockingError",
    "make_brick_hex8_mesh",
    "make_brick_tet4_mesh",
    "make_brick_tet10_mesh",
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
