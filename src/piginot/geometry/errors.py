"""Typed geometry validation failures."""

from __future__ import annotations

from enum import StrEnum


class GeometryReasonCode(StrEnum):
    AMBIGUOUS_BOUNDARY_TAG = "AMBIGUOUS_BOUNDARY_TAG"
    AMBIGUOUS_REGION_TAG = "AMBIGUOUS_REGION_TAG"
    BOUNDARY_MISMATCH = "BOUNDARY_MISMATCH"
    DEGENERATE_ELEMENT = "DEGENERATE_ELEMENT"
    DEGENERATE_VARIABLE_FAMILY = "DEGENERATE_VARIABLE_FAMILY"
    DUPLICATE_CELL = "DUPLICATE_CELL"
    DUPLICATE_NODE = "DUPLICATE_NODE"
    EMPTY_FAMILY = "EMPTY_FAMILY"
    INCONSISTENT_EDGE_ORIENTATION = "INCONSISTENT_EDGE_ORIENTATION"
    INVALID_CELL_INDEX = "INVALID_CELL_INDEX"
    INVALID_FACET_INDEX = "INVALID_FACET_INDEX"
    INVALID_FIELD_DATA = "INVALID_FIELD_DATA"
    INVALID_POINTS = "INVALID_POINTS"
    INVALID_QUERY = "INVALID_QUERY"
    INVERTED_ELEMENT = "INVERTED_ELEMENT"
    MISSING_PHYSICAL_TAGS = "MISSING_PHYSICAL_TAGS"
    MISSING_REGION_TAG = "MISSING_REGION_TAG"
    MIXED_CELL_TYPES = "MIXED_CELL_TYPES"
    NON_FINITE_POINT = "NON_FINITE_POINT"
    NON_MANIFOLD_EDGE = "NON_MANIFOLD_EDGE"
    QUERY_OUTSIDE_DOMAIN = "QUERY_OUTSIDE_DOMAIN"
    UNCLASSIFIED_BOUNDARY_FACET = "UNCLASSIFIED_BOUNDARY_FACET"
    UNIT_NOT_CANONICAL = "UNIT_NOT_CANONICAL"
    UNUSED_NODE = "UNUSED_NODE"
    UNSUPPORTED_BOUNDARY_TYPE = "UNSUPPORTED_BOUNDARY_TYPE"
    UNSUPPORTED_CELL_TYPE = "UNSUPPORTED_CELL_TYPE"
    UNSUPPORTED_GMSH_VERSION = "UNSUPPORTED_GMSH_VERSION"
    UNSUPPORTED_MESH_DIMENSION = "UNSUPPORTED_MESH_DIMENSION"


class GeometryValidationError(ValueError):
    """Raised when an input mesh cannot be made canonical."""

    def __init__(
        self,
        reason_code: GeometryReasonCode | str,
        message: str,
        entity_ids: tuple[int, ...] = (),
    ) -> None:
        code = GeometryReasonCode(reason_code)
        super().__init__(f"{code.value}: {message}")
        self.reason_code = code
        self.entity_ids = tuple(entity_ids)

    @property
    def offending_ids(self) -> tuple[int, ...]:
        """Compatibility alias for the pre-contract attribute name."""

        return self.entity_ids


class GeometryFormatError(GeometryValidationError):
    """Raised when a mesh file format is unsupported."""


class BoundaryTagError(GeometryValidationError):
    """Raised when boundary physical tags are missing or ambiguous."""

    @property
    def facet_ids(self) -> tuple[int, ...]:
        return self.entity_ids


class QueryContainmentError(GeometryValidationError):
    """Raised when query points are malformed or outside the mesh."""

    @property
    def query_ids(self) -> tuple[int, ...]:
        return self.entity_ids


class DegenerateFamilyError(GeometryValidationError):
    """Raised when a fixed mesh family is used where variability is required."""
