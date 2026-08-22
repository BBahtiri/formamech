"""Classification guards for fixed and variable canonical mesh families."""

from __future__ import annotations

from collections.abc import Sequence

from piginot.contracts import CanonicalMesh
from piginot.geometry.errors import (
    DegenerateFamilyError,
    GeometryReasonCode,
    GeometryValidationError,
)
from piginot.geometry.models import MeshFamilyKind


def classify_mesh_family(meshes: Sequence[CanonicalMesh]) -> MeshFamilyKind:
    """Classify a non-empty family solely by its canonical mesh identities."""

    if not meshes:
        raise GeometryValidationError(
            GeometryReasonCode.EMPTY_FAMILY,
            "mesh family must contain at least one canonical mesh",
        )

    unique_mesh_ids = frozenset(mesh.mesh_id for mesh in meshes)
    if len(unique_mesh_ids) == 1:
        return MeshFamilyKind.FIXED_DEGENERATE
    return MeshFamilyKind.VARIABLE


def require_variable_family(meshes: Sequence[CanonicalMesh]) -> None:
    """Reject a non-empty family that contains only one canonical identity."""

    if classify_mesh_family(meshes) is MeshFamilyKind.FIXED_DEGENERATE:
        raise DegenerateFamilyError(
            GeometryReasonCode.DEGENERATE_VARIABLE_FAMILY,
            "a fixed mesh family cannot be used as a variable family",
        )
