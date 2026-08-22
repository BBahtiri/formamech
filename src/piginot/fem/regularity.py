"""S7 peak-stress regularity and resolution screening."""

from __future__ import annotations

import math

from piginot.contracts import CanonicalMesh
from piginot.fem.models import FemFailureCode, FemRequest


def peak_stress_screen(
    request: FemRequest,
) -> tuple[FemFailureCode, str] | None:
    """Reject singular or under-resolved pointwise peak-stress requests."""

    reentrant = _reentrant_vertices(request.mesh)
    mixed = _mixed_boundary_vertices(request)
    for qoi in request.qois:
        is_peak = qoi.reduction in {"maximum", "pointwise_peak"} and (
            "stress" in qoi.field or "piola" in qoi.field
        )
        if not is_peak:
            continue
        region = _vertices_for_tag(request.mesh, qoi.region_tag)
        if region & reentrant:
            return (
                FemFailureCode.QOI_REGULARITY_REJECTED,
                "pointwise peak stress QoI is singular at a re-entrant boundary",
            )
        if region & mixed:
            return (
                FemFailureCode.QOI_REGULARITY_REJECTED,
                "pointwise peak stress QoI is singular at a mixed-BC junction",
            )
        graded = request.mesh.quality_metrics.get("qoi_graded_mesh", 0.0) >= 1.0
        if request.config.element_order != 2 and not graded:
            return (
                FemFailureCode.QOI_RESOLUTION_REJECTED,
                "pointwise peak stress requires P2 or a declared graded mesh",
            )
    return None


def _reentrant_vertices(mesh: CanonicalMesh) -> set[int]:
    boundary_vertices = {vertex for facet in mesh.boundary_facets for vertex in facet}
    result: set[int] = set()
    for vertex in boundary_vertices:
        angle_sum = math.fsum(
            _triangle_angle(mesh, cell, vertex) for cell in mesh.cells if vertex in cell
        )
        if angle_sum > math.pi + 1.0e-10:
            result.add(vertex)
    return result


def _triangle_angle(mesh: CanonicalMesh, cell: tuple[int, ...], vertex: int) -> float:
    others = [index for index in cell if index != vertex]
    origin = mesh.nodes[vertex]
    left = (
        mesh.nodes[others[0]][0] - origin[0],
        mesh.nodes[others[0]][1] - origin[1],
    )
    right = (
        mesh.nodes[others[1]][0] - origin[0],
        mesh.nodes[others[1]][1] - origin[1],
    )
    cross = left[0] * right[1] - left[1] * right[0]
    dot = left[0] * right[0] + left[1] * right[1]
    return math.atan2(abs(cross), dot)


def _mixed_boundary_vertices(request: FemRequest) -> set[int]:
    mesh = request.mesh
    classes = {index: "natural" for index in range(len(mesh.boundary_facets))}
    for condition in request.boundary_conditions:
        category = (
            "essential" if condition.kind in {"dirichlet", "symmetry"} else "natural"
        )
        for tagged_facet_index in mesh.semantic_tags.get(condition.boundary_tag, ()):
            classes[tagged_facet_index] = category

    incident: dict[int, set[str]] = {}
    for facet_index, facet in enumerate(mesh.boundary_facets):
        for vertex in facet:
            incident.setdefault(vertex, set()).add(classes[facet_index])
    return {
        vertex
        for vertex, categories in incident.items()
        if categories == {"essential", "natural"}
    }


def _vertices_for_tag(mesh: CanonicalMesh, tag: str) -> set[int]:
    return {
        vertex
        for facet_index in mesh.semantic_tags.get(tag, ())
        for vertex in mesh.boundary_facets[facet_index]
    }


__all__ = ("peak_stress_screen",)
