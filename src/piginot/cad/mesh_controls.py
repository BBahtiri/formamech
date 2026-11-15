"""Gmsh sizing fields and measured checks for a declared CAD mesh policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from piginot.cad._errors import CadImportError
from piginot.cad.mesh_policy import MeshPolicy
from piginot.contracts import CanonicalMesh


@dataclass(frozen=True, slots=True)
class Circle:
    center: tuple[float, float]
    radius: float
    span: float


def _circle(gmsh: Any, tag: int) -> Circle:
    low, high = gmsh.model.getParametrizationBounds(1, tag)
    start, span = float(low[0]), float(high[0] - low[0])
    points = [
        tuple(float(x) for x in gmsh.model.getValue(1, tag, [start + span * f])[:2])
        for f in (0, 1 / 3, 2 / 3)
    ]
    a, b, c = points
    bx, by = b[0] - a[0], b[1] - a[1]
    cx, cy = c[0] - a[0], c[1] - a[1]
    det = 2 * (bx * cy - by * cx)
    if det == 0:
        raise CadImportError("curve_geometry", "unqualified degenerate circular arc")
    center = (
        a[0] + ((bx * bx + by * by) * cy - (cx * cx + cy * cy) * by) / det,
        a[1] + ((cx * cx + cy * cy) * bx - (bx * bx + by * by) * cx) / det,
    )
    radius = math.dist(center, a)
    if not math.isfinite(radius) or radius <= 0 or not 0 < span <= 2 * math.pi + 1e-8:
        raise CadImportError("curve_geometry", "unqualified circular parametrization")
    return Circle(center, radius, span)


def _threshold(
    gmsh: Any, field: int, size: float, lo: float, hi: float, policy: MeshPolicy
) -> int:
    tag = int(gmsh.model.mesh.field.add("Threshold"))
    for name, value in {
        "InField": field,
        "SizeMin": size,
        "SizeMax": policy.global_size_mm,
        "DistMin": lo,
        "DistMax": hi,
        "Sigmoid": 1,
    }.items():
        gmsh.model.mesh.field.setNumber(tag, name, value)
    return tag


def _boundary_budget(segments: int, policy: MeshPolicy) -> None:
    # A connected triangulated planar domain has at least B-2 triangles.
    # Midpoint subdivision multiplies T by 16 and B by 4 at level 4;
    # P2 vector DOFs >= 2 * (boundary vertices + all edges).
    triangles = 16 * max(1, segments - 2)
    dofs = 3 * triangles + 12 * segments
    if triangles > policy.maximum_triangles or dofs > policy.maximum_p2_dofs:
        raise CadImportError(
            "budget", "requested boundary sizing exceeds the reference mesh budget"
        )


def configure(
    gmsh: Any, boundaries: tuple[Any, ...], policy: MeshPolicy
) -> dict[str, Circle]:
    """Apply fields before meshing; constrain supported circular chords exactly."""
    by_name = {b.name: b for b in boundaries}
    if any(item.boundary not in by_name for item in policy.feature_sizes):
        raise CadImportError("boundary", "mesh policy selects an unknown boundary")
    for option, value in {
        "Mesh.MeshSizeMin": policy.minimum_size_mm,
        "Mesh.MeshSizeMax": policy.global_size_mm,
        "Mesh.MeshSizeFromPoints": 0,
        "Mesh.MeshSizeExtendFromBoundary": 0,
        "Mesh.MeshSizeFromCurvature": policy.curvature_points_per_circle,
        "Mesh.MinimumCirclePoints": policy.curvature_points_per_circle,
    }.items():
        gmsh.option.setNumber(option, value)
    fields = []
    for item in policy.feature_sizes:
        distance = int(gmsh.model.mesh.field.add("Distance"))
        gmsh.model.mesh.field.setNumbers(
            distance, "CurvesList", [by_name[item.boundary].source_id]
        )
        # A sampled distance field guides density; it is not a geometry bound.
        gmsh.model.mesh.field.setNumber(distance, "Sampling", 200)
        fields.append(
            _threshold(
                gmsh,
                distance,
                item.size_mm,
                item.distance_min_mm,
                item.distance_max_mm,
                policy,
            )
        )
    for point in policy.point_sizes:
        distance = int(gmsh.model.mesh.field.add("MathEval"))
        x, y = point.center_mm
        gmsh.model.mesh.field.setString(
            distance, "F", f"Sqrt((x-({x:.17g}))^2+(y-({y:.17g}))^2)"
        )
        fields.append(
            _threshold(
                gmsh,
                distance,
                point.size_mm,
                point.distance_min_mm,
                point.distance_max_mm,
                policy,
            )
        )
    constant = int(gmsh.model.mesh.field.add("MathEval"))
    gmsh.model.mesh.field.setString(constant, "F", str(policy.global_size_mm))
    minimum = int(gmsh.model.mesh.field.add("Min"))
    gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [constant, *fields])
    gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
    selected = {item.boundary: item.size_mm for item in policy.feature_sizes}
    circles = {}
    boundary_segments = 0
    for boundary in boundaries:
        kind = gmsh.model.getType(1, boundary.source_id)
        if kind not in ("Circle", "Line"):
            raise CadImportError(
                "curve_geometry", f"unqualified curve {boundary.name}: {kind}"
            )
        size = selected.get(boundary.name, policy.global_size_mm)
        if kind == "Circle":
            circle = _circle(gmsh, boundary.source_id)
            circles[boundary.name] = circle
            tolerance = min(policy.curve_tolerance_mm, circle.radius)
            chord_limit = 2 * math.sqrt(tolerance * (2 * circle.radius - tolerance))
            if chord_limit < policy.minimum_size_mm:
                raise CadImportError(
                    "minimum_size",
                    "minimum_size_mm prevents the declared curve tolerance",
                )
            for point in policy.point_sizes:
                curve_distance = abs(
                    math.dist(point.center_mm, circle.center) - circle.radius
                )
                if curve_distance < point.distance_max_mm:
                    # Conservative near an arc endpoint: use the whole circle's distance.
                    size = min(size, point.size_mm)
            angle = min(
                2 * math.asin(min(1, chord_limit / (2 * circle.radius))),
                2 * math.pi / policy.curvature_points_per_circle,
                size / circle.radius,
            )
            segments = math.ceil(circle.span / angle)
            boundary_segments += segments
            _boundary_budget(boundary_segments, policy)
            gmsh.model.mesh.setTransfiniteCurve(boundary.source_id, segments + 1)
        elif boundary.name in selected:
            length = float(gmsh.model.occ.getMass(1, boundary.source_id))
            segments = math.ceil(length / size)
            boundary_segments += segments
            _boundary_budget(boundary_segments, policy)
            gmsh.model.mesh.setTransfiniteCurve(boundary.source_id, segments + 1)
    return circles


def projected_counts(mesh: CanonicalMesh, level: int = 4) -> dict[str, int]:
    """Count uniform midpoint descendants before allocating their arrays."""
    vertices, triangles, boundary = (
        len(mesh.nodes),
        len(mesh.cells),
        len(mesh.boundary_facets),
    )
    for _ in range({1: 0, 2: 1, 4: 2}[level]):
        vertices += (3 * triangles + boundary) // 2
        triangles *= 4
        boundary *= 2
    edge_count = (3 * triangles + boundary) // 2
    return {
        "reference_level": level,
        "reference_triangles": triangles,
        "reference_p2_dofs": 2 * (vertices + edge_count),
    }


def _quality(mesh: CanonicalMesh) -> dict[str, float]:
    angles: list[float] = []
    ratios: list[float] = []
    lengths: list[float] = []
    for cell in mesh.cells:
        a, b, c = (mesh.nodes[i] for i in cell)
        sides = (math.dist(a, b), math.dist(b, c), math.dist(c, a))
        twice_area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        ratios.append(max(sides) ** 2 / twice_area)
        lengths.extend(sides)
        angles.extend(
            math.degrees(
                math.acos(
                    max(
                        -1,
                        min(
                            1,
                            (
                                sides[(i + 1) % 3] ** 2
                                + sides[(i + 2) % 3] ** 2
                                - sides[i] ** 2
                            )
                            / (2 * sides[(i + 1) % 3] * sides[(i + 2) % 3]),
                        ),
                    )
                )
            )
            for i in range(3)
        )
    return {
        "minimum_angle_deg": min(angles),
        "maximum_aspect_ratio": max(ratios),
        "minimum_edge_mm": min(lengths),
        "maximum_edge_mm": max(lengths),
    }


def measure(
    gmsh: Any,
    mesh: CanonicalMesh,
    boundaries: tuple[Any, ...],
    circles: dict[str, Circle],
    policy: MeshPolicy,
) -> dict[str, Any]:
    """Check achieved geometry/quality separately from solution convergence."""
    counts = {"base_triangles": len(mesh.cells), **projected_counts(mesh)}
    if (
        counts["reference_triangles"] > policy.maximum_triangles
        or counts["reference_p2_dofs"] > policy.maximum_p2_dofs
    ):
        raise CadImportError(
            "budget", "prepared reference ladder exceeds triangle/DOF budget"
        )
    quality = _quality(mesh)
    if (
        quality["minimum_angle_deg"] < policy.minimum_angle_deg
        or quality["maximum_aspect_ratio"] > policy.maximum_aspect_ratio
    ):
        raise CadImportError(
            "quality", "achieved mesh violates the declared quality limits"
        )
    achieved = {}
    for boundary in boundaries:
        segments = [
            tuple(mesh.nodes[i] for i in mesh.boundary_facets[j])
            for j in mesh.semantic_tags[boundary.name]
        ]
        lengths = [math.dist(a, b) for a, b in segments]
        entry: dict[str, Any] = {
            "segment_count": len(segments),
            "max_chord_mm": max(lengths),
            "min_chord_mm": min(lengths),
            "status": "PASS",
        }
        if boundary.name in circles:
            circle = circles[boundary.name]
            angles, deviations = [], []
            for a, b in segments:
                ax, ay = a[0] - circle.center[0], a[1] - circle.center[1]
                bx, by = b[0] - circle.center[0], b[1] - circle.center[1]
                angle = abs(math.atan2(ax * by - ay * bx, ax * bx + ay * by))
                angles.append(angle)
                radial_error = max(
                    abs(math.hypot(ax, ay) - circle.radius),
                    abs(math.hypot(bx, by) - circle.radius),
                )
                deviations.append(
                    2 * circle.radius * math.sin(angle / 4) ** 2 + radial_error
                )
            if not math.isclose(sum(angles), circle.span, rel_tol=1e-7, abs_tol=1e-9):
                raise CadImportError(
                    "curve_geometry",
                    "circular segments do not cover the original CAD arc",
                )
            entry.update(
                radius_mm=circle.radius,
                max_deviation_mm=max(deviations),
                method="analytic-circle-chord-with-endpoint-error",
            )
        else:
            low, high = gmsh.model.getParametrizationBounds(1, boundary.source_id)
            start = gmsh.model.getValue(1, boundary.source_id, [float(low[0])])
            end = gmsh.model.getValue(1, boundary.source_id, [float(high[0])])
            a = (float(start[0]), float(start[1]))
            b = (float(end[0]), float(end[1]))
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            if not math.isclose(sum(lengths), length, rel_tol=1e-7, abs_tol=1e-9):
                raise CadImportError(
                    "curve_geometry", "segments do not cover the original CAD line"
                )
            entry.update(
                max_deviation_mm=max(
                    abs(dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / length
                    for s in segments
                    for p in s
                ),
                method="analytic-straight-segment",
            )
        if entry["max_deviation_mm"] > policy.curve_tolerance_mm:
            raise CadImportError(
                "curve_geometry",
                "achieved CAD deviation exceeds the declared tolerance",
            )
        achieved[boundary.name] = entry
    return {
        "schema": "piginot.cad-mesh-evidence-v1",
        "policy_sha256": policy.sha256,
        "geometry": {
            "status": "PASS",
            "boundaries": achieved,
            "scope": "CAD-curve-to-polygon only; no solution error bound",
        },
        "quality": quality,
        "mesh": counts,
        "limits_status": "PASS",
        "solution_convergence": "NOT_CHECKED",
    }
