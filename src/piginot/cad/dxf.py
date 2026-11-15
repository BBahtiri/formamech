"""Strict closed LINE/ARC/CIRCLE/LWPOLYLINE DXF profiles through OCC."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from piginot.cad._errors import CadImportError
from piginot.contracts import Unit

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class _Edge:
    kind: str
    start: Point
    end: Point
    layer: str
    center: Point = (0.0, 0.0)
    radius: float = 0.0
    angle: float = 0.0
    sweep: float = 0.0


def _point(value: Any, scale: float) -> Point:
    coords = tuple(float(number) for number in value)
    if (
        len(coords) != 3
        or not all(math.isfinite(number) for number in coords)
        or abs(coords[2]) > 1e-12
    ):
        raise CadImportError(
            "planarity", "DXF coordinates must be finite and lie in XY"
        )
    return coords[0] * scale, coords[1] * scale


def _entities(entity: Any) -> tuple[Any, ...]:
    kind = entity.dxftype()
    if kind not in ("LINE", "ARC", "CIRCLE", "LWPOLYLINE"):
        raise CadImportError("entity", f"unsupported DXF entity {kind}")
    if kind != "LINE" and tuple(entity.dxf.extrusion) != (0, 0, 1):
        raise CadImportError(
            "planarity", "DXF extrusion must be the positive XY normal"
        )
    if kind != "LWPOLYLINE":
        return (entity,)
    if not entity.closed or entity.has_width or entity.dxf.elevation != 0:
        raise CadImportError(
            "polyline", "DXF polyline must be closed, zero-width and in XY"
        )
    return tuple(entity.virtual_entities())


def _edge(entity: Any, scale: float, layer: str) -> _Edge:
    kind = entity.dxftype()
    if kind == "LINE":
        start, end = _point(entity.dxf.start, scale), _point(entity.dxf.end, scale)
        if start == end:
            raise CadImportError("topology", "DXF line has zero length")
        return _Edge("line", start, end, layer)
    center = _point(entity.dxf.center, scale)
    radius = float(entity.dxf.radius) * scale
    if not math.isfinite(radius) or radius <= 0:
        raise CadImportError("geometry", "DXF radius must be positive and finite")
    angle = 0.0 if kind == "CIRCLE" else math.radians(float(entity.dxf.start_angle))
    sweep = (
        2 * math.pi
        if kind == "CIRCLE"
        else math.radians(
            (float(entity.dxf.end_angle) - float(entity.dxf.start_angle)) % 360
        )
    )
    if not math.isfinite(angle) or not math.isfinite(sweep) or sweep <= 0:
        raise CadImportError("geometry", "DXF arc must have a finite nonzero sweep")
    start = center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)
    end = (
        start
        if kind == "CIRCLE"
        else (
            center[0] + radius * math.cos(angle + sweep),
            center[1] + radius * math.sin(angle + sweep),
        )
    )
    return _Edge(
        "circle" if kind == "CIRCLE" else "arc",
        start,
        end,
        layer,
        center,
        radius,
        angle,
        sweep,
    )


def _loops(
    edges: tuple[_Edge, ...], tolerance: float
) -> tuple[tuple[tuple[int, int], ...], ...]:
    vertices: list[Point] = []

    def vertex(point: Point) -> int:
        for index, existing in enumerate(vertices):
            if math.dist(point, existing) <= tolerance:
                return index
        vertices.append(point)
        return len(vertices) - 1

    ends = {
        index: (vertex(edge.start), vertex(edge.end))
        for index, edge in enumerate(edges)
        if edge.kind != "circle"
    }
    adjacent = {
        index: tuple(edge_id for edge_id, pair in ends.items() if index in pair)
        for index in range(len(vertices))
    }
    if any(len(ids) != 2 for ids in adjacent.values()):
        raise CadImportError(
            "topology", "DXF edges must form closed nonbranching loops"
        )
    loops: list[tuple[tuple[int, int], ...]] = [
        ((index, 1),) for index, edge in enumerate(edges) if edge.kind == "circle"
    ]
    unused = set(ends)
    while unused:
        first = min(unused)
        current = finish = ends[first][0]
        loop: list[tuple[int, int]] = []
        while True:
            candidates = [index for index in adjacent[current] if index in unused]
            if not candidates:
                raise CadImportError("topology", "DXF loop does not close")
            index = min(candidates)
            start, end = ends[index]
            direction = 1 if start == current else -1
            loop.append((index, direction))
            unused.remove(index)
            current = end if direction == 1 else start
            if current == finish:
                break
        loops.append(tuple(loop))
    return tuple(loops)


def _curves(gmsh: Any, edges: tuple[_Edge, ...], tolerance: float) -> tuple[int, ...]:
    points: dict[Point, int] = {}

    def point(coords: Point) -> int:
        for existing, tag in points.items():
            if math.dist(coords, existing) <= tolerance:
                return tag
        tag = int(gmsh.model.occ.addPoint(*coords, 0))
        points[coords] = tag
        return tag

    def curve(edge: _Edge) -> int:
        if edge.kind == "line":
            return int(gmsh.model.occ.addLine(point(edge.start), point(edge.end)))
        if edge.kind == "circle":
            return int(gmsh.model.occ.addCircle(*edge.center, 0, edge.radius))
        angle = edge.angle + edge.sweep / 2
        middle = (
            edge.center[0] + edge.radius * math.cos(angle),
            edge.center[1] + edge.radius * math.sin(angle),
        )
        return int(
            gmsh.model.occ.addCircleArc(
                point(edge.start), point(middle), point(edge.end), center=False
            )
        )

    return tuple(curve(edge) for edge in edges)


def _closed_surface(
    gmsh: Any,
    edges: tuple[_Edge, ...],
    loops: tuple[tuple[tuple[int, int], ...], ...],
    curves: tuple[int, ...],
    tolerance: float,
) -> int:
    for i, loop in enumerate(loops):
        for j, (first, _) in enumerate(loop):
            for second, _ in loop[j + 1 :]:
                adjacent = any(
                    math.dist(a, b) <= tolerance
                    for a in (edges[first].start, edges[first].end)
                    for b in (edges[second].start, edges[second].end)
                )
                if (
                    not adjacent
                    and gmsh.model.occ.getDistance(1, curves[first], 1, curves[second])[
                        0
                    ]
                    <= tolerance
                ):
                    raise CadImportError("topology", "DXF loop self-intersects")
        for other in loops[i + 1 :]:
            if any(
                gmsh.model.occ.getDistance(1, curves[a], 1, curves[b])[0] <= tolerance
                for a, _ in loop
                for b, _ in other
            ):
                raise CadImportError("topology", "DXF loops touch or intersect")
    wires = tuple(
        gmsh.model.occ.addCurveLoop(
            [curves[index] * direction for index, direction in loop]
        )
        for loop in loops
    )
    faces = tuple(gmsh.model.occ.addPlaneSurface([wire]) for wire in wires)
    gmsh.model.occ.synchronize()
    areas = tuple(float(gmsh.model.occ.getMass(2, face)) for face in faces)
    if any(area <= tolerance * tolerance for area in areas):
        raise CadImportError("topology", "DXF loop has zero area")
    outer = max(range(len(areas)), key=areas.__getitem__)

    def inside(index: int, container: int) -> bool:
        p = edges[loops[index][0][0]].start
        return bool(gmsh.model.isInside(2, faces[container], [*p, 0]))

    holes = tuple(index for index in range(len(loops)) if index != outer)
    if any(not inside(index, outer) for index in holes):
        raise CadImportError(
            "topology", "DXF must have one outer loop with contained holes"
        )
    if any(
        inside(a, b) or inside(b, a)
        for i, a in enumerate(holes)
        for b in holes[i + 1 :]
    ):
        raise CadImportError(
            "topology", "nested hole/island DXF profiles are unsupported"
        )
    gmsh.model.occ.remove([(2, face) for face in faces], recursive=False)
    return int(
        gmsh.model.occ.addPlaneSurface(
            [wires[outer], *(wires[index] for index in holes)]
        )
    )


def add_dxf_profile(gmsh: Any, path: Path, unit: Unit) -> tuple[int, Mapping[int, str]]:
    try:
        from ezdxf.filemanagement import readfile
    except ImportError as error:
        raise CadImportError(
            "dependency", "DXF import requires the cad extra (ezdxf)"
        ) from error

    drawing = readfile(path)
    declared = {Unit.MM: 4, Unit.CM: 5, Unit.M: 6}[unit]
    if drawing.units not in (0, declared):
        raise CadImportError("unit", "DXF $INSUNITS conflicts with the declared unit")
    edges = tuple(
        _edge(primitive, unit.canonical_factor, entity.dxf.layer)
        for entity in drawing.modelspace()
        for primitive in _entities(entity)
    )
    if not edges:
        raise CadImportError("topology", "DXF contains no profile geometry")
    xs = tuple(
        value
        for edge in edges
        for value in (
            edge.start[0],
            edge.end[0],
            edge.center[0] - edge.radius,
            edge.center[0] + edge.radius,
        )
        if edge.kind != "line"
    ) + tuple(
        value
        for edge in edges
        if edge.kind == "line"
        for value in (edge.start[0], edge.end[0])
    )
    ys = tuple(
        value
        for edge in edges
        for value in (
            edge.start[1],
            edge.end[1],
            edge.center[1] - edge.radius,
            edge.center[1] + edge.radius,
        )
        if edge.kind != "line"
    ) + tuple(
        value
        for edge in edges
        if edge.kind == "line"
        for value in (edge.start[1], edge.end[1])
    )
    tolerance = 1e-9 * max(1.0, max(xs) - min(xs), max(ys) - min(ys))
    loops = _loops(edges, tolerance)
    curves = _curves(gmsh, edges, tolerance)
    surface = _closed_surface(gmsh, edges, loops, curves, tolerance)
    gmsh.model.occ.synchronize()
    return surface, {tag: edge.layer for tag, edge in zip(curves, edges, strict=True)}
