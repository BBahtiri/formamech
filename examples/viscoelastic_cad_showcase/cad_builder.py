"""OpenCASCADE STEP CAD generator for a genus-2 viscoelastic elastomer mount bracket."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from piginot.contracts import CanonicalMesh


@dataclass(frozen=True, slots=True)
class BracketGeometryParams:
    """Parametric dimensions (mm) of the dual-hole relief-notched mount bracket."""

    length_mm: float = 40.0
    height_mm: float = 20.0
    notch_radius_mm: float = 4.0
    left_hole_radius_mm: float = 3.2
    right_hole_radius_mm: float = 2.6
    left_hole_x_mm: float = 11.0
    right_hole_x_mm: float = 29.0

    @property
    def analytical_area_mm2(self) -> float:
        return (
            self.length_mm * self.height_mm
            - 0.5 * math.pi * self.notch_radius_mm**2
            - math.pi * self.left_hole_radius_mm**2
            - math.pi * self.right_hole_radius_mm**2
        )

    def document(self) -> dict[str, float]:
        return {
            "length_mm": self.length_mm,
            "height_mm": self.height_mm,
            "notch_radius_mm": self.notch_radius_mm,
            "left_hole_radius_mm": self.left_hole_radius_mm,
            "right_hole_radius_mm": self.right_hole_radius_mm,
            "left_hole_x_mm": self.left_hole_x_mm,
            "right_hole_x_mm": self.right_hole_x_mm,
            "analytical_area_mm2": self.analytical_area_mm2,
        }


def generate_bracket_step(
    output_path: Path,
    params: BracketGeometryParams | None = None,
) -> Path:
    """Generate a planar STEP CAD model via Gmsh OpenCASCADE kernel.

    Geometry:
    - Rectangular elastomer mount plate ``[0, L] x [0, H]``
    - Semi-circular top relief notch centered at ``(L/2, H)``
    - Two internal circular mounting/lightening holes at ``(x_left, H/2)`` and
      ``(x_right, H/2)`` (genus-2: 3 closed boundary loops)
    """
    import gmsh  # type: ignore[import-untyped]

    p = BracketGeometryParams() if params is None else params
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already_init = bool(gmsh.isInitialized())
    if not already_init:
        gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("viscoelastic_mount_bracket")
        occ = gmsh.model.occ

        def _circle_disk(cx: float, cy: float, r: float) -> int:
            c = occ.addCircle(cx, cy, 0.0, r)
            loop = occ.addCurveLoop([c])
            return int(occ.addPlaneSurface([loop]))

        plate = occ.addRectangle(0.0, 0.0, 0.0, p.length_mm, p.height_mm)
        tools: list[tuple[int, int]] = []
        if p.notch_radius_mm > 0.0:
            notch = _circle_disk(0.5 * p.length_mm, p.height_mm, p.notch_radius_mm)
            tools.append((2, notch))
        if p.left_hole_radius_mm > 0.0:
            hole_l = _circle_disk(
                p.left_hole_x_mm, 0.5 * p.height_mm, p.left_hole_radius_mm
            )
            tools.append((2, hole_l))
        if p.right_hole_radius_mm > 0.0:
            hole_r = _circle_disk(
                p.right_hole_x_mm, 0.5 * p.height_mm, p.right_hole_radius_mm
            )
            tools.append((2, hole_r))

        if tools:
            occ.cut([(2, plate)], tools, removeObject=True, removeTool=True)
        occ.synchronize()
        gmsh.write(str(output_path))
    finally:
        if not already_init and gmsh.isInitialized():
            gmsh.finalize()
    return output_path


def classify_bracket_boundaries(
    mesh: CanonicalMesh,
    params: BracketGeometryParams | None = None,
    *,
    tol_mm: float = 1.0e-3,
) -> dict[str, str]:
    """Map imported ``edge-NNNNNN`` tags to engineering roles on the mount bracket.

    Returns ``{edge_tag: semantic_role}`` where ``semantic_role`` is one of:
    ``"left_clamp"``, ``"right_load"``, ``"bottom_free"``, ``"top_free"``,
    ``"notch_free"``, ``"left_hole"``, ``"right_hole"``.
    """
    p = BracketGeometryParams() if params is None else params
    mapping: dict[str, str] = {}
    for tag, facet_ids in sorted(mesh.semantic_tags.items()):
        pts = [mesh.nodes[idx] for f in facet_ids for idx in mesh.boundary_facets[f]]
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        if all(abs(x) <= tol_mm for x in xs):
            mapping[tag] = "left_clamp"
        elif all(abs(x - p.length_mm) <= tol_mm for x in xs):
            mapping[tag] = "right_load"
        elif all(abs(y) <= tol_mm for y in ys):
            mapping[tag] = "bottom_free"
        elif all(abs(y - p.height_mm) <= tol_mm for y in ys):
            mapping[tag] = "top_free"
        elif (
            my > 0.5 * p.height_mm + 0.5 * p.left_hole_radius_mm
            and abs(mx - 0.5 * p.length_mm) <= p.notch_radius_mm + 1.0
        ):
            mapping[tag] = "notch_free"
        elif mx < 0.5 * p.length_mm:
            mapping[tag] = "left_hole"
        else:
            mapping[tag] = "right_hole"
    return mapping


def relabel_bracket_mesh(
    mesh: CanonicalMesh,
    params: BracketGeometryParams | None = None,
) -> tuple[CanonicalMesh, dict[str, str]]:
    """Return a canonical mesh with human-readable boundary group names merged cleanly."""
    from piginot.contracts import Unit
    from piginot.geometry import CellType, MeshPayload, canonicalize_mesh

    edge_map = classify_bracket_boundaries(mesh, params)
    merged: dict[str, list[int]] = {}
    for edge_tag, role_name in edge_map.items():
        merged.setdefault(role_name, []).extend(mesh.semantic_tags[edge_tag])

    relabeled = canonicalize_mesh(
        MeshPayload(
            nodes=mesh.nodes,
            cells=mesh.cells,
            cell_type=CellType.TRIANGLE,
            boundary_facets=mesh.boundary_facets,
            boundary_tags={k: tuple(sorted(v)) for k, v in sorted(merged.items())},
            region_ids=mesh.region_ids,
            unit=Unit.MM,
            source_provenance=dict(mesh.source_provenance),
        )
    )
    return relabeled, edge_map


def summarize_bracket_project(project_doc: dict[str, Any]) -> dict[str, Any]:
    """Extract key topological stats from ``inspect_geometry`` output."""
    return {
        "mesh_id": project_doc["mesh"]["mesh_id"],
        "node_count": project_doc["mesh"]["node_count"],
        "triangle_count": project_doc["mesh"]["triangle_count"],
        "boundary_count": len(project_doc["boundaries"]),
    }
