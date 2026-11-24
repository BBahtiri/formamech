"""3D solid STEP CAD import, OpenCASCADE solid generators, and TET10/TET4 meshing (M5 / F1).

Imports single-solid 3D STEP/STP models via OpenCASCADE (``gmsh.model.occ``),
classifies and tags every 2D boundary face deterministically, generates
positively oriented quadratic ``TET10`` (or linear ``TET4``) meshes conforming
to :class:`~piginot.geometry.mesh3d.CanonicalMesh3D`, and persists hash-bound
3D geometry project directories.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from piginot.cad._errors import CadImportError
from piginot.contracts import Unit, canonical_json_bytes
from piginot.geometry.mesh3d import (
    CanonicalMesh3D,
    CellType3D,
    _tet_signed_volume,
)

CAD_3D_PROJECT_SCHEMA: Final = "formamech.cad-3d-project-v1"
BUILTIN_3D_TEMPLATES: Final[tuple[str, ...]] = (
    "box",
    "l-bracket",
    "mounting-lug",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_doc(document: object) -> str:
    return _sha256_bytes(canonical_json_bytes(document))


def _gmsh_module() -> Any:
    try:
        return __import__("gmsh")
    except (ImportError, OSError) as exc:
        raise CadImportError(
            "dependency",
            "3D CAD import requires formamech[cad] and Gmsh OpenCASCADE libraries",
        ) from exc


@dataclass(frozen=True, slots=True)
class FaceInfo3D:
    """Canonical metadata for one 2D boundary face of a 3D solid CAD model."""

    name: str
    kind: str
    source_id: int
    source_label: str
    area_mm2: float
    centroid_mm: tuple[float, float, float]
    normal_mm: tuple[float, float, float]
    triangle_count: int

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "source_id": self.source_id,
            "source_label": self.source_label,
            "area_mm2": self.area_mm2,
            "centroid_mm": list(self.centroid_mm),
            "normal_mm": list(self.normal_mm),
            "triangle_count": self.triangle_count,
        }


@dataclass(frozen=True, slots=True)
class ImportedCad3D:
    """Validated 3D solid CAD import result with ``CanonicalMesh3D`` and face tags."""

    mesh: CanonicalMesh3D
    faces: tuple[FaceInfo3D, ...]
    source_sha256: str
    source_format: str
    volume_mm3: float
    surface_area_mm2: float
    mesh_size_mm: float
    provenance: Mapping[str, str]

    def summary_document(self) -> dict[str, Any]:
        xs = [p[0] for p in self.mesh.nodes]
        ys = [p[1] for p in self.mesh.nodes]
        zs = [p[2] for p in self.mesh.nodes]
        return {
            "schema": CAD_3D_PROJECT_SCHEMA,
            "mesh_id": self.mesh.mesh_id,
            "cell_type": self.mesh.cell_type.value,
            "unit": self.mesh.unit.value,
            "n_nodes": len(self.mesh.nodes),
            "n_cells": len(self.mesh.cells),
            "n_surface_triangles": len(self.mesh.surface_triangles),
            "volume_mm3": self.volume_mm3,
            "surface_area_mm2": self.surface_area_mm2,
            "mesh_size_mm": self.mesh_size_mm,
            "bbox_min_mm": [min(xs), min(ys), min(zs)],
            "bbox_max_mm": [max(xs), max(ys), max(zs)],
            "source_sha256": self.source_sha256,
            "source_format": self.source_format,
            "faces": [f.document() for f in self.faces],
            "provenance": dict(sorted(self.provenance.items())),
        }


def _orient_tet4(
    nodes: list[tuple[float, float, float]], v0: int, v1: int, v2: int, v3: int
) -> tuple[int, int, int, int]:
    if _tet_signed_volume(nodes[v0], nodes[v1], nodes[v2], nodes[v3]) < 0.0:
        return (v0, v2, v1, v3)
    return (v0, v1, v2, v3)


def _orient_gmsh_tet10(
    nodes: list[tuple[float, float, float]], raw: tuple[int, ...]
) -> tuple[int, ...]:
    """Map Gmsh TET10 (type 11) to positive-volume ``CanonicalMesh3D`` ordering.

    Gmsh ordering: ``(v0, v1, v2, v3, e01, e12, e20, e03, e23, e13)``.
    FormaMech ordering: ``(v0, v1, v2, v3, e01, e12, e20, e03, e13, e23)``.
    Swapping ``v1 <-> v2`` to restore positive volume also swaps ``e01 <-> e20``
    and ``e13 <-> e23``.
    """
    v0, v1, v2, v3, e01, e12, e20, e03, e23_gmsh, e13_gmsh = raw
    if _tet_signed_volume(nodes[v0], nodes[v1], nodes[v2], nodes[v3]) < 0.0:
        return (v0, v2, v1, v3, e20, e12, e01, e03, e23_gmsh, e13_gmsh)
    return (v0, v1, v2, v3, e01, e12, e20, e03, e13_gmsh, e23_gmsh)


def _face_kind_name(gmsh_type: str) -> str:
    return {
        "Plane": "planar",
        "Cylinder": "cylindrical",
        "Cone": "conical",
        "Torus": "toroidal",
        "Sphere": "spherical",
        "BSpline": "bspline",
    }.get(gmsh_type, "other")


def _sanitize_face_label(raw: str, fallback: str) -> str:
    cleaned = "".join(
        ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in raw.strip()
    ).strip("_")
    return cleaned if cleaned else fallback


def import_cad_3d(
    source: Path | str,
    *,
    unit: Unit = Unit.MM,
    mesh_size_mm: float = 3.0,
    element_type: CellType3D = CellType3D.TET10,
    poisson_ratio: float | None = None,
    second_order_linear: bool = True,
) -> ImportedCad3D:
    """Import a single-solid 3D STEP/STP file into a ``CanonicalMesh3D``.

    When ``poisson_ratio`` is provided, immediately enforces
    :meth:`CanonicalMesh3D.validate_incompressibility_locking` (refusing ``TET4``
    at ``poisson_ratio >= 0.45``).
    """
    path = Path(source)
    if path.is_symlink() or not path.is_file():
        raise CadImportError("source", f"3D CAD file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in (".step", ".stp"):
        raise CadImportError("format", "3D CAD import supports STEP and STP files only")
    if (
        isinstance(mesh_size_mm, bool)
        or not isinstance(mesh_size_mm, (int, float))
        or not math.isfinite(float(mesh_size_mm))
        or float(mesh_size_mm) <= 0.0
    ):
        raise CadImportError("mesh_size", "mesh_size_mm must be finite and positive")
    if unit not in (Unit.MM, Unit.CM, Unit.M):
        raise CadImportError("unit", "unit must be mm, cm, or m")
    ctype = CellType3D(element_type)
    if ctype not in (CellType3D.TET4, CellType3D.TET10):
        raise CadImportError(
            "element", f"Gmsh 3D importer supports tet4 or tet10, got {ctype.value!r}"
        )

    source_sha = _sha256_bytes(path.read_bytes())
    gmsh = _gmsh_module()
    order = 2 if ctype is CellType3D.TET10 else 1

    gmsh.initialize([], readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.add("formamech-3d-import")
        gmsh.model.occ.importShapes(str(path), highestDimOnly=False, format="step")
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(3)
        if len(volumes) != 1:
            raise CadImportError(
                "topology",
                f"3D STEP import requires exactly 1 solid volume, found {len(volumes)}",
            )
        vol_tag = int(volumes[0][1])

        boundary_faces = sorted(
            abs(int(tag))
            for dim, tag in gmsh.model.getBoundary([(3, vol_tag)], oriented=False)
            if dim == 2
        )
        if not boundary_faces:
            raise CadImportError("topology", "3D solid has no boundary faces")

        # Assign unique deterministic semantic face names
        used_names: set[str] = set()
        face_records: list[tuple[int, str, str, str]] = []
        for idx, f_tag in enumerate(boundary_faces, start=1):
            raw_label = str(gmsh.model.getEntityName(2, f_tag) or "")
            fallback = f"face-{idx:04d}"
            candidate = _sanitize_face_label(raw_label, fallback)
            if candidate in used_names:
                candidate = f"{candidate}_{idx:04d}"
            used_names.add(candidate)
            kind = _face_kind_name(str(gmsh.model.getType(2, f_tag)))
            face_records.append((f_tag, candidate, kind, raw_label))

        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMin", 0.35 * float(mesh_size_mm)
        )
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(mesh_size_mm))
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 16)
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1 if second_order_linear else 0)
        if order == 2 and not second_order_linear:
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)

        gmsh.model.addPhysicalGroup(3, [vol_tag], tag=1)
        for f_tag, fname, _, _ in face_records:
            pg = gmsh.model.addPhysicalGroup(2, [f_tag])
            gmsh.model.setPhysicalName(2, pg, fname)

        gmsh.model.mesh.generate(3)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        tag_to_idx: dict[int, int] = {}
        raw_nodes: list[tuple[float, float, float]] = []
        for i, ntag in enumerate(node_tags):
            tag_to_idx[int(ntag)] = i
            raw_nodes.append(
                (
                    float(coords[3 * i]),
                    float(coords[3 * i + 1]),
                    float(coords[3 * i + 2]),
                )
            )

        # Extract 3D tetrahedral cells (type 4 = TET4, type 11 = TET10)
        expected_gmsh_type = 11 if ctype is CellType3D.TET10 else 4
        n_per_cell = 10 if ctype is CellType3D.TET10 else 4
        e_types, _, e_nodes = gmsh.model.mesh.getElements(3, vol_tag)
        cells_raw: list[tuple[int, ...]] = []
        for etype, flat in zip(e_types, e_nodes, strict=True):
            if int(etype) != expected_gmsh_type:
                continue
            for off in range(0, len(flat), n_per_cell):
                mapped = tuple(
                    tag_to_idx[int(flat[off + k])] for k in range(n_per_cell)
                )
                if ctype is CellType3D.TET10:
                    cells_raw.append(_orient_gmsh_tet10(raw_nodes, mapped))
                else:
                    cells_raw.append(
                        _orient_tet4(
                            raw_nodes, mapped[0], mapped[1], mapped[2], mapped[3]
                        )
                    )
        if not cells_raw:
            raise CadImportError("mesh", f"3D mesher produced zero {ctype.value} cells")

        # Extract boundary surface corner triangles (type 2 for TET4, type 9 first 3 corners for TET10)
        surface_tris_raw: list[tuple[int, int, int]] = []
        face_tags_map: dict[str, list[int]] = {}
        face_infos: list[FaceInfo3D] = []

        # Compute solid centroid to orient surface triangles outward
        vol_sum = 0.0
        cx_w, cy_w, cz_w = 0.0, 0.0, 0.0
        for cell in cells_raw:
            p0, p1, p2, p3 = (
                raw_nodes[cell[0]],
                raw_nodes[cell[1]],
                raw_nodes[cell[2]],
                raw_nodes[cell[3]],
            )
            v = _tet_signed_volume(p0, p1, p2, p3)
            vol_sum += v
            cx_w += v * 0.25 * (p0[0] + p1[0] + p2[0] + p3[0])
            cy_w += v * 0.25 * (p0[1] + p1[1] + p2[1] + p3[1])
            cz_w += v * 0.25 * (p0[2] + p1[2] + p2[2] + p3[2])
        solid_center = (
            cx_w / max(vol_sum, 1.0e-12),
            cy_w / max(vol_sum, 1.0e-12),
            cz_w / max(vol_sum, 1.0e-12),
        )

        # Build map from boundary corner-tri frozenset -> owning tet opposite 4th vertex
        face_to_opp: dict[frozenset[int], int] = {}
        for cell in cells_raw:
            v0, v1, v2, v3 = cell[0], cell[1], cell[2], cell[3]
            for f_nodes, opp in (
                ((v0, v1, v2), v3),
                ((v0, v1, v3), v2),
                ((v0, v2, v3), v1),
                ((v1, v2, v3), v0),
            ):
                key = frozenset(f_nodes)
                if key in face_to_opp:
                    # Interior shared face
                    face_to_opp[key] = -1
                else:
                    face_to_opp[key] = opp

        total_surf_area = 0.0
        for f_tag, fname, fkind, flabel in face_records:
            f_etypes, _, f_enodes = gmsh.model.mesh.getElements(2, f_tag)
            tri_indices: list[int] = []
            f_area = 0.0
            fcx, fcy, fcz = 0.0, 0.0, 0.0
            fnx, fny, fnz = 0.0, 0.0, 0.0
            for ft, fflat in zip(f_etypes, f_enodes, strict=True):
                ft_int = int(ft)
                if ft_int == 2:
                    stride = 3
                elif ft_int == 9:
                    stride = 6  # P2 triangle: first 3 nodes are corners
                else:
                    continue
                for off in range(0, len(fflat), stride):
                    a = tag_to_idx[int(fflat[off])]
                    b = tag_to_idx[int(fflat[off + 1])]
                    c = tag_to_idx[int(fflat[off + 2])]
                    pa, pb, pc = raw_nodes[a], raw_nodes[b], raw_nodes[c]
                    e1 = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
                    e2 = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2])
                    nx = e1[1] * e2[2] - e1[2] * e2[1]
                    ny = e1[2] * e2[0] - e1[0] * e2[2]
                    nz = e1[0] * e2[1] - e1[1] * e2[0]
                    opp_idx = face_to_opp.get(frozenset((a, b, c)), -1)
                    ref_pt = raw_nodes[opp_idx] if opp_idx >= 0 else solid_center
                    tcx = (pa[0] + pb[0] + pc[0]) / 3.0
                    tcy = (pa[1] + pb[1] + pc[1]) / 3.0
                    tcz = (pa[2] + pb[2] + pc[2]) / 3.0
                    out_dot = (
                        nx * (tcx - ref_pt[0])
                        + ny * (tcy - ref_pt[1])
                        + nz * (tcz - ref_pt[2])
                    )
                    if out_dot < 0.0:
                        b, c = c, b
                        nx, ny, nz = -nx, -ny, -nz
                    tri_area = 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)
                    f_area += tri_area
                    fcx += tri_area * tcx
                    fcy += tri_area * tcy
                    fcz += tri_area * tcz
                    fnx += 0.5 * nx
                    fny += 0.5 * ny
                    fnz += 0.5 * nz
                    tri_indices.append(len(surface_tris_raw))
                    surface_tris_raw.append((a, b, c))
            face_tags_map[fname] = tri_indices
            total_surf_area += f_area
            n_mag = math.sqrt(fnx * fnx + fny * fny + fnz * fnz)
            avg_n = (
                (fnx / n_mag, fny / n_mag, fnz / n_mag)
                if n_mag > 1.0e-12
                else (0.0, 0.0, 1.0)
            )
            avg_c = (
                (fcx / f_area, fcy / f_area, fcz / f_area)
                if f_area > 1.0e-12
                else (0.0, 0.0, 0.0)
            )
            face_infos.append(
                FaceInfo3D(
                    name=fname,
                    kind=fkind,
                    source_id=f_tag,
                    source_label=flabel,
                    area_mm2=f_area,
                    centroid_mm=avg_c,
                    normal_mm=avg_n,
                    triangle_count=len(tri_indices),
                )
            )

        # Compact node array to nodes referenced by cells (corner vertices first!)
        corner_set = {nid for cell in cells_raw for nid in cell[:4]}
        all_used = {nid for cell in cells_raw for nid in cell}
        midside_set = all_used - corner_set
        ordered_old = sorted(corner_set) + sorted(midside_set)
        remap = {old: new for new, old in enumerate(ordered_old)}

        compact_nodes = tuple(raw_nodes[i] for i in ordered_old)
        compact_cells = tuple(tuple(remap[nid] for nid in cell) for cell in cells_raw)
        compact_tris = tuple(
            (remap[a], remap[b], remap[c]) for a, b, c in surface_tris_raw
        )

        mesh = CanonicalMesh3D(
            nodes=compact_nodes,
            cells=compact_cells,
            cell_type=ctype,
            surface_triangles=compact_tris,
            semantic_face_tags={k: tuple(v) for k, v in face_tags_map.items()},
            unit=Unit.MM,
            metadata={
                "source_sha256": source_sha,
                "mesh_size_mm": float(mesh_size_mm),
                "n_corner_nodes": len(corner_set),
                "second_order_linear": bool(second_order_linear),
            },
        )
        if poisson_ratio is not None:
            mesh.validate_incompressibility_locking(poisson_ratio)

        return ImportedCad3D(
            mesh=mesh,
            faces=tuple(face_infos),
            source_sha256=source_sha,
            source_format="step",
            volume_mm3=vol_sum,
            surface_area_mm2=total_surf_area,
            mesh_size_mm=float(mesh_size_mm),
            provenance={
                "source_file": path.name,
                "element_type": ctype.value,
                "unit": unit.value,
            },
        )
    finally:
        gmsh.finalize()


def generate_3d_box_step(
    destination: Path | str,
    *,
    lx: float = 20.0,
    ly: float = 10.0,
    lz: float = 10.0,
) -> Path:
    """Generate a 3D rectangular prism STEP solid for uniaxial patch tests."""
    dest = Path(destination).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    gmsh = _gmsh_module()
    gmsh.initialize([], readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box-3d")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, float(lx), float(ly), float(lz))
        gmsh.model.occ.synchronize()
        gmsh.write(str(dest))
    finally:
        gmsh.finalize()
    return dest


def generate_3d_l_bracket_step(
    destination: Path | str,
    *,
    arm_x: float = 36.0,
    arm_y: float = 28.0,
    thickness: float = 10.0,
    depth: float = 12.0,
    hole_radius: float = 3.2,
) -> Path:
    """Generate a 3D cantilevered L-bracket STEP solid with a through-hole."""
    dest = Path(destination).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    gmsh = _gmsh_module()
    gmsh.initialize([], readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("l-bracket-3d")
        v_arm = gmsh.model.occ.addBox(0.0, 0.0, 0.0, thickness, arm_y, depth)
        h_arm = gmsh.model.occ.addBox(0.0, 0.0, 0.0, arm_x, thickness, depth)
        fused, _ = gmsh.model.occ.fuse([(3, v_arm)], [(3, h_arm)])
        bore_x = thickness + 0.55 * (arm_x - thickness)
        bore_z = 0.5 * depth
        cyl = gmsh.model.occ.addCylinder(
            bore_x, -1.0, bore_z, 0.0, thickness + 2.0, 0.0, hole_radius
        )
        gmsh.model.occ.cut(fused, [(3, cyl)])
        gmsh.model.occ.synchronize()
        gmsh.write(str(dest))
    finally:
        gmsh.finalize()
    return dest


def generate_3d_mounting_lug_step(
    destination: Path | str,
    *,
    base_lx: float = 42.0,
    base_ly: float = 22.0,
    base_lz: float = 7.0,
    boss_height: float = 16.0,
    boss_radius: float = 9.0,
    boss_thickness: float = 10.0,
    bore_radius: float = 4.5,
    bolt_radius: float = 2.4,
) -> Path:
    """Generate a complex 3D industrial mounting lug / clevis STEP solid.

    Features a rectangular mounting flange with two symmetric vertical bolt
    holes fused to a rounded upright clevis boss with a horizontal through-bore.
    """
    dest = Path(destination).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    gmsh = _gmsh_module()
    gmsh.initialize([], readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("mounting-lug-3d")
        cx = 0.5 * base_lx
        cy = 0.5 * base_ly

        # 1. Rectangular mounting base flange
        base = gmsh.model.occ.addBox(0.0, 0.0, 0.0, base_lx, base_ly, base_lz)

        # 2. Upright rectangular neck + rounded cylindrical crown centered on flange
        neck = gmsh.model.occ.addBox(
            cx - boss_radius,
            cy - 0.5 * boss_thickness,
            base_lz,
            2.0 * boss_radius,
            boss_thickness,
            boss_height - boss_radius,
        )
        crown_z = base_lz + boss_height - boss_radius
        crown = gmsh.model.occ.addCylinder(
            cx,
            cy - 0.5 * boss_thickness,
            crown_z,
            0.0,
            boss_thickness,
            0.0,
            boss_radius,
        )
        fused, _ = gmsh.model.occ.fuse([(3, base)], [(3, neck), (3, crown)])

        # 3. Subtract horizontal clevis bore + two vertical flange bolt holes
        bore = gmsh.model.occ.addCylinder(
            cx,
            cy - boss_thickness,
            crown_z,
            0.0,
            2.0 * boss_thickness,
            0.0,
            bore_radius,
        )
        bolt_offset = 0.5 * base_lx - 5.5
        bolt_l = gmsh.model.occ.addCylinder(
            cx - bolt_offset, cy, -1.0, 0.0, 0.0, base_lz + 2.0, bolt_radius
        )
        bolt_r = gmsh.model.occ.addCylinder(
            cx + bolt_offset, cy, -1.0, 0.0, 0.0, base_lz + 2.0, bolt_radius
        )
        gmsh.model.occ.cut(fused, [(3, bore), (3, bolt_l), (3, bolt_r)])
        gmsh.model.occ.synchronize()
        gmsh.write(str(dest))
    finally:
        gmsh.finalize()
    return dest


def prepare_geometry_3d(
    source: Path | str,
    output_dir: Path | str,
    *,
    unit: Unit = Unit.MM,
    mesh_size_mm: float = 3.0,
    element_type: CellType3D = CellType3D.TET10,
    poisson_ratio: float | None = None,
) -> dict[str, Any]:
    """Import a 3D STEP solid and persist a hash-bound 3D geometry project."""
    src = Path(source).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    imported = import_cad_3d(
        src,
        unit=unit,
        mesh_size_mm=mesh_size_mm,
        element_type=element_type,
        poisson_ratio=poisson_ratio,
    )

    mesh_doc = {
        "schema": "formamech.canonical-mesh-3d-v1",
        "mesh_id": imported.mesh.mesh_id,
        "cell_type": imported.mesh.cell_type.value,
        "unit": imported.mesh.unit.value,
        "nodes": [list(p) for p in imported.mesh.nodes],
        "cells": [list(c) for c in imported.mesh.cells],
        "surface_triangles": [list(t) for t in imported.mesh.surface_triangles],
        "semantic_face_tags": {
            k: list(v) for k, v in imported.mesh.semantic_face_tags.items()
        },
        "metadata": dict(imported.mesh.metadata),
    }
    mesh_bytes = canonical_json_bytes(mesh_doc) + b"\n"
    mesh_path = out / "mesh3d.json"
    mesh_path.write_bytes(mesh_bytes)

    summary = imported.summary_document()
    summary["mesh_file"] = "mesh3d.json"
    summary["mesh_sha256"] = _sha256_bytes(mesh_bytes)
    summary["project_hash"] = _sha256_doc(summary)
    (out / "geometry3d.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    return summary


def inspect_geometry_3d(project_dir: Path | str) -> dict[str, Any]:
    """Verify SHA-256 hashes and return summary metadata for a 3D geometry project."""
    proj = Path(project_dir)
    manifest_path = proj / "geometry3d.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"geometry3d.json not found in {proj}")
    doc: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if doc.get("schema") != CAD_3D_PROJECT_SCHEMA:
        raise ValueError(f"invalid 3D geometry project schema {doc.get('schema')!r}")
    mesh_path = proj / str(doc["mesh_file"])
    if _sha256_bytes(mesh_path.read_bytes()) != doc["mesh_sha256"]:
        raise ValueError("mesh3d.json SHA-256 mismatch (tampered 3D project)")
    return doc


def load_geometry_3d(project_dir: Path | str) -> CanonicalMesh3D:
    """Verify integrity and reload ``CanonicalMesh3D`` from a prepared 3D project."""
    doc = inspect_geometry_3d(project_dir)
    mesh_path = Path(project_dir) / str(doc["mesh_file"])
    raw = json.loads(mesh_path.read_text(encoding="utf-8"))
    mesh = CanonicalMesh3D(
        nodes=tuple((float(p[0]), float(p[1]), float(p[2])) for p in raw["nodes"]),
        cells=tuple(tuple(int(i) for i in c) for c in raw["cells"]),
        cell_type=CellType3D(raw["cell_type"]),
        surface_triangles=tuple(
            (int(t[0]), int(t[1]), int(t[2])) for t in raw["surface_triangles"]
        ),
        semantic_face_tags={
            str(k): tuple(int(i) for i in v)
            for k, v in raw["semantic_face_tags"].items()
        },
        unit=Unit(raw.get("unit", "mm")),
        metadata=dict(raw.get("metadata", {})),
    )
    if mesh.mesh_id != doc["mesh_id"]:
        raise ValueError("reloaded CanonicalMesh3D mesh_id mismatch")
    return mesh


def classify_faces_by_bbox(
    imported: ImportedCad3D, *, tol_rel: float = 1.0e-3
) -> dict[str, list[str]]:
    """Group face names lying on the 6 axis-aligned bounding-box planes."""
    xs = [p[0] for p in imported.mesh.nodes]
    ys = [p[1] for p in imported.mesh.nodes]
    zs = [p[2] for p in imported.mesh.nodes]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    tol = tol_rel * max(1.0, xmax - xmin, ymax - ymin, zmax - zmin)
    groups: dict[str, list[str]] = {
        "x_min": [],
        "x_max": [],
        "y_min": [],
        "y_max": [],
        "z_min": [],
        "z_max": [],
        "cylindrical": [],
    }
    for f in imported.faces:
        if f.kind == "cylindrical":
            groups["cylindrical"].append(f.name)
        if f.kind != "planar":
            continue
        cx, cy, cz = f.centroid_mm
        nx, ny, nz = f.normal_mm
        if abs(cx - xmin) <= tol and abs(abs(nx) - 1.0) <= 0.1:
            groups["x_min"].append(f.name)
        elif abs(cx - xmax) <= tol and abs(abs(nx) - 1.0) <= 0.1:
            groups["x_max"].append(f.name)
        elif abs(cy - ymin) <= tol and abs(abs(ny) - 1.0) <= 0.1:
            groups["y_min"].append(f.name)
        elif abs(cy - ymax) <= tol and abs(abs(ny) - 1.0) <= 0.1:
            groups["y_max"].append(f.name)
        elif abs(cz - zmin) <= tol and abs(abs(nz) - 1.0) <= 0.1:
            groups["z_min"].append(f.name)
        elif abs(cz - zmax) <= tol and abs(abs(nz) - 1.0) <= 0.1:
            groups["z_max"].append(f.name)
    return groups


__all__ = (
    "BUILTIN_3D_TEMPLATES",
    "CAD_3D_PROJECT_SCHEMA",
    "FaceInfo3D",
    "ImportedCad3D",
    "classify_faces_by_bbox",
    "generate_3d_box_step",
    "generate_3d_l_bracket_step",
    "generate_3d_mounting_lug_step",
    "import_cad_3d",
    "inspect_geometry_3d",
    "load_geometry_3d",
    "prepare_geometry_3d",
)
