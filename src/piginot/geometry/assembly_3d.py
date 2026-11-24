"""Multi-part 3D assembly contracts, multi-material regions & interfaces (V3-4A)."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np

from piginot.contracts import MaterialSpec, canonical_json_bytes
from piginot.geometry.mesh3d import CanonicalMesh3D

ASSEMBLY_3D_SCHEMA: Final = "formamech.assembly-mesh-3d-v1"


class InterfaceKind(StrEnum):
    """Supported physical interface models between 3D assembly bodies."""

    TIED = "tied"
    FRICTIONLESS_CONTACT = "frictionless_contact"
    COULOMB_CONTACT = "coulomb_contact"
    THIRD_MEDIUM_CONTACT = "third_medium_contact"


@dataclass(frozen=True, slots=True)
class PartRegionSpec:
    """Binding of one 3D solid part's cell indices to its constitutive material."""

    part_id: str
    cell_indices: tuple[int, ...]
    material: MaterialSpec

    def __post_init__(self) -> None:
        if not isinstance(self.part_id, str) or not self.part_id.strip():
            raise ValueError("part_id must be non-empty text")
        if not self.cell_indices:
            raise ValueError(f"part {self.part_id!r} must own at least one cell")

    def document(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "cell_count": len(self.cell_indices),
            "material_name": self.material.name,
            "material_model": self.material.model,
        }


@dataclass(frozen=True, slots=True)
class InterfacePairSpec:
    """Declared master-slave interface between two 3D assembly face tags."""

    name: str
    kind: InterfaceKind
    master_face_tag: str
    slave_face_tag: str
    penalty_mpa_per_mm: float = 1.0e3
    friction_mu: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("interface name must be non-empty text")
        object.__setattr__(self, "kind", InterfaceKind(self.kind))
        if not math.isfinite(self.penalty_mpa_per_mm) or self.penalty_mpa_per_mm <= 0.0:
            raise ValueError("penalty_mpa_per_mm must be finite and positive")
        if not math.isfinite(self.friction_mu) or self.friction_mu < 0.0:
            raise ValueError("friction_mu must be finite and nonnegative")

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "master_face_tag": self.master_face_tag,
            "slave_face_tag": self.slave_face_tag,
            "penalty_mpa_per_mm": float(self.penalty_mpa_per_mm),
            "friction_mu": float(self.friction_mu),
        }


@dataclass(frozen=True, slots=True)
class AssemblyMesh3D:
    """Content-addressed multi-part, multi-material 3D assembly mesh."""

    mesh: CanonicalMesh3D
    parts: tuple[PartRegionSpec, ...]
    interfaces: tuple[InterfacePairSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("AssemblyMesh3D requires at least one PartRegionSpec")
        n_cells = len(self.mesh.cells)
        assigned: set[int] = set()
        seen_ids: set[str] = set()
        for part in self.parts:
            if part.part_id in seen_ids:
                raise ValueError(f"duplicate part_id {part.part_id!r}")
            seen_ids.add(part.part_id)
            nu = part.material.properties.get("poisson_ratio")
            if nu is not None:
                self.mesh.validate_incompressibility_locking(float(nu.value))
            for cid in part.cell_indices:
                if cid < 0 or cid >= n_cells:
                    raise ValueError(
                        f"part {part.part_id!r} cell index {cid} out of range"
                    )
                if cid in assigned:
                    raise ValueError(f"cell {cid} assigned to multiple parts")
                assigned.add(cid)
        if len(assigned) != n_cells:
            raise ValueError(
                f"every assembly cell must belong to a part ({len(assigned)}/{n_cells} assigned)"
            )
        for iface in self.interfaces:
            for tag in (iface.master_face_tag, iface.slave_face_tag):
                if tag not in self.mesh.semantic_face_tags:
                    raise ValueError(
                        f"interface {iface.name!r} references missing face tag {tag!r}"
                    )

    @property
    def assembly_id(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()

    def document(self) -> dict[str, Any]:
        return {
            "schema": ASSEMBLY_3D_SCHEMA,
            "mesh_id": self.mesh.mesh_id,
            "part_count": len(self.parts),
            "parts": [p.document() for p in self.parts],
            "interfaces": [i.document() for i in self.interfaces],
        }


def merge_3d_parts(
    named_parts: Sequence[tuple[str, CanonicalMesh3D, MaterialSpec]],
    *,
    interfaces: Sequence[InterfacePairSpec] = (),
    merge_coincident_tolerance_mm: float | None = None,
) -> AssemblyMesh3D:
    """Merge multiple ``CanonicalMesh3D`` solids into one multi-part ``AssemblyMesh3D``.

    When ``merge_coincident_tolerance_mm`` is provided (conformal bonded assembly),
    coincident interface nodes within tolerance are welded automatically; when
    ``None`` (contacting / non-conformal tied assembly), each part keeps disjoint
    nodal DOFs coupled via ``interfaces``.
    """
    if not named_parts:
        raise ValueError("named_parts must be non-empty")
    ctype = named_parts[0][1].cell_type
    unit = named_parts[0][1].unit

    all_nodes: list[tuple[float, float, float]] = []
    all_cells: list[tuple[int, ...]] = []
    all_tris: list[tuple[int, int, int]] = []
    all_tags: dict[str, list[int]] = {}
    part_specs: list[PartRegionSpec] = []

    coord_map: dict[tuple[int, int, int], int] = {}
    tol = merge_coincident_tolerance_mm

    for part_id, part_mesh, material in named_parts:
        if part_mesh.cell_type is not ctype:
            raise ValueError(
                f"assembly requires uniform cell_type ({ctype.value}), "
                f"got {part_mesh.cell_type.value} in {part_id!r}"
            )
        local_to_global: list[int] = []
        for pt in part_mesh.nodes:
            if tol is not None and tol > 0.0:
                key = (
                    round(pt[0] / tol),
                    round(pt[1] / tol),
                    round(pt[2] / tol),
                )
                existing = coord_map.get(key)
                if existing is not None:
                    local_to_global.append(existing)
                    continue
                new_idx = len(all_nodes)
                all_nodes.append(pt)
                coord_map[key] = new_idx
                local_to_global.append(new_idx)
            else:
                local_to_global.append(len(all_nodes))
                all_nodes.append(pt)

        c_start = len(all_cells)
        for cell in part_mesh.cells:
            all_cells.append(tuple(local_to_global[i] for i in cell))
        c_end = len(all_cells)
        part_specs.append(
            PartRegionSpec(
                part_id=part_id,
                cell_indices=tuple(range(c_start, c_end)),
                material=material,
            )
        )

        t_start = len(all_tris)
        for a, b, c in part_mesh.surface_triangles:
            all_tris.append(
                (local_to_global[a], local_to_global[b], local_to_global[c])
            )
        for tag, tri_ids in part_mesh.semantic_face_tags.items():
            prefixed = f"{part_id}:{tag}"
            all_tags[prefixed] = [t_start + tid for tid in tri_ids]

    merged_mesh = CanonicalMesh3D(
        nodes=tuple(all_nodes),
        cells=tuple(all_cells),
        cell_type=ctype,
        surface_triangles=tuple(all_tris),
        semantic_face_tags={k: tuple(v) for k, v in all_tags.items()},
        unit=unit,
        metadata={"assembly_parts": [p[0] for p in named_parts]},
    )
    return AssemblyMesh3D(
        mesh=merged_mesh,
        parts=tuple(part_specs),
        interfaces=tuple(interfaces),
    )


def generate_3d_bimaterial_assembly_step(
    destination: Path | str,
    *,
    lx_each: float = 10.0,
    ly: float = 6.0,
    lz: float = 6.0,
    gap_mm: float = 0.0,
) -> Path:
    """Generate a two-solid 3D STEP assembly (bonded when gap=0, contacting when gap>0)."""
    import gmsh  # type: ignore[import-untyped]

    dest = Path(destination).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("bimaterial-assembly-3d")
        b1 = gmsh.model.occ.addBox(0.0, 0.0, 0.0, float(lx_each), float(ly), float(lz))
        b2 = gmsh.model.occ.addBox(
            float(lx_each) + float(gap_mm),
            0.0,
            0.0,
            float(lx_each),
            float(ly),
            float(lz),
        )
        if abs(gap_mm) <= 1.0e-12:
            gmsh.model.occ.fragment([(3, b1)], [(3, b2)])
        gmsh.model.occ.synchronize()
        gmsh.write(str(dest))
    finally:
        gmsh.finalize()
    return dest


def classify_assembly_faces_by_bbox(
    assembly: AssemblyMesh3D, *, tol: float = 1.0e-2
) -> Mapping[str, tuple[str, ...]]:
    """Group assembly semantic face tags lying on global bounding-box extremes."""
    nodes = np.asarray(assembly.mesh.nodes, dtype=np.float64)
    lo = nodes.min(axis=0)
    hi = nodes.max(axis=0)
    groups: dict[str, list[str]] = {
        "x_min": [],
        "x_max": [],
        "y_min": [],
        "y_max": [],
        "z_min": [],
        "z_max": [],
    }
    for tag, tids in assembly.mesh.semantic_face_tags.items():
        if not tids:
            continue
        pts = np.array(
            [
                assembly.mesh.nodes[i]
                for tid in tids
                for i in assembly.mesh.surface_triangles[tid]
            ],
            dtype=np.float64,
        )
        if float(np.max(np.abs(pts[:, 0] - lo[0]))) <= tol:
            groups["x_min"].append(tag)
        if float(np.max(np.abs(pts[:, 0] - hi[0]))) <= tol:
            groups["x_max"].append(tag)
        if float(np.max(np.abs(pts[:, 1] - lo[1]))) <= tol:
            groups["y_min"].append(tag)
        if float(np.max(np.abs(pts[:, 1] - hi[1]))) <= tol:
            groups["y_max"].append(tag)
        if float(np.max(np.abs(pts[:, 2] - lo[2]))) <= tol:
            groups["z_min"].append(tag)
        if float(np.max(np.abs(pts[:, 2] - hi[2]))) <= tol:
            groups["z_max"].append(tag)
    return {k: tuple(sorted(v)) for k, v in groups.items()}


__all__ = (
    "ASSEMBLY_3D_SCHEMA",
    "AssemblyMesh3D",
    "InterfaceKind",
    "InterfacePairSpec",
    "PartRegionSpec",
    "classify_assembly_faces_by_bbox",
    "generate_3d_bimaterial_assembly_step",
    "merge_3d_parts",
)
