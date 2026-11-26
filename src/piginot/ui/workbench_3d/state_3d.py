"""CLI-reproducible 3D CAD workspace, face-BC persistence, and viewer payload contract (V3-5)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from piginot.cad.importing_3d import (
    classify_faces_by_bbox,
    generate_3d_box_step,
    import_cad_3d,
)
from piginot.contracts import MaterialSpec, Quantity, Unit, canonical_json_bytes
from piginot.fem.backend_3d import (
    DirichletBC3D,
    FemRequest3D,
    FemResult3D,
    TorchFemBackend3D,
    TractionBC3D,
)
from piginot.fem.linear_solvers import LinearSolverConfig
from piginot.geometry.mesh3d import CanonicalMesh3D, CellType3D

VIEWER_3D_PREVIEW_SCHEMA: Final = "formamech.cad-preview-3d-v1"
VALID_FACE_BC_KINDS: Final = frozenset(
    {"fixed", "dirichlet", "traction", "contact", "traction_free"}
)
VALID_DIRICHLET_COMPONENTS: Final = frozenset({"x", "y", "z"})
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{0,63}$")


@dataclass(frozen=True, slots=True)
class FaceAssignment3D:
    """One semantic boundary condition assigned to a 3D B-Rep face tag."""

    face_tag: str
    name: str
    kind: str = "traction_free"
    components: tuple[str, ...] = ()
    values_mm: tuple[float, ...] = ()
    traction_mpa: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def document(self) -> dict[str, Any]:
        return {
            "face_tag": self.face_tag,
            "name": self.name,
            "kind": self.kind,
            "components": list(self.components),
            "values_mm": [float(v) for v in self.values_mm],
            "traction_mpa": [float(v) for v in self.traction_mpa],
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> FaceAssignment3D:
        tr = doc.get("traction_mpa", [0.0, 0.0, 0.0])
        if not isinstance(tr, Sequence) or isinstance(tr, (str, bytes)) or len(tr) != 3:
            raise ValueError("traction_mpa must be a 3-element numeric vector")
        tr_t = (float(tr[0]), float(tr[1]), float(tr[2]))
        return cls(
            face_tag=str(doc["face_tag"]),
            name=str(doc["name"]),
            kind=str(doc["kind"]),
            components=tuple(str(c) for c in doc.get("components", ())),
            values_mm=tuple(float(v) for v in doc.get("values_mm", ())),
            traction_mpa=tr_t,
        )


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _surface_face_groups_from_mesh(mesh: CanonicalMesh3D) -> list[dict[str, Any]]:
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    tris = np.asarray(mesh.surface_triangles, dtype=np.int64)
    groups: list[dict[str, Any]] = []
    for tag in sorted(mesh.semantic_face_tags.keys()):
        tri_indices = list(mesh.semantic_face_tags[tag])
        area = 0.0
        centroid = np.zeros(3, dtype=np.float64)
        if tri_indices:
            sub = tris[tri_indices]
            p0 = nodes[sub[:, 0]]
            p1 = nodes[sub[:, 1]]
            p2 = nodes[sub[:, 2]]
            areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
            area = float(np.sum(areas))
            ctrs = (p0 + p1 + p2) / 3.0
            if area > 1.0e-12:
                centroid = np.sum(ctrs * areas[:, None], axis=0) / area
            else:
                centroid = np.mean(ctrs, axis=0)
        groups.append(
            {
                "face_tag": tag,
                "triangle_indices": [int(i) for i in tri_indices],
                "area_mm2": float(area),
                "centroid_mm": [
                    float(centroid[0]),
                    float(centroid[1]),
                    float(centroid[2]),
                ],
            }
        )
    return groups


def validate_face_assignments_3d(
    face_tags: Sequence[str],
    assignments: Sequence[FaceAssignment3D],
) -> tuple[bool, tuple[str, ...]]:
    """Validate 3D B-Rep face BC assignments for FEM readiness."""
    errors: list[str] = []
    tag_set = set(face_tags)
    seen_tags: set[str] = set()
    has_essential = False
    has_load = False

    for a in assignments:
        if a.face_tag not in tag_set:
            errors.append(f"unknown 3D face_tag {a.face_tag!r}")
        if a.face_tag in seen_tags:
            errors.append(f"duplicate assignment for {a.face_tag!r}")
        seen_tags.add(a.face_tag)
        if not _NAME_RE.match(a.name):
            errors.append(f"invalid BC name {a.name!r}")
        if a.kind not in VALID_FACE_BC_KINDS:
            errors.append(f"unsupported BC kind {a.kind!r}")
        if a.kind == "fixed":
            has_essential = True
        elif a.kind == "dirichlet":
            has_essential = True
            if len(a.components) != len(a.values_mm) or not a.components:
                errors.append(
                    f"dirichlet BC {a.name!r} requires matching components/values"
                )
            elif any(c not in VALID_DIRICHLET_COMPONENTS for c in a.components):
                errors.append(
                    f"dirichlet BC {a.name!r} has unsupported components {a.components!r}; use 'x', 'y', 'z'"
                )
            elif len(set(a.components)) != len(a.components):
                errors.append(
                    f"dirichlet BC {a.name!r} has duplicate components {a.components!r}"
                )
            if any(not math.isfinite(v) for v in a.values_mm):
                errors.append(f"dirichlet BC {a.name!r} has non-finite values")
            if any(abs(v) > 0.0 for v in a.values_mm):
                has_load = True
        elif a.kind == "traction":
            if any(not math.isfinite(v) for v in a.traction_mpa):
                errors.append(f"traction BC {a.name!r} has non-finite traction")
            if any(abs(v) > 0.0 for v in a.traction_mpa):
                has_load = True

    missing = sorted(tag_set - seen_tags)
    if missing:
        errors.append(f"unassigned 3D faces: {', '.join(missing)}")
    if not has_essential:
        errors.append("assign at least one fixed or dirichlet support face")
    if not has_load:
        errors.append("assign at least one non-zero displacement or traction load face")
    return len(errors) == 0, tuple(errors)


class Workbench3DWorkspace:
    """Canonical file-backed workspace for 3D solid & assembly interactive studies."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.geometry_dir = self.root / "geometry_3d"
        self.results_dir = self.root / "results_3d"
        self.state_path = self.root / "state_3d.json"
        self.preview_path = self.geometry_dir / "preview_3d.json"
        self.mesh_path = self.geometry_dir / "mesh_3d.json"

    @property
    def workspace_id(self) -> str:
        return _sha256_hex(str(self.root).encode("utf-8"))[:16]

    def _ensure_dirs(self) -> None:
        self.geometry_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def import_step_path(
        self,
        step_path: Path,
        *,
        mesh_size_mm: float = 4.0,
        element_type: CellType3D = CellType3D.TET10,
    ) -> dict[str, Any]:
        """Import a 3D STEP solid, persist canonical mesh & preview, and initialize BCs."""
        src = Path(step_path)
        if not src.is_file():
            raise FileNotFoundError(f"3D STEP file not found: {src}")
        if src.suffix.lower() not in {".step", ".stp"}:
            raise ValueError(
                f"unsupported 3D CAD extension {src.suffix!r}; use .step or .stp"
            )
        self._ensure_dirs()
        target_step = self.geometry_dir / src.name
        if src.resolve() != target_step.resolve():
            target_step.write_bytes(src.read_bytes())

        imp = import_cad_3d(
            target_step,
            mesh_size_mm=mesh_size_mm,
            element_type=element_type,
        )
        mesh = imp.mesh
        face_groups = _surface_face_groups_from_mesh(mesh)
        bbox_faces = classify_faces_by_bbox(imp)

        # Persist mesh + preview
        mesh_doc = {
            "nodes": [[float(x), float(y), float(z)] for x, y, z in mesh.nodes],
            "cells": [[int(i) for i in c] for c in mesh.cells],
            "cell_type": mesh.cell_type.value,
            "surface_triangles": [
                [int(a), int(b), int(c)] for a, b, c in mesh.surface_triangles
            ],
            "semantic_face_tags": {
                k: [int(i) for i in v] for k, v in mesh.semantic_face_tags.items()
            },
            "unit": mesh.unit.value,
        }
        self.mesh_path.write_bytes(canonical_json_bytes(mesh_doc))
        preview = {
            "schema": VIEWER_3D_PREVIEW_SCHEMA,
            "geometry_id": mesh.mesh_id,
            "step_file": target_step.name,
            "cell_type": mesh.cell_type.value,
            "node_count": len(mesh.nodes),
            "cell_count": len(mesh.cells),
            "nodes_mm": [[float(x), float(y), float(z)] for x, y, z in mesh.nodes],
            "triangles": [
                [int(a), int(b), int(c)] for a, b, c in mesh.surface_triangles
            ],
            "face_groups": face_groups,
            "bbox_classification": {k: list(v) for k, v in bbox_faces.items()},
        }
        self.preview_path.write_bytes(canonical_json_bytes(preview))

        # Auto-seed sensible default face BCs (x_min fixed, x_max pulled +0.2 mm)
        x_min_set = set(bbox_faces.get("x_min", ()))
        x_max_set = set(bbox_faces.get("x_max", ()))
        defaults: list[FaceAssignment3D] = []
        for fg in face_groups:
            tag = str(fg["face_tag"])
            if tag in x_min_set:
                defaults.append(
                    FaceAssignment3D(
                        face_tag=tag,
                        name=f"fix_{tag}",
                        kind="fixed",
                        components=("x", "y", "z"),
                        values_mm=(0.0, 0.0, 0.0),
                    )
                )
            elif tag in x_max_set:
                defaults.append(
                    FaceAssignment3D(
                        face_tag=tag,
                        name=f"pull_{tag}",
                        kind="dirichlet",
                        components=("x",),
                        values_mm=(0.2,),
                    )
                )
            else:
                defaults.append(FaceAssignment3D(face_tag=tag, name=f"free_{tag}"))

        self._save_state(
            geometry_id=mesh.mesh_id,
            assignments=defaults,
            last_event_id=None,
            last_result=None,
        )
        return preview

    def import_sample_box(
        self,
        *,
        lx: float = 20.0,
        ly: float = 8.0,
        lz: float = 8.0,
        mesh_size_mm: float = 5.0,
        element_type: CellType3D = CellType3D.TET10,
    ) -> dict[str, Any]:
        """Generate a built-in OpenCASCADE 3D box STEP and import it."""
        self._ensure_dirs()
        step_path = generate_3d_box_step(
            self.geometry_dir / "sample_box_3d.step",
            lx=lx,
            ly=ly,
            lz=lz,
        )
        return self.import_step_path(
            step_path, mesh_size_mm=mesh_size_mm, element_type=element_type
        )

    def load_preview(self) -> dict[str, Any] | None:
        if not self.preview_path.is_file():
            return None
        return dict(json.loads(self.preview_path.read_text(encoding="utf-8")))

    def load_mesh(self) -> CanonicalMesh3D | None:
        if not self.mesh_path.is_file():
            return None
        doc = json.loads(self.mesh_path.read_text(encoding="utf-8"))
        return CanonicalMesh3D(
            nodes=tuple((float(p[0]), float(p[1]), float(p[2])) for p in doc["nodes"]),
            cells=tuple(tuple(int(i) for i in c) for c in doc["cells"]),
            cell_type=CellType3D(str(doc["cell_type"])),
            surface_triangles=tuple(
                (int(t[0]), int(t[1]), int(t[2])) for t in doc["surface_triangles"]
            ),
            semantic_face_tags={
                str(k): tuple(int(i) for i in v)
                for k, v in doc["semantic_face_tags"].items()
            },
            unit=Unit(str(doc.get("unit", "mm"))),
        )

    def load_state(self) -> dict[str, Any] | None:
        if not self.state_path.is_file():
            return None
        return dict(json.loads(self.state_path.read_text(encoding="utf-8")))

    def _save_state(
        self,
        *,
        geometry_id: str,
        assignments: Sequence[FaceAssignment3D],
        last_event_id: str | None,
        last_result: Mapping[str, Any] | None,
    ) -> str:
        assign_docs = [a.document() for a in assignments]
        rev = _sha256_hex(
            canonical_json_bytes(
                {"geometry_id": geometry_id, "assignments": assign_docs}
            )
        )[:16]
        state = {
            "geometry_id": geometry_id,
            "draft_revision": rev,
            "assignments": assign_docs,
            "last_event_id": last_event_id,
            "last_result": dict(last_result) if last_result is not None else None,
        }
        self.state_path.write_bytes(canonical_json_bytes(state))
        return rev

    def viewer_payload(
        self,
        *,
        result: Mapping[str, Any] | None = None,
        displacements_mm: Sequence[Sequence[float]] | None = None,
        fields: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build the JSON payload consumed by ``frontend/viewer-3d``."""
        preview = self.load_preview()
        state = self.load_state()
        if preview is None or state is None:
            raise ValueError("3D workspace has no imported STEP geometry yet")
        assignments = [FaceAssignment3D.from_document(d) for d in state["assignments"]]
        face_tags = [str(fg["face_tag"]) for fg in preview["face_groups"]]
        _, readiness_errors = validate_face_assignments_3d(face_tags, assignments)
        payload: dict[str, Any] = {
            "workspace_id": self.workspace_id,
            "geometry_id": str(preview["geometry_id"]),
            "draft_revision": str(state["draft_revision"]),
            "nodes_mm": preview["nodes_mm"],
            "triangles": preview["triangles"],
            "face_groups": preview["face_groups"],
            "assignments": state["assignments"],
            "readiness_errors": list(readiness_errors),
            "last_apply": dict(result)
            if result is not None
            else state.get("last_result"),
        }
        if displacements_mm is not None:
            payload["displacements_mm"] = [
                [float(v[0]), float(v[1]), float(v[2])] for v in displacements_mm
            ]
        if fields is not None:
            payload["fields"] = {k: dict(v) for k, v in fields.items()}
        return payload

    def apply_face_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Apply a 3D face BC edit with optimistic-concurrency and deduplication guards."""
        preview = self.load_preview()
        state = self.load_state()
        if preview is None or state is None:
            return {
                "event_id": str(event.get("event_id", "")),
                "face_tag": str(event.get("face_tag", "")),
                "status": "invalid",
                "errors": {"form": "No 3D CAD geometry is imported."},
            }

        event_id = str(event.get("event_id", ""))
        face_tag = str(event.get("face_tag", ""))
        if (
            state.get("last_event_id") == event_id
            and state.get("last_result") is not None
        ):
            return dict(state["last_result"])

        if str(event.get("geometry_id", "")) != str(state["geometry_id"]) or str(
            event.get("draft_revision", "")
        ) != str(state["draft_revision"]):
            return {
                "event_id": event_id,
                "face_tag": face_tag,
                "status": "stale",
                "reload": True,
            }

        valid_tags = {str(fg["face_tag"]) for fg in preview["face_groups"]}
        raw_assign = event.get("assignment")
        if not isinstance(raw_assign, Mapping):
            return {
                "event_id": event_id,
                "face_tag": face_tag,
                "status": "invalid",
                "errors": {"assignment": "assignment object is required"},
            }
        errors: dict[str, str] = {}
        try:
            proposed = FaceAssignment3D.from_document(raw_assign)
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "event_id": event_id,
                "face_tag": face_tag,
                "status": "invalid",
                "errors": {"assignment": str(exc)},
            }

        if proposed.face_tag not in valid_tags or proposed.face_tag != face_tag:
            errors["face_tag"] = f"unknown or mismatched face_tag {proposed.face_tag!r}"
        if not _NAME_RE.match(proposed.name):
            errors["name"] = f"invalid BC name {proposed.name!r}"
        if proposed.kind not in VALID_FACE_BC_KINDS:
            errors["kind"] = f"unsupported BC kind {proposed.kind!r}"
        if proposed.kind == "dirichlet":
            if (
                len(proposed.components) != len(proposed.values_mm)
                or not proposed.components
            ):
                errors["values_mm"] = (
                    "dirichlet requires matching components and values"
                )
            elif any(c not in VALID_DIRICHLET_COMPONENTS for c in proposed.components):
                errors["components"] = (
                    "dirichlet components must be chosen from 'x', 'y', 'z'"
                )
            elif len(set(proposed.components)) != len(proposed.components):
                errors["components"] = (
                    "dirichlet components must not contain duplicates"
                )
            elif any(not math.isfinite(v) for v in proposed.values_mm):
                errors["values_mm"] = "dirichlet values must be finite"
        if proposed.kind == "traction" and any(
            not math.isfinite(v) for v in proposed.traction_mpa
        ):
            errors["traction_mpa"] = "traction values must be finite"

        if errors:
            return {
                "event_id": event_id,
                "face_tag": face_tag,
                "status": "invalid",
                "errors": errors,
            }

        updated: list[FaceAssignment3D] = []
        for item in state["assignments"]:
            cur = FaceAssignment3D.from_document(item)
            updated.append(proposed if cur.face_tag == proposed.face_tag else cur)

        geom_id = str(state["geometry_id"])
        new_rev = _sha256_hex(
            canonical_json_bytes(
                {
                    "geometry_id": geom_id,
                    "assignments": [a.document() for a in updated],
                }
            )
        )[:16]
        ack = {
            "event_id": event_id,
            "face_tag": face_tag,
            "status": "accepted",
            "draft_revision": new_rev,
        }
        self._save_state(
            geometry_id=geom_id,
            assignments=updated,
            last_event_id=event_id,
            last_result=ack,
        )
        return ack

    def solve_preview_fem(
        self,
        *,
        youngs_modulus_mpa: float = 100.0,
        poisson_ratio: float = 0.30,
        load_steps: int = 2,
        linear_solver: LinearSolverConfig | None = None,
    ) -> FemResult3D:
        """Run an authoritative 3D ``TorchFemBackend3D`` solve from persisted face BCs."""
        mesh = self.load_mesh()
        state = self.load_state()
        if mesh is None or state is None:
            raise ValueError("3D workspace has no imported geometry/state to solve")
        assignments = [FaceAssignment3D.from_document(d) for d in state["assignments"]]
        ok, errs = validate_face_assignments_3d(
            list(mesh.semantic_face_tags.keys()), assignments
        )
        if not ok:
            raise ValueError("3D boundary setup incomplete: " + "; ".join(errs))

        contact_faces = [a.face_tag for a in assignments if a.kind == "contact"]
        if contact_faces:
            raise ValueError(
                "3D preview FEM does not solve contact constraints "
                f"(contact faces: {', '.join(contact_faces)}); use ContactSpec3D "
                "or change those faces to fixed, dirichlet, traction, or traction_free"
            )

        dirichlet_bcs: list[DirichletBC3D] = []
        traction_bcs: list[TractionBC3D] = []
        for a in assignments:
            if a.kind == "fixed":
                dirichlet_bcs.append(
                    DirichletBC3D(
                        name=a.name,
                        face_tags=(a.face_tag,),
                        components=("x", "y", "z"),
                        values_mm=(0.0, 0.0, 0.0),
                    )
                )
            elif a.kind == "dirichlet":
                dirichlet_bcs.append(
                    DirichletBC3D(
                        name=a.name,
                        face_tags=(a.face_tag,),
                        components=a.components,
                        values_mm=a.values_mm,
                    )
                )
            elif a.kind == "traction":
                traction_bcs.append(
                    TractionBC3D(
                        name=a.name,
                        face_tags=(a.face_tag,),
                        traction_mpa=a.traction_mpa,
                    )
                )

        mat = MaterialSpec(
            name="workbench-3d-nh",
            model="compressible_neo_hookean",
            properties={
                "elastic_modulus": Quantity(
                    float(youngs_modulus_mpa), Unit.MPA
                ).to_canonical(),
                "poisson_ratio": Quantity(
                    float(poisson_ratio), Unit.ONE
                ).to_canonical(),
            },
        )
        req = FemRequest3D(
            request_id=f"wb3d-{self.workspace_id}",
            mesh=mesh,
            material=mat,
            dirichlet_bcs=tuple(dirichlet_bcs),
            traction_bcs=tuple(traction_bcs),
            load_steps=load_steps,
            linear_solver=linear_solver,
        )
        return TorchFemBackend3D().solve(req)


__all__ = (
    "VALID_FACE_BC_KINDS",
    "VIEWER_3D_PREVIEW_SCHEMA",
    "FaceAssignment3D",
    "Workbench3DWorkspace",
    "validate_face_assignments_3d",
)
