"""Canonical workspace management for the Interactive 2D Engineering Workbench.

Every workbench workspace reads and writes the exact canonical FormaMech
artifacts (``geometry/``, ``physics.json``, ``inputs/``, ``preview/``,
``run/``, ``qualification/``, ``jobs/``) so projects authored in the UI can
be reproduced headlessly via ``formamech compile`` and ``formamech qualify``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from piginot.cad.mesh_policy import MeshPolicy
from piginot.cad.project import prepare_geometry
from piginot.contracts import Unit, canonical_json_bytes

WORKBENCH_STATE_SCHEMA: Final = "formamech.workbench-state-v1"
SUPPORTED_CAD_EXTENSIONS: Final = (".step", ".stp", ".dxf")


@dataclass(frozen=True, slots=True)
class StageProgress:
    """Boolean progress tracker across the five guided workbench stages."""

    cad_imported: bool
    physics_saved: bool
    inputs_prepared: bool
    plan_previewed: bool
    bundle_trained: bool
    bundle_qualified: bool

    def document(self) -> dict[str, bool]:
        return {
            "cad_imported": self.cad_imported,
            "physics_saved": self.physics_saved,
            "inputs_prepared": self.inputs_prepared,
            "plan_previewed": self.plan_previewed,
            "bundle_trained": self.bundle_trained,
            "bundle_qualified": self.bundle_qualified,
        }


class WorkbenchWorkspace:
    """Directory-backed project state for one FormaMech CAD modeling session."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def geometry_dir(self) -> Path:
        return self.root / "geometry"

    @property
    def physics_path(self) -> Path:
        return self.root / "physics.json"

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def preview_dir(self) -> Path:
        return self.root / "preview"

    @property
    def run_dir(self) -> Path:
        return self.root / "run"

    @property
    def qualification_dir(self) -> Path:
        return self.root / "qualification"

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def state_path(self) -> Path:
        return self.root / "workbench-state.json"

    @property
    def candidate_bundle_path(self) -> Path | None:
        candidate = self.run_dir / "candidate-bundle"
        if (candidate / "bundle.json").is_file():
            return candidate
        return None

    @property
    def qualification_manifest_path(self) -> Path | None:
        manifest = self.qualification_dir / "qualification-manifest.json"
        if manifest.is_file():
            return manifest
        return None

    def stage_progress(self) -> StageProgress:
        return StageProgress(
            cad_imported=(self.geometry_dir / "preview.json").is_file(),
            physics_saved=self.physics_path.is_file(),
            inputs_prepared=(
                (self.inputs_dir / "problem.json").is_file()
                and (self.inputs_dir / "config.json").is_file()
            ),
            plan_previewed=(self.preview_dir / "plan.json").is_file(),
            bundle_trained=self.candidate_bundle_path is not None,
            bundle_qualified=self.qualification_manifest_path is not None,
        )

    def load_preview(self) -> dict[str, Any] | None:
        preview_file = self.geometry_dir / "preview.json"
        if not preview_file.is_file():
            return None
        loaded = json.loads(preview_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("geometry/preview.json must be a JSON object")
        return loaded

    def load_plan(self) -> dict[str, Any] | None:
        plan_file = self.preview_dir / "plan.json"
        if not plan_file.is_file():
            return None
        loaded = json.loads(plan_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("preview/plan.json must be a JSON object")
        return loaded

    def save_state(self, data: Mapping[str, Any]) -> None:
        payload = {
            "schema": WORKBENCH_STATE_SCHEMA,
            **{str(k): v for k, v in data.items()},
        }
        self.state_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"schema": WORKBENCH_STATE_SCHEMA}
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {"schema": WORKBENCH_STATE_SCHEMA}
        return raw

    def import_cad_bytes(
        self,
        payload: bytes,
        filename: str,
        *,
        unit: Unit = Unit.MM,
        mesh_size_mm: float | None = 2.5,
        mesh_policy: MeshPolicy | None = None,
    ) -> dict[str, Any]:
        """Store uploaded CAD bytes, mesh via ``prepare_geometry``, and return preview."""
        safe_name = Path(filename).name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_CAD_EXTENSIONS:
            raise ValueError(
                f"unsupported CAD extension {suffix!r}; expected one of "
                f"{SUPPORTED_CAD_EXTENSIONS}"
            )
        if not payload:
            raise ValueError("CAD file payload must be non-empty")

        self.source_dir.mkdir(parents=True, exist_ok=True)
        src_path = self.source_dir / safe_name
        src_path.write_bytes(payload)

        # Clean stale downstream artifacts when re-importing geometry
        for stale_dir in (self.geometry_dir, self.inputs_dir, self.preview_dir):
            if stale_dir.exists():
                shutil.rmtree(stale_dir)

        prepare_geometry(
            src_path,
            self.geometry_dir,
            unit=unit,
            mesh_size_mm=mesh_size_mm if mesh_policy is None else None,
            mesh_policy=mesh_policy,
        )
        preview = self.load_preview()
        if preview is None:
            raise RuntimeError("prepare_geometry did not emit preview.json")
        state = self.load_state()
        state["source_filename"] = safe_name
        state["unit"] = unit.value
        state["mesh_size_mm"] = mesh_size_mm
        self.save_state(state)
        return preview

    def import_cad_path(
        self,
        source_path: Path | str,
        *,
        unit: Unit = Unit.MM,
        mesh_size_mm: float | None = 2.5,
        mesh_policy: MeshPolicy | None = None,
    ) -> dict[str, Any]:
        """Copy an existing local STEP/DXF file into the workspace and mesh it."""
        path = Path(source_path)
        if not path.is_file():
            raise FileNotFoundError(f"CAD file not found: {path}")
        return self.import_cad_bytes(
            path.read_bytes(),
            path.name,
            unit=unit,
            mesh_size_mm=mesh_size_mm,
            mesh_policy=mesh_policy,
        )

    def reset_prepared_inputs(self) -> None:
        """Clear prepared inputs and plan preview so updated physics can be re-prepared."""
        for target in (self.inputs_dir, self.preview_dir):
            if target.exists():
                shutil.rmtree(target)


def generate_sample_bracket_step(
    destination: Path | str,
    *,
    width_mm: float = 40.0,
    height_mm: float = 20.0,
    hole_radius_mm: float = 4.0,
) -> Path:
    """Generate a planar OpenCASCADE STEP plate with a central circular cutout.

    Useful for instant zero-setup workbench demonstrations and automated UI tests.
    Requires the optional ``formamech[cad]`` extra (``gmsh``).
    """
    if width_mm <= 0.0 or height_mm <= 0.0:
        raise ValueError("width_mm and height_mm must be positive")
    if hole_radius_mm <= 0.0 or 2.0 * hole_radius_mm >= min(width_mm, height_mm) * 0.85:
        raise ValueError("hole_radius_mm must fit strictly inside the plate")

    dest = Path(destination).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    gmsh = __import__("gmsh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("workbench-sample-bracket")
        plate = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, width_mm, height_mm)
        hole = gmsh.model.occ.addDisk(
            0.5 * width_mm, 0.5 * height_mm, 0.0, hole_radius_mm, hole_radius_mm
        )
        gmsh.model.occ.cut([(2, plate)], [(2, hole)])
        gmsh.model.occ.synchronize()
        gmsh.write(str(dest))
    finally:
        gmsh.finalize()
    return dest


def canonical_settings_bytes(settings_doc: Mapping[str, Any]) -> bytes:
    """Serialize a settings document deterministically for hashing and CLI parity."""
    return canonical_json_bytes(dict(settings_doc)) + b"\n"


__all__ = (
    "SUPPORTED_CAD_EXTENSIONS",
    "WORKBENCH_STATE_SCHEMA",
    "StageProgress",
    "WorkbenchWorkspace",
    "canonical_settings_bytes",
    "generate_sample_bracket_step",
)
