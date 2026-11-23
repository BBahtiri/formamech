"""Canonical workspace management for the Interactive 2D Engineering Workbench.

Every workbench workspace reads and writes the exact canonical FormaMech
artifacts (``geometry/``, ``physics.json``, ``inputs/``, ``preview/``,
``run/``, ``qualification/``, ``jobs/``) so projects authored in the UI can
be reproduced headlessly via ``formamech compile`` and ``formamech qualify``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from piginot.cad.mesh_policy import MeshPolicy
from piginot.cad.project import prepare_geometry
from piginot.contracts import Unit, canonical_json_bytes
from piginot.contracts.state_machine import (
    StateMachineSpec,
    StateSnapshot,
    TransitionSpec,
    register_state_machine,
)

WORKBENCH_STATE_SCHEMA: Final = "formamech.workbench-state-v1"
SUPPORTED_CAD_EXTENSIONS: Final = (".step", ".stp", ".dxf")


class WorkbenchStage(StrEnum):
    """Explicit lifecycle stage of a FormaMech 2D workbench workspace."""

    EMPTY = "EMPTY"
    CAD_IMPORTED = "CAD_IMPORTED"
    PHYSICS_SAVED = "PHYSICS_SAVED"
    INPUTS_PREPARED = "INPUTS_PREPARED"
    PLAN_PREVIEWED = "PLAN_PREVIEWED"
    BUNDLE_TRAINED = "BUNDLE_TRAINED"
    BUNDLE_QUALIFIED = "BUNDLE_QUALIFIED"


WORKBENCH_STAGE_MACHINE: Final[StateMachineSpec[WorkbenchStage]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.workbench.workspace",
            description=(
                "Guided 2D CAD workbench artifact progression from raw CAD import "
                "through physics definition, campaign planning, training, and "
                "numerical qualification."
            ),
            states=tuple(WorkbenchStage),
            initial_states=(WorkbenchStage.EMPTY,),
            terminal_states=(),
            transitions=(
                TransitionSpec(
                    WorkbenchStage.EMPTY,
                    WorkbenchStage.CAD_IMPORTED,
                    "import_cad",
                    "Import a planar STEP or DXF profile and build its canonical mesh.",
                ),
                TransitionSpec(
                    WorkbenchStage.CAD_IMPORTED,
                    WorkbenchStage.CAD_IMPORTED,
                    "import_cad",
                    "Re-import CAD geometry and invalidate stale downstream artifacts.",
                ),
                TransitionSpec(
                    WorkbenchStage.CAD_IMPORTED,
                    WorkbenchStage.PHYSICS_SAVED,
                    "save_physics",
                    "Assign boundary conditions, material model, and parameter ranges.",
                ),
                TransitionSpec(
                    WorkbenchStage.CAD_IMPORTED,
                    WorkbenchStage.PLAN_PREVIEWED,
                    "prepare_and_plan",
                    "Save physics, prepare canonical campaign inputs, and preview plan.",
                ),
                TransitionSpec(
                    WorkbenchStage.PHYSICS_SAVED,
                    WorkbenchStage.INPUTS_PREPARED,
                    "prepare_inputs",
                    "Generate canonical problem.json and config.json inputs.",
                ),
                TransitionSpec(
                    WorkbenchStage.PHYSICS_SAVED,
                    WorkbenchStage.PLAN_PREVIEWED,
                    "prepare_and_plan",
                    "Prepare inputs and generate --plan-only execution plan.",
                ),
                TransitionSpec(
                    WorkbenchStage.PHYSICS_SAVED,
                    WorkbenchStage.CAD_IMPORTED,
                    "import_cad",
                    "Re-import CAD geometry and invalidate stale physics.",
                ),
                TransitionSpec(
                    WorkbenchStage.INPUTS_PREPARED,
                    WorkbenchStage.PLAN_PREVIEWED,
                    "preview_plan",
                    "Generate preview/plan.json without running numerical work.",
                ),
                TransitionSpec(
                    WorkbenchStage.INPUTS_PREPARED,
                    WorkbenchStage.CAD_IMPORTED,
                    "import_cad",
                    "Re-import CAD geometry and invalidate prepared inputs.",
                ),
                TransitionSpec(
                    WorkbenchStage.PLAN_PREVIEWED,
                    WorkbenchStage.BUNDLE_TRAINED,
                    "compile_bundle",
                    "Run formamech compile to train and export candidate-bundle.",
                ),
                TransitionSpec(
                    WorkbenchStage.PLAN_PREVIEWED,
                    WorkbenchStage.PLAN_PREVIEWED,
                    "prepare_and_plan",
                    "Update physics settings and regenerate preview/plan.json.",
                ),
                TransitionSpec(
                    WorkbenchStage.PLAN_PREVIEWED,
                    WorkbenchStage.CAD_IMPORTED,
                    "import_cad",
                    "Re-import CAD geometry and reset downstream campaign artifacts.",
                ),
                TransitionSpec(
                    WorkbenchStage.BUNDLE_TRAINED,
                    WorkbenchStage.BUNDLE_QUALIFIED,
                    "qualify_bundle",
                    "Run formamech qualify and obtain a passing QUALIFIED manifest.",
                ),
                TransitionSpec(
                    WorkbenchStage.BUNDLE_TRAINED,
                    WorkbenchStage.PLAN_PREVIEWED,
                    "prepare_and_plan",
                    "Modify physics settings and regenerate plan for retraining.",
                ),
                TransitionSpec(
                    WorkbenchStage.BUNDLE_TRAINED,
                    WorkbenchStage.CAD_IMPORTED,
                    "import_cad",
                    "Re-import CAD geometry and reset downstream campaign artifacts.",
                ),
                TransitionSpec(
                    WorkbenchStage.BUNDLE_QUALIFIED,
                    WorkbenchStage.BUNDLE_QUALIFIED,
                    "qualify_bundle",
                    "Re-run qualification against an updated policy or sample suite.",
                ),
                TransitionSpec(
                    WorkbenchStage.BUNDLE_QUALIFIED,
                    WorkbenchStage.PLAN_PREVIEWED,
                    "prepare_and_plan",
                    "Modify physics settings and regenerate plan for retraining.",
                ),
                TransitionSpec(
                    WorkbenchStage.BUNDLE_QUALIFIED,
                    WorkbenchStage.CAD_IMPORTED,
                    "import_cad",
                    "Re-import CAD geometry and reset downstream campaign artifacts.",
                ),
            ),
        )
    )
)


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text via a sibling temporary file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_artifacts(staging: Path, changes: Mapping[Path, Path | None]) -> None:
    """Install complete staged files and restore originals if publication fails."""
    backups = Path(tempfile.mkdtemp(dir=staging.parent, prefix=".previous-"))
    moved: dict[Path, Path] = {}
    installed: list[tuple[Path, Path]] = []
    try:
        for target, replacement in changes.items():
            if target.exists():
                backup = backups / target.relative_to(staging.parent)
                backup.parent.mkdir(parents=True, exist_ok=True)
                target.replace(backup)
                moved[target] = backup
            if replacement is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                replacement.replace(target)
                installed.append((target, replacement))
    except OSError:
        try:
            for target, replacement in reversed(installed):
                target.replace(replacement)
            for target, backup in moved.items():
                backup.replace(target)
        except OSError as exc:
            raise OSError(f"Study recovery files are in {backups}.") from exc
        shutil.rmtree(backups, ignore_errors=True)
        raise
    shutil.rmtree(backups, ignore_errors=True)


def _run_cad_child(args: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = "\n".join(
            filter(None, (completed.stderr.strip(), completed.stdout.strip()))
        )
        raise RuntimeError(
            diagnostic or f"CAD process exited with status {completed.returncode}"
        )


def _state_text(data: Mapping[str, Any]) -> str:
    return (
        json.dumps({**data, "schema": WORKBENCH_STATE_SCHEMA}, indent=2, sort_keys=True)
        + "\n"
    )


@dataclass(frozen=True, slots=True)
class StageProgress:
    """Boolean progress tracker across the guided workbench stages."""

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


@dataclass(frozen=True, slots=True)
class ReplacementReview:
    revision: str
    part: str | None
    artifacts: tuple[str, ...]
    populated: bool


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

    def _qualification_verdict(self) -> str | None:
        manifest = self.qualification_manifest_path
        if manifest is None:
            return None
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "INVALID"
        if not isinstance(raw, dict):
            return "INVALID"
        verdict = raw.get("verdict")
        return None if verdict is None else str(verdict)

    def stage_progress(self) -> StageProgress:
        verdict = self._qualification_verdict()
        # Only a genuinely QUALIFIED manifest (or legacy manifest lacking a verdict key)
        # counts as qualified; an explicit NOT_QUALIFIED or corrupt manifest does not.
        is_qualified = self.qualification_manifest_path is not None and verdict in (
            "QUALIFIED",
            None,
        )
        return StageProgress(
            cad_imported=(self.geometry_dir / "preview.json").is_file(),
            physics_saved=self.physics_path.is_file()
            or "physics_draft" in self.load_state(),
            inputs_prepared=(
                (self.inputs_dir / "problem.json").is_file()
                and (self.inputs_dir / "config.json").is_file()
            ),
            plan_previewed=(self.preview_dir / "plan.json").is_file(),
            bundle_trained=self.candidate_bundle_path is not None,
            bundle_qualified=is_qualified,
        )

    def workflow_snapshot(self) -> StateSnapshot[WorkbenchStage]:
        """Return a self-describing FSM snapshot with staleness and next-action hints."""
        progress = self.stage_progress()
        state_doc = self.load_state()
        stale: list[str] = []
        blockers: list[str] = []

        # Determine highest contiguous valid stage (never skip missing prerequisites).
        if not progress.cad_imported:
            stage = WorkbenchStage.EMPTY
            if progress.physics_saved:
                stale.append("physics.json exists without imported geometry")
            if progress.inputs_prepared:
                stale.append("inputs/ exists without imported geometry")
            if progress.bundle_trained:
                stale.append("run/candidate-bundle exists without imported geometry")
            rec_trigger: str | None = "import_cad"
            rec_hint = "Import a 2D STEP or DXF profile in Step 1."
        elif not progress.physics_saved:
            stage = WorkbenchStage.CAD_IMPORTED
            if progress.inputs_prepared:
                stale.append("inputs/ exists without physics.json")
            if progress.bundle_trained:
                stale.append("run/candidate-bundle is stale relative to new geometry")
            rec_trigger = "prepare_and_plan"
            rec_hint = (
                "Assign boundaries in Step 2, then prepare inputs & plan in Step 3."
            )
        elif not progress.inputs_prepared:
            stage = WorkbenchStage.PHYSICS_SAVED
            if progress.plan_previewed:
                stale.append("preview/plan.json exists without prepared inputs/")
            if progress.bundle_trained:
                stale.append("run/candidate-bundle predates updated physics.json")
            rec_trigger = "prepare_and_plan"
            rec_hint = (
                "Prepare canonical campaign inputs and preview the plan in Step 3."
            )
        elif not progress.plan_previewed:
            stage = WorkbenchStage.INPUTS_PREPARED
            rec_trigger = "preview_plan"
            rec_hint = "Generate preview/plan.json in Step 3 before training."
        elif not progress.bundle_trained:
            stage = WorkbenchStage.PLAN_PREVIEWED
            rec_trigger = "compile_bundle"
            rec_hint = "Launch training via formamech compile in Step 4."
        elif not progress.bundle_qualified:
            stage = WorkbenchStage.BUNDLE_TRAINED
            verdict = self._qualification_verdict()
            if verdict == "NOT_QUALIFIED":
                blockers.append("latest qualification manifest returned NOT_QUALIFIED")
            elif verdict == "INVALID":
                blockers.append("qualification-manifest.json is malformed")
            rec_trigger = "qualify_bundle"
            rec_hint = (
                "Run numerical qualification in Step 4 to obtain a QUALIFIED manifest."
            )
        else:
            stage = WorkbenchStage.BUNDLE_QUALIFIED
            rec_trigger = "qualify_bundle"
            rec_hint = (
                "Inspect verified fields, Trust Receipts, or export FMU in Step 5."
            )

        # Detect artifact drift relative to recorded geometry / physics digests.
        current_geom_sha = _file_sha256(self.geometry_dir / "geometry.json")
        recorded_geom_sha = state_doc.get("geometry_sha256")
        if (
            current_geom_sha is not None
            and isinstance(recorded_geom_sha, str)
            and recorded_geom_sha != current_geom_sha
            and progress.inputs_prepared
        ):
            stale.append("prepared inputs/ may be stale relative to geometry.json")

        current_phys_sha = _file_sha256(self.physics_path)
        recorded_phys_sha = state_doc.get("physics_sha256")
        if (
            current_phys_sha is not None
            and isinstance(recorded_phys_sha, str)
            and recorded_phys_sha != current_phys_sha
            and progress.inputs_prepared
        ):
            stale.append("prepared inputs/ are stale relative to edited physics.json")

        if self.qualification_manifest_path is not None and not progress.bundle_trained:
            stale.append("qualification/ exists without a trained candidate-bundle")

        if progress.plan_previewed:
            from piginot.ui.workbench.study_plan import plan_blocker

            if message := plan_blocker(self):
                stale.append(message)
                rec_trigger = "prepare_and_plan"
                rec_hint = "Update the plan in Step 3 before training."

        return WORKBENCH_STAGE_MACHINE.snapshot(
            stage,
            blockers=blockers,
            stale_items=stale,
            recommended_trigger=rec_trigger,
            recommended_hint=rec_hint,
            metadata={
                "progress": progress.document(),
                "qualification_verdict": self._qualification_verdict(),
                "workspace_root": str(self.root),
            },
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
        _atomic_write_text(self.state_path, _state_text(data))

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"schema": WORKBENCH_STATE_SCHEMA}
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schema": WORKBENCH_STATE_SCHEMA}
        if not isinstance(raw, dict):
            return {"schema": WORKBENCH_STATE_SCHEMA}
        if raw.get("schema") not in (None, WORKBENCH_STATE_SCHEMA):
            return {"schema": WORKBENCH_STATE_SCHEMA}
        return raw

    def locked(self) -> AbstractContextManager[None]:
        from piginot.compiler.records import run_lock

        return run_lock(self.root, lock_name=".workbench.lock", resource="this study")

    def replacement_review(self) -> ReplacementReview:
        state = self.load_state()
        source = state.get("source_filename")
        has_geometry = (self.geometry_dir / "geometry.json").is_file()
        part = str(source or "current part") if has_geometry else None
        artifacts = tuple(
            label
            for directory, label in (
                (self.inputs_dir, "Prepared campaign inputs"),
                (self.preview_dir, "Plan preview"),
                (self.run_dir, "Training run and checkpoints"),
                (self.qualification_dir, "Qualification evidence"),
            )
            if directory.is_dir() and any(directory.iterdir())
        )
        if (
            self.physics_path.is_file()
            or state.get("boundaries")
            or state.get("physics_draft")
        ):
            artifacts = ("Saved physics and boundary conditions", *artifacts)
        revision = hashlib.sha256(
            canonical_json_bytes(
                {
                    "root": str(self.root),
                    "artifacts": artifacts,
                    "files": [
                        _file_sha256(path)
                        for path in (
                            self.state_path,
                            self.geometry_dir / "geometry.json",
                            self.physics_path,
                            self.preview_dir / "plan.json",
                            self.run_dir / "report.json",
                            self.qualification_dir / "qualification-manifest.json",
                        )
                    ],
                }
            )
        ).hexdigest()
        populated = bool(
            part
            or artifacts
            or any(self.jobs_dir.glob("*.json"))
            or (self.source_dir.is_dir() and any(self.source_dir.iterdir()))
        )
        return ReplacementReview(revision, part, artifacts, populated)

    def sample_workspace(self, review: ReplacementReview) -> WorkbenchWorkspace:
        if not review.populated:
            return self
        return WorkbenchWorkspace(
            tempfile.mkdtemp(
                prefix=f"{self.root.name}-sample-",
                dir=self.root.parent,
            )
        )

    def _check_replacement(self, expected: ReplacementReview | None) -> None:
        from piginot.ui.workbench.job_runner import WorkbenchJobRunner

        if expected is not None and self.replacement_review() != expected:
            raise RuntimeError(
                "The study changed. Review the current study before replacing geometry."
            )
        if WorkbenchJobRunner(self.jobs_dir, cwd=self.root).active_jobs():
            raise RuntimeError(
                "Finish or cancel the active study job before replacing geometry."
            )

    def import_cad_bytes(
        self,
        payload: bytes,
        filename: str,
        *,
        unit: Unit = Unit.MM,
        mesh_size_mm: float | None = 2.5,
        mesh_policy: MeshPolicy | None = None,
        isolated: bool = False,
        expected_study: ReplacementReview | None = None,
    ) -> dict[str, Any]:
        """Mesh uploaded CAD before replacing any files in the workspace."""
        safe_name = Path(filename).name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_CAD_EXTENSIONS:
            raise ValueError(
                f"unsupported CAD extension {suffix!r}; expected one of "
                f"{SUPPORTED_CAD_EXTENSIONS}"
            )
        if not payload:
            raise ValueError("CAD file payload must be non-empty")

        self._check_replacement(expected_study)
        with tempfile.TemporaryDirectory(
            dir=self.root, prefix=".geometry-staging-"
        ) as tmp:
            source = Path(tmp) / safe_name
            source.write_bytes(payload)
            return self._mesh_staged_cad(
                source, unit, mesh_size_mm, mesh_policy, isolated, expected_study
            )

    def import_sample(
        self, *, mesh_size_mm: float, expected_study: ReplacementReview | None = None
    ) -> dict[str, Any]:
        """Generate and mesh the sample outside the Streamlit script thread."""
        with tempfile.TemporaryDirectory(
            dir=self.root, prefix=".geometry-staging-"
        ) as tmp:
            source = Path(tmp) / "sample-bracket.step"
            _run_cad_child(
                [
                    "-c",
                    "import sys; from piginot.ui.workbench import generate_sample_bracket_step; "
                    "generate_sample_bracket_step(sys.argv[1])",
                    str(source),
                ]
            )
            return self._mesh_staged_cad(
                source, Unit.MM, mesh_size_mm, None, True, expected_study
            )

    def _mesh_staged_cad(
        self,
        source: Path,
        unit: Unit,
        mesh_size_mm: float | None,
        mesh_policy: MeshPolicy | None,
        isolated: bool,
        expected_study: ReplacementReview | None,
    ) -> dict[str, Any]:
        staging = source.parent
        geometry = staging / "geometry"
        if isolated:
            if mesh_policy is None:
                sizing = ["--mesh-size-mm", str(mesh_size_mm)]
            else:
                policy_path = staging / "mesh-policy.json"
                policy_path.write_bytes(
                    canonical_settings_bytes(mesh_policy.document())
                )
                sizing = ["--mesh-policy", str(policy_path)]
            _run_cad_child(
                [
                    "-m",
                    "formamech",
                    "cad",
                    "import",
                    str(source),
                    "--unit",
                    unit.value,
                    *sizing,
                    "--output",
                    str(geometry),
                ]
            )
        else:
            prepare_geometry(
                source,
                geometry,
                unit=unit,
                mesh_size_mm=mesh_size_mm if mesh_policy is None else None,
                mesh_policy=mesh_policy,
            )
        preview = json.loads((geometry / "preview.json").read_text(encoding="utf-8"))
        if not isinstance(preview, dict):
            raise ValueError("geometry/preview.json must be a JSON object")
        with self.locked():
            self._check_replacement(expected_study)
            self._publish_geometry(source, geometry, unit, mesh_size_mm)
        return preview

    def _publish_geometry(
        self, source: Path, geometry: Path, unit: Unit, mesh_size_mm: float | None
    ) -> None:
        staging = source.parent
        state = self.load_state()
        for key in (
            "boundaries",
            "physics_sha256",
            "physics_draft",
            "draft_revision",
            "prepared_draft_revision",
            "changed_boundary",
            "preparation_pending",
            "project_name",
            "material_model",
            "execution_profile",
            "reviewed_boundaries",
            "boundary_events",
        ):
            state.pop(key, None)
        state_file = staging / self.state_path.name
        state_file.write_text(
            _state_text(
                {
                    **state,
                    "source_filename": source.name,
                    "unit": unit.value,
                    "mesh_size_mm": mesh_size_mm,
                    "geometry_sha256": _file_sha256(geometry / "geometry.json"),
                    "geometry_revision": uuid.uuid4().hex,
                }
            ),
            encoding="utf-8",
        )
        publish_artifacts(
            staging,
            {
                self.source_dir / source.name: source,
                self.geometry_dir: geometry,
                self.inputs_dir: None,
                self.preview_dir: None,
                self.run_dir: None,
                self.qualification_dir: None,
                self.physics_path: None,
                self.state_path: state_file,
            },
        )

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
    "WORKBENCH_STAGE_MACHINE",
    "WORKBENCH_STATE_SCHEMA",
    "StageProgress",
    "WorkbenchStage",
    "WorkbenchWorkspace",
    "canonical_settings_bytes",
    "generate_sample_bracket_step",
)
