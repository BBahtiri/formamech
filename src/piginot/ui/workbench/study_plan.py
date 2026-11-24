"""Verify the link from applied study settings to prepared inputs and a plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from piginot.cad.settings import load_settings
from piginot.compiler.execution_records import digest, load_bound
from piginot.compiler.planning import build_plan
from piginot.ui.workbench.state import WorkbenchWorkspace, _file_sha256

_UPDATE = "Update plan in Material & Plan before training."


def _bound_inputs(
    workspace: WorkbenchWorkspace, plan: dict[str, Any], receipt: dict[str, Any]
) -> bool:
    inputs = workspace.inputs_dir
    geometry = workspace.geometry_dir
    resource = json.loads((geometry / "geometry.json").read_bytes())
    paths = {
        "problem": inputs / "problem.json",
        "config": inputs / "config.json",
        "manifest": inputs / "source.json",
        "geometry-resource": geometry / "geometry.json",
        "geometry-source": geometry / resource["source"]["path"],
        **{
            f"geometry-mesh-{level}": geometry / doc["path"]
            for level, doc in resource["meshes"].items()
        },
    }
    hashes = {
        "problem": receipt["input_sha256"]["problem.json"],
        "config": receipt["input_sha256"]["config.json"],
        "manifest": receipt["input_sha256"]["source.json"],
        **receipt["geometry_input_sha256"],
    }
    return (
        all(
            path.parent == geometry
            for key, path in paths.items()
            if key.startswith("geometry-")
        )
        and plan["input_paths"] == [str(path) for path in paths.values()]
        and plan["input_sha256"] == hashes
        and all(_file_sha256(path) == hashes[key] for key, path in paths.items())
    )


def plan_blocker(workspace: WorkbenchWorkspace) -> str | None:
    """Return an actionable reason when the compile inputs cannot be trusted."""
    try:
        plan = workspace.load_plan()
        if not plan:
            return (
                "Prepare inputs and preview a plan in Material & Plan before training."
            )
        state = workspace.load_state()
        raw = workspace.physics_path.read_bytes()
        settings = load_settings(raw)
        applied = state.get("physics_draft")
        legacy_boundaries = state.get("boundaries", settings["boundary_conditions"])
        changed = (
            applied != settings
            if applied is not None
            else legacy_boundaries != settings["boundary_conditions"]
        )
        if changed or state.get("draft_revision") != state.get(
            "prepared_draft_revision"
        ):
            edge = state.get("changed_boundary")
            message = (
                f"Boundary conditions changed ({edge})."
                if edge
                else "The saved draft has changed."
            )
            return f"{message} {_UPDATE}"
        if state.get("preparation_pending"):
            return f"The last preparation did not complete. {_UPDATE}"
        receipt = json.loads((workspace.inputs_dir / "preparation.json").read_bytes())
        if (
            receipt["settings_sha256"] == hashlib.sha256(raw).hexdigest()
            and receipt["geometry_resource_sha256"]
            == receipt["geometry_input_sha256"]["geometry-resource"]
            and plan["output_path"] == str(workspace.preview_dir)
            and plan["status"] == "PLANNED"
            and plan["plan_sha256"]
            == digest({k: v for k, v in plan.items() if k != "plan_sha256"})
            and _bound_inputs(workspace, plan, receipt)
        ):
            return None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return f"The prepared files no longer match this study. {_UPDATE}"


def has_compatible_checkpoint(
    workspace: WorkbenchWorkspace, output: Path | None = None
) -> bool:
    """Check the current run binding and the checkpoint roster before dispatch."""
    output = output or workspace.run_dir
    try:
        checkpoints = sorted((output / "checkpoints").iterdir())
        if not checkpoints:
            return False
        plan = build_plan(
            workspace.inputs_dir / "problem.json",
            workspace.inputs_dir / "config.json",
            output,
        )
        if json.loads((output / "plan.json").read_bytes()) != plan.document():
            return False
        prepared = load_bound(output / "stage-train-preparation.json", plan.plan_sha256)
        if not prepared or prepared["status"] != "PASS":
            return False
        training = plan.config.training
        steps = (
            *range(
                training.checkpoint_interval,
                training.optimizer_steps,
                training.checkpoint_interval,
            ),
            training.optimizer_steps,
        )
        names = tuple(f"step-{step:09d}" for step in steps)
        if tuple(path.name for path in checkpoints) != names[: len(checkpoints)]:
            return False
        return all(
            _checkpoint_matches(path, step, prepared["run_hash"])
            for path, step in zip(checkpoints, steps, strict=False)
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _checkpoint_matches(path: Path, step: int, run_hash: str) -> bool:
    from piginot.training._checkpoint_codec import _MANIFEST_FILENAME
    from piginot.training.checkpoint import (
        TrainingCheckpointManifest,
        _read_exact_checkpoint_files,
        _validate_file_hashes,
    )

    payloads = _read_exact_checkpoint_files(path)
    manifest = TrainingCheckpointManifest.from_bytes(payloads[_MANIFEST_FILENAME])
    _validate_file_hashes(manifest, payloads)
    return manifest.run_hash == run_hash and manifest.step == step
