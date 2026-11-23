"""Revisioned Apply contract for the workbench Boundary Picker."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from piginot.compiler.execution_records import digest
from piginot.contracts import Unit
from piginot.contracts.problem_spec import BoundaryCondition, Expression, Value
from piginot.ui.workbench.physics_forms import (
    _IDENT_RE,
    BoundaryAssignment,
    boundary_draft_update,
    load_applied_settings,
    validate_boundaries_against_preview,
)
from piginot.ui.workbench.state import WorkbenchWorkspace, _file_sha256


def workspace_id(workspace: WorkbenchWorkspace) -> str:
    return digest({"workspace": str(workspace.root)})


def _geometry_id(workspace: WorkbenchWorkspace, state: Mapping[str, Any]) -> str:
    return digest(
        {
            "geometry": _file_sha256(workspace.geometry_dir / "geometry.json"),
            "version": state.get("geometry_revision"),
        }
    )


def _revision(state: Mapping[str, Any], settings: Mapping[str, Any]) -> str:
    return digest({"settings": settings, "version": state.get("draft_revision")})


def picker_payload(
    workspace: WorkbenchWorkspace,
    *,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with workspace.locked():
        preview = workspace.load_preview()
        if preview is None:
            raise ValueError("Import a CAD profile to open the Boundary Picker.")
        state = workspace.load_state()
        settings = load_applied_settings(workspace, preview)
        geometry = _geometry_id(workspace, state)
        prepared = workspace.physics_path.is_file()
    assignments = tuple(
        BoundaryAssignment.from_document(item)
        for item in settings["boundary_conditions"]
    )
    _, errors = validate_boundaries_against_preview(preview, assignments)
    reviewed = set(state.get("reviewed_boundaries", ()))
    suggested = (
        []
        if prepared or "boundaries" in state
        else [item.region for item in assignments if item.region not in reviewed]
    )
    return {
        "workspace_id": workspace_id(workspace),
        "geometry_id": geometry,
        "draft_revision": _revision(state, settings),
        "preview": preview,
        "assignments": [item.document() for item in assignments],
        "suggested_regions": suggested,
        "readiness_errors": errors,
        "last_apply": result,
    }


def _value(raw: object, component: str, settings: Mapping[str, Any]) -> Value:
    if (
        not isinstance(raw, dict)
        or set(raw) != {"unit", "expression"}
        or raw["unit"] != "mm"
    ):
        raise ValueError("Enter a displacement expression in mm.")
    expression = Expression.parse(raw["expression"], f"u_{component}")
    dimensions = {
        item["name"]: Unit(item["unit"]).dimension
        for item in (*settings["parameters"], *settings.get("constants", ()))
    }
    dimensions.update(
        {
            name: Unit.MM.dimension
            for name in (
                "cad_origin_x",
                "cad_origin_y",
                "cad_extent_x",
                "cad_extent_y",
            )
        }
    )
    expression.check(dimensions, Unit.MM.dimension, f"u_{component}")
    if not expression.names:
        try:
            number = expression.constant
        except ZeroDivisionError as exc:
            raise ValueError("The expression divides by zero.") from exc
        if number is None or not -float("inf") < number < float("inf"):
            raise ValueError("The displacement must be finite.")
    return Value(expression=expression, unit=Unit.MM)


def _validate_proposal(
    raw: dict[str, Any], region: str, boundary_kind: str, settings: Mapping[str, Any]
) -> dict[str, str]:
    name, kind = raw.get("name"), raw.get("kind")
    if not isinstance(name, str) or not _IDENT_RE.fullmatch(name):
        return {
            "name": "Use a condition name with letters, digits, or underscores, starting with a letter or underscore."
        }
    if any(
        b["name"] == name and b["region"] != region
        for b in settings["boundary_conditions"]
    ):
        return {"name": "Another boundary already uses this condition name."}
    if kind not in ("dirichlet", "symmetry", "traction_free"):
        return {"kind": "Choose a supported boundary condition."}
    if kind != "traction_free" and boundary_kind not in ("line", "arc"):
        return {"kind": "A prescribed condition requires a line or circular arc."}
    components, raw_values = raw.get("components", []), raw.get("values", [])
    if (
        not isinstance(components, list)
        or any(component not in ("x", "y") for component in components)
        or len(components) != len(set(components))
    ):
        return {"components": "Choose each displacement component at most once."}
    if not isinstance(raw_values, list) or (
        kind == "dirichlet" and len(components) != len(raw_values)
    ):
        return {"components": "Enter a displacement for each prescribed component."}
    values = []
    errors = {}
    for component, raw_value in zip(components, raw_values, strict=False):
        try:
            values.append(_value(raw_value, component, settings))
        except (ValueError, RecursionError) as exc:
            errors[component] = str(exc)
    if errors:
        return errors
    if kind != "dirichlet" and raw_values:
        return {"components": "Only a prescribed displacement takes expressions."}
    try:
        BoundaryCondition(name, region, kind, tuple(components), tuple(values))
    except ValueError as exc:
        return {"components": str(exc)}
    return {}


def _stale(event_id: str, region: Any) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "region": region,
        "status": "stale",
        "reload": True,
        "message": "This study changed in another session. Reload the accepted conditions before applying your draft.",
    }


def apply_picker_event(workspace: WorkbenchWorkspace, event: object) -> dict[str, Any]:
    """Validate and atomically acknowledge one Apply for the active server-side study."""
    event_id = event.get("event_id") if isinstance(event, dict) else None
    if not isinstance(event_id, str) or not re.fullmatch(
        r"[a-zA-Z0-9_-]{1,128}", event_id
    ):
        return {"status": "invalid", "errors": {"form": "Invalid Apply request."}}
    assert isinstance(event, dict)
    region = event.get("region")
    envelope = {
        "event_id": event_id,
        "region": region if isinstance(region, str) else None,
    }
    with workspace.locked():
        preview = workspace.load_preview()
        state = workspace.load_state()
        if (
            preview is None
            or event.get("workspace_id") != workspace_id(workspace)
            or event.get("geometry_id") != _geometry_id(workspace, state)
        ):
            return _stale(event_id, envelope["region"])
        accepted = state.get("boundary_events", {}).get(event_id)
        if isinstance(accepted, dict):
            return accepted
        settings = load_applied_settings(workspace, preview)
        if event.get("draft_revision") != _revision(state, settings):
            return _stale(event_id, envelope["region"])
        boundary = next((b for b in preview["boundaries"] if b["name"] == region), None)
        raw = event.get("assignment")
        if boundary is None or not isinstance(raw, dict) or raw.get("region") != region:
            return {
                **envelope,
                "status": "invalid",
                "errors": {"region": "Select a boundary in the current geometry."},
            }
        assert isinstance(region, str)
        if errors := _validate_proposal(raw, region, boundary["kind"], settings):
            return {**envelope, "status": "invalid", "errors": errors}
        assignment = BoundaryAssignment.from_document(raw)
        updated = boundary_draft_update(state, settings, assignment)
        accepted = {
            **envelope,
            "status": "accepted",
            "draft_revision": _revision(
                updated, updated.get("physics_draft", settings)
            ),
            "message": f"Applied boundary condition to {region}.",
        }
        workspace.save_state(
            {
                **updated,
                "reviewed_boundaries": sorted(
                    {*state.get("reviewed_boundaries", ()), region}
                ),
                "boundary_events": {
                    **state.get("boundary_events", {}),
                    event_id: accepted,
                },
            }
        )
        return accepted
