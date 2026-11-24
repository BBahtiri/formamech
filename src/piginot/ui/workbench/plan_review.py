"""Read-only engineering comparison with the last prepared physics settings."""

from __future__ import annotations

from math import isfinite
from typing import Any

from piginot.cad.settings import load_settings
from piginot.contracts import Quantity, Unit
from piginot.ui.workbench.physics_forms import (
    BoundaryAssignment,
    WorkbenchPhysicsDraft,
    physics_draft_from_settings,
)
from piginot.ui.workbench.state import WorkbenchWorkspace


def prepared_physics(workspace: WorkbenchWorkspace) -> WorkbenchPhysicsDraft | None:
    try:
        settings = load_settings(workspace.physics_path.read_bytes())
    except FileNotFoundError:
        return None
    return physics_draft_from_settings(settings)


def _quantity(value: float, unit: str) -> tuple[float, str]:
    canonical = Quantity(value, Unit(unit)).to_canonical()
    return canonical.value, canonical.unit.value


def _expression_value(expression: str, unit: str) -> tuple[float | str, str]:
    try:
        number = float(expression)
    except ValueError:
        return expression.strip(), unit
    return _quantity(number, unit) if isfinite(number) else (expression, unit)


def _boundary_value(boundary: BoundaryAssignment) -> tuple[Any, ...]:
    components = tuple(sorted(boundary.components))
    values = dict(zip(boundary.components, boundary.values, strict=False))
    return (
        boundary.kind,
        components,
        tuple(_expression_value(*values[c]) for c in components if c in values),
        boundary.name,
    )


def _settings(draft: WorkbenchPhysicsDraft) -> dict[str, dict[str, Any]]:
    return {
        "Boundary conditions": {b.region: _boundary_value(b) for b in draft.boundaries},
        "Material and loads": {
            "Campaign / model name": draft.project_name,
            "Material model": draft.material_model,
            "Material parameter mapping": tuple(
                sorted(draft.material_param_map.items())
            ),
            **{
                f"Parameter {p.name}": (
                    p.role,
                    _quantity(p.lo, p.unit),
                    _quantity(p.hi, p.unit),
                )
                for p in draft.parameters
            },
        },
        "Training budget": {
            "Optimizer steps": draft.optimizer_steps,
            "Case batch size": draft.case_batch_size,
            "TRAIN-fit cases": draft.fit_cases,
            "TRAIN-screen cases": draft.screen_cases,
            "Checkpoint interval": draft.checkpoint_interval,
            "Random seed": draft.seed,
            "Learning rate": draft.learning_rate,
            "Final learning rate": draft.final_learning_rate,
            "Constant steps": draft.constant_steps,
        },
        "Tolerances": {
            "Displacement relative L2": draft.u_tol,
            "Stress relative L2": draft.p_tol,
            "Absolute peak error": draft.peak_tol,
            "Quantile": draft.quantile,
        },
    }


def _format(value: Any) -> str:
    return str(value).removesuffix(".0") if isinstance(value, float) else str(value)


def _display(area: str, setting: str, value: Any) -> str:
    if value is None:
        return "—"
    if area == "Boundary conditions":
        kind, components, values, name = value
        details = (
            ", ".join(
                f"{component} = {_format(number)} {unit}"
                for component, (number, unit) in zip(components, values, strict=False)
            )
            if values
            else ", ".join(components)
        )
        label = kind.replace("_", "-")
        return f"{label} · {details} ({name})" if details else f"{label} ({name})"
    if setting.startswith("Parameter "):
        role, (lo, unit), (hi, _) = value
        return f"{role}: {_format(lo)} to {_format(hi)} {unit}"
    if setting == "Material parameter mapping":
        return ", ".join(f"{prop} = {name}" for prop, name in value)
    return _format(value)


def plan_review_rows(
    prepared: WorkbenchPhysicsDraft | None, proposed: WorkbenchPhysicsDraft
) -> list[dict[str, str]]:
    previous = _settings(prepared) if prepared is not None else {}
    rows: list[dict[str, str]] = []
    for area, current in _settings(proposed).items():
        before = previous.get(area, {})
        for setting in dict.fromkeys((*before, *current)):
            old, new = before.get(setting), current.get(setting)
            if old != new:
                row = {"Area": area, "Setting": setting}
                if prepared is not None:
                    row["Prepared"] = _display(area, setting, old)
                rows.append({**row, "Proposed": _display(area, setting, new)})
    return rows
