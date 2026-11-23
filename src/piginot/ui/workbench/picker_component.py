"""Mount the packaged Streamlit v2 Boundary Picker."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache, partial
from inspect import signature
from pathlib import Path
from typing import Any

from piginot.compiler.planning import PlanError
from piginot.ui.workbench.boundary_picker import (
    apply_picker_event,
    picker_payload,
    workspace_id,
)
from piginot.ui.workbench.cad_viewer import build_cad_boundary_figure
from piginot.ui.workbench.physics_forms import BoundaryAssignment
from piginot.ui.workbench.state import WorkbenchWorkspace


@lru_cache(maxsize=1)
def _component() -> Any:
    import streamlit as st

    register = st.components.v2.component
    # Streamlit 1.51 sets style isolation when mounting rather than registering.
    registration_options: dict[str, Any] = (
        {"isolate_styles": False}
        if "isolate_styles" in signature(register).parameters
        else {}
    )
    mount: Callable[..., Any] = register(
        "formamech.boundary_picker",
        html='<div class="react-root"></div>',
        js="index.js",
        css="index.css",
        **registration_options,
    )
    return mount if registration_options else partial(mount, isolate_styles=False)


def render_boundary_picker(st: Any, workspace: WorkbenchWorkspace) -> None:
    """Resolve Apply events on the server before rendering the next snapshot."""
    key = f"boundary_picker::{workspace_id(workspace)}"
    result_key = f"{key}::last_apply"

    def on_apply() -> None:
        event = st.session_state[key].get("apply")
        if event is None:
            return
        active_path = str(st.session_state.get("wb_workspace_path", "")).strip()
        if active_path and Path(active_path).resolve() != workspace.root:
            return
        try:
            st.session_state[result_key] = apply_picker_event(workspace, event)
        except (OSError, PlanError):
            st.session_state[result_key] = {
                "event_id": event.get("event_id"),
                "region": event.get("region"),
                "status": "invalid",
                "errors": {"form": "The study could not be updated. Try Apply again."},
            }

    try:
        payload = picker_payload(workspace, result=st.session_state.get(result_key))
    except PlanError:
        st.info(
            "The study is being updated. Reopen the picker after the operation finishes."
        )
        return
    assignments = [
        BoundaryAssignment.from_document(item) for item in payload["assignments"]
    ]
    figure = build_cad_boundary_figure(
        payload["preview"], assignments=assignments, show_mesh=False
    )
    figure.layout.template = None
    _component()(
        key=key,
        data={**payload, "figure": figure.to_plotly_json()},
        on_apply_change=on_apply,
    )
