"""Mount the packaged Streamlit v2 3D WebGL Viewer & Face Picker component (V3-5)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache, partial
from inspect import signature
from typing import Any

from piginot.ui.workbench_3d.state_3d import Workbench3DWorkspace


@lru_cache(maxsize=1)
def _viewer_3d_component() -> Any:
    import streamlit as st

    register = st.components.v2.component
    registration_options: dict[str, Any] = (
        {"isolate_styles": False}
        if "isolate_styles" in signature(register).parameters
        else {}
    )
    mount: Callable[..., Any] = register(
        "formamech.viewer_3d",
        html='<div class="react-root"></div>',
        js="index.js",
        css="index.css",
        **registration_options,
    )
    return mount if registration_options else partial(mount, isolate_styles=False)


def render_viewer_3d(
    st: Any,
    workspace: Workbench3DWorkspace,
    *,
    displacements_mm: Sequence[Sequence[float]] | None = None,
    fields: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Render the interactive 3D WebGL viewer and process face-BC apply events."""
    key = f"viewer_3d::{workspace.workspace_id}"
    result_key = f"{key}::last_apply"

    def on_apply() -> None:
        event = st.session_state[key].get("apply")
        if event is None:
            return
        st.session_state[result_key] = workspace.apply_face_event(event)

    payload = workspace.viewer_payload(
        result=st.session_state.get(result_key),
        displacements_mm=displacements_mm,
        fields=fields,
    )
    _viewer_3d_component()(
        key=key,
        data=payload,
        on_apply_change=on_apply,
    )


__all__ = ("render_viewer_3d",)
