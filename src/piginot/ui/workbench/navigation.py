"""Stage navigation and resumption for the mounted Streamlit workbench."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

from piginot.contracts.state_machine import StateSnapshot
from piginot.ui.workbench.state import WorkbenchStage, WorkbenchWorkspace


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    anchor: str
    milestones: tuple[tuple[str, str], ...] = ()


IMPORT = Stage("Import CAD", "import-cad", (("CAD & mesh imported", "cad_imported"),))
BOUNDARIES = Stage(
    "Boundary Picker",
    "boundary-picker",
    (("Physics & boundaries saved", "physics_saved"),),
)
PLAN = Stage(
    "Material & Plan",
    "material-and-plan",
    (
        ("Campaign inputs prepared", "inputs_prepared"),
        ("Plan previewed", "plan_previewed"),
    ),
)
TRAIN = Stage(
    "Train & Qualify",
    "train-and-qualify",
    (
        ("Candidate bundle trained", "bundle_trained"),
        ("Model Qualified", "bundle_qualified"),
    ),
)
INSPECT = Stage("Trust Inspector", "trust-inspector")
STAGES = (IMPORT, BOUNDARIES, PLAN, TRAIN, INSPECT)
STAGE_KEY = "wb-stage"
FOCUS_KEY = "wb-stage-focus"


def next_action(snapshot: StateSnapshot[WorkbenchStage]) -> tuple[Stage, str, str]:
    if (
        snapshot.recommended_trigger == "prepare_and_plan"
        and snapshot.metadata["progress"]["plan_previewed"]
    ):
        return PLAN, "Update plan", ".st-key-wb-prepare-control button:not([disabled])"
    actions = {
        WorkbenchStage.EMPTY: (IMPORT, "Import CAD", ""),
        WorkbenchStage.CAD_IMPORTED: (BOUNDARIES, "Review boundaries", ""),
        WorkbenchStage.PHYSICS_SAVED: (
            PLAN,
            "Prepare plan",
            ".st-key-wb-prepare-control button:not([disabled])",
        ),
        WorkbenchStage.INPUTS_PREPARED: (
            PLAN,
            "Preview plan",
            ".st-key-wb-prepare-control button:not([disabled])",
        ),
        WorkbenchStage.PLAN_PREVIEWED: (
            TRAIN,
            "Start compile",
            ".st-key-wb-start-compile button:not([disabled])",
        ),
        WorkbenchStage.BUNDLE_TRAINED: (
            TRAIN,
            "Review qualification",
            "#numerical-qualification",
        ),
        WorkbenchStage.BUNDLE_QUALIFIED: (
            INSPECT,
            "Inspect qualification",
            "#deployment-evidence",
        ),
    }
    return actions[snapshot.state]


def render_progress(st: Any, snapshot: StateSnapshot[WorkbenchStage]) -> None:
    st.subheader("Study progress")
    progress = snapshot.metadata["progress"]
    for stage in STAGES:
        st.markdown(f"**{stage.name}**")
        if stage.milestones:
            st.markdown(
                "\n".join(
                    f"- {label} · {'Complete' if progress[key] else 'Pending'}"
                    for label, key in stage.milestones
                )
            )


def render_navigation(
    st: Any, workspace: WorkbenchWorkspace, snapshot: StateSnapshot[WorkbenchStage]
) -> Stage:
    state = st.session_state
    if state.get("wb-navigation-workspace") != str(workspace.root):
        state["wb-navigation-workspace"] = str(workspace.root)
        state[STAGE_KEY] = IMPORT.name

    def navigate(stage: Stage, control: str = "") -> None:
        state[STAGE_KEY] = stage.name
        state[FOCUS_KEY] = {"anchor": stage.anchor, "control": control}

    def select_stage() -> None:
        navigate(next(stage for stage in STAGES if stage.name == state[STAGE_KEY]))

    saved = workspace.load_state().get("physics_draft")
    name = saved.get("name") if isinstance(saved, dict) else None
    st.title(name or workspace.root.name or "Study")
    selected = st.selectbox(
        "Workbench stage",
        tuple(stage.name for stage in STAGES),
        key=STAGE_KEY,
        on_change=select_stage,
    )
    stage = next(stage for stage in STAGES if stage.name == selected)
    destination, action, control = next_action(snapshot)
    # Keep the stage containers at fixed positions when the contextual action changes.
    with st.container(key="wb-contextual-action"):
        if destination != stage:
            st.button(
                action,
                on_click=navigate,
                args=(destination, control),
                key="wb-next-action",
            )

    hidden = ", ".join(panel_selector(item) for item in STAGES if item != stage)
    st.html(
        f"<style>{hidden}, .st-key-wb-navigation-focus {{ display: none !important; }}</style>"
    )
    return stage


def panel_selector(stage: Stage) -> str:
    return f".st-key-wb-panel-{stage.anchor}"


def stage_panel(st: Any, stage: Stage) -> Any:
    return st.container(key=f"wb-panel-{stage.anchor}")


@cache
def _focus_component() -> Any:
    from streamlit.components.v2 import component

    return component(
        "formamech.stage_focus",
        js="""
export default function({data}) {
  if (!data) return;
  const panel = `.st-key-wb-panel-${data.anchor}`;
  let frame;
  let attempts = 0;
  function moveFocus() {
    const section = document.querySelector(panel);
    const heading = section?.querySelector(`#${data.anchor}`);
    const control = data.control && section?.querySelector(data.control);
    const target = control || heading;
    if (target?.getClientRects().length) {
      if (!target.matches('button, input')) target.tabIndex = -1;
      target.focus();
    } else if (++attempts < 60) {
      frame = requestAnimationFrame(moveFocus);
    }
  }
  frame = requestAnimationFrame(moveFocus);
  return () => cancelAnimationFrame(frame);
}
""",
    )


def focus_destination(st: Any) -> None:
    if request := st.session_state.pop(FOCUS_KEY, None):
        with st.container(key="wb-navigation-focus"):
            _focus_component()(data=request, key="wb-stage-focus-bridge")
