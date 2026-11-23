from dataclasses import replace
from pathlib import Path

import pytest
from formamech.ui import BoundaryAssignment, WorkbenchJobRunner, WorkbenchWorkspace

from piginot.ui.workbench import physics_forms
from piginot.ui.workbench.physics_forms import default_draft_from_preview

PLATE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "cad" / "plate-with-hole.step"
)


@pytest.fixture
def study(tmp_path):
    workspace = WorkbenchWorkspace(tmp_path / "study")
    preview = workspace.import_cad_path(PLATE)
    draft = replace(
        default_draft_from_preview(preview), optimizer_steps=127, seed=37, u_tol=0.067
    )
    physics_forms.prepare_and_plan_workspace(workspace, draft, execution_profile=False)
    return workspace, preview, draft


def test_reopen_restores_saved_form(study):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    ws, _, draft = study
    app = AppTest.from_string(
        "from piginot.ui.app import run_streamlit_app\n"
        f"run_streamlit_app(default_workspace={str(ws.root)!r})\n",
        default_timeout=60,
    ).run()
    numbers = {widget.label: widget.value for widget in app.number_input}
    assert not app.exception
    assert numbers["Optimizer steps"] == 127
    assert numbers["Random seed"] == draft.seed
    assert numbers["u rel-L2"] == draft.u_tol


def test_applied_boundary_refuses_old_plan_and_reprepares(study, monkeypatch):
    from piginot.compiler.planning import build_plan
    from piginot.ui.workbench.study_plan import plan_blocker

    ws, preview, draft = study
    original = draft.boundaries[0]
    assert original.kind == "traction_free"
    prior_bytes = ws.state_path.read_bytes()
    assert not physics_forms.apply_boundary(ws, preview, original)
    assert ws.state_path.read_bytes() == prior_bytes
    assert plan_blocker(ws) is None
    updated = BoundaryAssignment(original.region, original.name, "symmetry", ("y",))
    assert physics_forms.apply_boundary(ws, preview, updated)
    assert original.region in plan_blocker(WorkbenchWorkspace(ws.root))

    spec, config = (ws.inputs_dir / f"{name}.json" for name in ("problem", "config"))
    runner = WorkbenchJobRunner(ws.jobs_dir, cwd=ws.root)
    monkeypatch.setattr(
        runner, "launch", lambda **_: pytest.fail("stale compile dispatched")
    )
    for resume in (False, True):
        with pytest.raises(ValueError, match="Update plan"):
            runner.start_compile(
                spec_path=spec, config_path=config, output_dir=ws.run_dir, resume=resume
            )

    applied = physics_forms.load_physics_draft(ws, preview)
    assert applied.boundaries[0] == updated
    result = physics_forms.prepare_and_plan_workspace(
        ws, applied, execution_profile=False
    )
    assert result["plan"] == build_plan(spec, config, ws.preview_dir).document()
    assert result["plan"]["problem"]["boundary_conditions"][0]["kind"] == "symmetry"
    assert plan_blocker(ws) is None
    with pytest.raises(ValueError, match="compatible checkpoint"):
        runner.start_compile(
            spec_path=spec, config_path=config, output_dir=ws.run_dir, resume=True
        )


def test_failed_prepare_preserves_last_prepared_artifacts_and_persists_draft(
    study, monkeypatch
):
    from piginot.ui.workbench.study_plan import plan_blocker

    ws, preview, draft = study
    previous = {
        p: p.read_bytes()
        for p in (
            *ws.inputs_dir.iterdir(),
            ws.preview_dir / "plan.json",
            ws.physics_path,
        )
    }

    def fail(*_args, **_kwargs):
        raise RuntimeError("deliberate preparation failure")

    monkeypatch.setattr(physics_forms, "build_plan", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        physics_forms.prepare_and_plan_workspace(
            ws, replace(draft, optimizer_steps=128), execution_profile=False
        )
    assert all(p.read_bytes() == content for p, content in previous.items())
    assert plan_blocker(WorkbenchWorkspace(ws.root)) is not None
    assert (
        physics_forms.load_physics_draft(
            WorkbenchWorkspace(ws.root), preview
        ).optimizer_steps
        == 128
    )


def test_existing_study_freshness_comes_from_artifacts(study):
    from piginot.ui.workbench.study_plan import plan_blocker

    ws, _, _ = study
    ws.state_path.unlink()
    assert plan_blocker(WorkbenchWorkspace(ws.root)) is None
    (ws.inputs_dir / "preparation.json").unlink()
    assert plan_blocker(WorkbenchWorkspace(ws.root)) is not None
