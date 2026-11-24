import hashlib
import json
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


def test_reopen_restores_saved_form(study, workbench_apptest):
    ws, _, draft = study
    app = workbench_apptest.from_string(
        "from piginot.ui.app import run_streamlit_app\n"
        f"run_streamlit_app(default_workspace={str(ws.root)!r})\n",
        default_timeout=60,
    ).run()
    numbers = {widget.label: widget.value for widget in app.number_input}
    assert not app.exception
    assert numbers["Optimizer steps"] == 127
    assert numbers["Random seed"] == draft.seed
    assert numbers["u rel-L2"] == draft.u_tol

    next(
        button for button in app.button if button.label == "Start compile"
    ).click().run()
    assert app.session_state["wb-stage"] == "Train & Qualify"
    navigator = next(
        widget for widget in app.selectbox if widget.label == "Workbench stage"
    )
    navigator.set_value("Boundary Picker").run()
    assert app.session_state["wb-stage"] == "Boundary Picker"
    assert not app.exception


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
    study, monkeypatch, workbench_apptest
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
    app = workbench_apptest.from_string(
        "from piginot.ui.app import run_streamlit_app\n"
        f"run_streamlit_app(default_workspace={str(ws.root)!r})\n",
        default_timeout=60,
    ).run()
    steps = next(
        widget for widget in app.number_input if widget.label == "Optimizer steps"
    )
    steps.set_value(128).run()
    next(button for button in app.button if button.label == "Update plan").click().run()
    assert any("deliberate preparation failure" in error.value for error in app.error)
    assert len(app.table) == 1
    assert app.table[0].value.to_dict("records") == [
        {
            "Area": "Training budget",
            "Setting": "Optimizer steps",
            "Prepared": str(draft.optimizer_steps),
            "Proposed": "128",
        }
    ]
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


def test_historical_resume_rechecks_job_identity_and_current_plan(study, monkeypatch):
    from piginot.compiler.execution_records import save_bound
    from piginot.compiler.planning import build_plan
    from piginot.compiler.records import write_plan
    from piginot.training.checkpoint import write_training_checkpoint
    from piginot.training.state import DeterministicSamplerState
    from piginot.ui.workbench.job_runner import JobRecord
    from tests.training.test_checkpoint import (
        _adaptive,
        _binding,
        _model,
        _optimizer,
        _scheduler,
    )

    ws, preview, draft = study
    runner = WorkbenchJobRunner(ws.jobs_dir, cwd=ws.root)
    token = "a" * 32
    job = JobRecord(
        job_id="interrupted-compile",
        kind="compile",
        command=(),
        pid=None,
        state="FAILED",
        exit_code=1,
        started_utc="2026-11-23T00:00:00Z",
        finished_utc="2026-11-23T00:01:00Z",
        log_path=str(ws.jobs_dir / "compile.log"),
        output_dir=str(ws.run_dir),
        latest_checkpoint_step=25,
        evidence_id=token,
    )
    runner._save(job)
    archive = ws.jobs_dir / "evidence" / token
    archive.mkdir(parents=True)
    (archive / "owner.json").write_text(
        json.dumps(
            {
                "job_id": job.job_id,
                "kind": "compile",
                "output_dir": job.output_dir,
            }
        ),
        encoding="utf-8",
    )
    plan = build_plan(
        ws.inputs_dir / "problem.json", ws.inputs_dir / "config.json", ws.run_dir
    )
    write_plan(plan)
    binding, model = _binding(), _model()
    save_bound(
        ws.run_dir / "stage-train-preparation.json",
        plan.plan_sha256,
        {
            "status": "PASS",
            "run_hash": binding.run_hash,
        },
    )
    path = ws.run_dir / "checkpoints" / "step-000000025"
    path.parent.mkdir()
    write_training_checkpoint(
        path,
        model=model,
        optimizer=_optimizer(model),
        binding=binding,
        step=25,
        sampler_state=DeterministicSamplerState(binding.sampler_plan_hash, 25),
        scheduler_state=_scheduler(25),
        adaptive_state=_adaptive(25),
    )
    checkpoint = path / "checkpoint.json"
    evidence = checkpoint.read_bytes()
    (archive / "checkpoint.json").write_text(
        json.dumps(
            {
                "step": 25,
                "sha256": hashlib.sha256(evidence).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    assert runner.job_checkpoint(job) == 25
    launched = []
    monkeypatch.setattr(
        runner, "launch", lambda **kwargs: launched.append(kwargs) or job
    )
    kwargs = {
        "spec_path": ws.inputs_dir / "problem.json",
        "config_path": ws.inputs_dir / "config.json",
        "output_dir": ws.run_dir,
        "resume": True,
        "resume_job_id": job.job_id,
    }
    assert runner.can_resume(job, ws)
    assert runner.start_compile(**kwargs) == job
    assert "--resume" in launched[0]["cli_args"]
    checkpoint.write_bytes(b"new run checkpoint in the same output path")
    assert not runner.can_resume(job, ws)
    with pytest.raises(ValueError, match="no compatible checkpoint"):
        runner.start_compile(**kwargs)
    checkpoint.write_bytes(evidence)
    (path / "optimizer.json").write_bytes(b"corrupted checkpoint payload")
    assert not runner.can_resume(job, ws)

    edge = draft.boundaries[0]
    assert physics_forms.apply_boundary(
        ws, preview, replace(edge, kind="symmetry", components=("y",))
    )
    assert not runner.can_resume(job, ws)
    with pytest.raises(ValueError, match="Update plan"):
        runner.start_compile(**kwargs)
    assert len(launched) == 1


def test_plan_comparison_normalizes_units_and_matches_record_identities(study):
    from piginot.ui.workbench.plan_review import plan_review_rows

    _, _, draft = study
    quantities = tuple(
        replace(p, lo=p.lo * 1_000_000, hi=p.hi * 1_000_000, unit="Pa")
        if p.unit == "MPa"
        else p
        for p in reversed(draft.parameters)
    )
    normalized = replace(
        draft, parameters=quantities, boundaries=draft.boundaries[::-1]
    )
    assert plan_review_rows(draft, normalized) == []
    proposed = replace(normalized, optimizer_steps=40, p_tol=0.12)
    assert [row["Area"] for row in plan_review_rows(draft, proposed)] == [
        "Training budget",
        "Tolerances",
    ]
    initial = plan_review_rows(None, proposed)
    assert {row["Area"] for row in initial} == {
        "Boundary conditions",
        "Material and loads",
        "Training budget",
        "Tolerances",
    }
    assert "Prepared" not in initial[0]
