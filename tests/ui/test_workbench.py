"""End-to-end headless & Streamlit AppTest suite for the Interactive 2D Workbench (M2)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from formamech.ui import (
    MATERIAL_PRESETS,
    BoundaryAssignment,
    ParameterSpecDraft,
    WorkbenchJobRunner,
    WorkbenchPhysicsDraft,
    WorkbenchWorkspace,
    build_cad_boundary_figure,
    build_settings_document,
    default_draft_from_preview,
    extract_selected_region,
    generate_sample_bracket_step,
    prepare_and_plan_workspace,
    render_workbench_cad_png,
    validate_boundaries_against_preview,
)

from piginot.contracts import Unit
from piginot.ui.workbench.job_runner import JobRecord
from piginot.ui.workbench.state import canonical_settings_bytes

ROOT = Path(__file__).resolve().parents[2]
NOTCHED_BUNDLE = ROOT / "bundles" / "notched-v1-cp-freeze"


def test_workspace_cad_import_boundaries_prepare_plan_and_plots(tmp_path: Path) -> None:
    ws = WorkbenchWorkspace(tmp_path / "ws-demo")
    assert ws.stage_progress().cad_imported is False
    assert ws.load_preview() is None
    assert ws.load_plan() is None
    assert ws.candidate_bundle_path is None
    assert ws.qualification_manifest_path is None

    # 1) Generate sample OpenCASCADE STEP and import via WorkbenchWorkspace
    step_path = generate_sample_bracket_step(tmp_path / "bracket.step")
    assert step_path.is_file()
    preview = ws.import_cad_path(step_path, unit=Unit.MM, mesh_size_mm=4.0)
    assert preview["schema"] == "piginot.cad-preview-v1"
    assert len(preview["boundaries"]) >= 5  # 4 outer edges + internal circular hole
    assert ws.stage_progress().cad_imported is True
    assert ws.stage_progress().document()["cad_imported"] is True

    # Invalid extension / empty payload / invalid sample dims must fail closed
    with pytest.raises(ValueError, match="unsupported CAD extension"):
        ws.import_cad_bytes(b"abc", "model.stl")
    with pytest.raises(ValueError, match="non-empty"):
        ws.import_cad_bytes(b"", "model.step")
    with pytest.raises(FileNotFoundError):
        ws.import_cad_path(tmp_path / "no-such.step")
    with pytest.raises(ValueError, match="positive"):
        generate_sample_bracket_step(tmp_path / "bad.step", width_mm=-1.0)
    with pytest.raises(ValueError, match="fit strictly"):
        generate_sample_bracket_step(tmp_path / "bad2.step", hole_radius_mm=15.0)

    # 2) Build default physics draft & validate boundaries against preview
    draft = default_draft_from_preview(preview, project_name="demo-bracket")
    ok, errs = validate_boundaries_against_preview(preview, draft.boundaries)
    assert ok is True, errs
    assert errs == ()

    # Boundary validation catches missing/duplicate/invalid-name/non-essential cases
    ok_miss, err_miss = validate_boundaries_against_preview(
        preview, draft.boundaries[:-1]
    )
    assert ok_miss is False
    assert any("unassigned" in e for e in err_miss)

    bad_all_free = tuple(
        BoundaryAssignment(region=b.region, name=f"free_{i}", kind="traction_free")
        for i, b in enumerate(draft.boundaries)
    )
    ok_free, err_free = validate_boundaries_against_preview(preview, bad_all_free)
    assert ok_free is False
    assert any("at least one dirichlet" in e for e in err_free)

    bad_dup = (
        BoundaryAssignment(
            region=draft.boundaries[0].region,
            name="123-bad",
            kind="dirichlet",
            components=(),
            values=(),
        ),
        BoundaryAssignment(
            region=draft.boundaries[0].region,
            name="123-bad",
            kind="symmetry",
            components=("z",),
            values=(("0", "mm"),),
        ),
    )
    ok_dup, err_dup = validate_boundaries_against_preview(preview, bad_dup)
    assert ok_dup is False
    assert any("duplicate" in e for e in err_dup)

    # 3) Prepare inputs & --plan-only plan with zero manual JSON editing
    result = prepare_and_plan_workspace(ws, draft, execution_profile=False)
    assert result["preparation"]["status"] == "PREPARED"
    assert result["plan"]["status"] == "PLANNED"
    assert ws.stage_progress().physics_saved is True
    assert ws.stage_progress().inputs_prepared is True
    assert ws.stage_progress().plan_previewed is True
    assert ws.load_plan() is not None
    assert len(canonical_settings_bytes(result["settings"])) > 10

    # Round-trip BoundaryAssignment serialization & material presets
    for b in draft.boundaries:
        assert BoundaryAssignment.from_document(b.document()) == b
    for model_key in MATERIAL_PRESETS:
        d_m = default_draft_from_preview(preview, material_model=model_key)
        doc_m = build_settings_document(d_m)
        assert doc_m["material"]["model"] == model_key

    # Invalid settings edge cases
    with pytest.raises(ValueError, match="unknown material_model"):
        default_draft_from_preview(preview, material_model="no_such_model")
    with pytest.raises(ValueError, match="vary strictly"):
        build_settings_document(
            WorkbenchPhysicsDraft(
                project_name="fixed-only",
                material_model="compressible_neo_hookean",
                material_param_map={"elastic_modulus": "E", "poisson_ratio": "nu"},
                parameters=(
                    ParameterSpecDraft("E", "material", 10.0, 10.0, "MPa"),
                    ParameterSpecDraft("nu", "material", 0.3, 0.3, "1"),
                ),
                boundaries=draft.boundaries,
            )
        )

    # 4) Interactive Plotly CAD viewer + selection extractor + Matplotlib PNG plot
    import importlib.util

    first_reg = draft.boundaries[0].region
    if importlib.util.find_spec("plotly") is not None:
        fig = build_cad_boundary_figure(
            preview,
            assignments=draft.boundaries,
            selected_region=first_reg,
            show_mesh=True,
        )
        assert len(fig.data) >= len(preview["boundaries"]) + 2
    assert (
        extract_selected_region({"selection": {"points": [{"customdata": first_reg}]}})
        == first_reg
    )
    assert (
        extract_selected_region(
            {"selection": {"points": [{"customdata": [first_reg]}]}}
        )
        == first_reg
    )
    assert extract_selected_region(None) is None
    assert extract_selected_region({"selection": {"points": []}}) is None

    png_out = render_workbench_cad_png(
        preview,
        tmp_path / "cad_preview.png",
        assignments=draft.boundaries,
        selected_region=first_reg,
    )
    assert png_out.is_file()
    assert png_out.stat().st_size > 5_000


def test_job_runner_lifecycle_poll_logs_and_cancel(tmp_path: Path) -> None:
    runner = WorkbenchJobRunner(tmp_path / "jobs", cwd=ROOT)
    assert runner.list_jobs() == ()
    with pytest.raises(FileNotFoundError):
        runner.load("missing-job")

    # Launch a fast real CLI command (`formamech --version`) via launch()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "checkpoints" / "step-000000010").mkdir(parents=True)
    rec = runner.launch(
        kind="compile", cli_args=["--version"], output_dir=out_dir, job_id="ver-check"
    )
    assert rec.state == "RUNNING"

    deadline = time.time() + 15.0
    final = rec
    while time.time() < deadline and final.state == "RUNNING":
        time.sleep(0.15)
        final = runner.poll(rec.job_id)

    assert final.state == "SUCCEEDED"
    assert final.exit_code == 0
    assert final.latest_checkpoint_step == 10
    assert "0.2.0" in runner.read_log_tail(rec.job_id)
    assert len(runner.list_jobs()) == 1

    # JobRecord round-trip and validation
    assert JobRecord.from_document(final.document()) == final
    with pytest.raises(ValueError, match="job record"):
        JobRecord.from_document({"schema": "wrong"})

    # Cancel a sleeping/long-ish command safely
    long_rec = runner.launch(
        kind="qualify",
        cli_args=["inspect", str(NOTCHED_BUNDLE)],
        output_dir=out_dir,
        job_id="cancel-test",
    )
    cancelled = runner.cancel(long_rec.job_id)
    assert cancelled.state == "CANCELLED"

    # Cover start_compile / start_qualify argument construction & external-PID inference
    from piginot.ui.workbench.job_runner import _state_from_exit_code

    assert _state_from_exit_code(0) == "SUCCEEDED"
    assert _state_from_exit_code(3) == "REFUSED_SCIENTIFIC"
    assert _state_from_exit_code(1) == "FAILED"

    captured_cmds: list[tuple[str, list[str]]] = []
    orig_launch = runner.launch

    def _fake_launch(
        *,
        kind: str,
        cli_args: list[str],
        output_dir: Path | str,
        job_id: str | None = None,
    ) -> JobRecord:
        captured_cmds.append((kind, list(cli_args)))
        return orig_launch(
            kind="compile" if kind == "compile" else "qualify",
            cli_args=["--version"],
            output_dir=output_dir,
            job_id=job_id,
        )

    runner.launch = _fake_launch  # type: ignore[method-assign]
    r_cmp = runner.start_compile(
        spec_path=tmp_path / "p.json",
        config_path=tmp_path / "c.json",
        output_dir=out_dir,
        resume=True,
    )
    r_q = runner.start_qualify(
        bundle_path=NOTCHED_BUNDLE,
        output_dir=out_dir,
        policy_path=tmp_path / "pol.json",
        emit_qualified_bundle=tmp_path / "qbundle",
    )
    assert "--resume" in captured_cmds[0][1]
    assert "--policy" in captured_cmds[1][1]
    assert "--emit-qualified-bundle" in captured_cmds[1][1]
    runner.cancel(r_cmp.job_id)
    runner.cancel(r_q.job_id)

    # Simulate an externally exited job from a prior UI session
    ext_dir = tmp_path / "ext-qual"
    ext_dir.mkdir()
    (ext_dir / "qualification-manifest.json").write_text("{}", encoding="utf-8")
    ext_rec = JobRecord(
        job_id="ext-1",
        kind="qualify",
        command=("python", "-m", "formamech"),
        pid=99999999,
        state="RUNNING",
        exit_code=None,
        started_utc="2026-11-22T00:00:00Z",
        finished_utc=None,
        log_path=str(tmp_path / "missing.log"),
        output_dir=str(ext_dir),
    )
    (tmp_path / "jobs" / "ext-1.json").write_text(
        json.dumps(ext_rec.document()), encoding="utf-8"
    )
    inferred = runner.poll("ext-1")
    assert inferred.state == "SUCCEEDED"
    assert runner.read_log_tail("ext-1") == ""


def test_streamlit_apptest_workbench_renders_cleanly(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    ws = WorkbenchWorkspace(tmp_path / "apptest-ws")
    step_path = generate_sample_bracket_step(tmp_path / "sample.step")
    preview = ws.import_cad_path(step_path, unit=Unit.MM, mesh_size_mm=5.0)
    draft = default_draft_from_preview(preview, project_name="apptest-model")
    prepare_and_plan_workspace(ws, draft, execution_profile=False)

    script = (
        "from piginot.ui.app import run_streamlit_app\n"
        f"run_streamlit_app(default_bundle={str(NOTCHED_BUNDLE)!r}, "
        f"default_workspace={str(ws.root)!r})\n"
    )
    at = AppTest.from_string(script, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Interactive Engineering Workbench" in t.value for t in at.title)
    assert len(at.tabs) >= 5
    # Verify canonical workspace files exist and match CLI contract
    assert (ws.inputs_dir / "problem.json").is_file()
    assert (ws.preview_dir / "plan.json").is_file()
    plan_doc = json.loads((ws.preview_dir / "plan.json").read_text(encoding="utf-8"))
    assert plan_doc["status"] == "PLANNED"


def test_workbench_state_machines_staleness_and_job_invariants(tmp_path: Path) -> None:
    from formamech.contracts import InvalidTransitionError
    from formamech.ui import JobLifecycleState, WorkbenchStage
    from formamech.ui.workbench.cad_viewer import _classify_assignment
    from formamech.ui.workbench.job_runner import _infer_external_terminal

    # 1) Boundary visual role classifier edge cases
    assert _classify_assignment(None)[0] == "unassigned"
    assert (
        _classify_assignment(
            BoundaryAssignment("e1", "c", "dirichlet", ("x",), (("0.00", "mm"),))
        )[0]
        == "dirichlet_fixed"
    )
    assert (
        _classify_assignment(BoundaryAssignment("e1", "bad", "dirichlet", (), ()))[0]
        == "unassigned"
    )
    assert (
        _classify_assignment(
            BoundaryAssignment("e1", "c", "dirichlet", ("x",), (("", "mm"),))
        )[0]
        == "dirichlet_loaded"
    )

    # 2) Workspace FSM progression across all stages, stale items, and blockers
    ws = WorkbenchWorkspace(tmp_path / "ws-fsm")
    assert ws.workflow_snapshot().state is WorkbenchStage.EMPTY

    # Stale downstream artifacts without geometry are flagged cleanly
    ws.physics_path.write_text("{}", encoding="utf-8")
    ws.inputs_dir.mkdir(parents=True, exist_ok=True)
    (ws.inputs_dir / "problem.json").write_text("{}", encoding="utf-8")
    (ws.inputs_dir / "config.json").write_text("{}", encoding="utf-8")
    (ws.run_dir / "candidate-bundle").mkdir(parents=True, exist_ok=True)
    (ws.run_dir / "candidate-bundle" / "bundle.json").write_text("{}", encoding="utf-8")
    snap_empty_stale = ws.workflow_snapshot()
    assert snap_empty_stale.state is WorkbenchStage.EMPTY
    assert len(snap_empty_stale.stale_items) >= 3

    # Corrupt workbench-state.json falls back cleanly to default schema
    ws.state_path.write_text("{corrupt", encoding="utf-8")
    assert ws.load_state()["schema"] == "formamech.workbench-state-v1"
    ws.state_path.write_text('{"schema": "other"}', encoding="utf-8")
    assert ws.load_state()["schema"] == "formamech.workbench-state-v1"
    ws.state_path.write_text("[]", encoding="utf-8")
    assert ws.load_state()["schema"] == "formamech.workbench-state-v1"

    # Import CAD clears stale downstream artifacts atomically
    step_path = generate_sample_bracket_step(tmp_path / "fsm.step")
    preview = ws.import_cad_path(step_path, unit=Unit.MM, mesh_size_mm=5.0)
    assert ws.workflow_snapshot().state is WorkbenchStage.CAD_IMPORTED

    # Stale inputs/bundle relative to fresh CAD
    ws.inputs_dir.mkdir(parents=True, exist_ok=True)
    (ws.inputs_dir / "problem.json").write_text("{}", encoding="utf-8")
    (ws.inputs_dir / "config.json").write_text("{}", encoding="utf-8")
    (ws.run_dir / "candidate-bundle").mkdir(parents=True, exist_ok=True)
    (ws.run_dir / "candidate-bundle" / "bundle.json").write_text("{}", encoding="utf-8")
    assert len(ws.workflow_snapshot().stale_items) >= 2

    # PHYSICS_SAVED without inputs (and stale preview/bundle)
    ws.reset_prepared_inputs()
    ws.physics_path.write_text("{}", encoding="utf-8")
    ws.preview_dir.mkdir(parents=True, exist_ok=True)
    (ws.preview_dir / "plan.json").write_text("{}", encoding="utf-8")
    snap_phys = ws.workflow_snapshot()
    assert snap_phys.state is WorkbenchStage.PHYSICS_SAVED
    assert len(snap_phys.stale_items) >= 2

    # INPUTS_PREPARED without plan
    ws.inputs_dir.mkdir(parents=True, exist_ok=True)
    (ws.inputs_dir / "problem.json").write_text("{}", encoding="utf-8")
    (ws.inputs_dir / "config.json").write_text("{}", encoding="utf-8")
    (ws.preview_dir / "plan.json").unlink()
    assert ws.workflow_snapshot().state is WorkbenchStage.INPUTS_PREPARED

    # Full prepare + plan -> PLAN_PREVIEWED
    (ws.run_dir / "candidate-bundle" / "bundle.json").unlink()
    draft = default_draft_from_preview(preview, project_name="fsm-full")
    prepare_and_plan_workspace(ws, draft, execution_profile=False)
    assert ws.workflow_snapshot().state is WorkbenchStage.PLAN_PREVIEWED

    # Mutating physics.json or geometry.json after prepare flags staleness
    orig_phys = ws.physics_path.read_text(encoding="utf-8")
    ws.physics_path.write_text(orig_phys + "\n", encoding="utf-8")
    assert any("physics.json" in s for s in ws.workflow_snapshot().stale_items)
    ws.physics_path.write_text(orig_phys, encoding="utf-8")

    # Qualification without trained bundle flags staleness
    ws.qualification_dir.mkdir(parents=True, exist_ok=True)
    qfile = ws.qualification_dir / "qualification-manifest.json"
    qfile.write_text(json.dumps({"verdict": "QUALIFIED"}), encoding="utf-8")
    assert any("without a trained" in s for s in ws.workflow_snapshot().stale_items)

    # Trained bundle + NOT_QUALIFIED / invalid manifest -> BUNDLE_TRAINED with blocker
    (ws.run_dir / "candidate-bundle").mkdir(parents=True, exist_ok=True)
    (ws.run_dir / "candidate-bundle" / "bundle.json").write_text("{}", encoding="utf-8")
    qfile.write_text("{bad-json", encoding="utf-8")
    assert any("malformed" in b for b in ws.workflow_snapshot().blockers)
    qfile.write_text("[]", encoding="utf-8")
    assert any("malformed" in b for b in ws.workflow_snapshot().blockers)
    qfile.write_text(json.dumps({"verdict": "NOT_QUALIFIED"}), encoding="utf-8")
    snap_nq = ws.workflow_snapshot()
    assert snap_nq.state is WorkbenchStage.BUNDLE_TRAINED
    assert any("NOT_QUALIFIED" in b for b in snap_nq.blockers)

    # Passing QUALIFIED manifest -> BUNDLE_QUALIFIED
    qfile.write_text(json.dumps({"verdict": "QUALIFIED"}), encoding="utf-8")
    assert ws.workflow_snapshot().state is WorkbenchStage.BUNDLE_QUALIFIED

    # 3) Job runner FSM, snapshots, active_jobs, external inference, and terminal immutability
    runner = WorkbenchJobRunner(tmp_path / "jobs-fsm", cwd=ROOT)
    assert runner.active_jobs() == ()

    # External inference paths for compile and qualify
    c_dir = tmp_path / "ext-compile"
    c_dir.mkdir()
    assert _infer_external_terminal("compile", c_dir) == (JobLifecycleState.FAILED, 1)
    (c_dir / "candidate-bundle").mkdir()
    (c_dir / "candidate-bundle" / "bundle.json").write_text("{}", encoding="utf-8")
    assert _infer_external_terminal("compile", c_dir) == (
        JobLifecycleState.SUCCEEDED,
        0,
    )
    (c_dir / "report.json").write_text(json.dumps({"exit_code": 3}), encoding="utf-8")
    assert _infer_external_terminal("compile", c_dir) == (
        JobLifecycleState.REFUSED_SCIENTIFIC,
        3,
    )
    (c_dir / "report.json").write_text("{corrupt", encoding="utf-8")
    assert _infer_external_terminal("compile", c_dir) == (
        JobLifecycleState.SUCCEEDED,
        0,
    )

    q_dir = tmp_path / "ext-qual-fsm"
    q_dir.mkdir()
    assert _infer_external_terminal("qualify", q_dir) == (JobLifecycleState.FAILED, 1)
    (q_dir / "qualification-manifest.json").write_text("{bad", encoding="utf-8")
    assert _infer_external_terminal("qualify", q_dir) == (JobLifecycleState.FAILED, 1)
    (q_dir / "qualification-manifest.json").write_text(
        json.dumps({"verdict": "NOT_QUALIFIED"}), encoding="utf-8"
    )
    assert _infer_external_terminal("qualify", q_dir) == (
        JobLifecycleState.REFUSED_SCIENTIFIC,
        3,
    )

    # Pending & terminal record snapshots + terminal cancel refusal
    pending_rec = JobRecord(
        job_id="pend-1",
        kind="compile",
        command=("python",),
        pid=None,
        state="PENDING",
        exit_code=None,
        started_utc="2026-11-22T00:00:00Z",
        finished_utc=None,
        log_path=str(tmp_path / "p.log"),
        output_dir=str(c_dir),
    )
    assert pending_rec.is_terminal is False
    assert pending_rec.snapshot().state is JobLifecycleState.PENDING
    (tmp_path / "jobs-fsm" / "pend-1.json").write_text(
        json.dumps(pending_rec.document()), encoding="utf-8"
    )
    assert runner.poll("pend-1").lifecycle_state is JobLifecycleState.PENDING
    assert len(runner.active_jobs(kind="compile")) == 1
    assert runner.active_jobs(kind="qualify") == ()
    cancelled_p = runner.cancel("pend-1")
    assert cancelled_p.is_terminal is True
    assert cancelled_p.snapshot().terminal is True
    with pytest.raises(InvalidTransitionError):
        runner.cancel("pend-1")
    # Polling a terminal job is a strict no-op
    assert runner.poll("pend-1") == cancelled_p

    # Invalid kind / state rejected cleanly
    with pytest.raises(ValueError, match="job kind"):
        JobRecord.from_document({**pending_rec.document(), "kind": "unknown"})
    with pytest.raises(ValueError, match="job state"):
        JobRecord.from_document({**pending_rec.document(), "state": "BOGUS"})
