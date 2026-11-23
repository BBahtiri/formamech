"""Streamlit entrypoint for the FormaMech Interactive Engineering Workbench (`formamech ui`).

Unifies the guided zero-JSON 2D CAD workflow (Import -> Boundary Picker ->
Material & Plan -> Train & Qualify) with the Verification & Trust Inspector
using the exact same canonical backend contracts as the CLI and MCP server.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from piginot.contracts import Unit
from piginot.ui.inspector import (
    build_inspector_snapshot,
    build_plotly_figures,
    render_trust_dossier_markdown,
)
from piginot.ui.workbench import (
    MATERIAL_PRESETS,
    BoundaryAssignment,
    BoundaryKind,
    ParameterSpecDraft,
    WorkbenchJobRunner,
    WorkbenchPhysicsDraft,
    WorkbenchWorkspace,
    build_cad_boundary_figure,
    default_draft_from_preview,
    extract_selected_region,
    generate_sample_bracket_step,
    prepare_and_plan_workspace,
    validate_boundaries_against_preview,
)


def _load_or_init_boundaries(
    st: Any, workspace: WorkbenchWorkspace, preview: dict[str, Any]
) -> list[BoundaryAssignment]:
    state_key = f"wb_boundaries::{workspace.root}"
    preview_regions = [str(b["name"]) for b in preview.get("boundaries", ())]
    if state_key in st.session_state:
        existing: list[BoundaryAssignment] = list(st.session_state[state_key])
        if [b.region for b in existing] == preview_regions:
            return existing

    persisted = workspace.load_state().get("boundaries")
    if isinstance(persisted, list):
        try:
            loaded = [BoundaryAssignment.from_document(item) for item in persisted]
            if [b.region for b in loaded] == preview_regions:
                st.session_state[state_key] = loaded
                return loaded
        except (KeyError, TypeError, ValueError):
            pass

    defaults = list(default_draft_from_preview(preview).boundaries)
    st.session_state[state_key] = defaults
    return defaults


def _render_stage_1_import(st: Any, workspace: WorkbenchWorkspace) -> None:
    st.subheader("Step 1 — Import 2D CAD Geometry & Mesh")
    st.caption(
        "Upload a planar STEP/STP/DXF profile or generate a built-in OpenCASCADE "
        "sample bracket to inspect its mesh and semantic boundary groups."
    )

    col_cfg, col_view = st.columns([1.0, 1.5], gap="large")
    with col_cfg:
        uploaded = st.file_uploader(
            "Upload 2D STEP / STP / DXF profile",
            type=["step", "stp", "dxf"],
            key="wb_cad_uploader",
        )
        unit_str = st.selectbox("CAD Length Unit", options=["mm", "cm", "m"], index=0)
        mesh_size = st.number_input(
            "Target Mesh Size (mm)",
            min_value=0.25,
            max_value=50.0,
            value=3.0,
            step=0.5,
        )

        btn_a, btn_b = st.columns(2)
        if btn_a.button("Import Uploaded CAD", type="primary", width="stretch"):
            if uploaded is None:
                st.warning("Upload a .step, .stp, or .dxf file first.")
            else:
                try:
                    workspace.import_cad_bytes(
                        uploaded.getvalue(),
                        uploaded.name,
                        unit=Unit(unit_str),
                        mesh_size_mm=float(mesh_size),
                    )
                    st.session_state.pop(f"wb_boundaries::{workspace.root}", None)
                    st.success(f"Imported and meshed {uploaded.name}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"CAD import failed: {exc}")

        if btn_b.button("Generate Sample 2D Bracket", width="stretch"):
            try:
                sample_step = generate_sample_bracket_step(
                    workspace.source_dir / "sample-bracket.step"
                )
                workspace.import_cad_path(
                    sample_step,
                    unit=Unit.MM,
                    mesh_size_mm=float(mesh_size),
                )
                st.session_state.pop(f"wb_boundaries::{workspace.root}", None)
                st.success("Generated and meshed sample 40 x 20 mm bracket STEP.")
                st.rerun()
            except Exception as exc:
                st.error(f"Sample generation failed: {exc}")

        preview = workspace.load_preview()
        if preview is not None:
            mesh_info = preview.get("mesh", {})
            bbox = preview.get("bounding_box_mm", {})
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Nodes", int(mesh_info.get("node_count", 0)))
            m2.metric("Triangles", int(mesh_info.get("triangle_count", 0)))
            m3.metric("Boundaries", len(preview.get("boundaries", ())))
            st.caption(
                f"Bounding box (mm): x=[{bbox.get('min', [0, 0])[0]:.2f}, "
                f"{bbox.get('max', [0, 0])[0]:.2f}], "
                f"y=[{bbox.get('min', [0, 0])[1]:.2f}, "
                f"{bbox.get('max', [0, 0])[1]:.2f}]"
            )
            st.caption(f"Mesh ID: `{str(mesh_info.get('mesh_id', ''))[:16]}...`")

    with col_view:
        preview = workspace.load_preview()
        if preview is None:
            st.info(
                "No CAD geometry imported yet. Upload a STEP/DXF file or click "
                "**Generate Sample 2D Bracket** to begin."
            )
        else:
            assignments = _load_or_init_boundaries(st, workspace, preview)
            fig = build_cad_boundary_figure(
                preview, assignments=assignments, show_mesh=True
            )
            st.plotly_chart(fig, width="stretch")


def _render_stage_2_boundaries(st: Any, workspace: WorkbenchWorkspace) -> None:
    st.subheader("Step 2 — Visual Boundary Condition Picker")
    preview = workspace.load_preview()
    if preview is None:
        st.info("Import a 2D CAD profile in Step 1 first.")
        return

    assignments = _load_or_init_boundaries(st, workspace, preview)
    regions = [b.region for b in assignments]
    kinds_by_region = {
        str(entry["name"]): str(entry.get("kind", "other"))
        for entry in preview.get("boundaries", ())
    }
    sel_key = f"wb_selected_region::{workspace.root}"
    select_widget_key = f"wb_region_select::{workspace.root}"
    if sel_key not in st.session_state or st.session_state[sel_key] not in regions:
        st.session_state[sel_key] = regions[0]
        st.session_state[select_widget_key] = regions[0]
    elif (
        select_widget_key in st.session_state
        and st.session_state[select_widget_key] not in regions
    ):
        st.session_state[select_widget_key] = st.session_state[sel_key]

    col_canvas, col_editor = st.columns([1.45, 1.0], gap="large")
    with col_canvas:
        show_mesh = st.checkbox(
            "Show mesh wireframe", value=True, key=f"wb_show_mesh::{workspace.root}"
        )
        fig = build_cad_boundary_figure(
            preview,
            assignments=assignments,
            selected_region=str(st.session_state[sel_key]),
            show_mesh=show_mesh,
        )
        selection = st.plotly_chart(
            fig,
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key=f"wb_cad_picker_chart::{workspace.root}",
        )
        clicked = extract_selected_region(selection)
        if (
            clicked is not None
            and clicked in regions
            and clicked != st.session_state[sel_key]
        ):
            st.session_state[sel_key] = clicked
            st.session_state[select_widget_key] = clicked
            st.rerun()

    with col_editor:
        active_region = st.selectbox(
            "Selected CAD Boundary Group",
            options=regions,
            index=regions.index(str(st.session_state[sel_key])),
            key=select_widget_key,
        )
        st.session_state[sel_key] = active_region
        bkind = kinds_by_region.get(active_region, "other")
        st.caption(f"Geometry entity kind: `{bkind}`")

        current_idx = regions.index(active_region)
        current = assignments[current_idx]
        ws_ns = str(workspace.root)

        bc_name = st.text_input(
            "Condition Name",
            value=current.name,
            key=f"bc_name::{ws_ns}::{active_region}",
        )
        kind_options: list[BoundaryKind] = ["dirichlet", "symmetry", "traction_free"]
        bc_kind_str = st.selectbox(
            "Boundary Condition Type",
            options=kind_options,
            index=kind_options.index(current.kind),
            key=f"bc_kind::{ws_ns}::{active_region}",
        )
        bc_kind: BoundaryKind = (
            "dirichlet"
            if bc_kind_str == "dirichlet"
            else ("symmetry" if bc_kind_str == "symmetry" else "traction_free")
        )

        comps: list[str] = []
        vals: list[tuple[str, str]] = []
        if bc_kind != "traction_free":
            cur_map = {
                c: v for c, v in zip(current.components, current.values, strict=False)
            }
            use_x = st.checkbox(
                "Prescribe u_x",
                value="x" in cur_map or not cur_map,
                key=f"use_x::{ws_ns}::{active_region}",
            )
            expr_x = st.text_input(
                "u_x expression (mm)",
                value=cur_map.get("x", ("0", "mm"))[0],
                disabled=not use_x,
                key=f"expr_x::{ws_ns}::{active_region}",
            )
            use_y = st.checkbox(
                "Prescribe u_y",
                value="y" in cur_map,
                key=f"use_y::{ws_ns}::{active_region}",
            )
            expr_y = st.text_input(
                "u_y expression (mm)",
                value=cur_map.get("y", ("0", "mm"))[0],
                disabled=not use_y,
                key=f"expr_y::{ws_ns}::{active_region}",
            )
            if use_x:
                comps.append("x")
                vals.append((expr_x.strip() or "0", "mm"))
            if use_y:
                comps.append("y")
                vals.append((expr_y.strip() or "0", "mm"))

        if st.button("Apply Boundary Condition", type="primary", width="stretch"):
            updated = BoundaryAssignment(
                region=active_region,
                name=bc_name.strip(),
                kind=bc_kind,
                components=tuple(comps),
                values=tuple(vals),
            )
            assignments[current_idx] = updated
            st.session_state[f"wb_boundaries::{workspace.root}"] = assignments
            state = workspace.load_state()
            state["boundaries"] = [b.document() for b in assignments]
            workspace.save_state(state)
            st.success(f"Updated {active_region} -> {updated.name} ({updated.kind}).")
            st.rerun()

        st.divider()
        ok, errors = validate_boundaries_against_preview(preview, assignments)
        if ok:
            st.success("All boundary groups are explicitly assigned and valid.")
        else:
            for err in errors:
                st.error(err)

    st.subheader("Assigned Boundary Table")
    st.table(
        [
            {
                "Region": b.region,
                "Kind": kinds_by_region.get(b.region, "other"),
                "Name": b.name,
                "BC Type": b.kind,
                "Prescribed": ", ".join(
                    f"u_{c}={e} {u}"
                    for c, (e, u) in zip(b.components, b.values, strict=False)
                )
                if b.kind != "traction_free"
                else "traction-free",
            }
            for b in assignments
        ]
    )


def _render_stage_3_physics_plan(st: Any, workspace: WorkbenchWorkspace) -> None:
    st.subheader("Step 3 — Material Model, Load Ranges & Plan Preview")
    preview = workspace.load_preview()
    if preview is None:
        st.info("Import a 2D CAD profile in Step 1 first.")
        return

    assignments = _load_or_init_boundaries(st, workspace, preview)
    ok_bc, err_bc = validate_boundaries_against_preview(preview, assignments)
    if not ok_bc:
        st.warning("Fix boundary assignments in Step 2 before preparing inputs:")
        for err in err_bc:
            st.error(err)

    saved_state = workspace.load_state()
    model_keys = list(MATERIAL_PRESETS.keys())
    default_model = str(saved_state.get("material_model", "compressible_neo_hookean"))
    if default_model not in model_keys:
        default_model = "compressible_neo_hookean"

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        project_name = st.text_input(
            "Campaign / Model Name",
            value=str(saved_state.get("project_name", "cad-workbench-model")),
        )
        mat_model = st.selectbox(
            "Hyperelastic Material Model",
            options=model_keys,
            index=model_keys.index(default_model),
            format_func=lambda k: f"{MATERIAL_PRESETS[k].display_name} ({k})",
        )
        preset = MATERIAL_PRESETS[mat_model]
        st.caption(preset.description)

        st.markdown("**Material Parameter Bounds**")
        param_drafts: list[ParameterSpecDraft] = []
        param_map: dict[str, str] = {}
        for prop_name, pname, default_lo, default_hi, unit in preset.parameters:
            c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
            c1.markdown(f"`{pname}` ({prop_name}, {unit})")
            lo_v = c2.number_input(
                f"{pname} min",
                value=float(default_lo),
                format="%.4f",
                key=f"mat_lo_{mat_model}_{pname}",
            )
            hi_v = c3.number_input(
                f"{pname} max",
                value=float(default_hi),
                format="%.4f",
                key=f"mat_hi_{mat_model}_{pname}",
            )
            param_drafts.append(
                ParameterSpecDraft(
                    name=pname,
                    role="material",
                    lo=float(lo_v),
                    hi=float(hi_v),
                    unit=unit,
                )
            )
            param_map[prop_name] = pname

        st.markdown("**Prescribed Load Parameter Bounds**")
        bbox = preview.get("bounding_box_mm", {})
        extent_x = max(
            float(bbox.get("max", [10.0, 10.0])[0])
            - float(bbox.get("min", [0.0, 0.0])[0]),
            1.0,
        )
        lc1, lc2, lc3 = st.columns([1.2, 1.0, 1.0])
        load_name = lc1.text_input("Load symbol", value="ux")
        load_lo = lc2.number_input(
            "Load min (mm)", value=round(0.02 * extent_x, 4), format="%.4f"
        )
        load_hi = lc3.number_input(
            "Load max (mm)", value=round(0.08 * extent_x, 4), format="%.4f"
        )
        param_drafts.append(
            ParameterSpecDraft(
                name=load_name.strip() or "ux",
                role="load",
                lo=float(load_lo),
                hi=float(load_hi),
                unit="mm",
            )
        )

    with col_right:
        st.markdown("**Training & Sampling Budget**")
        t1, t2 = st.columns(2)
        opt_steps = t1.number_input(
            "Optimizer steps", min_value=1, max_value=50000, value=40
        )
        batch_sz = t2.number_input(
            "Case batch size", min_value=1, max_value=128, value=2
        )
        t3, t4 = st.columns(2)
        fit_cases = t3.number_input(
            "TRAIN-fit cases", min_value=2, max_value=1024, value=4
        )
        screen_cases = t4.number_input(
            "TRAIN-screen cases", min_value=0, max_value=256, value=2
        )
        t5, t6 = st.columns(2)
        ckpt_int = t5.number_input(
            "Checkpoint interval", min_value=1, max_value=5000, value=20
        )
        seed_val = t6.number_input(
            "Random seed", min_value=0, max_value=1000000, value=42
        )

        st.markdown("**Verification Tolerances**")
        v1, v2, v3 = st.columns(3)
        u_tol = v1.number_input(
            "u rel-L2", min_value=0.001, max_value=0.5, value=0.05, format="%.3f"
        )
        p_tol = v2.number_input(
            "P rel-L2", min_value=0.001, max_value=0.5, value=0.10, format="%.3f"
        )
        peak_tol = v3.number_input(
            "Peak err", min_value=0.001, max_value=0.5, value=0.10, format="%.3f"
        )
        exec_profile = st.checkbox(
            "Capture installed execution profile (ready for `formamech compile`)",
            value=True,
        )

        if st.button(
            "Prepare Inputs & Preview Plan (No Training Yet)",
            type="primary",
            disabled=not ok_bc,
            width="stretch",
        ):
            try:
                draft = WorkbenchPhysicsDraft(
                    project_name=project_name.strip(),
                    material_model=mat_model,
                    material_param_map=param_map,
                    parameters=tuple(param_drafts),
                    boundaries=tuple(assignments),
                    optimizer_steps=int(opt_steps),
                    case_batch_size=int(batch_sz),
                    checkpoint_interval=int(ckpt_int),
                    seed=int(seed_val),
                    fit_cases=int(fit_cases),
                    screen_cases=int(screen_cases),
                    u_tol=float(u_tol),
                    p_tol=float(p_tol),
                    peak_tol=float(peak_tol),
                )
                prepare_and_plan_workspace(
                    workspace, draft, execution_profile=exec_profile
                )
                st.success(
                    "Prepared canonical `physics.json`, `inputs/`, and `preview/plan.json`."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Preparation failed: {exc}")

    plan = workspace.load_plan()
    if plan is not None:
        st.divider()
        st.success(
            f"Campaign plan verified (`plan_sha256`: `{str(plan.get('plan_sha256', ''))[:16]}...`). "
            "Proceed to Step 4 to train or qualify."
        )
        c_a, c_b = st.columns(2)
        with c_a.expander("Generated Canonical `physics.json` (read-only)"):
            if workspace.physics_path.is_file():
                st.json(json.loads(workspace.physics_path.read_text(encoding="utf-8")))
        with c_b.expander("Compiler `--plan-only` Artifact (`preview/plan.json`)"):
            st.json(plan)


def _render_stage_4_train_qualify(
    st: Any, workspace: WorkbenchWorkspace, default_bundle: str
) -> None:
    st.subheader("Step 4 — Background Compile & Continuous Model Qualification")
    runner = WorkbenchJobRunner(workspace.jobs_dir, cwd=workspace.root)
    prog = workspace.stage_progress()
    active_compile = runner.active_jobs(kind="compile")
    active_qualify = runner.active_jobs(kind="qualify")

    col_train, col_qual = st.columns(2, gap="large")
    with col_train:
        st.markdown("### A. Campaign Compiler (`formamech compile`)")
        if not prog.inputs_prepared:
            st.info("Prepare inputs in Step 3 before launching compilation.")
        else:
            if active_compile:
                st.info(
                    f"Compile job `{active_compile[0].job_id}` is currently RUNNING."
                )
            b1, b2 = st.columns(2)
            if b1.button(
                "Start Compile Job",
                type="primary",
                disabled=bool(active_compile),
                width="stretch",
            ):
                job = runner.start_compile(
                    spec_path=workspace.inputs_dir / "problem.json",
                    config_path=workspace.inputs_dir / "config.json",
                    output_dir=workspace.run_dir,
                    resume=False,
                )
                st.success(
                    f"Launched background compile job `{job.job_id}` (PID {job.pid})."
                )
                st.rerun()
            if b2.button(
                "Resume Compile (`--resume`)",
                disabled=bool(active_compile),
                width="stretch",
            ):
                job = runner.start_compile(
                    spec_path=workspace.inputs_dir / "problem.json",
                    config_path=workspace.inputs_dir / "config.json",
                    output_dir=workspace.run_dir,
                    resume=True,
                )
                st.success(f"Resumed compile job `{job.job_id}`.")
                st.rerun()

    with col_qual:
        st.markdown("### B. Numerical Qualification (`formamech qualify`)")
        candidate_default = (
            str(workspace.candidate_bundle_path)
            if workspace.candidate_bundle_path is not None
            else default_bundle
        )
        qual_bundle_key = f"wb_qual_bundle_path::{workspace.root}"
        if (
            qual_bundle_key in st.session_state
            and not str(st.session_state[qual_bundle_key]).strip()
            and candidate_default
        ):
            st.session_state[qual_bundle_key] = candidate_default
        qual_bundle_str = st.text_input(
            "Bundle to Qualify",
            value=candidate_default,
            key=qual_bundle_key,
        )
        q1, q2, q3 = st.columns(3)
        preset = q1.selectbox(
            "Policy Preset",
            options=["dev", "holdout"],
            index=0,
            key=f"wb_qual_preset::{workspace.root}",
        )
        q_backend = q2.selectbox(
            "FEM Backend",
            options=["torch-fem", "scikit-fem"],
            index=0,
            key=f"wb_qual_backend::{workspace.root}",
        )
        q_dev_n = q3.number_input(
            "DEV samples",
            min_value=3,
            max_value=100,
            value=9,
            key=f"wb_qual_dev_n::{workspace.root}",
        )

        if active_qualify:
            st.info(
                f"Qualification job `{active_qualify[0].job_id}` is currently RUNNING."
            )
        if st.button(
            "Run Qualification Pipeline",
            type="primary",
            disabled=not qual_bundle_str.strip() or bool(active_qualify),
            width="stretch",
        ):
            job = runner.start_qualify(
                bundle_path=qual_bundle_str.strip(),
                output_dir=workspace.qualification_dir,
                preset=preset,
                sample_dev=int(q_dev_n),
                sample_holdout=5 if preset == "holdout" else 0,
                backend=q_backend,
            )
            st.success(f"Launched qualification job `{job.job_id}` (PID {job.pid}).")
            st.rerun()

    # Live Job Monitor — poll all non-terminal jobs plus recent history
    jobs = runner.list_jobs()
    if jobs:
        st.divider()
        hdr_a, hdr_b = st.columns([3, 1])
        hdr_a.markdown("### Background Jobs")
        if hdr_b.button("Refresh Job Status", width="stretch"):
            st.rerun()
        refreshed_all = [
            runner.poll(j.job_id) if not j.is_terminal else j for j in jobs
        ]
        refreshed = refreshed_all[:6]
        st.table(
            [
                {
                    "Job ID": j.job_id,
                    "Kind": j.kind,
                    "State": j.state,
                    "Exit": j.exit_code if j.exit_code is not None else "—",
                    "Checkpoint": j.latest_checkpoint_step
                    if j.latest_checkpoint_step is not None
                    else "—",
                    "Started (UTC)": j.started_utc,
                }
                for j in refreshed
            ]
        )
        running_jobs = [j for j in refreshed_all if not j.is_terminal]
        for running in running_jobs:
            if st.button(
                f"Cancel `{running.job_id}` ({running.kind})",
                key=f"cancel_job::{running.job_id}",
            ):
                runner.cancel(running.job_id)
                st.rerun()
        latest = refreshed[0]
        log_tail = runner.read_log_tail(latest.job_id, max_lines=40)
        if log_tail:
            with st.expander(f"Log tail — `{latest.job_id}`"):
                st.code(log_tail, language="text")

    # Qualification Manifest Display
    mpath = workspace.qualification_manifest_path
    if mpath is not None:
        st.divider()
        raw_manifest = json.loads(mpath.read_text(encoding="utf-8"))
        verdict = str(raw_manifest.get("verdict", "NOT_QUALIFIED"))
        granted = str(raw_manifest.get("granted_claim_level", "NOT_VERIFIED"))
        k1, k2, k3 = st.columns(3)
        k1.metric("Qualification Verdict", verdict)
        k2.metric("Granted Claim Level", granted)
        k3.metric("Policy", str(raw_manifest.get("policy_id", "")))
        st.table(list(raw_manifest.get("families", ())))
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "Download Qualification Manifest (JSON)",
            data=mpath.read_bytes(),
            file_name="qualification-manifest.json",
            mime="application/json",
            width="stretch",
        )
        rpath = workspace.qualification_dir / "qualification-report.md"
        if rpath.is_file():
            dl2.download_button(
                "Download Qualification Dossier (Markdown)",
                data=rpath.read_bytes(),
                file_name="qualification-report.md",
                mime="text/markdown",
                width="stretch",
            )


def _render_stage_5_inspector(
    st: Any, workspace: WorkbenchWorkspace, default_bundle: str
) -> None:
    from piginot.bundle import load_bundle
    from piginot.export import export_bundle_to_fmu
    from piginot.qualify import check_deployment_gate

    st.subheader("Step 5 — Verification & Trust Inspector")
    active_default = (
        str(workspace.candidate_bundle_path)
        if workspace.candidate_bundle_path is not None
        else default_bundle
    )
    insp_bundle_key = f"wb_inspector_bundle::{workspace.root}"
    if (
        insp_bundle_key in st.session_state
        and not str(st.session_state[insp_bundle_key]).strip()
        and active_default
    ):
        st.session_state[insp_bundle_key] = active_default
    c_path, c_mode, c_back = st.columns([2.2, 1.0, 1.0])
    bundle_str = c_path.text_input(
        "Bundle Directory",
        value=active_default,
        help="Path to a FormaMech verified or compiled candidate bundle directory.",
        key=insp_bundle_key,
    )
    mode = c_mode.selectbox(
        "Verification Mode",
        options=["fast", "certified", "exact"],
        index=1,
        key=f"wb_insp_mode::{workspace.root}",
    )
    is_fast_mode = mode == "fast"
    backend = c_back.selectbox(
        "FEM Backend",
        options=["torch-fem", "scikit-fem"],
        index=0,
        disabled=is_fast_mode,
        key=f"wb_insp_backend::{workspace.root}",
    )

    o1, o2, o3, o4 = st.columns([1.2, 1.2, 1.4, 1.6])
    spectral_repair = (
        o1.checkbox(
            "Spectral Warm-Start Repair",
            value=not is_fast_mode,
            disabled=is_fast_mode,
            key=f"wb_insp_repair::{workspace.root}",
        )
        and not is_fast_mode
    )
    escalate = (
        o2.checkbox(
            "Auto-Escalate to FEM",
            value=False,
            disabled=mode != "certified",
            key=f"wb_insp_escalate::{workspace.root}",
        )
        and mode == "certified"
    )
    tta_steps = (
        0
        if is_fast_mode
        else int(
            o3.slider(
                "Tier 1.5 Physics LoRA Steps",
                0,
                50,
                0,
                5,
                key=f"wb_insp_tta::{workspace.root}",
            )
        )
    )
    qual_manifest_str = o4.text_input(
        "Qualification Manifest (optional)",
        value=(
            str(workspace.qualification_manifest_path)
            if workspace.qualification_manifest_path is not None
            else ""
        ),
        key=f"wb_insp_qual_manifest::{workspace.root}",
    )

    if not bundle_str.strip():
        st.info(
            "Enter a bundle directory (for example `bundles/notched-v1-cp-freeze`) "
            "or train a candidate in Step 4 to inspect predictions and trust receipts."
        )
        return

    bundle_path = Path(bundle_str.strip())
    if not bundle_path.exists():
        st.error(f"Bundle path does not exist: {bundle_path}")
        return

    try:
        bundle = load_bundle(bundle_path)
    except Exception as exc:
        st.error(f"Failed to load bundle `{bundle_path}`: {exc}")
        return

    gate_badge = "UNCHECKED"
    gate_refused = False
    if qual_manifest_str.strip():
        gate_res = check_deployment_gate(bundle_path, Path(qual_manifest_str.strip()))
        gate_refused = not gate_res.allowed
        gate_badge = (
            f"QUALIFIED ({gate_res.granted_claim_level.value})"
            if gate_res.allowed
            else f"REFUSED ({', '.join(c.value for c in gate_res.reason_codes)})"
        )

    st.markdown("**Envelope Parameter Controls**")
    slider_cols = st.columns(max(1, min(len(bundle.envelope.primitive_bounds), 4)))
    inputs: dict[str, dict[str, object]] = {}
    for idx, bound in enumerate(bundle.envelope.primitive_bounds):
        lo = float(bound.lower_bound.value)
        hi = float(bound.upper_bound.value)
        mid = 0.5 * (lo + hi)
        col = slider_cols[idx % len(slider_cols)]
        if hi <= lo:
            col.metric(
                f"{bound.name} (fixed)", f"{lo:.4g} {bound.lower_bound.unit.value}"
            )
            val = lo
        else:
            val = float(
                col.slider(
                    f"{bound.name} ({bound.lower_bound.unit.value})",
                    min_value=lo,
                    max_value=hi,
                    value=mid,
                    key=f"wb_param_{bundle.manifest.name}_{bound.name}",
                )
            )
        inputs[bound.name] = {"value": val, "unit": bound.lower_bound.unit.value}

    act1, act2 = st.columns([1.2, 1.2])
    one_click_exact = act1.button("Run Warm-Start FEM & Compare", type="primary")
    export_fmu_clicked = act2.button(
        "Export FMI 3.0 Co-Simulation FMU",
        disabled=gate_refused,
    )
    effective_mode = "exact" if one_click_exact else mode
    if gate_refused:
        st.warning(
            f"Deployment gate {gate_badge}: FMU export is blocked until qualification passes."
        )

    if export_fmu_clicked and not gate_refused:
        try:
            fmu_dest = workspace.root / f"{bundle.manifest.name}.fmu"
            fmu_res = export_bundle_to_fmu(bundle_path, fmu_dest)
            st.success(f"Exported FMI 3.0 FMU to `{fmu_res.fmu_path}`.")
        except Exception as exc:
            st.error(f"FMU export failed: {exc}")

    cal_default = workspace.qualification_dir / "cad-calibration.json"
    cad_cal_path: Path | None = None
    if cal_default.is_file():
        from piginot.bounds import CadCalibrationRecord

        try:
            CadCalibrationRecord.load(
                cal_default,
                expected_manifest_sha256=bundle.manifest.manifest_sha256,
            )
            cad_cal_path = cal_default
        except (OSError, ValueError):
            st.caption(
                "Workspace CAD calibration belongs to a different bundle; "
                "serving without workspace conformal bounds."
            )
    try:
        snapshot = build_inspector_snapshot(
            bundle_path,
            inputs,
            mode=effective_mode,
            cad_calibration_path=cad_cal_path,
            backend=backend,
            spectral_repair=spectral_repair if effective_mode != "fast" else False,
            tta_steps=tta_steps if effective_mode != "fast" else 0,
            escalate=escalate if effective_mode == "certified" else False,
        )
    except Exception as exc:
        st.error(f"Inspector evaluation failed: {exc}")
        return
    summary = snapshot.summary()

    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Verification Tier", snapshot.verification_tier)
    r1c2.metric("Serving Path", snapshot.serving_path)
    r1c3.metric("Qualification Gate", gate_badge)

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric(
        "Min det(F_2D)",
        f"{summary['min_det_f_2d']:.4f}"
        if summary["min_det_f_2d"] is not None
        else "N/A",
    )
    r2c2.metric(
        "Receipt Integrity", "VERIFIED" if snapshot.receipt_verified else "FAILED"
    )
    r2c3.metric("Claim Level", snapshot.claim_level)

    sub_fields, sub_verify, sub_receipt = st.tabs(
        [
            "Field & Admissibility Maps",
            "Verification, QoIs & FEM Diff",
            "Trust Receipt & Dossier",
        ]
    )
    figs = build_plotly_figures(snapshot)
    with sub_fields:
        col_a, col_b = st.columns(2)
        if "displacement_mm" in figs:
            col_a.plotly_chart(figs["displacement_mm"], width="stretch")
        if "von_mises_mpa" in figs:
            col_b.plotly_chart(figs["von_mises_mpa"], width="stretch")
        if "det_f_2d" in figs:
            st.plotly_chart(figs["det_f_2d"], width="stretch")

    with sub_verify:
        if "fem_diff_mm" in figs:
            st.plotly_chart(figs["fem_diff_mm"], width="stretch")
        elif "corrected_diff_mm" in figs:
            st.plotly_chart(figs["corrected_diff_mm"], width="stretch")
        if snapshot.qois:
            st.subheader("Quantities of Interest")
            st.table(list(snapshot.qois))
        if snapshot.certificate is not None:
            st.subheader("Physics Certificate")
            st.json(snapshot.certificate)
        if snapshot.trust_report is not None:
            st.subheader("CAD Trust Report")
            st.json(snapshot.trust_report)

    with sub_receipt:
        dossier_md = render_trust_dossier_markdown(snapshot)
        st.download_button(
            "Download Verification Dossier (Markdown)",
            data=dossier_md.encode("utf-8"),
            file_name=f"{snapshot.bundle_name}-trust-dossier.md",
            mime="text/markdown",
        )
        if snapshot.trust_receipt is not None:
            receipt_json = json.dumps(snapshot.trust_receipt, indent=2, sort_keys=True)
            st.download_button(
                "Download RFC 8785 Trust Receipt (JSON)",
                data=receipt_json.encode("utf-8"),
                file_name=f"{snapshot.bundle_name}-trust-receipt.json",
                mime="application/json",
            )
            st.json(snapshot.trust_receipt)


def run_streamlit_app(
    default_bundle: str = "",
    default_workspace: str = ".formamech-runs/workbench/demo-project",
) -> None:
    """Render the Interactive 2D Engineering Workbench & Trust Inspector."""
    st: Any = importlib.import_module("streamlit")

    st.set_page_config(
        page_title="FormaMech — Interactive Engineering Workbench",
        layout="wide",
    )
    st.title("FormaMech — Interactive Engineering Workbench")
    st.caption(
        "Physics-informed simulation intelligence with FEM-backed trust: "
        "CAD import, visual boundary picking, campaign planning, numerical "
        "qualification, and RFC 8785 Trust Receipts."
    )

    with st.sidebar:
        st.header("Workbench Session")
        ws_str = st.text_input(
            "Workspace Directory",
            value=default_workspace,
            help="Canonical project directory storing geometry/, physics.json, inputs/, run/, qualification/.",
        )
        workspace = WorkbenchWorkspace(Path(ws_str.strip() or default_workspace))
        progress = workspace.stage_progress()
        wf_snapshot = workspace.workflow_snapshot()

        st.subheader("Guided Workflow Progress")
        st.caption(f"Current stage: `{wf_snapshot.state.value}`")
        steps = [
            ("1. CAD & Mesh Imported", progress.cad_imported),
            ("2. Physics & Boundaries Saved", progress.physics_saved),
            ("3. Campaign Inputs Prepared", progress.inputs_prepared),
            ("4. Plan Previewed", progress.plan_previewed),
            ("5. Candidate Bundle Trained", progress.bundle_trained),
            ("6. Model Qualified", progress.bundle_qualified),
        ]
        for label, done in steps:
            st.markdown(f"{'[x]' if done else '[ ]'} **{label}**")
        if wf_snapshot.stale_items:
            for item in wf_snapshot.stale_items:
                st.warning(f"Stale: {item}")
        if wf_snapshot.blockers:
            for blocker in wf_snapshot.blockers:
                st.error(f"Blocker: {blocker}")
        if wf_snapshot.recommended_hint:
            st.info(f"Next: {wf_snapshot.recommended_hint}")

        st.divider()
        st.caption(
            "Epistemic scope (ASME V&V 10): Numerical verification against "
            "the declared discrete FEM model only; never experimental validation."
        )

    default_tab_names = [
        "1. Import CAD",
        "2. Boundary Picker",
        "3. Material & Plan",
        "4. Train & Qualify",
        "5. Trust Inspector",
    ]
    tabs = st.tabs(default_tab_names)
    with tabs[0]:
        _render_stage_1_import(st, workspace)
    with tabs[1]:
        _render_stage_2_boundaries(st, workspace)
    with tabs[2]:
        _render_stage_3_physics_plan(st, workspace)
    with tabs[3]:
        _render_stage_4_train_qualify(st, workspace, default_bundle)
    with tabs[4]:
        _render_stage_5_inspector(st, workspace, default_bundle)


def _parse_cli_args(argv: list[str]) -> tuple[str, str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("bundle", nargs="?", default="")
    parser.add_argument("--bundle", dest="bundle_opt", default="")
    parser.add_argument(
        "--workspace",
        dest="workspace",
        default=".formamech-runs/workbench/demo-project",
    )
    ns, _ = parser.parse_known_args(argv)
    bundle = str(ns.bundle_opt or ns.bundle or "")
    return bundle, str(ns.workspace)


if __name__ == "__main__":
    arg_bundle, arg_workspace = _parse_cli_args(sys.argv[1:])
    run_streamlit_app(default_bundle=arg_bundle, default_workspace=arg_workspace)
