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
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from piginot.contracts import Unit
from piginot.contracts.canonical import strict_json_loads
from piginot.qualify.gate import GateReasonCode, check_deployment_gate
from piginot.qualify.manifest import QualificationManifest
from piginot.ui.field_probes import render_field_maps
from piginot.ui.inspector import (
    build_inspector_snapshot,
    render_trust_dossier_markdown,
)
from piginot.ui.job_history import render_job_completion, render_job_monitor
from piginot.ui.workbench import (
    MATERIAL_PRESETS,
    BoundaryAssignment,
    ParameterSpecDraft,
    WorkbenchJobRunner,
    WorkbenchWorkspace,
    build_cad_boundary_figure,
    prepare_and_plan_workspace,
    validate_boundaries_against_preview,
)
from piginot.ui.workbench.navigation import (
    BOUNDARIES,
    IMPORT,
    INSPECT,
    PLAN,
    TRAIN,
    focus_destination,
    render_navigation,
    render_progress,
    stage_panel,
)
from piginot.ui.workbench.physics_forms import (
    load_applied_settings,
    load_physics_draft,
)
from piginot.ui.workbench.plan_review import plan_review_rows, prepared_physics
from piginot.ui.workbench.state import _file_sha256
from piginot.ui.workbench.study_plan import has_compatible_checkpoint, plan_blocker


def _load_or_init_boundaries(
    workspace: WorkbenchWorkspace, preview: dict[str, Any]
) -> list[BoundaryAssignment]:
    settings = load_applied_settings(workspace, preview)
    return [
        BoundaryAssignment.from_document(item)
        for item in settings["boundary_conditions"]
    ]


def _render_stage_1_import(st: Any, workspace: WorkbenchWorkspace) -> None:
    st.subheader(IMPORT.name, anchor=IMPORT.anchor)
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

        review = workspace.replacement_review()
        pending_key = "wb_cad_replacement"
        choices = (
            str(workspace.root),
            getattr(uploaded, "file_id", None),
            unit_str,
            mesh_size,
        )
        pending = st.session_state.get(pending_key)
        if pending is not None and pending[0] != choices:
            st.session_state.pop(pending_key)
            pending = None
        btn_a, btn_b = st.columns(2)
        upload_clicked = btn_a.button(
            "Import Uploaded CAD", type="primary", width="stretch"
        )
        sample_label = (
            "Open Sample in New Workspace"
            if review.populated
            else "Generate Sample 2D Bracket"
        )
        sample_clicked = btn_b.button(sample_label, width="stretch")
        expected = None
        if upload_clicked:
            if uploaded is None:
                st.warning("Upload a .step, .stp, or .dxf file first.")
            elif review.populated:
                pending = (choices, review, uuid.uuid4().hex)
                st.session_state[pending_key] = pending
            else:
                expected = review
        if pending is not None and uploaded is not None:
            previous = pending[1]
            st.markdown(
                f"**Replace {previous.part or 'geometry in the current study'}?**"
            )
            if previous.artifacts:
                st.markdown("\n".join(f"- {item}" for item in previous.artifacts))
            keep, replace = st.columns(2)
            keep.button(
                "Keep current study",
                width="stretch",
                key=f"keep_cad::{pending[2]}",
                on_click=st.session_state.pop,
                args=(pending_key, None),
            )
            if replace.button(
                "Replace geometry", width="stretch", key=f"replace_cad::{pending[2]}"
            ):
                st.session_state.pop(pending_key, None)
                expected = previous

        if expected is not None and uploaded is not None:
            try:
                with st.spinner("Meshing…"):
                    workspace.import_cad_bytes(
                        uploaded.getvalue(),
                        uploaded.name,
                        unit=Unit(unit_str),
                        mesh_size_mm=float(mesh_size),
                        isolated=True,
                        expected_study=expected,
                    )
                st.session_state[f"wb_import_notice::{workspace.root}"] = (
                    f"Imported and meshed {uploaded.name}."
                )
                st.rerun()
            except Exception as exc:
                st.error("CAD import failed. Check the diagnostic and retry.")
                with st.expander("Technical details"):
                    st.code(str(exc), language=None)

        if sample_clicked:
            try:
                with st.spinner("Meshing…"):
                    target = workspace.sample_workspace(review)
                    target.import_sample(
                        mesh_size_mm=float(mesh_size),
                        expected_study=(
                            review
                            if target is workspace
                            else target.replacement_review()
                        ),
                    )
                st.session_state["wb_open_workspace"] = str(target.root)
                st.session_state[f"wb_import_notice::{target.root}"] = (
                    "Generated and meshed sample 40 x 20 mm bracket STEP."
                )
                st.rerun()
            except Exception as exc:
                st.error(
                    "Sample generation failed. Check the meshing settings, then retry."
                )
                with st.expander("Technical details"):
                    st.code(str(exc), language=None)

        if notice := st.session_state.pop(f"wb_import_notice::{workspace.root}", None):
            st.success(notice)
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
            assignments = _load_or_init_boundaries(workspace, preview)
            fig = build_cad_boundary_figure(
                preview, assignments=assignments, show_mesh=True
            )
            st.plotly_chart(fig, width="stretch")


def _render_stage_2_boundaries(st: Any, workspace: WorkbenchWorkspace) -> None:
    st.subheader(BOUNDARIES.name, anchor=BOUNDARIES.anchor)
    if not (workspace.geometry_dir / "preview.json").is_file():
        st.info("Import a 2D profile in Import CAD.")
        return

    from piginot.ui.workbench.picker_component import render_boundary_picker

    render_boundary_picker(st, workspace)


def _render_stage_3_physics_plan(st: Any, workspace: WorkbenchWorkspace) -> bool:
    st.subheader(PLAN.name, anchor=PLAN.anchor)
    preview = workspace.load_preview()
    if preview is None:
        st.info("Import a 2D profile in Import CAD.")
        return False

    saved = load_physics_draft(workspace, preview)
    assignments = saved.boundaries
    ok_bc, err_bc = validate_boundaries_against_preview(preview, assignments)
    if not ok_bc:
        st.warning(
            "Fix the boundary assignments in Boundary Picker before preparing inputs."
        )
        for err in err_bc:
            st.error(err)

    saved_state = workspace.load_state()
    scope = f"{workspace.root}::{saved_state.get('geometry_revision', '')}"
    model_keys = list(MATERIAL_PRESETS.keys())
    if saved.material_model not in model_keys:
        model_keys.append(saved.material_model)
    saved_params = {p.name: p for p in saved.parameters}
    plan = workspace.load_plan()
    blocker = plan_blocker(workspace) if plan is not None else None
    if blocker:
        st.warning(blocker)

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        project_name = st.text_input(
            "Campaign / Model Name",
            value=saved.project_name,
            key=f"project_name::{scope}",
        )
        mat_model = st.selectbox(
            "Hyperelastic Material Model",
            options=model_keys,
            index=model_keys.index(saved.material_model),
            format_func=lambda k: (
                f"{MATERIAL_PRESETS[k].display_name} ({k})"
                if k in MATERIAL_PRESETS
                else k
            ),
            key=f"material_model::{scope}",
        )
        preset = MATERIAL_PRESETS.get(mat_model)
        if preset:
            st.caption(preset.description)

        st.markdown("**Material Parameter Bounds**")
        param_drafts: list[ParameterSpecDraft] = []
        same_material = mat_model == saved.material_model
        param_map = dict(saved.material_param_map) if same_material else {}
        defaults = (
            preset.parameters
            if preset
            else tuple(
                (
                    prop,
                    name,
                    saved_params[name].lo,
                    saved_params[name].hi,
                    saved_params[name].unit,
                )
                for prop, name in param_map.items()
                if name in saved_params
            )
        )
        for prop_name, pname, default_lo, default_hi, unit in defaults:
            if same_material:
                pname = param_map.get(prop_name, pname)
                if pname not in saved_params:
                    continue
                parameter = saved_params[pname]
                default_lo, default_hi, unit = (
                    parameter.lo,
                    parameter.hi,
                    parameter.unit,
                )
            c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
            c1.markdown(f"`{pname}` ({prop_name}, {unit})")
            lo_v = c2.number_input(
                f"{pname} min",
                value=float(default_lo),
                format="%.4f",
                key=f"mat_lo::{scope}::{mat_model}::{prop_name}",
            )
            hi_v = c3.number_input(
                f"{pname} max",
                value=float(default_hi),
                format="%.4f",
                key=f"mat_hi::{scope}::{mat_model}::{prop_name}",
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
        lc1, lc2, lc3 = st.columns([1.2, 1.0, 1.0])
        load = next((p for p in saved.parameters if p.role == "load"), None)
        edited_params = {p.name: p for p in param_drafts}
        if load is not None:
            load_name = lc1.text_input(
                "Load symbol", value=load.name, key=f"load_name::{scope}"
            )
            load_lo = lc2.number_input(
                f"Load min ({load.unit})",
                value=float(load.lo),
                format="%.4f",
                key=f"load_lo::{scope}",
            )
            load_hi = lc3.number_input(
                f"Load max ({load.unit})",
                value=float(load.hi),
                format="%.4f",
                key=f"load_hi::{scope}",
            )
            edited_params[load.name] = replace(
                load,
                name=load_name.strip() or load.name,
                lo=float(load_lo),
                hi=float(load_hi),
            )
        old_material_names = (
            set(saved.material_param_map.values()) if not same_material else set()
        )
        parameters = tuple(
            edited_params.pop(p.name, p)
            for p in saved.parameters
            if p.name not in old_material_names
        ) + tuple(edited_params.values())

    with col_right:
        st.markdown("**Training & Sampling Budget**")
        t1, t2 = st.columns(2)
        opt_steps = t1.number_input(
            "Optimizer steps",
            min_value=1,
            max_value=max(50000, saved.optimizer_steps),
            value=saved.optimizer_steps,
            key=f"steps::{scope}",
        )
        batch_sz = t2.number_input(
            "Case batch size",
            min_value=1,
            max_value=max(128, saved.case_batch_size),
            value=saved.case_batch_size,
            key=f"batch::{scope}",
        )
        t3, t4 = st.columns(2)
        fit_cases = t3.number_input(
            "TRAIN-fit cases",
            min_value=2,
            max_value=max(1024, saved.fit_cases),
            value=saved.fit_cases,
            key=f"fit::{scope}",
        )
        screen_cases = t4.number_input(
            "TRAIN-screen cases",
            min_value=0,
            max_value=max(256, saved.screen_cases),
            value=saved.screen_cases,
            key=f"screen::{scope}",
        )
        t5, t6 = st.columns(2)
        ckpt_int = t5.number_input(
            "Checkpoint interval",
            min_value=1,
            max_value=max(5000, saved.checkpoint_interval),
            value=saved.checkpoint_interval,
            key=f"checkpoint::{scope}",
        )
        seed_val = t6.number_input(
            "Random seed",
            min_value=0,
            max_value=max(1000000, saved.seed),
            value=saved.seed,
            key=f"seed::{scope}",
        )

        st.markdown("**Verification Tolerances**")
        v1, v2, v3 = st.columns(3)
        u_tol = v1.number_input(
            "u rel-L2",
            min_value=0.0,
            value=float(saved.u_tol),
            format="%.3f",
            key=f"u_tol::{scope}",
        )
        p_tol = v2.number_input(
            "P rel-L2",
            min_value=0.0,
            value=float(saved.p_tol),
            format="%.3f",
            key=f"p_tol::{scope}",
        )
        peak_tol = v3.number_input(
            "Peak err",
            min_value=0.0,
            value=float(saved.peak_tol),
            format="%.3f",
            key=f"peak_tol::{scope}",
        )
        prepared_profile = bool(saved_state.get("execution_profile", True))
        exec_profile = st.checkbox(
            "Capture installed execution profile (ready for `formamech compile`)",
            value=prepared_profile,
            key=f"profile::{scope}",
        )

        draft = replace(
            saved,
            project_name=project_name.strip(),
            material_model=mat_model,
            material_param_map=param_map,
            parameters=parameters,
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
    with st.container(key="wb-plan-review"):
        try:
            prepared = prepared_physics(workspace)
            st.subheader(
                "Initial plan summary"
                if prepared is None
                else "Changes since last prepared plan"
            )
            rows = plan_review_rows(prepared, draft)
            if prepared is None or exec_profile != prepared_profile:
                row = {"Area": "Preparation", "Setting": "Capture execution profile"}
                if prepared is not None:
                    row["Prepared"] = "Yes" if prepared_profile else "No"
                rows.append({**row, "Proposed": "Yes" if exec_profile else "No"})
            if rows:
                st.table(rows)
            else:
                st.write("No settings changes.")
        except (OSError, ValueError) as exc:
            st.warning(f"Plan comparison unavailable: {exc}")
    with st.container(key="wb-prepare-control"):
        prepare_clicked = st.button(
            "Update plan"
            if plan is not None
            else "Prepare Inputs & Preview Plan (No Training Yet)",
            type="primary",
            disabled=not ok_bc,
            width="stretch",
            key=f"prepare::{scope}",
        )
    if prepare_clicked:
        try:
            prepare_and_plan_workspace(workspace, draft, execution_profile=exec_profile)
            st.rerun()
        except Exception as exc:
            st.error(f"Preparation failed: {exc}")
            blocker = plan_blocker(workspace)

    if plan is not None:
        st.divider()
        if not blocker:
            st.success(
                f"Campaign plan verified (`plan_sha256`: `{str(plan.get('plan_sha256', ''))[:16]}...`). "
                "Open Train & Qualify to continue."
            )
        c_a, c_b = st.columns(2)
        with c_a.expander("Generated Canonical `physics.json` (read-only)"):
            if workspace.physics_path.is_file():
                st.json(json.loads(workspace.physics_path.read_text(encoding="utf-8")))
        with c_b.expander("Compiler `--plan-only` Artifact (`preview/plan.json`)"):
            st.json(plan)
    return draft != saved or exec_profile != prepared_profile


def _render_stage_4_train_qualify(
    st: Any, workspace: WorkbenchWorkspace, default_bundle: str, pending_edits: bool
) -> None:
    st.subheader(TRAIN.name, anchor=TRAIN.anchor)
    runner = WorkbenchJobRunner(workspace.jobs_dir, cwd=workspace.root)
    prog = workspace.stage_progress()
    active_compile = runner.active_jobs(kind="compile")
    active_qualify = runner.active_jobs(kind="qualify")

    col_train, col_qual = st.columns(2, gap="large")
    with col_train:
        st.markdown("### A. Campaign Compiler (`formamech compile`)")
        if not prog.inputs_prepared:
            st.info("Prepare inputs in Material & Plan before launching compilation.")
        else:
            blocker = plan_blocker(workspace)
            if blocker:
                st.warning(blocker)
            prepared_plan = workspace.load_plan() or {}
            if revision := prepared_plan.get("plan_sha256"):
                st.markdown(f"**Prepared revision** `{revision}`")
                if pending_edits and not blocker:
                    st.info(
                        "Material & Plan has pending edits. Start and Resume use the prepared revision."
                    )
            if active_compile:
                st.info(
                    f"Compile job `{active_compile[0].job_id}` is currently RUNNING."
                )
            b1, b2 = st.columns(2)
            disabled = bool(active_compile or blocker)
            start = b1.button(
                "Start Compile Job",
                type="primary",
                disabled=disabled,
                width="stretch",
                key="wb-start-compile",
            )
            resume = b2.button(
                "Resume Compile (`--resume`)",
                disabled=disabled or not has_compatible_checkpoint(workspace),
                width="stretch",
            )
            if start or resume:
                try:
                    job = runner.start_compile(
                        spec_path=workspace.inputs_dir / "problem.json",
                        config_path=workspace.inputs_dir / "config.json",
                        output_dir=workspace.run_dir,
                        resume=resume,
                    )
                    st.success(
                        f"Launched background compile job `{job.job_id}` (PID {job.pid})."
                    )
                    st.rerun()
                except (OSError, ValueError) as exc:
                    st.error(str(exc))

    with col_qual:
        st.subheader("Numerical qualification", anchor="numerical-qualification")
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
            try:
                job = runner.start_qualify(
                    bundle_path=qual_bundle_str.strip(),
                    output_dir=workspace.qualification_dir,
                    preset=preset,
                    sample_dev=int(q_dev_n),
                    sample_holdout=5 if preset == "holdout" else 0,
                    backend=q_backend,
                )
                st.success(
                    f"Launched qualification job `{job.job_id}` (PID {job.pid})."
                )
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(str(exc))

    _render_workspace_qualification(st, workspace)
    render_job_monitor(st, runner, workspace)


def _render_workspace_qualification(st: Any, workspace: WorkbenchWorkspace) -> None:
    mpath = workspace.qualification_manifest_path
    if mpath is None:
        return
    st.subheader("Workspace qualification evidence")
    try:
        manifest_bytes = mpath.read_bytes()
        raw = strict_json_loads(manifest_bytes)
        if not isinstance(raw, dict) or "qualification_manifest_sha256" not in raw:
            raise ValueError("Qualification manifest integrity evidence is missing.")
        manifest = QualificationManifest.from_document(raw)
    except (OSError, ValueError):
        st.warning(
            "Qualification evidence unreadable. Rerun qualification to replace it."
        )
        return
    st.write(f"**Evidence bundle** {manifest.bundle_name}")
    st.write(f"**Verdict** {manifest.verdict.value}")
    st.write(f"**Granted claim** {manifest.granted_claim_level.value}")
    st.write(f"**Policy** {manifest.policy_id}")
    if manifest.families:
        st.table(list(manifest.families))
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Download Qualification Manifest (JSON)",
        data=manifest_bytes,
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


def _render_bundle_export(
    st: Any,
    workspace: WorkbenchWorkspace,
    bundle_path: Path,
    manifest_path: Path | None,
) -> None:
    from piginot.export import export_bundle_to_fmu

    gate = check_deployment_gate(bundle_path, manifest_path)
    reasons = gate.reason_codes
    missing = GateReasonCode.MANIFEST_MISSING in reasons
    mismatch = GateReasonCode.BUNDLE_MISMATCH in reasons
    unchecked = manifest_path is None and missing
    if gate.allowed:
        status, action = "Qualified for this policy", "Export qualified FMU"
    elif unchecked:
        status, action = "Qualification evidence missing", "Export unchecked FMU"
    else:
        action = "Export FMU"
        if mismatch:
            status = "Evidence belongs to another bundle"
        elif missing:
            status = "Qualification evidence missing"
        elif GateReasonCode.MANIFEST_INVALID in reasons:
            status = "Qualification evidence unreadable"
        else:
            status = "Qualification refused"

    with st.container(key="wb-deployment-evidence"):
        st.subheader("Deployment evidence")
        st.markdown(f"**{status}**")
        st.markdown(f"**Selected bundle** {gate.bundle_name}")
        if gate.bundle_manifest_sha256 is not None:
            st.markdown(f"Bundle manifest SHA-256 `{gate.bundle_manifest_sha256}`")
        st.markdown(f"**Evaluation policy**  \n{gate.policy_id or 'Unavailable'}")
        st.markdown(f"Required claim `{gate.min_claim_level.value}`")
        grant = (
            "Not granted for this bundle"
            if mismatch
            else gate.granted_claim_level.value
        )
        st.markdown(f"**Granted claim**  \n{grant}")
        if not gate.allowed:
            st.caption(gate.detail)
        permitted = gate.allowed or unchecked
        if not permitted:
            st.markdown(
                "FMU export blocked. Choose applicable evidence or rerun qualification."
            )
        clicked = st.button(
            action,
            key=f"wb_export_fmu::{gate.bundle_manifest_sha256}::{action}",
            disabled=not permitted,
        )
        if clicked and permitted:
            try:
                current = check_deployment_gate(bundle_path, manifest_path)
                if current != gate:
                    st.warning(
                        "Qualification evidence changed. Review the decision before exporting."
                    )
                    return
                name = current.bundle_name
                if Path(name).name != name or name in (".", ".."):
                    raise ValueError(
                        "The selected bundle name cannot be used as an FMU filename."
                    )
                fmu_dest = workspace.root / f"{name}.fmu"
                with tempfile.TemporaryDirectory(
                    prefix=".fmu-", dir=workspace.root
                ) as staging:
                    result = export_bundle_to_fmu(
                        bundle_path, Path(staging) / fmu_dest.name
                    )
                    if (
                        result.bundle_manifest_sha256 != current.bundle_manifest_sha256
                        or check_deployment_gate(bundle_path, manifest_path) != current
                    ):
                        st.warning(
                            "Bundle or evidence changed during packaging. Review the decision before exporting."
                        )
                        return
                    result.fmu_path.replace(fmu_dest)
                st.success(f"Exported FMI 3.0 FMU to `{fmu_dest}`.")
            except Exception as exc:
                st.error(f"FMU export failed: {exc}")


def _render_stage_5_inspector(
    st: Any, workspace: WorkbenchWorkspace, default_bundle: str, *, active: bool
) -> None:
    from piginot.bundle import load_bundle

    st.subheader(INSPECT.name, anchor=INSPECT.anchor)
    seen_key = f"wb_inspector_seen::{workspace.root}"
    seen = st.session_state.get(seen_key, False)
    st.session_state[seen_key] = True
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
    mode_key = f"wb_insp_mode::{workspace.root}"
    mode = c_mode.selectbox(
        "Verification Mode",
        options=["fast", "certified", "exact"],
        index=1,
        key=mode_key,
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
            "or train a candidate in Train & Qualify to inspect predictions and trust receipts."
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

    def show_exact() -> None:
        st.session_state[mode_key] = "exact"

    one_click_exact = st.button(
        "Run Warm-Start FEM & Compare", type="primary", on_click=show_exact
    )
    _render_bundle_export(
        st,
        workspace,
        bundle_path,
        Path(qual_manifest_str.strip()) if qual_manifest_str.strip() else None,
    )
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
        request = (
            bundle_path.resolve(),
            bundle.manifest.manifest_sha256,
            inputs,
            mode,
            backend,
            spectral_repair,
            tta_steps,
            escalate,
            cad_cal_path,
            _file_sha256(cad_cal_path) if cad_cal_path is not None else None,
        )
        cached = st.session_state.get("wb_inspector_snapshot")
        if cached is not None and cached[0] == request and not one_click_exact:
            snapshot = cached[1]
        else:
            deferred_key = f"wb_inspector_deferred::{workspace.root}"
            if (seen and not active) or st.session_state.get(deferred_key) == request:
                st.session_state[deferred_key] = request
                if not one_click_exact and not st.button(
                    "Evaluate selected bundle", disabled=not active
                ):
                    st.info(
                        "Evaluate the selected bundle to refresh the Inspector fields."
                    )
                    return
            st.session_state.pop(deferred_key, None)
            snapshot = build_inspector_snapshot(
                bundle_path,
                inputs,
                mode=mode,
                cad_calibration_path=cad_cal_path,
                backend=backend,
                spectral_repair=spectral_repair,
                tta_steps=tta_steps,
                escalate=escalate,
            )
            st.session_state["wb_inspector_snapshot"] = (request, snapshot)
    except Exception as exc:
        st.error(f"Inspector evaluation failed: {exc}")
        with st.container(key="wb-receipt-integrity"):
            st.markdown("**Receipt integrity**")
            st.write("Unavailable")
        return
    with st.container(key="wb-receipt-integrity"):
        st.markdown("**Receipt integrity**")
        st.write(snapshot.receipt_integrity)
        if snapshot.receipt_integrity == "Invalid":
            st.caption(". ".join(snapshot.receipt_errors))

    r1c1, r1c2 = st.columns(2)
    r1c1.metric("Verification Tier", snapshot.verification_tier)
    r1c2.metric("Serving Path", snapshot.serving_path)

    st.markdown(f"**Evaluation claim** {snapshot.claim_level}")

    sub_fields, sub_verify, sub_receipt = st.tabs(
        [
            "Field & Admissibility Maps",
            "Verification, QoIs & FEM Diff",
            "Trust Receipt & Dossier",
        ]
    )
    with sub_fields:
        figs = render_field_maps(
            st, snapshot, f"{workspace.root}::{bundle.manifest.manifest_sha256}"
        )

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


WORKFLOW_MODES: tuple[str, ...] = (
    "2D Fixed CAD",
    "3D Solid / Assembly",
    "Parametric Explore & Optimize",
    "Stateful Cyclic",
    "CAE Ingest & V&V 10",
)


def _render_3d_solid_mode(st: Any, workspace_root: Path) -> None:
    import numpy as np

    from piginot.ui.workbench_3d import Workbench3DWorkspace, render_viewer_3d

    st.title("3D Solid / Assembly Interactive Workbench")
    ws3 = Workbench3DWorkspace(workspace_root)
    c1, c2 = st.columns([1.0, 2.0], gap="large")
    with c1:
        mesh_size = float(
            st.number_input(
                "3D Mesh Size (mm)",
                min_value=1.0,
                max_value=50.0,
                value=6.0,
                step=1.0,
                key="wb3d_mesh_size",
            )
        )
        uploaded = st.file_uploader(
            "Upload 3D STEP solid / assembly",
            type=["step", "stp"],
            key="wb3d_step_uploader",
        )
        if (
            st.button("Import Uploaded 3D STEP", type="primary")
            and uploaded is not None
        ):
            tmp_step = ws3.root / "geometry_3d" / uploaded.name
            tmp_step.parent.mkdir(parents=True, exist_ok=True)
            tmp_step.write_bytes(uploaded.getvalue())
            ws3.import_step_path(tmp_step, mesh_size_mm=mesh_size)
            st.rerun()
        if st.button("Generate Sample 3D Box"):
            ws3.import_sample_box(mesh_size_mm=mesh_size)
            st.rerun()

        e_mpa = float(
            st.number_input(
                "Young's Modulus E (MPa)",
                min_value=1.0,
                max_value=10_000.0,
                value=100.0,
                key="wb3d_e_mpa",
            )
        )
        nu = float(
            st.number_input(
                "Poisson Ratio nu",
                min_value=0.0,
                max_value=0.49,
                value=0.30,
                key="wb3d_nu",
            )
        )
        if ws3.load_preview() is not None and st.button(
            "Solve 3D Finite-Strain FEM", type="primary"
        ):
            res = ws3.solve_preview_fem(youngs_modulus_mpa=e_mpa, poisson_ratio=nu)
            st.session_state["wb3d_last_fem"] = res
            st.success(
                f"Converged={res.converged} · max |u|={res.max_displacement_mm:.4f} mm · "
                f"peak von Mises={res.peak_von_mises_mpa:.2f} MPa"
            )

    with c2:
        if ws3.load_preview() is None:
            st.info(
                "Upload a 3D STEP part or click **Generate Sample 3D Box** to open the "
                "interactive WebGL 3D face picker and field explorer."
            )
        else:
            last_fem = st.session_state.get("wb3d_last_fem")
            disps = None
            fields = None
            if last_fem is not None and last_fem.converged:
                u_arr = np.asarray(last_fem.nodal_displacement, dtype=np.float64)
                disps = u_arr.tolist()
                u_mag = np.linalg.norm(u_arr, axis=1)
                fields = {
                    "von_mises_mpa": {
                        "name": "von_mises_mpa",
                        "label": "von Mises Stress",
                        "unit": "MPa",
                        "values": [float(v) for v in last_fem.nodal_von_mises_mpa],
                    },
                    "displacement_mag_mm": {
                        "name": "displacement_mag_mm",
                        "label": "Displacement |u|",
                        "unit": "mm",
                        "values": [float(v) for v in u_mag],
                    },
                }
            render_viewer_3d(st, ws3, displacements_mm=disps, fields=fields)


def run_streamlit_app(
    default_bundle: str = "",
    default_workspace: str = ".formamech-runs/workbench/demo-project",
    default_mode: str = "2D Fixed CAD",
) -> None:
    """Render the Interactive Engineering Workbench & Trust Inspector."""
    st: Any = importlib.import_module("streamlit")
    # Plotly's optional probe can see an incomplete pandas import from another session.
    importlib.import_module("pandas")

    st.set_page_config(
        page_title="FormaMech — Interactive Engineering Workbench",
        layout="wide",
    )
    with st.sidebar:
        if opened := st.session_state.pop("wb_open_workspace", None):
            st.session_state["wb_workspace_path"] = opened
        if "wb_workspace_path" not in st.session_state:
            st.session_state["wb_workspace_path"] = default_workspace
        if "wb_workflow_mode" not in st.session_state:
            st.session_state["wb_workflow_mode"] = (
                default_mode if default_mode in WORKFLOW_MODES else WORKFLOW_MODES[0]
            )
        with st.expander("Workspace details"):
            ws_str = st.text_input(
                "Workspace Directory",
                key="wb_workspace_path",
                help="Canonical project directory storing geometry/, physics.json, inputs/, run/, qualification/.",
            )
            workspace = WorkbenchWorkspace(Path(ws_str.strip() or default_workspace))
            wf_snapshot = workspace.workflow_snapshot()
            for item in wf_snapshot.stale_items:
                st.warning(f"Stale: {item}")
            for blocker in wf_snapshot.blockers:
                st.error(f"Blocker: {blocker}")
        workflow_mode = st.selectbox(
            "Workflow mode",
            options=list(WORKFLOW_MODES),
            key="wb_workflow_mode",
        )
        if workflow_mode == "2D Fixed CAD":
            render_progress(st, wf_snapshot)

        st.divider()
        st.caption(
            "Epistemic scope (ASME V&V 10): Numerical verification against "
            "the declared discrete FEM model only unless an experimental "
            "V&V 10 dossier is attached."
        )

    if workflow_mode == "3D Solid / Assembly":
        _render_3d_solid_mode(st, workspace.root)
        return
    if workflow_mode == "Parametric Explore & Optimize":
        from piginot.ui.workbench_3d import render_parametric_explorer_mode

        render_parametric_explorer_mode(st, workspace.root)
        return
    if workflow_mode == "Stateful Cyclic":
        from piginot.ui.workbench_3d import render_stateful_cyclic_mode

        render_stateful_cyclic_mode(st, workspace.root)
        return
    if workflow_mode == "CAE Ingest & V&V 10":
        from piginot.ui.workbench_3d import render_cae_vv10_mode

        render_cae_vv10_mode(st, workspace.root)
        return

    active_stage = render_navigation(st, workspace, wf_snapshot)
    render_job_completion(st, workspace)
    # Reserve a fixed-position container so focus requests fire immediately after
    # the active stage panel renders, rather than waiting behind hidden stages or
    # background-job fragment reruns.
    focus_slot = st.container(key="wb-navigation-focus")
    with stage_panel(st, IMPORT):
        _render_stage_1_import(st, workspace)
    if active_stage == IMPORT:
        focus_destination(st, container=focus_slot)
    with stage_panel(st, BOUNDARIES):
        _render_stage_2_boundaries(st, workspace)
    if active_stage == BOUNDARIES:
        focus_destination(st, container=focus_slot)
    with stage_panel(st, PLAN):
        pending_edits = _render_stage_3_physics_plan(st, workspace)
    if active_stage == PLAN:
        focus_destination(st, container=focus_slot)
    with stage_panel(st, TRAIN):
        _render_stage_4_train_qualify(st, workspace, default_bundle, pending_edits)
    if active_stage == TRAIN:
        focus_destination(st, container=focus_slot)
    with stage_panel(st, INSPECT):
        _render_stage_5_inspector(
            st, workspace, default_bundle, active=active_stage == INSPECT
        )
    if active_stage == INSPECT:
        focus_destination(st, container=focus_slot)


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
