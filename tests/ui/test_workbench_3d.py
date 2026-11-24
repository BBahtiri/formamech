"""Verification tests for Phase 5 (V3-5) 3D Interactive WebGL Workbench & Multi-Workflow Studio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from piginot.cad.parametric_3d import (
    default_holed_block_3d_spec,
    sample_parametric_cad_3d_doe,
)
from piginot.export.vtu_export import export_fields_to_vtu
from piginot.geometry.mesh3d import CellType3D
from piginot.training.parametric_3d_engine import (
    Parametric3DPhysicsConfig,
    Parametric3DTrainingConfig,
    save_parametric_3d_bundle,
    train_parametric_3d_operator,
)
from piginot.ui.workbench_3d import (
    VIEWER_3D_PREVIEW_SCHEMA,
    Workbench3DWorkspace,
    evaluate_parametric_3d_candidate,
    run_cae_ingest_summary,
    run_stateful_3d_cyclic_preview,
    run_vv10_calibration_summary,
)


def test_workbench_3d_import_face_bc_concurrency_and_fem_preview(
    tmp_path: Path,
) -> None:
    ws = Workbench3DWorkspace(tmp_path / "ws3d")
    preview = ws.import_sample_box(lx=20.0, ly=8.0, lz=8.0, mesh_size_mm=7.0)
    assert preview["schema"] == VIEWER_3D_PREVIEW_SCHEMA
    assert preview["node_count"] > 10
    assert len(preview["face_groups"]) == 6
    assert ws.preview_path.is_file()
    assert ws.mesh_path.is_file()
    assert ws.state_path.is_file()

    # Invalid extension / missing file fail closed
    with pytest.raises(FileNotFoundError):
        ws.import_step_path(tmp_path / "missing.step")
    bad_file = tmp_path / "bad.stl"
    bad_file.write_text("solid", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported 3D CAD extension"):
        ws.import_step_path(bad_file)

    payload = ws.viewer_payload()
    assert payload["workspace_id"] == ws.workspace_id
    assert payload["readiness_errors"] == []
    assert len(payload["assignments"]) == 6

    # Apply valid face edit -> accepted + revision changes + deduplicates replay
    loaded = next(a for a in payload["assignments"] if a["kind"] == "dirichlet")
    edited = {**loaded, "values_mm": [0.35]}
    ev1 = {
        "workspace_id": payload["workspace_id"],
        "geometry_id": payload["geometry_id"],
        "draft_revision": payload["draft_revision"],
        "event_id": "ev-3d-1",
        "face_tag": loaded["face_tag"],
        "assignment": edited,
    }
    ack1 = ws.apply_face_event(ev1)
    assert ack1["status"] == "accepted"
    assert ack1["draft_revision"] != payload["draft_revision"]
    # Duplicate event id is idempotent
    saved_bytes = ws.state_path.read_bytes()
    assert ws.apply_face_event(ev1) == ack1
    assert ws.state_path.read_bytes() == saved_bytes

    # Stale revision rejected without mutating workspace
    stale_ev = {**ev1, "event_id": "ev-3d-stale"}
    stale_ack = ws.apply_face_event(stale_ev)
    assert stale_ack["status"] == "stale"
    assert stale_ack["reload"] is True
    assert ws.state_path.read_bytes() == saved_bytes

    # Invalid BC name rejected without mutating workspace
    cur_payload = ws.viewer_payload()
    bad_ev = {
        "workspace_id": cur_payload["workspace_id"],
        "geometry_id": cur_payload["geometry_id"],
        "draft_revision": cur_payload["draft_revision"],
        "event_id": "ev-3d-bad",
        "face_tag": loaded["face_tag"],
        "assignment": {**edited, "name": "123 invalid"},
    }
    bad_ack = ws.apply_face_event(bad_ev)
    assert bad_ack["status"] == "invalid"
    assert "name" in bad_ack["errors"]
    assert ws.state_path.read_bytes() == saved_bytes

    # Solve authoritative 3D FEM preview directly from persisted workspace state
    res = ws.solve_preview_fem(
        youngs_modulus_mpa=100.0, poisson_ratio=0.30, load_steps=2
    )
    assert res.converged is True
    assert res.max_displacement_mm >= 0.34
    assert res.min_det_f > 0.0

    # Payload with field overlays ready for WebGL viewer
    field_payload = ws.viewer_payload(
        displacements_mm=res.nodal_displacement.tolist(),
        fields={
            "von_mises_mpa": {
                "name": "von_mises_mpa",
                "label": "von Mises Stress",
                "unit": "MPa",
                "values": res.nodal_von_mises_mpa.tolist(),
            }
        },
    )
    assert "von_mises_mpa" in field_payload["fields"]
    assert len(field_payload["displacements_mm"]) == preview["node_count"]


def test_parametric_3d_explorer_stateful_and_cae_vv10_controllers(
    tmp_path: Path,
) -> None:
    # 1) Parametric 3D explorer with OOD guard + warm-start FEM check
    spec = default_holed_block_3d_spec(mesh_size_mm=8.0)
    physics = Parametric3DPhysicsConfig(
        youngs_modulus_mpa=15.0,
        poisson_ratio=0.35,
        traction_mpa=(0.30, 0.0, 0.0),
        load_steps=2,
    )
    doe = sample_parametric_cad_3d_doe(spec, count=3, seed_offset=1)
    bundle = train_parametric_3d_operator(
        spec,
        physics,
        doe.instances,
        training_config=Parametric3DTrainingConfig(
            epochs=4,
            use_fem_supervision=False,
            quadrature_subsample=32,
            seed=11,
        ),
    )
    bdir = tmp_path / "p3d_bundle"
    save_parametric_3d_bundle(bundle, bdir)

    in_params = dict(spec.nominal_parameters())
    snap_ok = evaluate_parametric_3d_candidate(bdir, in_params, verify_with_fem=True)
    assert snap_ok.refused is False
    assert snap_ok.max_displacement_mm is not None and snap_ok.max_displacement_mm > 0.0
    assert snap_ok.fem_verified is True
    assert snap_ok.fem_rel_l2_error is not None

    # Out-of-bounds / topology OOD geometry -> refused
    ood_params = {**in_params, "lx": 120.0}
    snap_ood = evaluate_parametric_3d_candidate(bdir, ood_params, verify_with_fem=False)
    assert snap_ood.refused is True

    # 2) Stateful 3D cyclic preview controller
    _, cyc = run_stateful_3d_cyclic_preview(
        tmp_path / "cyc", ux_sequence_mm=(0.05, 0.20, 0.05), model_kind="j2_combined"
    )
    assert cyc.all_converged is True
    assert cyc.min_dissipation >= -1.0e-9
    assert len(cyc.step_table) == 3

    # 3) CAE archive screening & ASME V&V 10 summary controllers
    ws3 = Workbench3DWorkspace(tmp_path / "arch_src")
    ws3.import_sample_box(mesh_size_mm=8.0, element_type=CellType3D.TET4)
    mesh = ws3.load_mesh()
    assert mesh is not None
    arch_dir = tmp_path / "cae_archive"
    arch_dir.mkdir()
    for i in range(4):
        u = np.zeros((len(mesh.nodes), 3), dtype=np.float64)
        u[:, 0] = 0.01 * (i + 1) * np.asarray(mesh.nodes)[:, 0]
        export_fields_to_vtu(
            mesh, arch_dir / f"run_{i}.vtu", point_fields={"displacement_mm": u}
        )
    ingest_doc = run_cae_ingest_summary(arch_dir)
    assert ingest_doc["total_samples"] == 4
    assert ingest_doc["accepted_count"] == 4

    # Synthetic incompressible Neo-Hookean coupon JSON
    mu = 4.0
    lams = [1.05, 1.15, 1.30, 1.50]
    p11 = [mu * (lam - 1.0 / (lam * lam)) for lam in lams]
    ds_json = {
        "model_family": "incompressible_neo_hookean",
        "coupons": [
            {
                "specimen_id": f"c{i}",
                "role": "VALIDATION_HOLDOUT" if i == 3 else "CALIBRATION",
                "test_mode": "uniaxial_tension",
                "stretches_or_strains": lams,
                "measured_stress_mpa": [s * (1.0 + 0.002 * (i - 1)) for s in p11],
                "measurement_std_mpa": 0.10,
            }
            for i in range(4)
        ],
    }
    ds_path = tmp_path / "coupons.json"
    ds_path.write_text(json.dumps(ds_json), encoding="utf-8")
    vv_doc = run_vv10_calibration_summary(ds_path, u_num_mpa=0.05)
    assert vv_doc["experimental_validation_tier"] == "HOLDOUT_EXPERIMENTALLY_VALIDATED"
    assert vv_doc["passed"] is True


def test_streamlit_apptest_workflow_modes_render_cleanly(
    tmp_path: Path, workbench_apptest
) -> None:
    for mode in (
        "3D Solid / Assembly",
        "Parametric Explore & Optimize",
        "Stateful Cyclic",
        "CAE Ingest & V&V 10",
    ):
        script = (
            "from piginot.ui.app import run_streamlit_app\n"
            f"run_streamlit_app(default_workspace={str(tmp_path / 'ws')!r}, "
            f"default_mode={mode!r})\n"
        )
        at = workbench_apptest.from_string(script, default_timeout=60)
        at.run()
        assert not at.exception, (
            f"Mode {mode} raised: {[e.value for e in at.exception]}"
        )


def test_workbench_3d_worker_thread_gmsh_validation_contact_and_multi_solver_discovery(
    tmp_path: Path,
) -> None:
    import threading

    from piginot.ui.workbench_3d import FaceAssignment3D, validate_face_assignments_3d
    from piginot.ui.workbench_3d.vv10_panel import discover_cae_samples

    # 1) Gmsh 3D STEP generation + import works cleanly inside a non-main worker thread (Streamlit)
    worker_err: list[BaseException] = []
    worker_nodes: list[int] = []

    def _run_in_worker() -> None:
        try:
            ws_t = Workbench3DWorkspace(tmp_path / "ws_thread")
            prev = ws_t.import_sample_box(mesh_size_mm=8.0)
            worker_nodes.append(int(prev["node_count"]))
        except BaseException as exc:  # pragma: no cover - regression guard
            worker_err.append(exc)

    th = threading.Thread(target=_run_in_worker)
    th.start()
    th.join(timeout=30.0)
    assert not worker_err, f"Worker thread Gmsh import failed: {worker_err[0]}"
    assert worker_nodes and worker_nodes[0] > 10

    # 2) Invalid/duplicate Dirichlet components and contact fail-closed in preview FEM
    ok_bad_comp, errs_bad_comp = validate_face_assignments_3d(
        ["f1", "f2"],
        [
            FaceAssignment3D("f1", "fix1", "fixed", ("x", "y", "z"), (0.0, 0.0, 0.0)),
            FaceAssignment3D("f2", "pull2", "dirichlet", ("w",), (0.2,)),
        ],
    )
    assert not ok_bad_comp
    assert any("unsupported components" in e for e in errs_bad_comp)

    ok_dup_comp, errs_dup_comp = validate_face_assignments_3d(
        ["f1", "f2"],
        [
            FaceAssignment3D("f1", "fix1", "fixed", ("x", "y", "z"), (0.0, 0.0, 0.0)),
            FaceAssignment3D("f2", "pull2", "dirichlet", ("x", "x"), (0.2, 0.1)),
        ],
    )
    assert not ok_dup_comp
    assert any("duplicate components" in e for e in errs_dup_comp)

    ws = Workbench3DWorkspace(tmp_path / "ws_contact")
    ws.import_sample_box(mesh_size_mm=8.0)
    payload = ws.viewer_payload()
    loaded = next(a for a in payload["assignments"] if a["kind"] == "dirichlet")
    bad_comp_ack = ws.apply_face_event(
        {
            "workspace_id": payload["workspace_id"],
            "geometry_id": payload["geometry_id"],
            "draft_revision": payload["draft_revision"],
            "event_id": "ev-bad-comp",
            "face_tag": loaded["face_tag"],
            "assignment": {**loaded, "components": ["z", "z"], "values_mm": [0.1, 0.2]},
        }
    )
    assert bad_comp_ack["status"] == "invalid"
    assert "components" in bad_comp_ack["errors"]

    free_face = next(a for a in payload["assignments"] if a["kind"] == "traction_free")
    ack_contact = ws.apply_face_event(
        {
            "workspace_id": payload["workspace_id"],
            "geometry_id": payload["geometry_id"],
            "draft_revision": payload["draft_revision"],
            "event_id": "ev-contact-1",
            "face_tag": free_face["face_tag"],
            "assignment": {
                **free_face,
                "kind": "contact",
                "components": [],
                "values_mm": [],
            },
        }
    )
    assert ack_contact["status"] == "accepted"
    with pytest.raises(ValueError, match="ContactSpec3D"):
        ws.solve_preview_fem(load_steps=1)

    # 3) Multi-solver CAE archive discovery supports Abaqus .inp+.csv and Nastran .bdf+.csv
    #    without double-parsing CalculiX .inp+.frd pairs
    arch = tmp_path / "multi_solver_arch"
    arch.mkdir()
    tet_inp = (
        "*NODE\n"
        "1, 0.0, 0.0, 0.0\n2, 1.0, 0.0, 0.0\n3, 0.0, 1.0, 0.0\n4, 0.0, 0.0, 1.0\n"
        "*ELEMENT, TYPE=C3D4\n1, 1, 2, 3, 4\n"
    )
    (arch / "abq_run.inp").write_text(tet_inp, encoding="utf-8")
    (arch / "abq_run.csv").write_text(
        "node_id,ux,uy,uz\n1,0,0,0\n2,0.01,0,0\n3,0,0,0\n4,0,0,0\n",
        encoding="utf-8",
    )
    (arch / "nas_run.bdf").write_text(
        "GRID,1,,0.0,0.0,0.0\nGRID,2,,1.0,0.0,0.0\nGRID,3,,0.0,1.0,0.0\nGRID,4,,0.0,0.0,1.0\n"
        "CTETRA,1,1,1,2,3,4\n",
        encoding="utf-8",
    )
    (arch / "nas_run.csv").write_text(
        "node_id,ux,uy,uz\n1,0,0,0\n2,0.02,0,0\n3,0,0,0\n4,0,0,0\n",
        encoding="utf-8",
    )
    (arch / "ccx_run.inp").write_text(tet_inp, encoding="utf-8")
    (arch / "ccx_run.frd").write_text(
        " -4  DISP        4    1\n"
        " -1         1 0.00000E+00 0.00000E+00 0.00000E+00\n"
        " -1         2 1.50000E-02 0.00000E+00 0.00000E+00\n"
        " -1         3 0.00000E+00 0.00000E+00 0.00000E+00\n"
        " -1         4 0.00000E+00 0.00000E+00 0.00000E+00\n"
        " -3\n",
        encoding="utf-8",
    )
    samples = discover_cae_samples(arch)
    by_id = {s.sample_id: s.source_format.value for s in samples}
    assert by_id == {
        "abq_run": "abaqus-inp",
        "ccx_run": "calculix-frd",
        "nas_run": "nastran-bdf",
    }
