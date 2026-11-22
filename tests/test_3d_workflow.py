"""End-to-end M5 tests for 3D STEP CAD import, dual nonlinear FEM backends, and 3D Surface3DEnergyOperator."""

from __future__ import annotations

import json
import math
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pytest
import torch

from piginot.cad._errors import CadImportError
from piginot.cad.importing_3d import (
    CAD_3D_PROJECT_SCHEMA,
    classify_faces_by_bbox,
    generate_3d_box_step,
    generate_3d_l_bracket_step,
    generate_3d_mounting_lug_step,
    import_cad_3d,
    inspect_geometry_3d,
    load_geometry_3d,
    prepare_geometry_3d,
)
from piginot.cli import main as cli_main
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import (
    DirichletBC3D,
    FemRequest3D,
    ScikitFemBackend3D,
    TorchFemBackend3D,
    TractionBC3D,
    analytical_3d_uniaxial_hyperelastic,
    assemble_surface_tractions_3d,
    face_node_indices_3d,
)
from piginot.geometry.mesh3d import CellType3D, VolumetricLockingError
from piginot.operators.surface_operator_3d import (
    Surface3DConfig,
    Surface3DEnergyOperator,
)
from piginot.training.energy_engine_3d import (
    SURFACE_3D_BUNDLE_SCHEMA,
    SURFACE_3D_QUALIFICATION_SCHEMA,
    LoadCase3D,
    PlanarClampLifting3D,
    SoftMinFaceLifting3D,
    Surface3DTrainingConfig,
    build_surface_3d_tensors,
    compute_3d_break_even,
    lifted_displacement_and_gradient,
    lifting_from_document,
    load_surface_3d_bundle,
    predict_3d_bundle,
    qualify_3d_bundle,
    save_surface_3d_bundle,
    train_surface_3d_operator,
)


def _neo_hookean(youngs_mpa: float = 10.0, poisson: float = 0.35) -> MaterialSpec:
    return MaterialSpec(
        name=f"nh-E{youngs_mpa:g}-nu{poisson:g}",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(youngs_mpa, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(poisson, Unit.ONE).to_canonical(),
        },
    )


# ---------------------------------------------------------------------------
# 1. 3D CAD Import, Meshing, Locking Refusal & Project Persistence
# ---------------------------------------------------------------------------


def test_3d_step_import_persistence_tamper_and_locking_refusal(tmp_path: Path) -> None:
    box_step = generate_3d_box_step(tmp_path / "box.step", lx=20.0, ly=10.0, lz=10.0)
    imported = import_cad_3d(box_step, mesh_size_mm=6.0, element_type=CellType3D.TET10)
    assert imported.mesh.cell_type is CellType3D.TET10
    assert imported.mesh.unit is Unit.MM
    assert len(imported.faces) == 6
    assert abs(imported.volume_mm3 - 2000.0) < 1.0e-5
    assert abs(imported.surface_area_mm2 - 1000.0) < 1.0e-5

    classified = classify_faces_by_bbox(imported)
    for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        assert len(classified[key]) == 1

    # TET4 locking refusal for nearly incompressible nu >= 0.45
    with pytest.raises(VolumetricLockingError):
        import_cad_3d(
            box_step,
            mesh_size_mm=6.0,
            element_type=CellType3D.TET4,
            poisson_ratio=0.49,
        )

    # TET10 succeeds cleanly at nu = 0.49
    inc_tet10 = import_cad_3d(
        box_step,
        mesh_size_mm=8.0,
        element_type=CellType3D.TET10,
        poisson_ratio=0.49,
    )
    assert inc_tet10.mesh.cell_type is CellType3D.TET10

    # Prepare, inspect, reload, and tamper-check 3D geometry project
    proj_dir = tmp_path / "box_proj"
    summary = prepare_geometry_3d(box_step, proj_dir, mesh_size_mm=7.0)
    assert summary["schema"] == CAD_3D_PROJECT_SCHEMA
    inspected = inspect_geometry_3d(proj_dir)
    assert inspected["mesh_id"] == summary["mesh_id"]
    reloaded = load_geometry_3d(proj_dir)
    assert reloaded.mesh_id == summary["mesh_id"]
    assert reloaded.cell_type is CellType3D.TET10

    # Tamper mesh3d.json -> ValueError
    mesh_file = proj_dir / "mesh3d.json"
    mesh_file.write_bytes(mesh_file.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        inspect_geometry_3d(proj_dir)

    # Error handling for bad inputs
    with pytest.raises(CadImportError):
        import_cad_3d(tmp_path / "missing.step")
    bad_ext = tmp_path / "bad.stl"
    bad_ext.write_text("solid x\nendsolid x\n", encoding="utf-8")
    with pytest.raises(CadImportError):
        import_cad_3d(bad_ext)
    with pytest.raises(CadImportError):
        import_cad_3d(box_step, mesh_size_mm=-1.0)


# ---------------------------------------------------------------------------
# 2. Benchmark 1: 3D Uniaxial Finite-Strain Patch Test (3-Way Parity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell_type", [CellType3D.TET4, CellType3D.TET10])
def test_benchmark1_3d_uniaxial_patch_test_three_way_parity(
    tmp_path: Path, cell_type: CellType3D
) -> None:
    box_step = generate_3d_box_step(
        tmp_path / f"box_{cell_type.value}.step", lx=20.0, ly=10.0, lz=10.0
    )
    imported = import_cad_3d(box_step, mesh_size_mm=7.0, element_type=cell_type)
    faces = classify_faces_by_bbox(imported)
    mat = _neo_hookean(youngs_mpa=10.0, poisson=0.35)

    stretch = 1.15
    u_pull = (stretch - 1.0) * 20.0
    exact = analytical_3d_uniaxial_hyperelastic(mat, stretch)

    req = FemRequest3D(
        request_id=f"patch-{cell_type.value}",
        mesh=imported.mesh,
        material=mat,
        dirichlet_bcs=(
            DirichletBC3D("x0", faces["x_min"], ("x",), (0.0,)),
            DirichletBC3D("y0", faces["y_min"], ("y",), (0.0,)),
            DirichletBC3D("z0", faces["z_min"], ("z",), (0.0,)),
            DirichletBC3D("xL", faces["x_max"], ("x",), (u_pull,)),
        ),
        traction_bcs=(),
        load_steps=2,
        solver_tolerance=1.0e-10,
    )

    torch_res = TorchFemBackend3D().solve(req)
    skfem_res = ScikitFemBackend3D().solve(req)
    assert torch_res.converged
    assert skfem_res.converged

    nodes = np.asarray(imported.mesh.nodes, dtype=np.float64)
    u_exact = np.column_stack(
        [
            (exact["axial_stretch"] - 1.0) * nodes[:, 0],
            (exact["transverse_stretch"] - 1.0) * nodes[:, 1],
            (exact["transverse_stretch"] - 1.0) * nodes[:, 2],
        ]
    )
    expected_force_n = exact["axial_first_piola_mpa"] * (10.0 * 10.0)
    expected_energy = exact["strain_energy_density_mpa"] * (20.0 * 10.0 * 10.0)

    for res in (torch_res, skfem_res):
        err_u = float(np.max(np.abs(res.nodal_displacement - u_exact)))
        assert err_u < 1.0e-8, f"{res.backend_name} displacement err={err_u}"
        assert abs(res.reactions_n["xL"][0] - expected_force_n) < 1.0e-6
        assert abs(res.reactions_n["x0"][0] + expected_force_n) < 1.0e-6
        assert abs(res.total_strain_energy_n_mm - expected_energy) < 1.0e-6
        assert abs(res.peak_von_mises_mpa - exact["axial_cauchy_mpa"]) < 1.0e-6
        assert abs(res.min_det_f - exact["det_f"]) < 1.0e-8
        assert "backend_name" in res.summary_document()

    # Cross-backend agreement to machine precision on patch test
    diff_backends = float(
        np.max(np.abs(torch_res.nodal_displacement - skfem_res.nodal_displacement))
    )
    assert diff_backends < 1.0e-9

    # Also verify TET4 solver/request fail-closed refusal at nu = 0.49
    if cell_type is CellType3D.TET4:
        with pytest.raises(VolumetricLockingError):
            FemRequest3D(
                request_id="lock-refuse",
                mesh=imported.mesh,
                material=_neo_hookean(youngs_mpa=10.0, poisson=0.49),
                dirichlet_bcs=req.dirichlet_bcs,
                traction_bcs=(),
            )


# ---------------------------------------------------------------------------
# 3. Benchmark 2: 3D Nonlinear L-Bracket (>15% Deformation & Backend Parity)
# ---------------------------------------------------------------------------


def test_benchmark2_3d_l_bracket_nonlinear_and_cross_backend_parity(
    tmp_path: Path,
) -> None:
    step_path = generate_3d_l_bracket_step(tmp_path / "l_bracket.step")
    coarse = import_cad_3d(step_path, mesh_size_mm=6.2, element_type=CellType3D.TET10)
    fine = import_cad_3d(step_path, mesh_size_mm=4.6, element_type=CellType3D.TET10)
    assert len(fine.mesh.cells) > len(coarse.mesh.cells)

    mat = _neo_hookean(youngs_mpa=12.0, poisson=0.35)
    faces_c = classify_faces_by_bbox(coarse)
    faces_f = classify_faces_by_bbox(fine)

    req_c = FemRequest3D(
        request_id="lbracket-coarse",
        mesh=coarse.mesh,
        material=mat,
        dirichlet_bcs=(
            DirichletBC3D("clamp", faces_c["y_max"], ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        traction_bcs=(TractionBC3D("tip_load", faces_c["x_max"], (0.0, -0.055, 0.0)),),
        load_steps=3,
        solver_tolerance=1.0e-8,
    )
    t_res = TorchFemBackend3D().solve(req_c)
    s_res = ScikitFemBackend3D().solve(req_c)

    assert t_res.converged
    assert s_res.converged
    # >15% nonlinear tip deflection relative to arm thickness (10 mm -> >1.5 mm)
    assert t_res.max_displacement_mm > 5.0
    assert t_res.min_det_f > 0.5

    # Exact global static equilibrium: reaction + external force = 0
    f_ext_tot = assemble_surface_tractions_3d(coarse.mesh, req_c.traction_bcs).sum(
        axis=0
    )
    react_tot = np.asarray(t_res.reactions_n["clamp"], dtype=np.float64)
    assert float(np.linalg.norm(react_tot + f_ext_tot)) < 1.0e-6

    # Cross-backend parity between independent TorchFemBackend3D and ScikitFemBackend3D
    rel_u_diff = float(
        np.linalg.norm(t_res.nodal_displacement - s_res.nodal_displacement)
        / np.linalg.norm(s_res.nodal_displacement)
    )
    rel_e_diff = abs(
        t_res.total_strain_energy_n_mm - s_res.total_strain_energy_n_mm
    ) / max(s_res.total_strain_energy_n_mm, 1.0e-12)
    assert rel_u_diff < 5.0e-3, f"3D L-bracket backend rel L2 diff={rel_u_diff:.4e}"
    assert rel_e_diff < 5.0e-3, f"3D L-bracket backend energy diff={rel_e_diff:.4e}"

    # Mesh convergence check on fine TET10 mesh
    req_f = FemRequest3D(
        request_id="lbracket-fine",
        mesh=fine.mesh,
        material=mat,
        dirichlet_bcs=(
            DirichletBC3D("clamp", faces_f["y_max"], ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        traction_bcs=(TractionBC3D("tip_load", faces_f["x_max"], (0.0, -0.055, 0.0)),),
        load_steps=3,
        solver_tolerance=1.0e-8,
    )
    f_res = TorchFemBackend3D().solve(req_f)
    assert f_res.converged
    assert f_res.min_det_f > 0.5
    # Fine mesh tip displacement agrees within 10% of coarse mesh
    assert (
        abs(f_res.max_displacement_mm - t_res.max_displacement_mm)
        / f_res.max_displacement_mm
        < 0.10
    )


def test_3d_fem_backend_entry_points_registered() -> None:
    eps = {
        ep.name: ep
        for ep in importlib_metadata.entry_points(group="formamech.fem.backends")
    }
    assert "torch-fem-3d" in eps
    assert "scikit-fem-3d" in eps
    assert eps["torch-fem-3d"].load() is TorchFemBackend3D
    assert eps["scikit-fem-3d"].load() is ScikitFemBackend3D


# ---------------------------------------------------------------------------
# 4. Exact Autograd Product-Rule Check on 3D Lifting
# ---------------------------------------------------------------------------


def test_lifted_displacement_and_gradient_matches_finite_differences(
    tmp_path: Path,
) -> None:
    box_step = generate_3d_box_step(tmp_path / "grad_box.step", lx=10.0, ly=8.0, lz=6.0)
    imported = import_cad_3d(box_step, mesh_size_mm=6.0, element_type=CellType3D.TET10)
    faces = classify_faces_by_bbox(imported)

    torch.manual_seed(7)
    op = Surface3DEnergyOperator(
        Surface3DConfig(
            parameter_dim=2, hidden_dim=16, slice_count=4, num_blocks=1, num_heads=2
        )
    ).to(dtype=torch.float64)
    s_c, s_n, s_a, s_f = build_surface_3d_tensors(
        imported.mesh,
        clamped_face_tags=faces["z_min"],
        loaded_face_tags=faces["z_max"],
        dtype=torch.float64,
    )
    enc = op.encode_surface(s_c, s_n, s_a, s_f)
    p = torch.tensor([[0.4, -0.2]], dtype=torch.float64)
    pts = torch.tensor([[[3.2, 4.1, 2.7], [7.1, 2.3, 4.4]]], dtype=torch.float64)

    liftings = (
        PlanarClampLifting3D(
            axis="z", plane_coord=0.0, extent_mm=6.0, displacement_scale_mm=1.2
        ),
        SoftMinFaceLifting3D(
            anchor_points_mm=((0.0, 0.0, 0.0), (10.0, 8.0, 0.0)),
            length_scale_mm=4.0,
            displacement_scale_mm=1.2,
        ),
    )
    eps = 1.0e-6
    for lift in liftings:
        _, grad_auto = lifted_displacement_and_gradient(
            op,
            encoding=enc,
            query_xyz=pts,
            parameters=p,
            lifting=lift,
            create_graph=False,
        )
        grad_fd = torch.zeros_like(grad_auto)
        for j in range(3):
            dp = torch.zeros_like(pts)
            dp[..., j] = eps
            s_p, o_p = lift.scale_and_offset(pts + dp)
            s_m, o_m = lift.scale_and_offset(pts - dp)
            u_p = op.displacement(
                encoding=enc,
                query_xyz=pts + dp,
                parameters=p,
                lifting_offset=o_p,
                lifting_scale=s_p,
            )
            u_m = op.displacement(
                encoding=enc,
                query_xyz=pts - dp,
                parameters=p,
                lifting_offset=o_m,
                lifting_scale=s_m,
            )
            grad_fd[..., :, j] = (u_p - u_m) / (2.0 * eps)
        max_diff = float((grad_auto - grad_fd).abs().max().item())
        assert max_diff < 1.0e-6, f"{lift.document()['kind']} FD diff={max_diff}"
        assert lifting_from_document(lift.document()) == lift


# ---------------------------------------------------------------------------
# 5. Benchmark 3: 3D Industrial Mounting Lug Operator Train, Persist, Serve & Qualify
# ---------------------------------------------------------------------------


def test_benchmark3_3d_mounting_lug_end_to_end_pipeline(tmp_path: Path) -> None:
    lug_step = generate_3d_mounting_lug_step(tmp_path / "lug.step")
    imported = import_cad_3d(lug_step, mesh_size_mm=7.6, element_type=CellType3D.TET10)
    mesh = imported.mesh
    faces = classify_faces_by_bbox(imported)
    clamped = faces["z_min"]
    # Identify the horizontal clevis bore cylindrical face (highest z centroid among cylinders)
    cyl_faces = [f for f in imported.faces if f.kind == "cylindrical"]
    assert len(cyl_faces) >= 3  # 2 vertical bolt holes + clevis bore + crown
    bore_face = min(
        [f for f in cyl_faces if f.centroid_mm[2] > 10.0],
        key=lambda f: f.area_mm2,
    )
    loaded = (bore_face.name,)

    mat = _neo_hookean(youngs_mpa=15.0, poisson=0.35)
    solver = TorchFemBackend3D()

    # 5a. Label-free volumetric Deep Ritz training (0 FEM labels) with SoftMinFaceLifting3D
    clamp_nids = sorted(face_node_indices_3d(mesh, clamped))
    anchor_pts = tuple(
        mesh.nodes[i] for i in clamp_nids[:: max(1, len(clamp_nids) // 24)]
    )
    soft_lift = SoftMinFaceLifting3D(
        anchor_points_mm=anchor_pts, length_scale_mm=10.0, displacement_scale_mm=1.5
    )
    lf_cases = (
        LoadCase3D(
            "lf_1", (0.5, 0.0), (TractionBC3D("bore", loaded, (0.0, 0.0, 0.5)),)
        ),
        LoadCase3D(
            "lf_2", (1.0, 0.0), (TractionBC3D("bore", loaded, (0.0, 0.0, 1.0)),)
        ),
    )
    lf_bundle = train_surface_3d_operator(
        mesh,
        mat,
        clamped_face_tags=clamped,
        loaded_face_tags=loaded,
        lifting=soft_lift,
        train_cases=lf_cases,
        operator_config=Surface3DConfig(
            parameter_dim=2, hidden_dim=24, slice_count=6, num_blocks=1, num_heads=2
        ),
        training_config=Surface3DTrainingConfig(
            epochs=8, learning_rate=3.0e-3, quadrature_subsample=200, seed=0
        ),
        dtype=torch.float32,
    )
    assert lf_bundle.training_summary.training_mode == "variational-energy-label-free"
    assert lf_bundle.claim_level == "NOT_VERIFIED"
    assert (
        lf_bundle.training_summary.final_loss < lf_bundle.training_summary.initial_loss
    )
    assert lf_bundle.training_summary.min_det_f_seen > 0.0

    # 5b. Multi-fidelity training on 2D multi-axial bore loads + exact PlanarClampLifting3D
    planar_lift = PlanarClampLifting3D(
        axis="z", plane_coord=0.0, extent_mm=23.0, displacement_scale_mm=2.5
    )

    def _solve_case(cid: str, tz: float, tx: float) -> LoadCase3D:
        tbc = (TractionBC3D("bore", loaded, (tx, 0.0, tz)),)
        req = FemRequest3D(
            request_id=cid,
            mesh=mesh,
            material=mat,
            dirichlet_bcs=(
                DirichletBC3D("clamp", clamped, ("x", "y", "z"), (0.0, 0.0, 0.0)),
            ),
            traction_bcs=tbc,
            load_steps=2,
            solver_tolerance=1.0e-8,
        )
        res = solver.solve(req)
        assert res.converged
        return LoadCase3D(cid, (tz, tx), tbc, fem_result=res)

    train_cases = (
        _solve_case("tr_1", 0.4, -0.3),
        _solve_case("tr_2", 0.4, 0.3),
        _solve_case("tr_3", 0.9, -0.3),
        _solve_case("tr_4", 0.9, 0.3),
        _solve_case("tr_5", 0.65, 0.0),
    )
    holdout_cases = (
        _solve_case("ho_1", 0.55, 0.15),
        _solve_case("ho_2", 0.80, -0.15),
    )

    bundle = train_surface_3d_operator(
        mesh,
        mat,
        clamped_face_tags=clamped,
        loaded_face_tags=loaded,
        lifting=planar_lift,
        train_cases=train_cases,
        operator_config=Surface3DConfig(
            parameter_dim=2, hidden_dim=48, slice_count=8, num_blocks=2, num_heads=4
        ),
        training_config=Surface3DTrainingConfig(
            epochs=180,
            learning_rate=6.0e-3,
            supervision_weight=60.0,
            quadrature_subsample=140,
            seed=42,
        ),
        dtype=torch.float32,
    )
    assert bundle.training_summary.training_mode == "multi-fidelity-label-efficient"
    assert bundle.claim_level == "NOT_VERIFIED"

    # Persistence round-trip + tamper refusal
    b_dir = tmp_path / "lug_bundle"
    manifest = save_surface_3d_bundle(bundle, b_dir)
    assert manifest["schema"] == SURFACE_3D_BUNDLE_SCHEMA
    reloaded = load_surface_3d_bundle(b_dir)
    assert reloaded.bundle_hash == bundle.bundle_hash

    # Serving: fast, certified, and warm-started exact
    fast_rep = predict_3d_bundle(reloaded, holdout_cases[0], mode="fast")
    assert fast_rep.fem_result is None
    assert fast_rep.min_det_f > 0.0
    assert float(np.max(np.abs(fast_rep.nodal_displacement[clamp_nids]))) == 0.0

    cert_rep = predict_3d_bundle(
        reloaded, holdout_cases[0], mode="certified", backend=solver
    )
    assert cert_rep.displacement_rel_l2_error is not None
    assert cert_rep.displacement_rel_l2_error < 0.15

    exact_rep = predict_3d_bundle(
        reloaded, holdout_cases[0], mode="exact", backend=solver
    )
    assert exact_rep.fem_result is not None and exact_rep.fem_result.converged
    assert "mode" in exact_rep.summary_document()

    # Qualification & break-even N*
    qual = qualify_3d_bundle(
        reloaded,
        holdout_cases,
        max_mean_displacement_rel_l2=0.15,
        max_mean_peak_vm_rel_error=0.25,
        backend=solver,
    )
    assert qual["schema"] == SURFACE_3D_QUALIFICATION_SCHEMA
    assert qual["claim_level"] == "HOLDOUT_VERIFIED"
    assert qual["passed"] is True
    assert qual["break_even"]["fast_speedup_vs_cold_fem"] > 1.0

    # Tamper weights -> refused
    w_path = b_dir / "weights.safetensors"
    w_path.write_bytes(w_path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="weights.safetensors SHA-256 mismatch"):
        load_surface_3d_bundle(b_dir)

    # Pure helper test for compute_3d_break_even edge cases
    be = compute_3d_break_even(
        train_wall_time_s=60.0,
        cold_fem_s_per_query=2.0,
        fast_surrogate_s_per_query=0.02,
        warm_fem_s_per_query=0.8,
    )
    assert be["break_even_n_star_fast"] == float(math.ceil(60.0 / 1.98))
    assert be["break_even_n_star_warm"] == float(math.ceil(60.0 / 1.2))


# ---------------------------------------------------------------------------
# 6. CLI Wiring for 3D CAD Commands
# ---------------------------------------------------------------------------


def test_cli_3d_cad_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    step_out = tmp_path / "cli_lug.step"
    rc1 = cli_main(
        ["cad", "sample-3d", "--template", "mounting-lug", "--output", str(step_out)]
    )
    assert rc1 == 0
    assert step_out.is_file()
    out1 = json.loads(capsys.readouterr().out)
    assert out1["template"] == "mounting-lug"

    proj_dir = tmp_path / "cli_lug_proj"
    rc2 = cli_main(
        [
            "cad",
            "import-3d",
            str(step_out),
            "--output",
            str(proj_dir),
            "--mesh-size-mm",
            "6.5",
            "--element",
            "tet10",
        ]
    )
    assert rc2 == 0
    out2 = json.loads(capsys.readouterr().out)
    assert out2["schema"] == CAD_3D_PROJECT_SCHEMA
    assert out2["cell_type"] == "tet10"

    rc3 = cli_main(["cad", "inspect-3d", str(proj_dir)])
    assert rc3 == 0
    out3 = json.loads(capsys.readouterr().out)
    assert out3["mesh_id"] == out2["mesh_id"]


def test_3d_validation_guards(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        DirichletBC3D("", ("f1",), ("x",), (0.0,))
    with pytest.raises(ValueError):
        DirichletBC3D("c", (), ("x",), (0.0,))
    with pytest.raises(ValueError):
        DirichletBC3D("c", ("f1",), ("x",), (0.0, 1.0))
    with pytest.raises(ValueError):
        TractionBC3D("", ("f1",), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        TractionBC3D("t", (), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        PlanarClampLifting3D(axis="z", plane_coord=0.0, extent_mm=-1.0)
    with pytest.raises(ValueError):
        SoftMinFaceLifting3D(anchor_points_mm=(), length_scale_mm=1.0)
    with pytest.raises(ValueError):
        Surface3DTrainingConfig(epochs=0)
    with pytest.raises(ValueError):
        Surface3DTrainingConfig(learning_rate=-1.0)
    with pytest.raises(ValueError):
        lifting_from_document({"kind": "unknown-3d"})
    with pytest.raises(FileNotFoundError):
        load_surface_3d_bundle(tmp_path / "does_not_exist")
    with pytest.raises(ValueError):
        analytical_3d_uniaxial_hyperelastic(_neo_hookean(), -1.0)
