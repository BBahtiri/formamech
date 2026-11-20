"""End-to-end CLI, CAD (STEP/DXF), shipped-bundle, and artifact tests for P1-P6."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from formamech.cli import main as formamech_main
from piginot.active import (
    AcquisitionCandidate,
    AcquisitionPlan,
    ActiveFemSolveOutcome,
    run_active_fem_campaign,
)
from piginot.contracts import Unit
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh
from piginot.operators import (
    GEOMETRY_FEATURE_NAMES,
    BoundaryEnergyConfig,
    BoundaryEnergyOperator,
    BoundaryRoleMap,
    EnergyOperatorConfig,
    GeometryOODDetector,
    GeometryTokenizer,
    compute_geometry_descriptor,
    load_boundary_operator_checkpoint,
    load_geometry_adapter,
    load_geometry_ood_detector,
    save_boundary_operator_checkpoint,
    save_geometry_ood_detector,
)
from piginot.stateful import (
    GeneralizedMaxwellConfig,
    GeneralizedMaxwellViscoelastic,
    HistoryAwareViscoelasticOperator,
    MaxwellBranch,
    ViscoelasticOperatorConfig,
    load_viscoelastic_artifact,
    save_viscoelastic_artifact,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_rect_dxf(path: Path, width: float = 10.0, height: float = 6.0) -> Path:
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4  # millimetres
    msp = doc.modelspace()
    msp.add_line((0.0, 0.0), (width, 0.0))
    msp.add_line((width, 0.0), (width, height))
    msp.add_line((width, height), (0.0, height))
    msp.add_line((0.0, height), (0.0, 0.0))
    doc.saveas(str(path))
    return path


def _unit_rect_mesh(length: float = 4.0, height: float = 2.0):
    return canonicalize_mesh(
        MeshPayload(
            ((0.0, 0.0), (length, 0.0), (length, height), (0.0, height)),
            ((0, 1, 2), (0, 2, 3)),
            CellType.TRIANGLE,
            ((0, 1), (1, 2), (2, 3), (3, 0)),
            {"bottom": (0,), "right": (1,), "top": (2,), "left": (3,)},
            (1, 1),
            Unit.MM,
            {"kind": "rect"},
        )
    )


def test_p1_live_dxf_import_hierarchy_and_convergence_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("gmsh")
    pytest.importorskip("ezdxf")

    dxf_path = _write_rect_dxf(tmp_path / "plate.dxf", width=8.0, height=4.0)
    proj_dir = tmp_path / "cad-proj"
    code = formamech_main(
        [
            "cad",
            "import",
            str(dxf_path),
            "--unit",
            "mm",
            "--mesh-size-mm",
            "2.0",
            "--hierarchy",
            "--output",
            str(proj_dir),
        ]
    )
    assert code == 0
    out_import = json.loads(capsys.readouterr().out)
    assert out_import["status"] == "IMPORTED"
    assert (proj_dir / "hierarchy.json").is_file()
    assert len(out_import["hierarchy_hash"]) == 64

    hier_out = tmp_path / "hier-standalone.json"
    code_h = formamech_main(
        ["cad", "hierarchy", str(proj_dir), "--output", str(hier_out)]
    )
    assert code_h == 0
    hier_doc = json.loads(capsys.readouterr().out)
    assert hier_doc["hierarchy_hash"] == out_import["hierarchy_hash"]
    assert hier_out.is_file()

    conv_out = tmp_path / "geom-conv.json"
    code_c = formamech_main(
        [
            "cad",
            "convergence",
            str(dxf_path),
            "--unit",
            "mm",
            "--mesh-sizes",
            "3.0",
            "1.5",
            "--output",
            str(conv_out),
        ]
    )
    assert code_c == 0
    conv_doc = json.loads(capsys.readouterr().out)
    assert conv_doc["status"] == "GEOMETRY_CONVERGED"
    assert conv_out.is_file()


def test_p2_cad_adapt_cli_and_safetensors_tamper_detection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("gmsh")
    pytest.importorskip("ezdxf")

    dxf_path = _write_rect_dxf(tmp_path / "adapt-plate.dxf", width=6.0, height=3.0)
    proj_dir = tmp_path / "adapt-proj"
    assert (
        formamech_main(
            [
                "cad",
                "import",
                str(dxf_path),
                "--unit",
                "mm",
                "--mesh-size-mm",
                "2.0",
                "--output",
                str(proj_dir),
            ]
        )
        == 0
    )
    imported = json.loads(capsys.readouterr().out)
    b_names = imported["boundary_names"]
    roles = {name: "traction_free" for name in b_names}
    roles[b_names[0]] = "dirichlet_x"

    # Build and save base BoundaryEnergyOperator checkpoint + OOD detector.
    b_cfg = BoundaryEnergyConfig(
        energy=EnergyOperatorConfig(
            parameter_names=("strain",),
            hidden_dim=16,
            latent_token_count=4,
            parameter_hidden_dim=16,
            num_attention_heads=2,
            decoder_layers=1,
            positional_frequencies=2,
        ),
        boundary_feature_names=GEOMETRY_FEATURE_NAMES,
    )
    torch.manual_seed(0)
    base_op = BoundaryEnergyOperator(b_cfg).to(torch.float64)
    ckpt_dir = tmp_path / "base-op"
    save_boundary_operator_checkpoint(ckpt_dir, base_op)
    reloaded_base = load_boundary_operator_checkpoint(ckpt_dir)
    assert sum(p.numel() for p in reloaded_base.parameters()) == sum(
        p.numel() for p in base_op.parameters()
    )

    tok = GeometryTokenizer(subdivisions_per_facet=1)
    m1 = _unit_rect_mesh(5.0, 3.0)
    m2 = _unit_rect_mesh(7.0, 3.5)
    r_map = BoundaryRoleMap(
        {
            "left": "symmetry",
            "bottom": "symmetry",
            "right": "dirichlet_x",
            "top": "traction_free",
        }
    )
    det = GeometryOODDetector.fit(
        [
            compute_geometry_descriptor(m1, r_map, tokenizer=tok),
            compute_geometry_descriptor(m2, r_map, tokenizer=tok),
        ],
        descriptor_threshold=10.0,
    )
    ood_path = tmp_path / "ood.json"
    save_geometry_ood_detector(ood_path, det)
    assert load_geometry_ood_detector(ood_path) == det

    cfg_path = tmp_path / "adapt-cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "boundary_roles": roles,
                "nominal_strain": 0.08,
                "parameter_vector": [0.0],
                "material": {
                    "name": "nh",
                    "model": "compressible_neo_hookean",
                    "properties": {
                        "elastic_modulus": {"value": 100.0, "unit": "MPa"},
                        "poisson_ratio": {"value": 0.3, "unit": "1"},
                    },
                },
                "adapter": {
                    "rank": 2,
                    "alpha": 2.0,
                    "max_steps": 3,
                    "from_scratch_reference_steps": 40,
                    "max_step_budget_fraction": 0.10,
                    "learning_rate": 5e-3,
                },
                "energy": {"points_per_cell": 2},
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "adapted-out"
    code_ad = formamech_main(
        [
            "cad",
            "adapt",
            "--project",
            str(proj_dir),
            "--checkpoint",
            str(ckpt_dir),
            "--config",
            str(cfg_path),
            "--ood-detector",
            str(ood_path),
            "--output",
            str(out_dir),
        ]
    )
    assert code_ad == 0
    ad_manifest = json.loads(capsys.readouterr().out)
    assert ad_manifest["result"]["status"] == "COMPLETED"
    loaded_ad, _ = load_geometry_adapter(out_dir, base_op)
    assert loaded_ad.adapter_digest() == ad_manifest["adapter_digest"]

    # Tampering with adapter.safetensors is rejected immediately.
    raw = (out_dir / "adapter.safetensors").read_bytes()
    (out_dir / "adapter.safetensors").write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xFF]))
    with pytest.raises(ValueError, match="SHA-256"):
        load_geometry_adapter(out_dir, base_op)


def test_p3_active_select_cli_and_campaign_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "plan_id": "cli-active-test",
                "arm_kind": "physics_diversity",
                "budget_solves": 4,
                "batch_size": 2,
                "seed": 0,
                "supervises_training": True,
                "training_mode": "multi_fidelity_label_efficient",
            }
        ),
        encoding="utf-8",
    )
    pool_path = tmp_path / "pool.json"
    pool_data = [
        {
            "case_id": "t-1",
            "partition": "TRAIN",
            "feature_vector": [0.05, 0.25],
            "eta_energy": 0.02,
            "ood_distance": 0.1,
        },
        {
            "case_id": "t-2",
            "partition": "TRAIN",
            "feature_vector": [0.22, 0.40],
            "eta_energy": 0.09,
            "ood_distance": 1.5,
        },
        {
            "case_id": "t-3",
            "partition": "TRAIN",
            "feature_vector": [0.20, 0.16],
            "eta_energy": 0.07,
            "ood_distance": 1.2,
        },
    ]
    pool_path.write_text(json.dumps(pool_data), encoding="utf-8")
    out_path = tmp_path / "selected.json"
    assert (
        formamech_main(
            [
                "active",
                "select",
                "--plan",
                str(plan_path),
                "--pool",
                str(pool_path),
                "--output",
                str(out_path),
            ]
        )
        == 0
    )
    sel_doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(sel_doc["selected_case_ids"]) == 2
    assert len(sel_doc["selection_hash"]) == 64

    # DEV candidate in pool is refused with exit code 2.
    bad_pool = tmp_path / "bad_pool.json"
    bad_pool.write_text(
        json.dumps([{**pool_data[0], "partition": "DEVELOPMENT"}]), encoding="utf-8"
    )
    assert (
        formamech_main(
            ["active", "select", "--plan", str(plan_path), "--pool", str(bad_pool)]
        )
        == 2
    )
    capsys.readouterr()

    # Campaign runner persists all 4 auditable artifacts.
    plan = AcquisitionPlan(
        plan_id="camp-1",
        arm_kind="physics_diversity",
        budget_solves=2,
        batch_size=1,
        seed=0,
    )
    cands = [
        AcquisitionCandidate(
            case_id=r["case_id"],
            partition="TRAIN",
            feature_vector=tuple(r["feature_vector"]),
            eta_energy=r["eta_energy"],
            ood_distance=r["ood_distance"],
        )
        for r in pool_data
    ]
    camp_dir = tmp_path / "active-campaign"
    res = run_active_fem_campaign(
        plan=plan,
        score_pool=lambda _r, _acq: cands,
        solve_train_case=lambda c: ActiveFemSolveOutcome(
            succeeded=True, wall_seconds=0.05, dof_count=128, result_hash="a" * 64
        ),
        output_dir=camp_dir,
    )
    assert res.ledger.attempted_count == 2
    assert (
        res.training_mode_disclosure == "multi-fidelity-label-efficient-not-label-free"
    )
    for fname in (
        "plan.json",
        "selection-evidence.json",
        "fem-cost-ledger.json",
        "active-summary.json",
    ):
        assert (camp_dir / fname).is_file()


def test_p4_live_certified_dwr_trust_report_on_shipped_bundle(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "trust-sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "geometry": {
                    "remesh_status": "GEOMETRY_CONVERGED",
                    "declared_curve_tolerance_mm": 0.05,
                    "achieved_max_chord_error_mm": 0.01,
                },
                "discretization": {
                    "reference_status": "REFERENCE_CONVERGED",
                    "element_order": 2,
                    "m1_to_m2_u_rel_l2": 0.001,
                    "m1_to_m2_p_rel_l2": 0.005,
                },
                "indicators_qualified": False,
            }
        ),
        encoding="utf-8",
    )
    out_file = tmp_path / "certified_trust.json"
    code = formamech_main(
        [
            "predict",
            str(ROOT / "bundles" / "notched-v1-cp-freeze"),
            "--request",
            str(ROOT / "bundles" / "requests" / "notched-in-domain.json"),
            "--mode",
            "certified",
            "--trust-report",
            "--trust-sidecar",
            str(sidecar),
            "--output",
            str(out_file),
        ]
    )
    assert code == 0
    doc = json.loads(out_file.read_text(encoding="utf-8"))
    tr = doc["trust_report"]
    assert tr["schema"] == "piginot.cad-trust-report-v1"
    assert len(doc["trust_report_sha256"]) == 64
    assert len(tr["qoi_results"]) >= 2
    for q in tr["qoi_results"]:
        assert q["predicted_value"] is not None
        ec = q["error_components"]
        assert ec["surrogate_discrete"]["dwr_estimate"] is not None
        assert ec["surrogate_discrete"]["eta_energy"] is not None
        assert ec["geometry_representation"]["remesh_status"] == "GEOMETRY_CONVERGED"
        assert ec["fem_discretization"]["reference_status"] == "REFERENCE_CONVERGED"


def test_p5_live_optimize_cli_on_shipped_bundle(tmp_path: Path) -> None:
    prob_path = tmp_path / "design.json"
    prob_path.write_text(
        json.dumps(
            {
                "name": "notched-width-lightweighting",
                "parameters": [
                    {
                        "name": "width",
                        "lower": 18.0,
                        "upper": 28.0,
                        "initial": 24.0,
                        "unit": "mm",
                    }
                ],
                "fixed_inputs": {
                    "notch_radius": {"value": 4.0, "unit": "mm"},
                    "notch_depth": {"value": 3.0, "unit": "mm"},
                    "elastic_modulus": {"value": 760.0, "unit": "MPa"},
                    "poisson_ratio": {"value": 0.23, "unit": "1"},
                    "nominal_axial_strain": {"value": 0.15, "unit": "1"},
                },
                "objective_name": "width_proxy",
                "objective_metric": "expr: width",
                "objective_unit": "mm",
                "minimize": True,
                "constraints": [
                    {
                        "name": "avg_p11",
                        "metric": "avg_p11",
                        "kind": "inequality_le",
                        "limit": 140.0,
                        "tolerance": 0.5,
                        "unit": "MPa",
                    }
                ],
                "max_iterations": 2,
                "max_fem_evaluations": 3,
                "gradient_check_interval": 10,
                "accept_eta_tolerance": 0.5,
            }
        ),
        encoding="utf-8",
    )
    out_opt = tmp_path / "opt-report.json"
    code = formamech_main(
        [
            "optimize",
            str(ROOT / "bundles" / "notched-v1-cp-freeze"),
            "--problem",
            str(prob_path),
            "--output",
            str(out_opt),
        ]
    )
    assert code == 0
    rep = json.loads(out_opt.read_text(encoding="utf-8"))
    assert rep["status"] in (
        "CONVERGED_FEM_VERIFIED",
        "BUDGET_EXHAUSTED_FEM_VERIFIED",
    )
    assert rep["best_design"]["fem_feasible"] is True
    assert rep["best_design"]["fem_objective"] < 24.0
    assert rep["fem_evaluations"] >= 2

    # Refusal path returns exit code 2 without any FEM calls.
    out_ref = tmp_path / "opt-refused.json"
    code_ref = formamech_main(
        [
            "optimize",
            str(ROOT / "bundles" / "notched-v1-cp-freeze"),
            "--problem",
            str(prob_path),
            "--refuse-unsupported",
            "--output",
            str(out_ref),
        ]
    )
    assert code_ref == 2
    ref_doc = json.loads(out_ref.read_text(encoding="utf-8"))
    assert ref_doc["status"] == "REFUSED_UNSUPPORTED_GEOMETRY"
    assert ref_doc["fem_evaluations"] == 0


def test_p6_stateful_rollout_cli_checkpoint_resume_and_tamper(
    tmp_path: Path,
) -> None:
    from piginot.operators import QueryFrame

    mesh = _unit_rect_mesh(4.0, 2.0)
    visco = GeneralizedMaxwellViscoelastic(
        GeneralizedMaxwellConfig(
            equilibrium_modulus_mpa=100.0,
            poisson_ratio=0.3,
            branches=(MaxwellBranch(0.5, 1.0),),
        )
    )
    op = HistoryAwareViscoelasticOperator(
        ViscoelasticOperatorConfig(
            load_dim=1, n_branches=1, hidden_dim=16, num_layers=2
        )
    ).to(torch.float64)
    frame = QueryFrame(
        origin=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        extent=torch.tensor([[4.0, 2.0]], dtype=torch.float64),
    )
    art_dir = tmp_path / "visco-artifact"
    save_viscoelastic_artifact(
        art_dir, operator=op, material=visco, mesh=mesh, frame=frame
    )
    loaded = load_viscoelastic_artifact(art_dir)
    assert loaded.mesh.mesh_id == mesh.mesh_id

    # Run first half [0, 0.5] and save checkpoint, then resume through [0, 1.0].
    lp_half = tmp_path / "lp-half.json"
    lp_half.write_text(
        json.dumps(
            {
                "time_grid": {
                    "t0": 0.0,
                    "t1": 0.5,
                    "dt_initial": 0.25,
                    "dt_min": 0.05,
                    "dt_max": 0.25,
                },
                "segments": [
                    {
                        "start": 0.0,
                        "end": 0.5,
                        "kind": "ramp",
                        "boundary_values": {"ux": "u_max * s"},
                    },
                    {
                        "start": 0.5,
                        "end": 1.0,
                        "kind": "hold",
                        "boundary_values": {"ux": "u_max"},
                    },
                ],
                "parameters": {"u_max": 0.20},
                "load_key": "ux",
            }
        ),
        encoding="utf-8",
    )
    ckpt_dir = tmp_path / "ckpt-mid"
    out_half = tmp_path / "rollout-half.json"
    assert (
        formamech_main(
            [
                "stateful",
                "rollout",
                "--artifact",
                str(art_dir),
                "--load-path",
                str(lp_half),
                "--checkpoint-out",
                str(ckpt_dir),
                "--output",
                str(out_half),
            ]
        )
        == 0
    )
    r_half = json.loads(out_half.read_text(encoding="utf-8"))
    assert r_half["status"] == "COMPLETED"
    assert len(r_half["steps"]) == 2

    lp_full = tmp_path / "lp-full.json"
    lp_full_doc = json.loads(lp_half.read_text(encoding="utf-8"))
    lp_full_doc["time_grid"]["t1"] = 1.0
    lp_full.write_text(json.dumps(lp_full_doc), encoding="utf-8")
    out_res = tmp_path / "rollout-resumed.json"
    assert (
        formamech_main(
            [
                "stateful",
                "rollout",
                "--artifact",
                str(art_dir),
                "--load-path",
                str(lp_full),
                "--resume",
                str(ckpt_dir),
                "--output",
                str(out_res),
            ]
        )
        == 0
    )
    r_res = json.loads(out_res.read_text(encoding="utf-8"))
    assert r_res["status"] == "COMPLETED"
    assert [s["time_index"] for s in r_res["steps"]] == [3, 4]

    # Tampering with artifact weights is rejected immediately.
    raw = (art_dir / "weights.safetensors").read_bytes()
    (art_dir / "weights.safetensors").write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xFF]))
    with pytest.raises(ValueError, match="SHA-256"):
        load_viscoelastic_artifact(art_dir)
