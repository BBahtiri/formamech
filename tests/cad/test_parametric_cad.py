"""Comprehensive verification suite for M4 Parametric CAD & Geometry-Aware Exploration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from formamech.cad import (
    BUILTIN_TEMPLATE_GENERATOR_ID,
    PARAMETRIC_BUNDLE_SCHEMA,
    PARAMETRIC_CAD,
    PARAMETRIC_CAD_ADAPTER_ID,
    PARAMETRIC_CAD_SCHEMA,
    ExpectedTopology,
    GeneratedCadMesh,
    GeometricBound,
    ParametricCadSpec,
    ParametricPhysicsConfig,
    ParametricTopologyError,
    ParametricTrainingConfig,
    TopologyStabilityGuard,
    default_holed_plate_spec,
    default_notched_bracket_spec,
    evaluate_parametric_fem,
    init_parametric_project,
    is_parametric_bundle_dir,
    load_parametric_bundle,
    load_parametric_project,
    predict_parametric_cad,
    qualify_parametric_bundle,
    register_parametric_generator,
    sample_parametric_cad_doe,
    save_parametric_bundle,
    train_parametric_cad_surrogate,
)
from formamech.optimization import (
    DesignOptimizationStatus,
    ParametricBundleDesignBridge,
    parse_parametric_design_document,
)

from piginot.cli import main as cli_main
from piginot.contracts import Unit
from piginot.geometry import CellType, MeshFamilyKind, MeshPayload, canonicalize_mesh
from piginot.operators.geometry_tokenizer import GeometryOODStatus


def test_parametric_cad_spec_topology_guard_and_doe(tmp_path: Path) -> None:
    """Verify ParametricCadSpec contracts, TopologyStabilityGuard fail-closed checks, and DOE."""
    spec = default_notched_bracket_spec(mesh_size_mm=3.5)
    assert spec.document()["schema"] == PARAMETRIC_CAD_SCHEMA
    assert ParametricCadSpec.from_document(spec.document()) == spec
    assert len(spec.spec_hash) == 64
    assert PARAMETRIC_CAD.descriptor.adapter_id == PARAMETRIC_CAD_ADAPTER_ID

    holed_spec = default_holed_plate_spec(mesh_size_mm=3.5)
    assert ParametricCadSpec.from_document(holed_spec.document()) == holed_spec

    # Validation error guards
    with pytest.raises(ValueError, match="lower or upper"):
        GeometricBound(name="bad", expression="hole_radius")
    with pytest.raises(ValueError, match="upper must not be less"):
        GeometricBound(name="bad", expression="hole_radius", lower=5.0, upper=2.0)
    with pytest.raises(ValueError, match="component_count"):
        ExpectedTopology(
            component_count=0,
            hole_count=0,
            boundary_groups=("left",),
            boundary_kinds={"left": "line"},
        )
    with pytest.raises(ValueError, match="expected schema"):
        ParametricCadSpec.from_document({"schema": "wrong"})

    # Project init + reload + tamper detection
    proj_dir = tmp_path / "param-proj"
    manifest = init_parametric_project(spec, proj_dir, verify_nominal=True)
    assert manifest["nominal_verified"] is True
    assert (proj_dir / "nominal.step").is_file()
    reloaded_spec = load_parametric_project(proj_dir)
    assert reloaded_spec.spec_hash == spec.spec_hash

    # TopologyStabilityGuard: valid nominal passes; out-of-box & ligament bound refuse
    guard = TopologyStabilityGuard(spec)
    nom_inst = guard.generate(spec.nominal_parameters())
    assert nom_inst.topology.component_count == 1
    assert nom_inst.topology.hole_count == 1
    assert set(nom_inst.mesh.semantic_tags) == set(
        spec.expected_topology.boundary_groups
    )

    with pytest.raises(ParametricTopologyError, match="outside"):
        guard.generate({"hole_radius": 8.0, "notch_radius": 2.0, "plate_width": 20.0})

    # Ligament bound violation: 18/2 - 4.5 - 4.0 = 0.5 < 2.2 mm
    with pytest.raises(ParametricTopologyError, match="min_ligament_mm"):
        guard.generate({"hole_radius": 4.5, "notch_radius": 4.0, "plate_width": 18.0})

    # Force breakthrough (bypass derived bound) -> OpenCASCADE cuts plate in two -> genus error!
    with pytest.raises(ParametricTopologyError):
        guard.generate(
            {"hole_radius": 6.0, "notch_radius": 5.0, "plate_width": 18.0},
            enforce_box=False,
            enforce_bounds=False,
        )

    # Custom generator with wrong semantic boundary tags must fail closed
    def _bad_tags_gen(
        _spec: ParametricCadSpec,
        _params: dict[str, float] | object,
        _step_out: Path | None,
    ) -> GeneratedCadMesh:
        del _spec, _params, _step_out
        m = canonicalize_mesh(
            MeshPayload(
                nodes=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
                cells=((0, 1, 2), (0, 2, 3)),
                cell_type=CellType.TRIANGLE,
                boundary_facets=((0, 1), (1, 2), (2, 3), (3, 0)),
                boundary_tags={"outer": (0, 1, 2, 3)},
                region_ids=(1, 1),
                unit=Unit.MM,
                source_provenance={"source": "bad-tag-test"},
            )
        )
        return GeneratedCadMesh(mesh=m, boundary_kinds={"outer": "line"})

    register_parametric_generator("test-bad-tags-v1", _bad_tags_gen)
    bad_spec = ParametricCadSpec(
        name="bad-tag-test",
        generator_id="test-bad-tags-v1",
        template="custom",
        parameters=holed_spec.parameters,
        expected_topology=holed_spec.expected_topology,
        boundary_roles=holed_spec.boundary_roles,
    )
    with pytest.raises(ParametricTopologyError, match="hole_count"):
        TopologyStabilityGuard(bad_spec).generate(holed_spec.nominal_parameters())

    # Deterministic Halton DOE sampling produces VARIABLE family & exact replay hash
    doe_a = sample_parametric_cad_doe(spec, count=5, seed_offset=1)
    doe_b = sample_parametric_cad_doe(spec, count=5, seed_offset=1)
    assert doe_a.family_kind is MeshFamilyKind.VARIABLE
    assert len(doe_a.instances) == 5
    assert doe_a.report_hash == doe_b.report_hash
    assert BUILTIN_TEMPLATE_GENERATOR_ID == spec.generator_id


def test_parametric_surrogate_bundle_ood_qualification_and_tamper(
    tmp_path: Path,
) -> None:
    """Train BoundaryEnergyOperator + GeometryOODDetector, qualify on 10 unseen CADs & OOD."""
    spec = default_notched_bracket_spec(mesh_size_mm=3.6)
    physics = ParametricPhysicsConfig(
        elastic_modulus_mpa=10.0,
        poisson_ratio=0.35,
        nominal_strain=0.06,
        fem_element_order=1,
        fem_load_steps=2,
    )
    doe_train = sample_parametric_cad_doe(spec, count=6, seed_offset=1)
    t_cfg = ParametricTrainingConfig(
        hidden_dim=24,
        latent_token_count=4,
        parameter_hidden_dim=16,
        num_attention_heads=4,
        decoder_layers=2,
        positional_frequencies=2,
        epochs=15,
        learning_rate=4.0e-3,
        points_per_cell=1,
        seed=0,
        use_fem_supervision=False,
        ood_descriptor_threshold=5.0,
        ood_token_threshold=5.0,
    )
    bundle = train_parametric_cad_surrogate(
        spec, physics, doe_train.instances, config=t_cfg
    )
    assert bundle.claim_level == "NOT_VERIFIED"
    assert bundle.training_summary.training_mode == "variational-energy-label-free"
    assert bundle.training_summary.min_det_f_seen > 0.0

    # Save & reload tamper-checked bundle
    bundle_dir = tmp_path / "param-bundle"
    manifest = save_parametric_bundle(bundle, bundle_dir)
    assert manifest["schema"] == PARAMETRIC_BUNDLE_SCHEMA
    assert is_parametric_bundle_dir(bundle_dir) is True
    loaded = load_parametric_bundle(bundle_dir)
    assert loaded.bundle_hash == bundle.bundle_hash

    # In-envelope unseen geometry serves cleanly in fast, certified, and exact modes
    unseen_params = {"hole_radius": 3.1, "notch_radius": 2.2, "plate_width": 21.5}
    p_fast = predict_parametric_cad(loaded, parameters=unseen_params, mode="fast")
    assert p_fast.accepted is True
    assert p_fast.status == "SERVED_FAST"
    assert p_fast.ood_report is not None
    assert p_fast.ood_report.status is GeometryOODStatus.IN_DISTRIBUTION

    p_cert = predict_parametric_cad(loaded, parameters=unseen_params, mode="certified")
    assert p_cert.accepted is True
    assert p_cert.status == "SERVED_CERTIFIED"
    assert p_cert.displacement_rel_l2_error is not None
    assert p_cert.fem_metrics is not None and p_cert.fem_metrics.converged is True

    p_exact = predict_parametric_cad(loaded, parameters=unseen_params, mode="exact")
    assert p_exact.accepted is True
    assert p_exact.status == "SERVED_EXACT"

    # Held-out qualification across 10 unseen in-envelope geometries + 100% OOD refusal
    qual_path = tmp_path / "parametric-qualification.json"
    q_report = qualify_parametric_bundle(
        loaded,
        holdout_count=10,
        seed_offset=101,
        max_mean_displacement_rel_l2=0.35,
        max_mean_reaction_rel_error=0.35,
        output_path=qual_path,
    )
    assert qual_path.is_file()
    assert q_report.holdout_count == 10
    assert q_report.in_envelope_ood_pass_rate == 1.0
    assert q_report.ood_probe_refusal_rate == 1.0
    assert q_report.passed is True
    assert q_report.claim_level == "HOLDOUT_VERIFIED"

    # Tampering parametric-spec.json must fail closed on load
    spec_file = bundle_dir / "parametric-spec.json"
    spec_file.write_text(spec_file.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_parametric_bundle(bundle_dir)


def test_parametric_fem_guarded_design_optimization_mass_reduction() -> None:
    """Verify FEM-guarded trust-region optimization achieves FEM-verified area reduction."""
    spec = default_holed_plate_spec(mesh_size_mm=4.0)
    physics = ParametricPhysicsConfig(
        elastic_modulus_mpa=10.0,
        poisson_ratio=0.35,
        nominal_strain=0.06,
        fem_element_order=1,
        fem_load_steps=2,
    )
    doe_train = sample_parametric_cad_doe(spec, count=6, seed_offset=1)
    t_cfg = ParametricTrainingConfig(
        hidden_dim=24,
        latent_token_count=4,
        parameter_hidden_dim=16,
        num_attention_heads=4,
        decoder_layers=2,
        positional_frequencies=2,
        epochs=20,
        learning_rate=4.0e-3,
        points_per_cell=1,
        seed=1,
        use_fem_supervision=True,
        supervision_weight=15.0,
        ood_descriptor_threshold=5.0,
        ood_token_threshold=5.0,
    )
    bundle = train_parametric_cad_surrogate(
        spec, physics, doe_train.instances, config=t_cfg
    )
    assert bundle.training_summary.training_mode == "multi-fidelity-label-efficient"

    # Evaluate initial nominal geometry with exact FEM to set realistic constraints
    guard = TopologyStabilityGuard(spec)
    nom_inst = guard.generate(spec.nominal_parameters())
    nom_fem = evaluate_parametric_fem(nom_inst.mesh, physics, request_id="nom-ref")
    assert nom_fem.converged is True

    design_doc = {
        "name": "holed-plate-lightweight",
        "objective_name": "area_mm2",
        "objective_metric": "area_mm2",
        "objective_unit": "mm2",
        "minimize": True,
        "constraints": [
            {
                "name": "peak_vm_limit",
                "metric": "peak_von_mises_mpa",
                "kind": "inequality_le",
                "limit": nom_fem.peak_von_mises_mpa * 1.35,
                "tolerance": 0.05,
                "unit": "MPa",
            },
            {
                "name": "min_stiffness_rx",
                "metric": "reaction_force_n",
                "kind": "inequality_ge",
                "limit": nom_fem.reaction_force_n * 0.65,
                "tolerance": 0.05,
                "unit": "N",
            },
        ],
        "max_iterations": 8,
        "max_fem_evaluations": 10,
        "initial_trust_radius": 0.25,
        "min_trust_radius": 0.02,
        "gradient_check_interval": 3,
        "gradient_cos_min": 0.0,
    }
    p_spec = parse_parametric_design_document(design_doc, bundle=bundle)
    bridge = ParametricBundleDesignBridge(bundle, p_spec)
    report = bridge.run()

    assert report.status in (
        DesignOptimizationStatus.CONVERGED_FEM_VERIFIED,
        DesignOptimizationStatus.BUDGET_EXHAUSTED_FEM_VERIFIED,
    )
    assert report.best_design is not None
    assert report.best_design.fem_feasible is True
    assert report.best_design.reference_hash is not None
    # Genuine FEM-verified area/mass reduction relative to nominal starting design
    assert report.best_design.fem_objective < nom_fem.area_mm2 - 5.0
    assert report.fem_evaluations >= 2
    assert report.surrogate_evaluations >= 1


def test_cli_cad_parametric_end_to_end_and_optimize(tmp_path: Path) -> None:
    """Exercise the full M4 CLI workflow: init -> sample -> train -> predict -> qualify -> optimize."""
    proj_dir = tmp_path / "cli-proj"
    doe_dir = tmp_path / "cli-doe"
    bundle_dir = tmp_path / "cli-bundle"
    physics_file = tmp_path / "physics.json"
    physics_file.write_text(
        json.dumps(
            {
                "elastic_modulus_mpa": 10.0,
                "poisson_ratio": 0.35,
                "nominal_strain": 0.06,
                "fem_element_order": 1,
                "fem_load_steps": 2,
            }
        ),
        encoding="utf-8",
    )

    # 1. init
    assert (
        cli_main(
            [
                "cad",
                "parametric",
                "init",
                "--template",
                "holed-plate",
                "--mesh-size-mm",
                "4.5",
                "--output",
                str(proj_dir),
            ]
        )
        == 0
    )
    # 2. sample
    assert (
        cli_main(
            [
                "cad",
                "parametric",
                "sample",
                str(proj_dir),
                "--count",
                "4",
                "--output",
                str(doe_dir),
            ]
        )
        == 0
    )
    assert (doe_dir / "doe-manifest.json").is_file()

    # 3. train
    assert (
        cli_main(
            [
                "cad",
                "parametric",
                "train",
                str(proj_dir),
                "--physics",
                str(physics_file),
                "--count",
                "4",
                "--epochs",
                "8",
                "--points-per-cell",
                "1",
                "--output",
                str(bundle_dir),
            ]
        )
        == 0
    )

    # 4. predict via both `formamech predict --cad-params` and `formamech cad parametric predict`
    in_params = tmp_path / "in_variant.json"
    in_params.write_text(
        json.dumps({"hole_radius": 3.5, "plate_width": 20.5}), encoding="utf-8"
    )
    pred_out = tmp_path / "pred.json"
    assert (
        cli_main(
            [
                "predict",
                str(bundle_dir),
                "--cad-params",
                str(in_params),
                "--mode",
                "certified",
                "--output",
                str(pred_out),
            ]
        )
        == 0
    )
    pred_doc = json.loads(pred_out.read_text(encoding="utf-8"))
    assert pred_doc["status"] == "SERVED_CERTIFIED"

    # Out-of-envelope variant must return EXIT_REFUSED (2)
    ood_params = tmp_path / "ood_variant.json"
    ood_params.write_text(
        json.dumps({"hole_radius": 11.0, "plate_width": 20.0}), encoding="utf-8"
    )
    assert (
        cli_main(
            [
                "predict",
                str(bundle_dir),
                "--cad-params",
                str(ood_params),
                "--mode",
                "fast",
            ]
        )
        == 2
    )

    # 5. qualify
    qual_out = tmp_path / "qual.json"
    assert (
        cli_main(
            [
                "cad",
                "parametric",
                "qualify",
                str(bundle_dir),
                "--count",
                "4",
                "--max-disp-error",
                "0.5",
                "--max-reaction-error",
                "0.5",
                "--output",
                str(qual_out),
            ]
        )
        == 0
    )
    qual_doc = json.loads(qual_out.read_text(encoding="utf-8"))
    assert qual_doc["passed"] is True
    assert qual_doc["claim_level"] == "HOLDOUT_VERIFIED"

    # 6. optimize directly via `formamech optimize <parametric-bundle> --problem ...`
    opt_prob = tmp_path / "opt_problem.json"
    opt_prob.write_text(
        json.dumps(
            {
                "name": "cli-holed-plate-opt",
                "objective_name": "area_mm2",
                "objective_metric": "area_mm2",
                "objective_unit": "mm2",
                "minimize": True,
                "constraints": [
                    {
                        "name": "vm_cap",
                        "metric": "peak_von_mises_mpa",
                        "kind": "inequality_le",
                        "limit": 5.0,
                        "tolerance": 0.1,
                        "unit": "MPa",
                    }
                ],
                "max_iterations": 5,
                "max_fem_evaluations": 6,
            }
        ),
        encoding="utf-8",
    )
    opt_out = tmp_path / "opt_report.json"
    assert (
        cli_main(
            [
                "optimize",
                str(bundle_dir),
                "--problem",
                str(opt_prob),
                "--output",
                str(opt_out),
            ]
        )
        == 0
    )
    opt_doc = json.loads(opt_out.read_text(encoding="utf-8"))
    assert opt_doc["best_design"] is not None
    assert opt_doc["best_design"]["fem_feasible"] is True


def test_parametric_guards_metrics_and_bridge_edge_cases(tmp_path: Path) -> None:
    """Exercise remaining validation guards, metrics, and tamper/OOD paths (>90% cov)."""
    from piginot.cad.parametric import get_parametric_generator

    # GeometricBound operators (+, -, *, /, unary -) and upper-bound violation
    gb = GeometricBound(
        name="ratio",
        expression="-(a - b) * 2.0 + (a + b) / 2.0",
        lower=-100.0,
        upper=5.0,
    )
    ok_lo, val_lo, _ = gb.check({"a": 2.0, "b": 4.0})
    assert ok_lo is False and val_lo > 5.0
    with pytest.raises(ZeroDivisionError):
        GeometricBound(name="div0", expression="a / b", lower=0.0).evaluate(
            {"a": 1.0, "b": 0.0}
        )
    with pytest.raises(ValueError, match="unknown geometric bound symbol"):
        GeometricBound(name="miss", expression="x + 1", lower=0.0).evaluate({})

    # Generator registry error
    with pytest.raises(ValueError, match="unknown parametric CAD generator"):
        get_parametric_generator("nonexistent-gen-v99")

    # Project tamper error
    spec = default_holed_plate_spec(mesh_size_mm=4.5)
    proj = tmp_path / "tamper-proj"
    init_parametric_project(spec, proj, verify_nominal=False)
    (proj / "project.json").write_text(
        json.dumps({"spec_hash": "0" * 64}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="spec_hash mismatch"):
        load_parametric_project(proj)

    # Train tiny bundle and verify non-analytic objective metric, expr metric, OOD/envelope refusal in bridge
    physics = ParametricPhysicsConfig(
        elastic_modulus_mpa=10.0,
        poisson_ratio=0.35,
        nominal_strain=0.05,
        fem_element_order=1,
        fem_load_steps=2,
    )
    doe = sample_parametric_cad_doe(spec, count=3, seed_offset=1)
    bundle = train_parametric_cad_surrogate(
        spec,
        physics,
        doe.instances,
        config=ParametricTrainingConfig(
            hidden_dim=16,
            latent_token_count=4,
            parameter_hidden_dim=8,
            num_attention_heads=4,
            decoder_layers=2,
            positional_frequencies=1,
            epochs=4,
            points_per_cell=1,
            seed=0,
        ),
    )

    # Unsupported metric raises cleanly
    with pytest.raises(ValueError, match="unsupported objective_metric"):
        parse_parametric_design_document(
            {
                "name": "bad-metric",
                "objective_name": "bogus",
                "objective_metric": "bogus",
            },
            bundle=bundle,
        )
    # Unknown parameter raises cleanly
    with pytest.raises(ValueError, match="not declared in ParametricCadSpec"):
        parse_parametric_design_document(
            {
                "name": "bad-param",
                "objective_name": "area_mm2",
                "parameters": [
                    {
                        "name": "unknown_p",
                        "lower": 1.0,
                        "upper": 2.0,
                        "initial": 1.5,
                        "unit": "mm",
                    }
                ],
            },
            bundle=bundle,
        )

    # Exercise compliance_n_mm objective + expr/max_displacement_mm/min_det_f constraints
    p_spec = parse_parametric_design_document(
        {
            "name": "multi-metric-check",
            "objective_name": "compliance_n_mm",
            "objective_metric": "compliance_n_mm",
            "objective_unit": "N*mm",
            "minimize": True,
            "constraints": [
                {
                    "name": "max_u",
                    "metric": "max_displacement_mm",
                    "kind": "inequality_le",
                    "limit": 10.0,
                    "unit": "mm",
                },
                {
                    "name": "min_j",
                    "metric": "min_det_f",
                    "kind": "inequality_ge",
                    "limit": 0.5,
                    "unit": "1",
                },
                {
                    "name": "lig_expr",
                    "metric": "expr:plate_width / 2 - hole_radius",
                    "kind": "inequality_ge",
                    "limit": 2.5,
                    "unit": "mm",
                },
            ],
            "max_iterations": 2,
            "max_fem_evaluations": 4,
        },
        bundle=bundle,
    )
    bridge = ParametricBundleDesignBridge(bundle, p_spec)
    # Surrogate evaluation at nominal (exercises non-analytic FD gradient path!)
    s_ok = bridge.surrogate_evaluator(spec.nominal_parameters())
    assert s_ok.in_envelope is True and s_ok.admissible is True
    assert bridge.remesh_count >= 1
    # Out-of-envelope surrogate & FEM evaluation fail closed
    s_bad = bridge.surrogate_evaluator({"hole_radius": 20.0, "plate_width": 20.0})
    assert s_bad.in_envelope is False and s_bad.admissible is False
    f_bad = bridge.fem_evaluator({"hole_radius": 20.0, "plate_width": 20.0})
    assert f_bad.succeeded is False
    # Exact FEM evaluation at nominal succeeds
    f_ok = bridge.fem_evaluator(spec.nominal_parameters())
    assert f_ok.succeeded is True and f_ok.objective is not None
    assert f_ok.constraints["min_j"] > 0.5
