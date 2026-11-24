"""End-to-end verification tests for Phase 2 (V3-2) 3D parametric CAD, OOD & P5 optimization."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from piginot.cad.parametric_3d import (
    ParametricTopology3DError,
    TopologyStabilityGuard3D,
    default_holed_block_3d_spec,
    sample_parametric_cad_3d_doe,
)
from piginot.operators.geometry_3d import SURFACE_3D_GALE_FEATURE_DIM
from piginot.operators.surface_operator_3d import Surface3DConfig
from piginot.optimization.parametric_3d_bridge import (
    Parametric3DDesignBridge,
    Parametric3DDesignSpec,
)
from piginot.optimization.problem import DesignConstraint, DesignProblem
from piginot.optimization.trust_region import DesignOptimizationStatus
from piginot.training.parametric_3d_engine import (
    Parametric3DPhysicsConfig,
    Parametric3DTrainingConfig,
    load_parametric_3d_bundle,
    predict_parametric_3d,
    qualify_parametric_3d_bundle,
    save_parametric_3d_bundle,
    train_parametric_3d_operator,
)


def test_3d_parametric_topology_guard_refuses_ligament_and_bounds() -> None:
    spec = default_holed_block_3d_spec(mesh_size_mm=7.0)
    guard = TopologyStabilityGuard3D(spec)

    # Nominal passes with Euler chi == 0 (genus 1 through-hole)
    nom = guard.build_and_verify(spec.nominal_parameters(), instance_id="nom")
    assert nom.descriptor.surface_euler_characteristic == 0
    assert nom.imported.volume_mm3 > 500.0

    # Parameter out of box fails closed
    with pytest.raises(ParametricTopology3DError, match="outside"):
        guard.build_and_verify({"lx": 50.0, "ly": 12.0, "hole_radius": 2.4})

    # Algebraic ligament violation fails closed (ly/2 - hole_radius = 5.0 - 3.4 = 1.6 >= 1.5 passes;
    # but ly=10.0, hole_radius=3.4 -> 1.6 passes; let's force ly/2 - r < 1.5 with r=3.3, ly=10.0? No, 10/2-3.3=1.7;
    # use custom spec check via ly=10.0 with custom bound or r=3.4 & ly=9.5 -> box fail; test direct geometric bound)
    from dataclasses import replace

    tight_spec = replace(
        spec,
        parameters=(
            spec.parameters[0],
            spec.parameters[1],
            replace(spec.parameters[2], upper=4.2),
        ),
    )
    tight_guard = TopologyStabilityGuard3D(tight_spec)
    with pytest.raises(ParametricTopology3DError, match="min_wall_y"):
        tight_guard.build_and_verify({"lx": 20.0, "ly": 10.0, "hole_radius": 4.0})


def test_3d_parametric_surrogate_train_save_qualify_and_p5_optimize(
    tmp_path: Path,
) -> None:
    spec = default_holed_block_3d_spec(mesh_size_mm=7.5)
    doe = sample_parametric_cad_3d_doe(spec, count=3, seed_offset=1)
    assert len(doe.instances) == 3

    physics = Parametric3DPhysicsConfig(
        youngs_modulus_mpa=15.0,
        poisson_ratio=0.35,
        traction_mpa=(0.35, 0.0, 0.0),
        load_steps=1,
    )
    bundle = train_parametric_3d_operator(
        spec,
        physics,
        doe.instances,
        operator_config=Surface3DConfig(
            parameter_dim=3,
            surface_feature_dim=SURFACE_3D_GALE_FEATURE_DIM,
            hidden_dim=24,
            slice_count=4,
            num_blocks=1,
            num_heads=2,
        ),
        training_config=Parametric3DTrainingConfig(
            epochs=25,
            learning_rate=4e-3,
            quadrature_subsample=48,
            use_fem_supervision=True,
            seed=0,
        ),
        dtype=torch.float32,
    )
    assert bundle.training_mode == "multi-fidelity-label-efficient"
    assert bundle.claim_level == "NOT_VERIFIED"

    # Save and reload tamper-checked 3D parametric bundle
    b_dir = tmp_path / "param-3d-bundle"
    save_parametric_3d_bundle(bundle, b_dir)
    reloaded = load_parametric_3d_bundle(b_dir)
    assert reloaded.bundle_hash == bundle.bundle_hash

    # Predict unseen in-envelope 3D variant & refuse out-of-box OOD probe
    pred_in = predict_parametric_3d(
        reloaded, {"lx": 19.5, "ly": 12.5, "hole_radius": 2.5}, mode="certified"
    )
    assert pred_in.accepted is True
    assert pred_in.displacement_rel_l2_error is not None
    assert pred_in.displacement_rel_l2_error < 0.50

    pred_ood = predict_parametric_3d(
        reloaded, {"lx": 40.0, "ly": 12.5, "hole_radius": 2.5}, mode="fast"
    )
    assert pred_ood.accepted is False

    # Qualification report checks unseen 3D holdouts + 100% OOD refusal
    q_rep = qualify_parametric_3d_bundle(
        reloaded, holdout_count=2, seed_offset=20, max_mean_displacement_rel_l2=0.60
    )
    assert q_rep.ood_refusal_rate == 1.0
    assert q_rep.passed is True
    assert q_rep.granted_claim_level == "HOLDOUT_VERIFIED"

    # 3D P5 FEM-guarded lightweighting (minimize volume_mm3 with FEM stress & det(F) guard)
    prob = DesignProblem(
        name="lightweight-3d-holed-block",
        geometry_version="holed-block-3d-v1",
        problem_spec_hash=spec.spec_hash,
        parameters=spec.parameters,
        objective_name="volume_mm3",
        objective_unit="mm^3",
        constraints=(
            DesignConstraint(
                name="peak_stress_limit",
                kind="inequality_le",
                limit=3.5,
                unit="MPa",
            ),
        ),
        max_iterations=3,
        max_fem_evaluations=10,
        initial_trust_radius=0.25,
        gradient_check_interval=2,
    )
    bridge = Parametric3DDesignBridge(
        reloaded,
        Parametric3DDesignSpec(
            problem=prob,
            objective_metric="volume_mm3",
            constraint_metrics={"peak_stress_limit": "peak_von_mises_mpa"},
        ),
    )
    init_fem = bridge.evaluate_fem(spec.nominal_parameters())
    assert init_fem.succeeded and init_fem.objective is not None
    opt_rep = bridge.optimize()
    assert opt_rep.status in (
        DesignOptimizationStatus.CONVERGED_FEM_VERIFIED,
        DesignOptimizationStatus.BUDGET_EXHAUSTED_FEM_VERIFIED,
    )
    assert opt_rep.best_design is not None
    assert opt_rep.best_design.fem_feasible is True
    assert opt_rep.best_design.fem_objective < init_fem.objective
