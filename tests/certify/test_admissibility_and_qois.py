"""Unit tests for Tier 0 admissibility gate, stress/von-Mises DWR QoIs, and receipt edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from campaigns.vvuq_suite import rectangular_strip_mesh
from piginot.certify import (
    AdmissibilityThresholds,
    ArtifactDigests,
    ReactionForceQoi,
    RegionalAverageStressQoi,
    SmoothedPeakVonMisesQoi,
    TotalStrainEnergyQoi,
    TrustReceipt,
    VerificationTier,
    estimate_qoi_dwr,
    evaluate_tier0_admissibility,
    verify_receipt_document,
)
from piginot.contracts import (
    BoundaryConditionSpec,
    ExecutionProfile,
    MaterialSpec,
    Quantity,
    Unit,
)
from piginot.fem import FemRequest, FemSolverConfig, ScikitFemBackend, TorchFemBackend


def test_tier0_admissibility_passes_and_catches_all_violations() -> None:
    thresholds = AdmissibilityThresholds(
        min_det_f_2d=0.05,
        min_det_f_3d=0.05,
        max_dirichlet_deviation=1.0e-4,
        max_equilibrium_residual_rms=0.1,
        max_free_traction_rms=0.1,
        max_section_force_imbalance=0.05,
    )
    eye = np.broadcast_to(np.eye(2), (4, 2, 2)).copy()
    clean = evaluate_tier0_admissibility(
        {"u": np.zeros((4, 2)), "F_2D": eye, "J_3D": np.ones(4)},
        thresholds=thresholds,
        dirichlet_deviation=0.0,
        equilibrium_residual=np.array([0.01, -0.01]),
        free_traction_residual=np.array([]),
        section_forces=[10.0, 10.1, 9.95],
    )
    assert clean.passed is True
    assert clean.document()["passed"] is True

    # Inverted 2D & 3D + exceeded residuals + imbalance
    bad_f = eye.copy()
    bad_f[0, 0, 0] = -0.5
    bad = evaluate_tier0_admissibility(
        {"u": np.zeros((4, 2)), "F_2D": bad_f, "J_3D": np.array([1.0, -0.1])},
        thresholds=thresholds,
        dirichlet_deviation=0.5,
        equilibrium_residual=np.array([1.0, 2.0]),
        free_traction_residual=np.array([0.5, 0.5]),
        section_forces=[10.0, 20.0],
    )
    assert bad.passed is False
    for expected in (
        "det_f_2d_below_floor",
        "det_f_3d_below_floor",
        "dirichlet_deviation_exceeded",
        "equilibrium_residual_exceeded",
        "free_traction_residual_exceeded",
        "section_force_imbalance_exceeded",
    ):
        assert expected in bad.failed_checks

    # Non-finite checks across all inputs
    nan_f = eye.copy()
    nan_f[0, 0, 0] = float("nan")
    nan_res = evaluate_tier0_admissibility(
        {
            "u": np.array([float("nan")]),
            "F_2D": nan_f,
            "J_3D": np.array([float("nan")]),
        },
        thresholds=thresholds,
        equilibrium_residual=np.array([float("inf")]),
        free_traction_residual=np.array([float("nan")]),
        section_forces=[1.0, float("nan")],
    )
    assert nan_res.passed is False and nan_res.all_finite is False
    assert "non_finite:u" in nan_res.failed_checks
    assert "det_f_2d_non_finite" in nan_res.failed_checks
    assert "det_f_3d_non_finite" in nan_res.failed_checks
    assert "equilibrium_residual_non_finite" in nan_res.failed_checks
    assert "free_traction_residual_non_finite" in nan_res.failed_checks
    assert "section_force_non_finite" in nan_res.failed_checks

    # Threshold validation guards
    with pytest.raises(TypeError):
        AdmissibilityThresholds(min_det_f_2d=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AdmissibilityThresholds(min_det_f_2d=0.0)
    with pytest.raises(TypeError):
        AdmissibilityThresholds(max_equilibrium_residual_rms=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AdmissibilityThresholds(max_free_traction_rms=-1.0)


def test_all_four_dwr_qois_on_both_backends_and_receipt_error_paths() -> None:
    mesh = rectangular_strip_mesh(length_mm=10.0, height_mm=5.0, nx=4, ny=2)
    mat = MaterialSpec(
        name="nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(10.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.32, Unit.ONE).to_canonical(),
        },
    )
    req = FemRequest(
        request_id="certify-qois",
        mesh=mesh,
        material=mat,
        boundary_conditions=(
            BoundaryConditionSpec(
                name="left_fix",
                boundary_tag="left",
                kind="dirichlet",
                components=("x", "y"),
                values=(
                    Quantity(0.0, Unit.MM).to_canonical(),
                    Quantity(0.0, Unit.MM).to_canonical(),
                ),
            ),
            BoundaryConditionSpec(
                name="right_pull",
                boundary_tag="right",
                kind="dirichlet",
                components=("x",),
                values=(Quantity(0.6, Unit.MM).to_canonical(),),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(element_order=1, solver_tolerance=1.0e-11),
        execution_profile=ExecutionProfile(
            python_version="3.13",
            library_versions={"numpy": np.__version__},
            blas_threads=1,
            device="cpu",
            dtype="float64",
            rng_algorithm="none",
            seed=0,
        ),
    )
    sk = ScikitFemBackend()
    th = TorchFemBackend()
    att = sk.solve_many((req,))[0]
    assert att.success and att.result is not None
    u_fem = np.asarray(att.result.displacement, dtype=np.float64)
    u_pert = u_fem.copy()
    nodes_arr = np.asarray(mesh.nodes, dtype=np.float64)
    interior = (nodes_arr[:, 0] > 1.0e-9) & (nodes_arr[:, 0] < 10.0 - 1.0e-9)
    u_pert[interior, 0] += 0.015

    lin_sk = sk.linearize(req)
    lin_th = th.linearize(req)
    qois = (
        ReactionForceQoi("rxn", "right_pull"),
        RegionalAverageStressQoi("avg_p11", component=(0, 0)),
        TotalStrainEnergyQoi("energy"),
        SmoothedPeakVonMisesQoi("peak_vm"),
    )
    for qoi in qois:
        e_sk = estimate_qoi_dwr(lin_sk, u_pert, qoi)
        e_th = estimate_qoi_dwr(lin_th, u_pert, qoi)
        assert e_sk.dwr_estimate is not None and e_th.dwr_estimate is not None
        assert e_th.qoi_value == pytest.approx(e_sk.qoi_value, rel=1.0e-9)
        assert e_th.dwr_estimate == pytest.approx(e_sk.dwr_estimate, rel=1.0e-7)

    # Receipt error paths
    receipt = TrustReceipt(
        mode="fast",
        verification_tier=VerificationTier.SURROGATE_ADMISSIBLE,
        claim_level="NOT_VERIFIED",
        serving_path="fast_surrogate",
        artifact_digests=ArtifactDigests(
            cad_source_sha256="a" * 64,
            mesh_id="b" * 64,
            weights_sha256="c" * 64,
            solver_tolerances={"tol": 1.0e-10},
        ),
        tier0_admissibility={"passed": True},
    )
    doc = receipt.document()
    ok, errs = verify_receipt_document(doc)
    assert ok and not errs

    bad_schema = {**doc, "schema": "wrong"}
    ok2, errs2 = verify_receipt_document(bad_schema)
    assert not ok2 and any("schema" in e for e in errs2)

    bad_exp = {
        **doc,
        "epistemic_scope": {**doc["epistemic_scope"], "experimental_validation": True},
    }
    ok3, errs3 = verify_receipt_document(bad_exp)
    assert not ok3 and errs3
