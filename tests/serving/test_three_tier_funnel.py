"""Phase 2 tests: 3-tier verification funnel, CAD calibration, spectral repair, receipts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from piginot.bounds import (
    CadCalibrationRecord,
    NoFiniteBound,
    build_cad_calibration,
    calibrate_functional_band,
)
from piginot.bundle import load_bundle
from piginot.certify import (
    AdmissibilityThresholds,
    CertificateVerdict,
    SmoothedPeakVonMisesQoi,
    TotalStrainEnergyQoi,
    VerificationTier,
    estimate_qoi_dwr,
    evaluate_tier0_admissibility,
    verify_receipt_document,
)
from piginot.cli import main as cli_main
from piginot.cli import parse_request
from piginot.fem import ScikitFemBackend
from piginot.fem.linearization import FemLinearization
from piginot.serving import (
    ServingMode,
    ServingPath,
    SpectralRepairConfig,
    load_recipe,
    serve,
    spectrally_safe_repair,
)
from tests.fem.test_nonlinear_solver import _refined_unit_square, _traction_request

ROOT = Path(__file__).resolve().parents[2]
NOTCHED_BUNDLE = ROOT / "bundles" / "notched-v1-cp-freeze"
NOTCHED_REQUEST = ROOT / "bundles" / "requests" / "notched-in-domain.json"


def test_tier0_admissibility_checks_jacobian_finiteness_and_residuals() -> None:
    good_fields = {
        "u": np.zeros((4, 2)),
        "F_2D": np.broadcast_to(np.eye(2), (4, 2, 2)).copy(),
        "J_3D": np.ones(4),
        "P_served": np.zeros((4, 2, 2)),
    }
    res_ok = evaluate_tier0_admissibility(
        good_fields,
        thresholds=AdmissibilityThresholds(
            min_det_f_2d=0.1,
            max_dirichlet_deviation=1e-6,
            max_equilibrium_residual_rms=0.05,
            max_free_traction_rms=0.05,
            max_section_force_imbalance=0.1,
        ),
        dirichlet_deviation=1e-9,
        equilibrium_residual=np.full(4, 0.01),
        free_traction_residual=np.full(4, 0.01),
        section_forces=[100.0, 102.0, 99.0],
    )
    assert res_ok.passed is True
    assert res_ok.failed_checks == ()
    assert res_ok.min_det_f_2d == pytest.approx(1.0)

    bad_f = np.broadcast_to(np.eye(2), (4, 2, 2)).copy()
    bad_f[2, 0, 0] = -0.2  # negative det(F)
    res_bad = evaluate_tier0_admissibility(
        {**good_fields, "F_2D": bad_f},
        thresholds=AdmissibilityThresholds(
            min_det_f_2d=0.1,
            max_dirichlet_deviation=1e-6,
            max_section_force_imbalance=0.05,
        ),
        dirichlet_deviation=1e-2,
        section_forces=[100.0, 140.0],
    )
    assert res_bad.passed is False
    assert "det_f_2d_below_floor" in res_bad.failed_checks
    assert "dirichlet_deviation_exceeded" in res_bad.failed_checks
    assert "section_force_imbalance_exceeded" in res_bad.failed_checks


def test_total_strain_energy_and_smoothed_peak_vm_qoi_gradients_and_dwr() -> None:
    request = replace(_traction_request(element_order=1), mesh=_refined_unit_square())
    attempt = ScikitFemBackend().solve_many((request,))[0]
    assert attempt.success and attempt.result is not None
    lin = FemLinearization(request)
    u_fem = np.asarray(attempt.result.displacement, dtype=np.float64)
    x, y = lin.dof_coordinates.T
    u_surr = u_fem + 1e-2 * np.column_stack(
        (x * np.sin(np.pi * y + 0.2), y * np.cos(np.pi * x + 0.1))
    )
    vec = lin.with_exact_dirichlet(lin.vector(u_surr))
    state = lin.state(vec)

    qois = (
        TotalStrainEnergyQoi(),
        SmoothedPeakVonMisesQoi(p_norm=6.0),
    )
    rng = np.random.default_rng(7)
    d_free = rng.standard_normal(lin.constraints.free.size)
    d_free /= np.linalg.norm(d_free)
    d_full = np.zeros(lin.dof_count, dtype=np.float64)
    d_full[lin.constraints.free] = d_free
    eps = 1e-6
    plus = lin.state(vec + eps * d_full)
    minus = lin.state(vec - eps * d_full)

    for qoi in qois:
        grad = qoi.free_gradient(lin, state)
        analytic = float(grad @ d_free)
        fd = (qoi.evaluate(lin, plus) - qoi.evaluate(lin, minus)) / (2.0 * eps)
        assert analytic == pytest.approx(fd, rel=1e-5, abs=1e-7)

        est = estimate_qoi_dwr(lin, u_surr, qoi)
        assert est.verdict is CertificateVerdict.OK
        assert est.dwr_estimate is not None
        assert est.adjoint_residual_norm is not None
        assert est.adjoint_residual_norm < 1e-10


def test_cad_conformal_and_functional_band_calibration_round_trip(
    tmp_path: Path,
) -> None:
    # Fewer than 19 samples at alpha=0.05 must refuse finite bound.
    with pytest.raises(NoFiniteBound):
        build_cad_calibration(
            bundle_name="demo",
            bundle_manifest_sha256="a" * 64,
            problem_hash="b" * 64,
            qoi_scores={"avg_p11": ("MPa", [0.1] * 10)},
            alpha=0.05,
        )

    rng = np.random.default_rng(11)
    qoi_scores = {"avg_p11": ("MPa", rng.uniform(0.01, 0.5, size=25).tolist())}
    func_res = {
        "u": ("mm", rng.normal(0.0, 0.02, size=(25, 10)).tolist()),
    }
    record = build_cad_calibration(
        bundle_name="demo",
        bundle_manifest_sha256="a" * 64,
        problem_hash="b" * 64,
        qoi_scores=qoi_scores,
        alpha=0.05,
        functional_residuals=func_res,
        parameter_bounds={"strain": (0.05, 0.2)},
    )
    out = tmp_path / "cal.json"
    record.save(out)
    loaded = CadCalibrationRecord.load(out, expected_manifest_sha256="a" * 64)
    assert loaded.calibration_sha256 == record.calibration_sha256
    assert loaded.evaluate({"strain": 0.1})["refused"] is False
    assert loaded.evaluate({"strain": 0.5})["refused"] is True
    stat_in = loaded.statistical_record("avg_p11", in_envelope=True)
    assert stat_in is not None and stat_in.refused is False
    stat_out = loaded.statistical_record("avg_p11", in_envelope=False)
    assert stat_out is not None and stat_out.refused is True

    fb = calibrate_functional_band(
        field_name="P",
        unit="MPa",
        residuals=rng.normal(0.0, 0.1, size=(20, 5)).tolist(),
        alpha=0.05,
    )
    assert fb.nominal_coverage == pytest.approx(0.95)
    assert len(fb.band_half_widths([1.0, 2.0, 0.5])) == 3  # type: ignore[arg-type]


def test_spectrally_safe_repair_recovers_folded_start_to_spd_basin() -> None:
    request = replace(_traction_request(element_order=1), mesh=_refined_unit_square())
    backend = ScikitFemBackend()
    lin = backend.linearize(request)
    # Deliberately invert elements on free DOFs so direct state() is inadmissible.
    xy = lin.dof_coordinates
    folded_nodal = np.column_stack((-2.5 * xy[:, 0], 1.8 * xy[:, 1]))
    folded_vec = lin.with_exact_dirichlet(lin.vector(folded_nodal))

    repaired, report = spectrally_safe_repair(
        lin, folded_vec, config=SpectralRepairConfig(max_steps=30, min_det_f=0.05)
    )
    assert report.needed_repair is True
    assert report.initial_admissible is False
    assert report.succeeded is True
    assert report.final_positive_definite is True
    assert report.final_min_det_f is not None and report.final_min_det_f >= 0.05

    attempt = backend.solve_from(request, repaired, (1.0,))
    assert attempt.success is True

    # Already-converged solution needs zero repair steps.
    assert attempt.result is not None
    sol_vec = lin.vector(np.asarray(attempt.result.displacement, dtype=np.float64))
    _, noop = spectrally_safe_repair(lin, sol_vec)
    assert noop.needed_repair is False
    assert noop.steps_taken == 0


def test_serve_emits_verified_rfc8785_receipt_and_cli_calibrate(
    tmp_path: Path,
) -> None:
    bundle = load_bundle(NOTCHED_BUNDLE)
    inputs, queries = parse_request(json.loads(NOTCHED_REQUEST.read_text("utf-8")))

    # Build a CAD calibration sidecar via CLI and serve with --receipt --tier0.
    scores_path = tmp_path / "scores.json"
    cal_path = tmp_path / "notched-cal.json"
    scores_path.write_text(
        json.dumps(
            {
                "qoi_scores": {
                    "avg_p11": {
                        "unit": "MPa",
                        "scores": [0.2 + 0.01 * i for i in range(25)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    rc = cli_main(
        [
            "calibrate",
            str(NOTCHED_BUNDLE),
            "--scores",
            str(scores_path),
            "--alpha",
            "0.05",
            "--output",
            str(cal_path),
        ]
    )
    assert rc == 0
    assert cal_path.is_file()
    cad_cal = CadCalibrationRecord.load(
        cal_path, expected_manifest_sha256=bundle.manifest.manifest_sha256
    )

    served_fast = serve(
        bundle,
        inputs,
        queries,
        mode=ServingMode.FAST,
        cad_calibration=cad_cal,
        tier0_check=True,
        emit_receipt=True,
    )
    doc_fast = served_fast.document()
    assert doc_fast["trust_receipt"]["verification_tier"] == (
        VerificationTier.CONFORMAL_BOUNDED.value
    )
    ok_f, errs_f = verify_receipt_document(doc_fast["trust_receipt"])
    assert ok_f is True and errs_f == ()

    recipe = load_recipe(bundle, NOTCHED_BUNDLE)
    served_cert = serve(
        bundle,
        inputs,
        queries,
        mode=ServingMode.CERTIFIED,
        recipe=recipe,
        cad_calibration=cad_cal,
        trust_report=True,
        emit_receipt=True,
    )
    doc_cert = served_cert.document()
    assert doc_cert["path"] == ServingPath.OPERATOR_CERTIFICATE.value
    assert doc_cert["trust_receipt"]["verification_tier"] == (
        VerificationTier.DWR_BOUNDED.value
    )
    ok_c, errs_c = verify_receipt_document(doc_cert["trust_receipt"])
    assert ok_c is True and errs_c == ()
    # Statistical calibration record is populated inside CADTrustReport.
    qoi_by_name = {
        row["qoi_name"]: row for row in doc_cert["trust_report"]["qoi_results"]
    }
    assert (
        qoi_by_name["avg_p11"]["error_components"]["statistical_calibration"]["refused"]
        is False
    )


def test_portable_user_cad_bundle_serves_certified_and_exact_without_recipe(
    tmp_path: Path,
) -> None:
    from piginot.contracts import ProblemSpec
    from tests.bundle import test_portable_problem as tpp

    raw_doc = json.loads((ROOT / "campaigns/dogbone_v1/problem_spec.json").read_bytes())
    raw_doc["fem"]["element_order"] = 1
    raw_doc["fem"]["load_steps"] = 4
    spec = ProblemSpec.from_document(raw_doc)
    prob = tpp.portable_problem(spec)
    bundle_dir = tpp.build(tmp_path / "portable-cad-bundle", problem=prob)
    bundle = load_bundle(bundle_dir)
    inputs = tpp.material_inputs(spec)

    # Certified serving directly from embedded ProblemSpec (no FemRecipe needed!).
    cert = serve(
        bundle,
        inputs,
        mode=ServingMode.CERTIFIED,
        trust_report=True,
        emit_receipt=True,
    )
    cert_doc = cert.document()
    assert cert_doc["trust_receipt"]["claim_level"] == "NOT_VERIFIED"
    assert cert_doc["trust_receipt"]["artifact_digests"]["problem_hash"] == (
        prob.problem_hash
    )
    ok_cert, errs_cert = verify_receipt_document(cert_doc["trust_receipt"])
    assert ok_cert is True and errs_cert == ()

    # Exact serving with spectrally safe warm-start repair directly on portable bundle.
    exact = serve(
        bundle,
        inputs,
        mode=ServingMode.EXACT,
        spectral_repair=True,
        emit_receipt=True,
    )
    exact_doc = exact.document()
    assert exact_doc["exact"] is not None
    assert exact_doc["trust_receipt"]["claim_level"] == "NOT_VERIFIED"
    assert exact_doc["trust_receipt"]["verification_tier"] == (
        VerificationTier.DISCRETE_FEM_VERIFIED.value
    )
    ok_ex, errs_ex = verify_receipt_document(exact_doc["trust_receipt"])
    assert ok_ex is True and errs_ex == ()
