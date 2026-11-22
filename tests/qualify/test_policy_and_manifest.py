"""Unit tests for qualification policy, manifest, diff, and Markdown report contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from formamech.qualify import (
    QUALIFICATION_MANIFEST_SCHEMA,
    QUALIFICATION_POLICY_SCHEMA,
    CalibrationPolicy,
    DwrPolicy,
    FieldTolerance,
    QoiTolerance,
    QualificationCheckResult,
    QualificationIntegrityError,
    QualificationManifest,
    QualificationPolicy,
    QualificationVerdict,
    compare_qualifications,
    default_development_policy,
    default_holdout_policy,
    render_qualification_report_markdown,
    verify_manifest_document,
)

from piginot.verification.status import CheckStatus, ClaimLevel


def test_policy_roundtrip_presets_and_validation(tmp_path: Path) -> None:
    dev = default_development_policy()
    assert dev.target_claim_level is ClaimLevel.DEVELOPMENT_VERIFIED
    assert dev.document()["schema"] == QUALIFICATION_POLICY_SCHEMA

    ho = default_holdout_policy()
    assert ho.target_claim_level is ClaimLevel.HOLDOUT_VERIFIED
    assert ho.calibration.min_calibration_cases == 19

    pfile = tmp_path / "policy.json"
    ho.save(pfile)
    reloaded = QualificationPolicy.load(pfile)
    assert reloaded == ho
    assert reloaded.policy_sha256 == ho.policy_sha256

    # Tampered policy digest must fail closed
    tampered = json.loads(pfile.read_text(encoding="utf-8"))
    tampered["policy_id"] = "tampered"
    pfile.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        QualificationPolicy.load(pfile)

    # Validation edge cases
    with pytest.raises(ValueError, match="at least one"):
        QoiTolerance()
    with pytest.raises(ValueError, match="split-conformal"):
        CalibrationPolicy(enabled=True, alpha=0.05, min_calibration_cases=5)
    with pytest.raises(ValueError, match="effectivity_max"):
        DwrPolicy(effectivity_min=2.0, effectivity_max=1.0)
    with pytest.raises(ValueError, match="target_claim_level"):
        QualificationPolicy(
            policy_id="bad",
            target_claim_level=ClaimLevel.NOT_VERIFIED,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        FieldTolerance(disp_rel_l2_max=0.0)


def _sample_manifest(
    *,
    verdict: QualificationVerdict = QualificationVerdict.QUALIFIED,
    granted: ClaimLevel = ClaimLevel.DEVELOPMENT_VERIFIED,
    target: ClaimLevel = ClaimLevel.DEVELOPMENT_VERIFIED,
    disp_q: float = 0.015,
    stress_q: float = 0.04,
    fam_status: str = "PASS",
) -> QualificationManifest:
    return QualificationManifest(
        created_utc="2026-11-22T15:00:00Z",
        bundle_name="demo-cad",
        bundle_manifest_sha256="a" * 64,
        approved_envelope_sha256="b" * 64,
        problem_hash="c" * 64,
        policy_id="dev-standard",
        policy_sha256="d" * 64,
        target_claim_level=target,
        granted_claim_level=granted,
        verdict=verdict,
        solver_backend_id="scikit-fem-v1",
        families=(
            {
                "family_id": "integrity",
                "required": True,
                "status": "PASS",
                "missing_checks": [],
            },
            {
                "family_id": "fem_accuracy_dev",
                "required": True,
                "status": fam_status,
                "missing_checks": [],
            },
        ),
        checks=(
            QualificationCheckResult(
                check_id="integrity.bundle_verified",
                family_id="integrity",
                status=CheckStatus.PASS,
                detail="ok",
                measured={"bundle_manifest_sha256": "a" * 64},
            ),
            QualificationCheckResult(
                check_id="dev.disp_rel_l2",
                family_id="fem_accuracy_dev",
                status=CheckStatus(fam_status),
                detail="checked",
                measured={"value": disp_q},
            ),
        ),
        dev_metrics={
            "case_count": 10,
            "disp_rel_l2_quantile": disp_q,
            "disp_rel_l2_max": disp_q * 1.1,
            "stress_rel_l2_quantile": stress_q,
            "stress_rel_l2_max": stress_q * 1.1,
            "min_det_f": 0.95,
        },
        holdout_metrics=None,
        calibration_sha256="e" * 64,
        calibration_path="cad-calibration.json",
        amortization={
            "online_speedup": 15.0,
            "fem_seconds": 1.5,
            "inference_seconds": 0.1,
            "calibration_only_break_even_queries": 10.7,
            "break_even_queries": 32.1,
        },
    )


def test_manifest_roundtrip_tamper_invariants_and_report(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    assert manifest.document()["schema"] == QUALIFICATION_MANIFEST_SCHEMA

    mfile = tmp_path / "qualification-manifest.json"
    manifest.save(mfile)
    reloaded = QualificationManifest.load(mfile)
    assert reloaded.manifest_sha256 == manifest.manifest_sha256

    ok, errs = verify_manifest_document(
        manifest.full_document(),
        expected_bundle_sha256="a" * 64,
        expected_policy_sha256="d" * 64,
        expected_envelope_sha256="b" * 64,
    )
    assert ok is True
    assert errs == ()

    # Mismatches and tampering must fail closed
    ok_bad, errs_bad = verify_manifest_document(
        manifest.full_document(),
        expected_bundle_sha256="f" * 64,
        expected_policy_sha256="0" * 64,
        expected_envelope_sha256="1" * 64,
    )
    assert ok_bad is False
    assert len(errs_bad) == 3

    tampered = manifest.full_document()
    tampered["verdict"] = "NOT_QUALIFIED"
    ok_t, errs_t = verify_manifest_document(tampered)
    assert ok_t is False
    assert errs_t

    # Invariant checks: claim ceiling & verdict/claim alignment
    with pytest.raises(ValueError, match="NOT_QUALIFIED"):
        _sample_manifest(
            verdict=QualificationVerdict.NOT_QUALIFIED,
            granted=ClaimLevel.DEVELOPMENT_VERIFIED,
        )
    with pytest.raises(ValueError, match="QUALIFIED"):
        _sample_manifest(
            verdict=QualificationVerdict.QUALIFIED,
            granted=ClaimLevel.NOT_VERIFIED,
        )
    with pytest.raises(ValueError, match="exceed target_claim_level"):
        _sample_manifest(
            verdict=QualificationVerdict.QUALIFIED,
            granted=ClaimLevel.HOLDOUT_VERIFIED,
            target=ClaimLevel.DEVELOPMENT_VERIFIED,
        )
    with pytest.raises(QualificationIntegrityError):
        QualificationManifest.from_document({"schema": "wrong"})

    md = render_qualification_report_markdown(manifest)
    assert "# FormaMech Numerical Model Qualification Report" in md
    assert "`QUALIFIED`" in md
    assert "`DEVELOPMENT_VERIFIED`" in md
    assert "ASME V&V 10" in md
    assert "`experimental_validation`: `False`" in md
    assert "15x" in md


def test_compare_qualifications_detects_regressions() -> None:
    baseline = _sample_manifest(disp_q=0.010, stress_q=0.030)
    improved = _sample_manifest(disp_q=0.008, stress_q=0.025)
    diff_ok = compare_qualifications(baseline, improved)
    assert diff_ok.regressed is False
    assert diff_ok.regressions == ()

    regressed = _sample_manifest(
        verdict=QualificationVerdict.NOT_QUALIFIED,
        granted=ClaimLevel.NOT_VERIFIED,
        disp_q=0.025,
        stress_q=0.060,
        fam_status="FAIL",
    )
    diff_bad = compare_qualifications(baseline, regressed)
    assert diff_bad.regressed is True
    assert any("NOT_QUALIFIED" in msg for msg in diff_bad.regressions)
    assert any("granted_claim_level" in msg for msg in diff_bad.regressions)
    assert any("dev.disp_rel_l2_quantile" in msg for msg in diff_bad.regressions)
    assert any("fem_accuracy_dev" in msg for msg in diff_bad.regressions)


def test_qualify_import_boundary_and_contract_edge_cases(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    from formamech.qualify import (
        GateReasonCode,
        check_deployment_gate,
        parse_cases_document,
    )

    # 1) Runtime import boundary: importing formamech.qualify loads zero FEM/CAD stacks
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(root / "src"), os.environ.get("PYTHONPATH", "")))
        ),
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, formamech.qualify\n"
                "for mod in ('piginot.fem', 'skfem', 'scipy', 'sympy', 'gmsh', 'ezdxf'):\n"
                "    assert mod not in sys.modules, f'unexpected import: {mod}'\n"
            ),
        ],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr

    # 2) Policy validation branches
    assert QoiTolerance(max_abs_error=1.5, quantile=1.0).max_abs_error == 1.5
    with pytest.raises(ValueError, match="finite number"):
        FieldTolerance(disp_rel_l2_max=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        FieldTolerance(disp_rel_l2_max=float("inf"))
    with pytest.raises(ValueError, match="nonnegative"):
        FieldTolerance(max_dirichlet_residual=-1.0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        FieldTolerance(quantile=1.5)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        CalibrationPolicy(enabled=False, alpha=1.0, min_calibration_cases=1)
    with pytest.raises(ValueError, match="integer >= 1"):
        CalibrationPolicy(enabled=False, min_calibration_cases=0)
    with pytest.raises(TypeError, match="boolean"):
        CalibrationPolicy(enabled="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="boolean"):
        CalibrationPolicy(require_functional_band="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="boolean"):
        DwrPolicy(enabled="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="policy_id"):
        QualificationPolicy(policy_id="  ")
    with pytest.raises(TypeError, match="FieldTolerance"):
        QualificationPolicy(policy_id="x", field_tolerances="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CalibrationPolicy"):
        QualificationPolicy(policy_id="x", calibration="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DwrPolicy"):
        QualificationPolicy(policy_id="x", dwr="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        QualificationPolicy(
            policy_id="x", qoi_tolerances={" ": QoiTolerance(max_abs_error=1.0)}
        )
    with pytest.raises(TypeError, match="QoiTolerance"):
        QualificationPolicy(policy_id="x", qoi_tolerances={"q": "bad"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="boolean"):
        QualificationPolicy(policy_id="x", report_breakeven="no")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="notes"):
        QualificationPolicy(policy_id="x", notes=123)  # type: ignore[arg-type]

    bad_pol = tmp_path / "list.json"
    bad_pol.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        QualificationPolicy.load(bad_pol)

    # 3) Manifest validation branches
    with pytest.raises(ValueError, match="check_id"):
        QualificationCheckResult(
            check_id="", family_id="f", status=CheckStatus.PASS, detail=""
        )
    with pytest.raises(ValueError, match="family_id"):
        QualificationCheckResult(
            check_id="c", family_id="", status=CheckStatus.PASS, detail=""
        )
    with pytest.raises(ValueError, match="bundle_name"):
        QualificationManifest(
            created_utc="2026-11-22T00:00:00Z",
            bundle_name="",
            bundle_manifest_sha256="a" * 64,
            approved_envelope_sha256="b" * 64,
            problem_hash=None,
            policy_id="p",
            policy_sha256="c" * 64,
            target_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
            granted_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
            verdict=QualificationVerdict.QUALIFIED,
            solver_backend_id="s",
            families=(),
            checks=(),
            dev_metrics={},
        )
    with pytest.raises(ValueError, match="64-character"):
        QualificationManifest(
            created_utc="2026-11-22T00:00:00Z",
            bundle_name="b",
            bundle_manifest_sha256="short",
            approved_envelope_sha256="b" * 64,
            problem_hash=None,
            policy_id="p",
            policy_sha256="c" * 64,
            target_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
            granted_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
            verdict=QualificationVerdict.QUALIFIED,
            solver_backend_id="s",
            families=(),
            checks=(),
            dev_metrics={},
        )
    with pytest.raises(ValueError, match="experimental_validation"):
        QualificationManifest(
            created_utc="2026-11-22T00:00:00Z",
            bundle_name="b",
            bundle_manifest_sha256="a" * 64,
            approved_envelope_sha256="b" * 64,
            problem_hash=None,
            policy_id="p",
            policy_sha256="c" * 64,
            target_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
            granted_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
            verdict=QualificationVerdict.QUALIFIED,
            solver_backend_id="s",
            families=(),
            checks=(),
            dev_metrics={},
            epistemic_scope={
                "experimental_validation": True,
                "regulatory_certification": False,
            },
        )
    with pytest.raises(ValueError, match="regulatory_certification"):
        QualificationManifest(
            created_utc="2026-11-22T00:00:00Z",
            bundle_name="b",
            bundle_manifest_sha256="a" * 64,
            approved_envelope_sha256="b" * 64,
            problem_hash=None,
            policy_id="p",
            policy_sha256="c" * 64,
            target_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
            granted_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
            verdict=QualificationVerdict.QUALIFIED,
            solver_backend_id="s",
            families=(),
            checks=(),
            dev_metrics={},
            epistemic_scope={
                "experimental_validation": False,
                "regulatory_certification": True,
            },
        )
    with pytest.raises(ValueError, match="REQUIRED_LIMITATION_PARAGRAPH"):
        QualificationManifest(
            created_utc="2026-11-22T00:00:00Z",
            bundle_name="b",
            bundle_manifest_sha256="a" * 64,
            approved_envelope_sha256="b" * 64,
            problem_hash=None,
            policy_id="p",
            policy_sha256="c" * 64,
            target_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
            granted_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
            verdict=QualificationVerdict.QUALIFIED,
            solver_backend_id="s",
            families=(),
            checks=(),
            dev_metrics={},
            limitations="short",
        )
    ok_nondict, errs_nd = verify_manifest_document("bad")  # type: ignore[arg-type]
    assert ok_nondict is False and errs_nd
    ok_nodigest, errs_dg = verify_manifest_document(_sample_manifest().document())
    assert ok_nodigest is False and any("missing" in e for e in errs_dg)

    # 4) Gate edge cases: invalid bundle, invalid JSON manifest, bundle/envelope mismatch
    bad_bundle = check_deployment_gate(tmp_path / "nonexistent-bundle")
    assert bad_bundle.allowed is False
    assert GateReasonCode.BUNDLE_INVALID in bad_bundle.reason_codes

    notched = root / "bundles" / "notched-v1-cp-freeze"
    corrupt_m = tmp_path / "corrupt.json"
    corrupt_m.write_text("{not-json", encoding="utf-8")
    g_corrupt = check_deployment_gate(notched, corrupt_m)
    assert g_corrupt.allowed is False
    assert GateReasonCode.MANIFEST_INVALID in g_corrupt.reason_codes

    wrong_bundle_m = tmp_path / "wrong-bundle.json"
    _sample_manifest().save(wrong_bundle_m)
    g_mismatch = check_deployment_gate(notched, wrong_bundle_m)
    assert g_mismatch.allowed is False
    assert GateReasonCode.BUNDLE_MISMATCH in g_mismatch.reason_codes
    assert GateReasonCode.ENVELOPE_MISMATCH in g_mismatch.reason_codes

    # 5) Case parsing edge cases
    cases_list = parse_cases_document(
        [{"case_id": "x1", "inputs": {"a": {"value": 1.0, "unit": "mm"}}}]
    )
    assert len(cases_list) == 1
    with pytest.raises(ValueError, match="cases"):
        parse_cases_document({"wrong": []})
    with pytest.raises(ValueError, match="JSON object or list"):
        parse_cases_document("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid partition"):
        parse_cases_document([{"partition": "BAD", "inputs": {}}])
    with pytest.raises(ValueError, match="inputs"):
        parse_cases_document([{"case_id": "x", "inputs": "bad"}])
    with pytest.raises(ValueError, match="value"):
        parse_cases_document([{"case_id": "x", "inputs": {"a": {"value": 1.0}}}])
