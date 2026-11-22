"""End-to-end tests for qualification pipeline, deployment gate, CLI, export, and MCP."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from formamech.mcp_server import FormaMechMcpServer
from formamech.qualify import (
    CalibrationPolicy,
    DeploymentGateError,
    DwrPolicy,
    FieldTolerance,
    GateReasonCode,
    QoiTolerance,
    QualificationCase,
    QualificationPolicy,
    QualificationVerdict,
    check_deployment_gate,
    enforce_deployment_gate,
    input_fingerprint,
    load_cases_file,
    qualify_bundle,
    sample_qualification_cases,
)

from piginot.bundle import load_bundle
from piginot.cli import main as cli_main
from piginot.verification.status import ClaimLevel

ROOT = Path(__file__).resolve().parents[2]
NOTCHED_BUNDLE = ROOT / "bundles" / "notched-v1-cp-freeze"
NOTCHED_REQUEST = ROOT / "bundles" / "requests" / "notched-in-domain.json"


def _fast_dev_policy(
    *,
    disp_max: float = 0.08,
    stress_max: float = 0.15,
    min_cal: int = 3,
    alpha: float = 0.25,
    target: ClaimLevel = ClaimLevel.DEVELOPMENT_VERIFIED,
    min_holdout_coverage: float | None = None,
) -> QualificationPolicy:
    return QualificationPolicy(
        policy_id=f"test-{target.value.lower()}",
        target_claim_level=target,
        field_tolerances=FieldTolerance(
            disp_rel_l2_max=disp_max,
            stress_rel_l2_max=stress_max,
            quantile=0.90,
        ),
        qoi_tolerances={
            "avg_p11": QoiTolerance(max_rel_error=stress_max, quantile=0.90),
            "total_strain_energy": QoiTolerance(
                max_rel_error=stress_max, quantile=0.90
            ),
        },
        calibration=CalibrationPolicy(
            enabled=True,
            alpha=alpha,
            min_calibration_cases=min_cal,
            require_functional_band=True,
            min_holdout_coverage=min_holdout_coverage,
        ),
        dwr=DwrPolicy(
            enabled=True,
            effectivity_min=0.05,
            effectivity_max=20.0,
            min_valid_fraction=0.5,
        ),
        training_wall_seconds=12.0,
    )


def test_end_to_end_dev_and_holdout_qualification_with_promoted_bundle(
    tmp_path: Path,
) -> None:
    bundle = load_bundle(NOTCHED_BUNDLE)
    dev_cases, ho_cases = sample_qualification_cases(
        bundle, dev_count=3, holdout_count=2, seed=42
    )
    assert len(dev_cases) == 3
    assert len(ho_cases) == 2
    assert {c.fingerprint for c in dev_cases}.isdisjoint(
        {c.fingerprint for c in ho_cases}
    )

    # 1. HOLDOUT_VERIFIED qualification with promoted bundle emission
    policy_ho = _fast_dev_policy(
        target=ClaimLevel.HOLDOUT_VERIFIED,
        min_holdout_coverage=0.5,
    )
    out_dir = tmp_path / "qual-ho"
    promoted_dir = tmp_path / "promoted-bundle"
    res = qualify_bundle(
        NOTCHED_BUNDLE,
        policy_ho,
        dev_cases=dev_cases,
        holdout_cases=ho_cases,
        backend="torch-fem",
        output_dir=out_dir,
        emit_qualified_bundle=promoted_dir,
    )
    assert res.qualified is True
    assert res.manifest.verdict is QualificationVerdict.QUALIFIED
    assert res.manifest.granted_claim_level is ClaimLevel.HOLDOUT_VERIFIED
    assert res.calibration is not None
    assert "displacement_mm" in res.calibration.functional_bands
    assert res.manifest_path is not None and res.manifest_path.is_file()
    assert res.report_path is not None and res.report_path.is_file()
    assert res.calibration_path is not None and res.calibration_path.is_file()
    assert res.manifest.amortization is not None
    assert res.promoted_bundle_path == promoted_dir

    # Promoted bundle loads cleanly with upgraded ClaimLevel
    promoted = load_bundle(promoted_dir)
    assert promoted.claim_level is ClaimLevel.HOLDOUT_VERIFIED
    assert (
        promoted.verification["qualification_manifest_sha256"]
        == res.manifest.manifest_sha256
    )

    # 2. Deployment gate passes for target bundle & policy, fails for mismatched policy
    gate_ok = enforce_deployment_gate(
        NOTCHED_BUNDLE,
        res.manifest_path,
        policy=policy_ho,
        min_claim_level=ClaimLevel.HOLDOUT_VERIFIED,
    )
    assert gate_ok.allowed is True
    assert gate_ok.granted_claim_level is ClaimLevel.HOLDOUT_VERIFIED

    other_policy = _fast_dev_policy(disp_max=0.07)
    gate_bad_pol = check_deployment_gate(
        NOTCHED_BUNDLE, res.manifest_path, policy=other_policy
    )
    assert gate_bad_pol.allowed is False
    assert GateReasonCode.POLICY_MISMATCH in gate_bad_pol.reason_codes

    # Gate refuses missing manifest and tampered manifest
    gate_missing = check_deployment_gate(NOTCHED_BUNDLE, tmp_path / "missing.json")
    assert gate_missing.allowed is False
    assert GateReasonCode.MANIFEST_MISSING in gate_missing.reason_codes

    tampered_path = tmp_path / "tampered.json"
    tampered_doc = json.loads(res.manifest_path.read_text(encoding="utf-8"))
    tampered_doc["bundle_name"] = "tampered"
    tampered_path.write_text(json.dumps(tampered_doc), encoding="utf-8")
    with pytest.raises(DeploymentGateError) as exc_info:
        enforce_deployment_gate(NOTCHED_BUNDLE, tampered_path)
    assert GateReasonCode.MANIFEST_INVALID in exc_info.value.reason_codes


def test_fail_closed_refusals_tight_tol_leakage_and_missing_holdout(
    tmp_path: Path,
) -> None:
    bundle = load_bundle(NOTCHED_BUNDLE)
    dev_cases, _ = sample_qualification_cases(bundle, dev_count=3, seed=7)

    # A) Unachievably tight displacement tolerance -> NOT_QUALIFIED, no promoted bundle
    tight = _fast_dev_policy(disp_max=1e-9, stress_max=1e-9)
    promoted_refused = tmp_path / "should-not-exist"
    res_tight = qualify_bundle(
        NOTCHED_BUNDLE,
        tight,
        dev_cases=dev_cases,
        backend="torch-fem",
        output_dir=tmp_path / "qual-tight",
        emit_qualified_bundle=promoted_refused,
    )
    assert res_tight.qualified is False
    assert res_tight.manifest.granted_claim_level is ClaimLevel.NOT_VERIFIED
    assert res_tight.promoted_bundle_path is None
    assert not promoted_refused.exists()

    gate_refused = check_deployment_gate(NOTCHED_BUNDLE, res_tight.manifest_path)
    assert gate_refused.allowed is False
    assert GateReasonCode.NOT_QUALIFIED in gate_refused.reason_codes

    # B) TRAIN-DEV partition leakage -> partition_hygiene FAIL -> NOT_QUALIFIED
    leaked_fp = frozenset({input_fingerprint(dev_cases[0].inputs)})
    res_leak = qualify_bundle(
        NOTCHED_BUNDLE,
        _fast_dev_policy(),
        dev_cases=dev_cases,
        train_fingerprints=leaked_fp,
        backend="torch-fem",
    )
    assert res_leak.qualified is False
    assert res_leak.manifest.granted_claim_level is ClaimLevel.NOT_VERIFIED
    hygiene = next(
        f for f in res_leak.manifest.families if f["family_id"] == "partition_hygiene"
    )
    assert hygiene["status"] == "FAIL"

    # C) Target HOLDOUT_VERIFIED without holdout cases -> NOT_QUALIFIED
    res_no_ho = qualify_bundle(
        NOTCHED_BUNDLE,
        _fast_dev_policy(target=ClaimLevel.HOLDOUT_VERIFIED),
        dev_cases=dev_cases,
        holdout_cases=(),
        backend="torch-fem",
    )
    assert res_no_ho.qualified is False
    assert res_no_ho.manifest.granted_claim_level is ClaimLevel.NOT_VERIFIED


def test_cli_qualify_export_predict_and_mcp_gate(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    _fast_dev_policy().save(policy_path)
    qual_out = tmp_path / "qual-cli"

    # 1) CLI qualify run -> PASS (exit 0)
    rc_run = cli_main(
        [
            "qualify",
            "run",
            str(NOTCHED_BUNDLE),
            "--policy",
            str(policy_path),
            "--sample-dev",
            "3",
            "--seed",
            "11",
            "--backend",
            "torch-fem",
            "--output",
            str(qual_out),
        ]
    )
    assert rc_run == 0
    manifest_path = qual_out / "qualification-manifest.json"
    assert manifest_path.is_file()

    # 2) CLI qualify verify -> PASS (0) for DEV, REFUSED (2) for HOLDOUT
    rc_ver_ok = cli_main(
        [
            "qualify",
            "verify",
            str(NOTCHED_BUNDLE),
            "--manifest",
            str(manifest_path),
            "--policy",
            str(policy_path),
            "--min-claim",
            "DEVELOPMENT_VERIFIED",
        ]
    )
    assert rc_ver_ok == 0

    rc_ver_low = cli_main(
        [
            "qualify",
            "verify",
            str(NOTCHED_BUNDLE),
            "--manifest",
            str(manifest_path),
            "--min-claim",
            "HOLDOUT_VERIFIED",
        ]
    )
    assert rc_ver_low == 2

    # 3) CLI qualify diff against itself -> no regression (exit 0)
    diff_out = tmp_path / "diff.json"
    rc_diff = cli_main(
        [
            "qualify",
            "diff",
            str(manifest_path),
            str(manifest_path),
            "--output",
            str(diff_out),
        ]
    )
    assert rc_diff == 0
    assert json.loads(diff_out.read_text(encoding="utf-8"))["regressed"] is False

    # 4) CLI predict & export fmu with --require-qualified
    pred_out = tmp_path / "pred.json"
    assert (
        cli_main(
            [
                "predict",
                str(NOTCHED_BUNDLE),
                "--request",
                str(NOTCHED_REQUEST),
                "--mode",
                "fast",
                "--no-confidence",
                "--require-qualified",
                "--qualification-manifest",
                str(manifest_path),
                "--output",
                str(pred_out),
            ]
        )
        == 0
    )
    # Missing manifest under --require-qualified must refuse (exit 2)
    assert (
        cli_main(
            [
                "predict",
                str(NOTCHED_BUNDLE),
                "--request",
                str(NOTCHED_REQUEST),
                "--mode",
                "fast",
                "--no-confidence",
                "--require-qualified",
                "--qualification-manifest",
                str(tmp_path / "no-such-manifest.json"),
            ]
        )
        == 2
    )
    fmu_path = tmp_path / "gated.fmu"
    assert (
        cli_main(
            [
                "export",
                "fmu",
                str(NOTCHED_BUNDLE),
                "--output",
                str(fmu_path),
                "--require-qualified",
                "--qualification-manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    assert fmu_path.is_file()
    assert (
        cli_main(
            [
                "export",
                "fmu",
                str(NOTCHED_BUNDLE),
                "--output",
                str(tmp_path / "blocked.fmu"),
                "--require-qualified",
                "--qualification-manifest",
                str(tmp_path / "missing.json"),
            ]
        )
        == 2
    )

    # 5) Explicit cases JSON file round-trip
    raw_req = json.loads(NOTCHED_REQUEST.read_text(encoding="utf-8"))
    cases_file = tmp_path / "dev_cases.json"
    cases_file.write_text(
        json.dumps({"cases": [{"case_id": "c1", "inputs": raw_req["inputs"]}]}),
        encoding="utf-8",
    )
    loaded_cases = load_cases_file(cases_file)
    assert len(loaded_cases) == 1
    assert isinstance(loaded_cases[0], QualificationCase)

    # 6) MCP verify_qualification & require_qualified=True enforcement
    # Place manifest inside workspace temporarily under .formamech-runs/test-qual
    ws_manifest_dir = ROOT / ".formamech-runs" / "pytest-qual-mcp"
    ws_manifest_dir.mkdir(parents=True, exist_ok=True)
    ws_manifest = ws_manifest_dir / "qualification-manifest.json"
    ws_manifest.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        mcp = FormaMechMcpServer(ROOT, require_qualified=True)
        ver = mcp.call_tool(
            "verify_qualification",
            {
                "bundle_path": "bundles/notched-v1-cp-freeze",
                "manifest_path": str(ws_manifest.relative_to(ROOT)),
            },
        )
        assert ver["allowed"] is True

        # Calling predict_verified without manifest under require_qualified=True fails closed
        resp_blocked = mcp.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tools/call",
                "params": {
                    "name": "predict_verified",
                    "arguments": {
                        "bundle_path": "bundles/notched-v1-cp-freeze",
                        "inputs": raw_req["inputs"],
                        "mode": "fast",
                    },
                },
            }
        )
        assert resp_blocked is not None
        assert resp_blocked["result"]["isError"] is True
        assert "DeploymentGateError" in resp_blocked["result"]["content"][0]["text"]

        # Supplying the valid manifest succeeds
        ok_pred = mcp.call_tool(
            "predict_verified",
            {
                "bundle_path": "bundles/notched-v1-cp-freeze",
                "inputs": raw_req["inputs"],
                "mode": "fast",
                "qualification_manifest_path": str(ws_manifest.relative_to(ROOT)),
            },
        )
        assert ok_pred["verdict"] == "IN_DOMAIN"
    finally:
        if ws_manifest.exists():
            ws_manifest.unlink()
        if ws_manifest_dir.exists():
            ws_manifest_dir.rmdir()


def test_out_of_domain_case_refuses_and_zero_sampling_works() -> None:
    from piginot.contracts import Quantity, Unit

    bundle = load_bundle(NOTCHED_BUNDLE)
    d0, h0 = sample_qualification_cases(bundle, dev_count=0, holdout_count=0)
    assert d0 == () and h0 == ()
    with pytest.raises(ValueError, match="non-negative"):
        sample_qualification_cases(bundle, dev_count=-1)

    raw_req = json.loads(NOTCHED_REQUEST.read_text(encoding="utf-8"))
    ood_inputs = {
        k: Quantity(float(v["value"]), Unit(str(v["unit"]))).to_canonical()
        for k, v in raw_req["inputs"].items()
    }
    first_key = next(iter(ood_inputs))
    ood_inputs[first_key] = Quantity(1.0e9, ood_inputs[first_key].unit).to_canonical()
    ood_case = QualificationCase(case_id="ood-0", partition="DEV", inputs=ood_inputs)

    pol = QualificationPolicy(
        policy_id="ood-check",
        target_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
        calibration=CalibrationPolicy(enabled=False, min_calibration_cases=1),
        dwr=DwrPolicy(enabled=False),
        report_breakeven=False,
    )
    res = qualify_bundle(
        NOTCHED_BUNDLE, pol, dev_cases=(ood_case,), backend="torch-fem"
    )
    assert res.qualified is False
    assert res.manifest.granted_claim_level is ClaimLevel.NOT_VERIFIED
    adm = next(f for f in res.manifest.families if f["family_id"] == "admissibility")
    assert adm["status"] == "FAIL"
