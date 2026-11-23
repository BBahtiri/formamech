"""Unit tests for per-CAD split-conformal QoI and functional-band calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from piginot.bounds import (
    CAD_CALIBRATION_SCHEMA,
    CadCalibrationRecord,
    build_cad_calibration,
    calibrate_functional_band,
)


def test_functional_band_and_cad_calibration_roundtrip_and_guards(
    tmp_path: Path,
) -> None:
    residuals = [[0.01 * (i + 1), 0.005 * (i + 1)] for i in range(20)]
    scales = [[2.0, 1.0] for _ in range(20)]

    band_unscaled = calibrate_functional_band(
        field_name="u", unit="mm", residuals=residuals, alpha=0.05
    )
    assert band_unscaled.nominal_coverage == pytest.approx(0.95)
    assert band_unscaled.band_half_widths() == pytest.approx(
        band_unscaled.supremum_quantile
    )
    scaled_w = band_unscaled.band_half_widths([2.0, 0.0])
    assert isinstance(scaled_w, list) and len(scaled_w) == 2

    band_scaled = calibrate_functional_band(
        field_name="u", unit="mm", residuals=residuals, scales=scales, alpha=0.05
    )
    assert band_scaled.supremum_quantile < band_unscaled.supremum_quantile
    assert band_scaled.document()["field_name"] == "u"

    # Validation guards for calibrate_functional_band
    with pytest.raises(ValueError, match="at least one"):
        calibrate_functional_band(field_name="u", unit="mm", residuals=[])
    with pytest.raises(ValueError, match="scale_floor"):
        calibrate_functional_band(
            field_name="u", unit="mm", residuals=residuals, scale_floor=0.0
        )
    with pytest.raises(ValueError, match="non-empty and finite"):
        calibrate_functional_band(
            field_name="u", unit="mm", residuals=[[float("nan")]] * 20
        )
    with pytest.raises(ValueError, match="align with residual"):
        calibrate_functional_band(
            field_name="u", unit="mm", residuals=residuals, scales=[[1.0]] * 20
        )

    manifest_sha = "a" * 64
    rec = build_cad_calibration(
        bundle_name="bracket-v1",
        bundle_manifest_sha256=manifest_sha,
        problem_hash="b" * 64,
        qoi_scores={"reaction": ("N", [0.05 * (i + 1) for i in range(20)])},
        alpha=0.05,
        functional_residuals={"displacement": ("mm", residuals)},
        parameter_bounds={"strain": (0.05, 0.20)},
    )
    assert rec.in_envelope({"strain": 0.10}) is True
    assert rec.in_envelope({"strain": 0.50}) is False
    assert rec.in_envelope({}) is False

    st_in = rec.statistical_record("reaction", in_envelope=True)
    assert st_in is not None and not st_in.refused and st_in.half_width is not None
    st_out = rec.statistical_record("reaction", in_envelope=False)
    assert st_out is not None and st_out.refused and st_out.half_width is None
    assert rec.statistical_record("missing", in_envelope=True) is None

    ev_in = rec.evaluate({"strain": 0.12})
    assert ev_in["in_calibration_envelope"] is True and ev_in["refused"] is False
    assert "reaction" in ev_in["qoi_bounds"]
    ev_out = rec.evaluate({"strain": 1.0})
    assert ev_out["in_calibration_envelope"] is False and ev_out["refused"] is True

    # Save / load round-trip + tamper & bundle-mismatch detection
    out_file = tmp_path / "cad-cal.json"
    rec.save(out_file)
    loaded = CadCalibrationRecord.load(out_file, expected_manifest_sha256=manifest_sha)
    assert loaded.calibration_sha256 == rec.calibration_sha256

    with pytest.raises(ValueError, match="different bundle"):
        CadCalibrationRecord.load(out_file, expected_manifest_sha256="c" * 64)

    tampered = json.loads(out_file.read_text("utf-8"))
    tampered["bundle_name"] = "tampered-name"
    bad_file = tmp_path / "tampered.json"
    bad_file.write_text(json.dumps(tampered), "utf-8")
    with pytest.raises(ValueError, match="SHA-256 digest mismatch"):
        CadCalibrationRecord.load(bad_file)

    wrong_schema = tmp_path / "wrong-schema.json"
    wrong_schema.write_text(json.dumps({"schema": "other"}), "utf-8")
    with pytest.raises(ValueError, match=CAD_CALIBRATION_SCHEMA):
        CadCalibrationRecord.load(wrong_schema)

    # Post-init validation guards
    with pytest.raises(ValueError, match="bundle_name"):
        CadCalibrationRecord(
            bundle_name="   ",
            bundle_manifest_sha256=manifest_sha,
            problem_hash=None,
            exchangeability_scope="scope",
            alpha=0.05,
            qoi_calibrations={},
        )
    with pytest.raises(ValueError, match="64-char"):
        CadCalibrationRecord(
            bundle_name="ok",
            bundle_manifest_sha256="short",
            problem_hash=None,
            exchangeability_scope="scope",
            alpha=0.05,
            qoi_calibrations={},
        )
    with pytest.raises(ValueError, match="alpha"):
        CadCalibrationRecord(
            bundle_name="ok",
            bundle_manifest_sha256=manifest_sha,
            problem_hash=None,
            exchangeability_scope="scope",
            alpha=1.5,
            qoi_calibrations={},
        )

    # Wave 2: CAD_CALIBRATION_MACHINE, stripped calibration_sha256 refusal,
    # empty/invalid parameter_bounds refusal, and FunctionalBand/ScalarQoi post-init guards
    from piginot.bounds import (
        CAD_CALIBRATION_MACHINE,
        CadCalibrationGateState,
        FunctionalBandCalibration,
        ScalarQoiCalibration,
    )

    assert CAD_CALIBRATION_MACHINE.can_transition(
        CadCalibrationGateState.UNCHECKED,
        CadCalibrationGateState.BOUND_VERIFIED,
        trigger="verify_digest_and_bundle_binding",
    )
    assert CAD_CALIBRATION_MACHINE.is_terminal(
        CadCalibrationGateState.REFUSED_OUT_OF_ENVELOPE
    )

    # Stripping calibration_sha256 from sidecar JSON must fail closed
    stripped = json.loads(out_file.read_text("utf-8"))
    stripped.pop("calibration_sha256", None)
    stripped_file = tmp_path / "stripped.json"
    stripped_file.write_text(json.dumps(stripped), "utf-8")
    with pytest.raises(ValueError, match="SHA-256 digest mismatch"):
        CadCalibrationRecord.load(stripped_file)

    # Empty parameter_bounds fails closed on in_envelope(), evaluate(), and statistical_record()
    empty_env_rec = CadCalibrationRecord(
        bundle_name="ok",
        bundle_manifest_sha256=manifest_sha,
        problem_hash=None,
        exchangeability_scope="scope",
        alpha=0.05,
        qoi_calibrations=rec.qoi_calibrations,
        parameter_bounds={},
    )
    assert empty_env_rec.in_envelope({"strain": 0.10}) is False
    ev_empty = empty_env_rec.evaluate({"strain": 0.10})
    assert ev_empty["refused"] is True and "undeclared" in ev_empty["refusal_reason"]
    st_empty = empty_env_rec.statistical_record("reaction", in_envelope=True)
    assert st_empty is not None and st_empty.refused is True

    with pytest.raises(ValueError, match="lower <= upper"):
        CadCalibrationRecord(
            bundle_name="ok",
            bundle_manifest_sha256=manifest_sha,
            problem_hash=None,
            exchangeability_scope="scope",
            alpha=0.05,
            qoi_calibrations={},
            parameter_bounds={"strain": (0.5, 0.1)},
        )

    with pytest.raises(ValueError, match="rank_k"):
        FunctionalBandCalibration(
            field_name="u",
            unit="mm",
            alpha=0.05,
            sample_count=10,
            rank_k=11,
            supremum_quantile=0.1,
        )
    with pytest.raises(ValueError, match="quantile"):
        ScalarQoiCalibration(
            qoi_name="rx",
            unit="N",
            alpha=0.05,
            sample_count=20,
            rank_k=19,
            quantile=-0.1,
        )
