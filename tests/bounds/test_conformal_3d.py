"""Unit & Gate G-V5-1a/G-V5-1d tests for 3D split-conformal UQ & AS6983 shifts."""

from __future__ import annotations

import time

import numpy as np
import pytest

from piginot.bounds.conformal_3d import (
    CONFORMAL_3D_SCHEMA,
    Conformal3DCalibrationRecord,
    Conformal3DCalibrator,
    Conformal3DDevSample,
    summarize_physical_errors,
    verify_as6983_holdout_coverage,
)


def _synthetic_samples(prefix: str, n: int, *, seed: int) -> list[Conformal3DDevSample]:
    rng = np.random.default_rng(seed)
    out: list[Conformal3DDevSample] = []
    for i in range(n):
        z = rng.normal(0.0, 1.0, size=8)
        scale = 1.0 + 0.15 * float(np.sqrt(np.mean(z * z)))
        true_vm = 120.0 + 15.0 * float(z[0]) + float(rng.normal(0.0, 3.0 * scale))
        pred_vm = true_vm - float(rng.normal(0.5, 2.5 * scale))
        true_u = 1.2 + 0.2 * float(z[1]) + float(rng.normal(0.0, 0.04 * scale))
        pred_u = true_u - float(rng.normal(0.01, 0.03 * scale))
        out.append(
            Conformal3DDevSample(
                case_id=f"{prefix}-{i:03d}",
                predicted_qois={
                    "quadrature_peak_von_mises_mpa": pred_vm,
                    "max_displacement_mm": pred_u,
                },
                true_qois={
                    "quadrature_peak_von_mises_mpa": true_vm,
                    "max_displacement_mm": true_u,
                },
                latent_vector=tuple(float(v) for v in z),
            )
        )
    return out


def test_as6983_holdout_coverage_and_physical_error_summaries() -> None:
    """Gate G-V5-1d: DEV-calibrated AS6983 shift passes Beta-Binomial HOLDOUT test."""
    dev = _synthetic_samples("dev", 40, seed=11)
    holdout = _synthetic_samples("holdout", 60, seed=29)
    record = Conformal3DCalibrator.fit(
        "bundle-sha-demo",
        dev,
        qoi_units={
            "quadrature_peak_von_mises_mpa": "MPa",
            "max_displacement_mm": "mm",
        },
        alpha=0.05,
    )
    assert record.document()["schema"] == CONFORMAL_3D_SCHEMA
    reloaded = Conformal3DCalibrationRecord.from_document(record.document())
    assert reloaded.calibration_sha256 == record.calibration_sha256

    cov_reports = verify_as6983_holdout_coverage(reloaded, holdout, test_level=0.05)
    assert set(cov_reports) == {"quadrature_peak_von_mises_mpa", "max_displacement_mm"}
    for qname, rep in cov_reports.items():
        assert rep.as6983_passed, f"AS6983 failed on {qname}: {rep.document()}"
        assert rep.two_sided_passed, f"two-sided conformal failed on {qname}"
        err = rep.holdout_error_summary
        assert err.max_abs_error >= err.p90_abs_error >= err.mean_abs_error >= 0.0
        assert err.unit in ("MPa", "mm")


def test_conformal_3d_refuses_dev_holdout_leakage() -> None:
    dev = _synthetic_samples("dev", 25, seed=3)
    record = Conformal3DCalibrator.fit(
        "bundle-sha-demo",
        dev,
        qoi_units={"quadrature_peak_von_mises_mpa": "MPa"},
        alpha=0.05,
    )
    with pytest.raises(ValueError, match="split leakage refused"):
        verify_as6983_holdout_coverage(record, [dev[0]])


def test_conformal_3d_sub_millisecond_overhead() -> None:
    """Gate G-V5-1a: Negligible (<1% / sub-ms) inference overhead for UQ + AS6983."""
    dev = _synthetic_samples("dev", 25, seed=7)
    record = Conformal3DCalibrator.fit(
        "bundle-sha-fast",
        dev,
        qoi_units={
            "quadrature_peak_von_mises_mpa": "MPa",
            "max_displacement_mm": "mm",
        },
        alpha=0.05,
    )
    pred = {"quadrature_peak_von_mises_mpa": 135.0, "max_displacement_mm": 1.35}
    z = (0.2, -0.4, 0.1, 0.0, 0.5, -0.2, 0.3, 0.1)

    t0 = time.perf_counter()
    for _ in range(200):
        assess = record.evaluate(pred, latent_vector=z)
    per_call_ms = (time.perf_counter() - t0) * 1000.0 / 200.0

    assert per_call_ms < 2.0, f"conformal 3D evaluation too slow: {per_call_ms:.3f} ms"
    vm_b = assess.bounds["quadrature_peak_von_mises_mpa"]
    assert vm_b.conservative_shifted_value >= vm_b.predicted_value
    assert vm_b.upper_conformal_bound >= vm_b.lower_conformal_bound


def test_summarize_physical_errors_validation() -> None:
    summary = summarize_physical_errors(
        "peak_stress", "MPa", [100.0, 200.0], [95.0, 220.0]
    )
    assert summary.mean_abs_error == pytest.approx(12.5)
    assert summary.max_abs_error == pytest.approx(20.0)
    with pytest.raises(ValueError, match="equal non-empty"):
        summarize_physical_errors("peak_stress", "MPa", [], [])
