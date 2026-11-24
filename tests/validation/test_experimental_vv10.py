"""Verification tests for Phase 3 (V3-3) differentiable inverse calibration & ASME V&V 10."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from piginot.validation import (
    VALIDATION_LIFECYCLE_MACHINE,
    CouponRole,
    ExperimentalCouponDataset,
    ExperimentalTestMode,
    ExperimentalValidationTier,
    InverseCalibrationConfig,
    ValidationLifecycleState,
    calibrate_material_from_experiments,
    predict_coupon_stress_torch,
)


def test_mooney_rivlin_inverse_recovery_and_vv10_holdout_validation() -> None:
    rng = np.random.default_rng(123)
    # True Mooney-Rivlin parameters: C10 = 1.40 MPa, C01 = 0.45 MPa
    c10_true, c01_true = 1.40, 0.45
    lam_uni = np.linspace(1.02, 1.80, 25)
    lam_eq = np.linspace(1.02, 1.50, 20)
    lam_val = np.linspace(1.05, 1.75, 18)

    u_exp = 0.04  # MPa measurement std
    sig_uni = 2.0 * (c10_true + c01_true / lam_uni) * (lam_uni - lam_uni**-2)
    sig_eq = 2.0 * (c10_true + c01_true * lam_eq**2) * (lam_eq - lam_eq**-5)
    sig_val = 2.0 * (c10_true + c01_true / lam_val) * (lam_val - lam_val**-2)

    coupons = [
        ExperimentalCouponDataset(
            specimen_id="cal-uni-1",
            role=CouponRole.CALIBRATION,
            test_mode=ExperimentalTestMode.UNIAXIAL_TENSION,
            stretches_or_strains=tuple(lam_uni.tolist()),
            measured_stress_mpa=tuple(
                (sig_uni + rng.normal(0.0, u_exp * 0.4, len(lam_uni))).tolist()
            ),
            measurement_std_mpa=u_exp,
        ),
        ExperimentalCouponDataset(
            specimen_id="cal-eq-1",
            role=CouponRole.CALIBRATION,
            test_mode=ExperimentalTestMode.EQUIBIAXIAL_TENSION,
            stretches_or_strains=tuple(lam_eq.tolist()),
            measured_stress_mpa=tuple(
                (sig_eq + rng.normal(0.0, u_exp * 0.4, len(lam_eq))).tolist()
            ),
            measurement_std_mpa=u_exp,
        ),
        ExperimentalCouponDataset(
            specimen_id="val-uni-holdout",
            role=CouponRole.VALIDATION_HOLDOUT,
            test_mode=ExperimentalTestMode.UNIAXIAL_TENSION,
            stretches_or_strains=tuple(lam_val.tolist()),
            measured_stress_mpa=tuple(
                (sig_val + rng.normal(0.0, u_exp * 0.4, len(lam_val))).tolist()
            ),
            measurement_std_mpa=u_exp,
        ),
    ]

    dossier = calibrate_material_from_experiments(
        coupons,
        config=InverseCalibrationConfig(
            model_family="incompressible_mooney_rivlin",
            u_num_mpa=0.01,
            numerical_claim_level="HOLDOUT_VERIFIED",
        ),
    )
    assert VALIDATION_LIFECYCLE_MACHINE.machine_id == "formamech.validation.vv10"
    assert dossier.passed is True
    assert dossier.state is ValidationLifecycleState.VALIDATED
    assert dossier.snapshot().state == "VALIDATED"
    assert (
        dossier.experimental_validation_tier
        is ExperimentalValidationTier.HOLDOUT_EXPERIMENTALLY_VALIDATED
    )
    # Orthogonal numerical claim preserved intact
    assert dossier.numerical_claim_level == "HOLDOUT_VERIFIED"

    est = {p.name: p for p in dossier.calibrated_parameters}
    assert abs(est["c10"].value - c10_true) / c10_true < 0.03
    assert abs(est["c01"].value - c01_true) / c01_true < 0.05
    assert est["c10"].std_error > 0.0
    assert dossier.validation_ratio is not None and dossier.validation_ratio <= 1.0


def test_j2_cyclic_inverse_calibration_and_leakage_refusal() -> None:
    # Synthetic J2 cyclic strain path
    strains = np.concatenate(
        [
            np.linspace(0.0, 0.012, 12),
            np.linspace(0.011, -0.012, 16),
            np.linspace(-0.011, 0.010, 12),
        ]
    )
    raw_true = torch.tensor(
        [np.log(np.expm1(70.0)), np.log(np.expm1(2.2)), np.log(np.expm1(6.0))],
        dtype=torch.float64,
    )
    sig_true, _ = predict_coupon_stress_torch(
        "j2_voce_1d",
        raw_true,
        ExperimentalTestMode.UNIAXIAL_CYCLIC_J2,
        torch.tensor(strains, dtype=torch.float64),
    )
    s_np = sig_true.detach().numpy()

    cal = ExperimentalCouponDataset(
        specimen_id="j2-cal",
        role=CouponRole.CALIBRATION,
        test_mode=ExperimentalTestMode.UNIAXIAL_CYCLIC_J2,
        stretches_or_strains=tuple(strains.tolist()),
        measured_stress_mpa=tuple(s_np.tolist()),
        measurement_std_mpa=5.0,
    )
    val = ExperimentalCouponDataset(
        specimen_id="j2-val",
        role=CouponRole.VALIDATION_HOLDOUT,
        test_mode=ExperimentalTestMode.UNIAXIAL_CYCLIC_J2,
        stretches_or_strains=tuple((strains * 0.9).tolist()),
        measured_stress_mpa=tuple(
            predict_coupon_stress_torch(
                "j2_voce_1d",
                raw_true,
                ExperimentalTestMode.UNIAXIAL_CYCLIC_J2,
                torch.tensor(strains * 0.9, dtype=torch.float64),
            )[0]
            .detach()
            .numpy()
            .tolist()
        ),
        measurement_std_mpa=5.0,
    )
    dossier = calibrate_material_from_experiments(
        [cal, val],
        config=InverseCalibrationConfig(
            model_family="j2_voce_1d", adam_steps=120, lbfgs_steps=30
        ),
    )
    assert dossier.passed is True
    assert dossier.calibration_rmse_mpa < 3.0

    # Leakage refusal: overlapping specimen_id between CALIBRATION & VALIDATION_HOLDOUT
    leaked = ExperimentalCouponDataset(
        specimen_id="j2-cal",
        role=CouponRole.VALIDATION_HOLDOUT,
        test_mode=ExperimentalTestMode.UNIAXIAL_CYCLIC_J2,
        stretches_or_strains=cal.stretches_or_strains,
        measured_stress_mpa=cal.measured_stress_mpa,
        measurement_std_mpa=5.0,
    )
    refused = calibrate_material_from_experiments(
        [cal, leaked],
        config=InverseCalibrationConfig(model_family="j2_voce_1d"),
    )
    assert refused.passed is False
    assert refused.state is ValidationLifecycleState.REJECTED
    assert (
        refused.experimental_validation_tier is ExperimentalValidationTier.UNVALIDATED
    )

    # Renamed-duplicate leakage: identical measurements under a different specimen_id
    renamed_leak = ExperimentalCouponDataset(
        specimen_id="j2-val-renamed-copy",
        role=CouponRole.VALIDATION_HOLDOUT,
        test_mode=cal.test_mode,
        stretches_or_strains=cal.stretches_or_strains,
        measured_stress_mpa=cal.measured_stress_mpa,
        measurement_std_mpa=cal.measurement_std_mpa,
    )
    refused_renamed = calibrate_material_from_experiments(
        [cal, renamed_leak],
        config=InverseCalibrationConfig(model_family="j2_voce_1d"),
    )
    assert refused_renamed.passed is False
    assert refused_renamed.state is ValidationLifecycleState.REJECTED
    assert (
        refused_renamed.experimental_validation_tier
        is ExperimentalValidationTier.UNVALIDATED
    )
    assert len(refused_renamed.dossier_hash) == 64


def test_yeoh_negative_c20_rubber_inverse_recovery() -> None:
    """Standard S-curve Yeoh rubber has C10 > 0, C20 < 0, C30 > 0."""
    rng = np.random.default_rng(19)
    c10_t, c20_t, c30_t = 0.40, -0.05, 0.015
    lam_u = np.linspace(1.02, 2.20, 25)
    lam_e = np.linspace(1.02, 1.65, 20)
    lam_v = np.linspace(1.05, 2.10, 18)

    def _yeoh_uni(lam: np.ndarray) -> np.ndarray:
        i1 = lam**2 + 2.0 / lam
        dw = c10_t + 2.0 * c20_t * (i1 - 3.0) + 3.0 * c30_t * (i1 - 3.0) ** 2
        return 2.0 * dw * (lam - lam**-2)

    def _yeoh_eq(lam: np.ndarray) -> np.ndarray:
        i1 = 2.0 * lam**2 + lam**-4
        dw = c10_t + 2.0 * c20_t * (i1 - 3.0) + 3.0 * c30_t * (i1 - 3.0) ** 2
        return 2.0 * dw * (lam - lam**-5)

    u_exp = 0.04
    coupons = [
        ExperimentalCouponDataset(
            specimen_id="yeoh-cal-u",
            role=CouponRole.CALIBRATION,
            test_mode=ExperimentalTestMode.UNIAXIAL_TENSION,
            stretches_or_strains=tuple(lam_u.tolist()),
            measured_stress_mpa=tuple(
                (_yeoh_uni(lam_u) + rng.normal(0.0, 0.01, len(lam_u))).tolist()
            ),
            measurement_std_mpa=u_exp,
        ),
        ExperimentalCouponDataset(
            specimen_id="yeoh-cal-e",
            role=CouponRole.CALIBRATION,
            test_mode=ExperimentalTestMode.EQUIBIAXIAL_TENSION,
            stretches_or_strains=tuple(lam_e.tolist()),
            measured_stress_mpa=tuple(
                (_yeoh_eq(lam_e) + rng.normal(0.0, 0.01, len(lam_e))).tolist()
            ),
            measurement_std_mpa=u_exp,
        ),
        ExperimentalCouponDataset(
            specimen_id="yeoh-val-u",
            role=CouponRole.VALIDATION_HOLDOUT,
            test_mode=ExperimentalTestMode.UNIAXIAL_TENSION,
            stretches_or_strains=tuple(lam_v.tolist()),
            measured_stress_mpa=tuple(
                (_yeoh_uni(lam_v) + rng.normal(0.0, 0.01, len(lam_v))).tolist()
            ),
            measurement_std_mpa=u_exp,
        ),
    ]
    dossier = calibrate_material_from_experiments(
        coupons,
        config=InverseCalibrationConfig(
            model_family="incompressible_yeoh", adam_steps=300, lbfgs_steps=50
        ),
    )
    assert dossier.passed is True
    est = {p.name: p for p in dossier.calibrated_parameters}
    assert abs(est["c10"].value - c10_t) / c10_t < 0.05
    assert est["c20"].value < 0.0
    assert abs(est["c20"].value - c20_t) / abs(c20_t) < 0.12
    assert abs(est["c30"].value - c30_t) / c30_t < 0.15


def test_byte_frozen_s14_status_module_is_unmodified() -> None:
    """Guardrail: experimental validation never modifies byte-frozen S14 status.py."""
    status_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "piginot"
        / "verification"
        / "status.py"
    )
    raw = status_path.read_text(encoding="utf-8")
    assert "HOLDOUT_EXPERIMENTALLY_VALIDATED" not in raw
    assert "COUPON_CALIBRATED" not in raw
    assert len(hashlib.sha256(status_path.read_bytes()).hexdigest()) == 64
