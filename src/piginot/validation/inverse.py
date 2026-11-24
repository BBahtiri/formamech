"""Differentiable inverse material calibration & ASME V&V 10 validation (V3-3).

Identifies hyperelastic (Neo-Hookean, Mooney-Rivlin, Yeoh) and J2 elastoplastic
(Voce / linear hardening) constitutive parameters from physical coupon data
using PyTorch autograd (Adam + L-BFGS), computes Gauss-Newton parameter
covariance (``u_input``), and evaluates unseen ``VALIDATION_HOLDOUT`` specimens
against the ASME V&V 10 uncertainty budget::

    u_val = sqrt(u_num^2 + u_input^2 + u_exp^2)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from piginot.validation.contracts import (
    VALIDATION_LIFECYCLE_MACHINE,
    CalibratedParameterEstimate,
    CouponRole,
    ExperimentalCouponDataset,
    ExperimentalTestMode,
    ExperimentalValidationDossier,
    ExperimentalValidationTier,
    ValidationLifecycleState,
)

CalibrationModelFamily = Literal[
    "incompressible_neo_hookean",
    "incompressible_mooney_rivlin",
    "incompressible_yeoh",
    "j2_voce_1d",
]


@dataclass(frozen=True, slots=True)
class InverseCalibrationConfig:
    """Optimizer & ASME V&V 10 acceptance settings for inverse calibration."""

    model_family: CalibrationModelFamily = "incompressible_mooney_rivlin"
    adam_steps: int = 250
    lbfgs_steps: int = 40
    learning_rate: float = 5.0e-2
    coverage_factor_k: float = 2.0
    u_num_mpa: float = 0.01
    numerical_claim_level: str = "NOT_VERIFIED"
    seed: int = 0

    def __post_init__(self) -> None:
        if type(self.adam_steps) is not int or self.adam_steps < 1:
            raise ValueError("adam_steps must be >= 1")
        if not math.isfinite(self.coverage_factor_k) or self.coverage_factor_k <= 0.0:
            raise ValueError("coverage_factor_k must be positive")
        if not math.isfinite(self.u_num_mpa) or self.u_num_mpa < 0.0:
            raise ValueError("u_num_mpa must be nonnegative")


def predict_coupon_stress_torch(
    model_family: CalibrationModelFamily,
    raw_params: torch.Tensor,
    test_mode: ExperimentalTestMode,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, tuple[tuple[str, torch.Tensor, str], ...]]:
    """Differentiable constitutive nominal stress prediction for physical coupon tests.

    All physical parameters are mapped through ``softplus`` to strictly enforce
    thermodynamic positivity during unconstrained optimization.
    """
    sp = torch.nn.functional.softplus

    if model_family == "incompressible_neo_hookean":
        mu = sp(raw_params[0]) + 1.0e-6
        lam = inputs
        if test_mode is ExperimentalTestMode.UNIAXIAL_TENSION:
            stress = mu * (lam - lam.pow(-2))
        elif test_mode is ExperimentalTestMode.EQUIBIAXIAL_TENSION:
            stress = mu * (lam - lam.pow(-5))
        elif test_mode is ExperimentalTestMode.SIMPLE_SHEAR:
            stress = mu * lam
        else:
            raise ValueError(
                f"unsupported test_mode {test_mode.value!r} for Neo-Hookean"
            )
        return stress, (("mu", mu, "MPa"),)

    if model_family == "incompressible_mooney_rivlin":
        c10 = sp(raw_params[0]) + 1.0e-6
        c01 = sp(raw_params[1]) + 1.0e-6
        lam = inputs
        if test_mode is ExperimentalTestMode.UNIAXIAL_TENSION:
            stress = 2.0 * (c10 + c01 / lam) * (lam - lam.pow(-2))
        elif test_mode is ExperimentalTestMode.EQUIBIAXIAL_TENSION:
            stress = 2.0 * (c10 + c01 * lam.square()) * (lam - lam.pow(-5))
        elif test_mode is ExperimentalTestMode.SIMPLE_SHEAR:
            stress = 2.0 * (c10 + c01) * lam
        else:
            raise ValueError(
                f"unsupported test_mode {test_mode.value!r} for Mooney-Rivlin"
            )
        return stress, (("c10", c10, "MPa"), ("c01", c01, "MPa"))

    if model_family == "incompressible_yeoh":
        # Standard Yeoh rubber: C10 > 0, C30 > 0 for large-strain stability;
        # C20 is unconstrained so S-curve mid-strain softening (C20 < 0) is representable.
        c10 = sp(raw_params[0]) + 1.0e-6
        c20 = raw_params[1]
        c30 = sp(raw_params[2]) + 1.0e-8
        lam = inputs
        if test_mode is ExperimentalTestMode.UNIAXIAL_TENSION:
            i1 = lam.square() + 2.0 / lam
            dw_di1 = c10 + 2.0 * c20 * (i1 - 3.0) + 3.0 * c30 * (i1 - 3.0).square()
            stress = 2.0 * dw_di1 * (lam - lam.pow(-2))
        elif test_mode is ExperimentalTestMode.EQUIBIAXIAL_TENSION:
            i1 = 2.0 * lam.square() + lam.pow(-4)
            dw_di1 = c10 + 2.0 * c20 * (i1 - 3.0) + 3.0 * c30 * (i1 - 3.0).square()
            stress = 2.0 * dw_di1 * (lam - lam.pow(-5))
        elif test_mode is ExperimentalTestMode.SIMPLE_SHEAR:
            i1 = 3.0 + lam.square()
            dw_di1 = c10 + 2.0 * c20 * (i1 - 3.0) + 3.0 * c30 * (i1 - 3.0).square()
            stress = 2.0 * dw_di1 * lam
        else:
            raise ValueError(f"unsupported test_mode {test_mode.value!r} for Yeoh")
        return stress, (("c10", c10, "MPa"), ("c20", c20, "MPa"), ("c30", c30, "MPa"))

    if model_family == "j2_voce_1d":
        if test_mode is not ExperimentalTestMode.UNIAXIAL_CYCLIC_J2:
            raise ValueError("j2_voce_1d requires test_mode='uniaxial_cyclic_j2'")
        e_mod = sp(raw_params[0]) * 1_000.0 + 1.0
        sig_y0 = sp(raw_params[1]) * 100.0 + 1.0
        h_lin = sp(raw_params[2]) * 100.0
        # Differentiable 1D radial-return incremental plasticity along strain history
        eps_p = torch.zeros((), dtype=inputs.dtype)
        ep_bar = torch.zeros((), dtype=inputs.dtype)
        stresses: list[torch.Tensor] = []
        for k in range(inputs.shape[0]):
            eps_k = inputs[k]
            sig_trial = e_mod * (eps_k - eps_p)
            sig_y = sig_y0 + h_lin * ep_bar
            f_trial = torch.abs(sig_trial) - sig_y
            d_gamma = torch.relu(f_trial) / (e_mod + h_lin)
            sign_s = torch.sign(sig_trial)
            eps_p = eps_p + d_gamma * sign_s
            ep_bar = ep_bar + d_gamma
            sig_k = e_mod * (eps_k - eps_p)
            stresses.append(sig_k)
        return (
            torch.stack(stresses, dim=0),
            (
                ("youngs_modulus", e_mod, "MPa"),
                ("yield_stress", sig_y0, "MPa"),
                ("linear_hardening", h_lin, "MPa"),
            ),
        )

    raise ValueError(f"unsupported calibration model family {model_family!r}")


def _initial_raw_params(
    model_family: CalibrationModelFamily,
    initial_guess: Mapping[str, float] | None,
) -> torch.Tensor:
    def _inv_sp(val: float) -> float:
        v = max(float(val), 1.0e-4)
        return math.log(math.expm1(v)) if v < 20.0 else v

    g = {} if initial_guess is None else initial_guess
    if model_family == "incompressible_neo_hookean":
        return torch.tensor([_inv_sp(g.get("mu", 1.5))], dtype=torch.float64)
    if model_family == "incompressible_mooney_rivlin":
        return torch.tensor(
            [_inv_sp(g.get("c10", 1.0)), _inv_sp(g.get("c01", 0.3))],
            dtype=torch.float64,
        )
    if model_family == "incompressible_yeoh":
        return torch.tensor(
            [
                _inv_sp(g.get("c10", 1.0)),
                float(g.get("c20", 0.0)),
                _inv_sp(g.get("c30", 0.02)),
            ],
            dtype=torch.float64,
        )
    if model_family == "j2_voce_1d":
        return torch.tensor(
            [
                _inv_sp(g.get("youngs_modulus", 50_000.0) / 1_000.0),
                _inv_sp(g.get("yield_stress", 180.0) / 100.0),
                _inv_sp(g.get("linear_hardening", 400.0) / 100.0),
            ],
            dtype=torch.float64,
        )
    raise ValueError(f"unsupported model family {model_family!r}")


def calibrate_material_from_experiments(
    coupons: Sequence[ExperimentalCouponDataset],
    *,
    config: InverseCalibrationConfig | None = None,
    initial_guess: Mapping[str, float] | None = None,
) -> ExperimentalValidationDossier:
    """Identify constitutive parameters on CALIBRATION coupons & validate on HOLDOUT coupons."""
    cfg = InverseCalibrationConfig() if config is None else config
    state = ValidationLifecycleState.INGESTED

    cal_coupons = [c for c in coupons if c.role is CouponRole.CALIBRATION]
    val_coupons = [c for c in coupons if c.role is CouponRole.VALIDATION_HOLDOUT]

    # Strict anti-leakage check: zero specimen ID, dataset hash, or measurement-content overlap
    cal_ids = {c.specimen_id for c in cal_coupons}
    val_ids = {c.specimen_id for c in val_coupons}
    cal_hashes = {c.dataset_hash for c in cal_coupons}
    val_hashes = {c.dataset_hash for c in val_coupons}
    cal_meas_hashes = {c.measurement_hash for c in cal_coupons}
    val_meas_hashes = {c.measurement_hash for c in val_coupons}
    if (
        not cal_coupons
        or (cal_ids & val_ids)
        or (cal_hashes & val_hashes)
        or (cal_meas_hashes & val_meas_hashes)
    ):
        state = VALIDATION_LIFECYCLE_MACHINE.require_transition(
            state,
            ValidationLifecycleState.REJECTED,
            trigger="reject_leakage_or_invalid",
        ).to_state
        return ExperimentalValidationDossier(
            model_family=cfg.model_family,
            numerical_claim_level=cfg.numerical_claim_level,
            experimental_validation_tier=ExperimentalValidationTier.UNVALIDATED,
            state=state,
            calibrated_parameters=(),
            calibration_rmse_mpa=float("inf"),
            holdout_comparison_error_e_mpa=None,
            u_num_mpa=cfg.u_num_mpa,
            u_input_mpa=0.0,
            u_exp_mpa=0.0,
            u_val_mpa=cfg.u_num_mpa,
            validation_ratio=None,
            passed=False,
            calibration_specimen_ids=tuple(sorted(cal_ids)),
            holdout_specimen_ids=tuple(sorted(val_ids)),
            reason="missing CALIBRATION coupons or detected calibration/holdout leakage",
        )

    torch.manual_seed(cfg.seed)
    raw = (
        _initial_raw_params(cfg.model_family, initial_guess)
        .clone()
        .detach()
        .requires_grad_(True)
    )

    cal_tensors = [
        (
            c.test_mode,
            torch.tensor(c.stretches_or_strains, dtype=torch.float64),
            torch.tensor(c.measured_stress_mpa, dtype=torch.float64),
            float(c.measurement_std_mpa),
        )
        for c in cal_coupons
    ]

    def _weighted_nll() -> torch.Tensor:
        sq_terms: list[torch.Tensor] = []
        for mode, x_t, y_t, u_exp in cal_tensors:
            pred, _ = predict_coupon_stress_torch(cfg.model_family, raw, mode, x_t)
            sq_terms.append(((pred - y_t) / u_exp).square().mean())
        return torch.stack(sq_terms).mean()

    opt_adam = torch.optim.Adam([raw], lr=cfg.learning_rate)
    for _ in range(cfg.adam_steps):
        opt_adam.zero_grad(set_to_none=True)
        loss = _weighted_nll()
        loss.backward()  # type: ignore[no-untyped-call]
        opt_adam.step()

    opt_lbfgs = torch.optim.LBFGS(
        [raw], lr=0.8, max_iter=cfg.lbfgs_steps, line_search_fn="strong_wolfe"
    )

    def _closure() -> torch.Tensor:
        opt_lbfgs.zero_grad(set_to_none=True)
        val = _weighted_nll()
        val.backward()  # type: ignore[no-untyped-call]
        return val

    opt_lbfgs.step(_closure)  # type: ignore[no-untyped-call]

    # Compute calibration RMSE & Gauss-Newton parameter covariance in physical parameter space
    n_p = int(raw.shape[0])
    jac_rows: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]] = []
    res_list: list[float] = []
    param_meta: tuple[tuple[str, float, str], ...] = ()
    raw_fixed = raw.detach().clone().requires_grad_(True)
    # Also compute d(phys_param)/d(raw_param) diagonal Jacobian
    _, named_t = predict_coupon_stress_torch(
        cfg.model_family, raw_fixed, cal_tensors[0][0], cal_tensors[0][1]
    )
    phys_jac = np.zeros((n_p, n_p), dtype=np.float64)
    meta_list: list[tuple[str, float, str]] = []
    for j, (pname, ptens, punit) in enumerate(named_t):
        g_p = torch.autograd.grad(ptens, raw_fixed, retain_graph=True)[0]
        phys_jac[j, :] = g_p.detach().numpy()
        meta_list.append((pname, float(ptens.detach().item()), punit))
    param_meta = tuple(meta_list)

    for mode, x_t, y_t, u_exp in cal_tensors:
        pred, _ = predict_coupon_stress_torch(cfg.model_family, raw_fixed, mode, x_t)
        diff = (pred - y_t).detach().numpy()
        res_list.extend(diff.tolist())
        j_mat = np.zeros((int(x_t.shape[0]), n_p), dtype=np.float64)
        for row_i in range(int(x_t.shape[0])):
            g_r = torch.autograd.grad(pred[row_i], raw_fixed, retain_graph=True)[0]
            j_mat[row_i, :] = g_r.detach().numpy() / u_exp
        jac_rows.append(j_mat)

    j_all = np.vstack(jac_rows)
    cal_rmse = float(np.sqrt(np.mean(np.square(res_list))))
    jtj = j_all.T @ j_all + 1.0e-8 * np.eye(n_p, dtype=np.float64)
    cov_raw = np.linalg.inv(jtj)
    cov_phys = phys_jac @ cov_raw @ phys_jac.T
    param_estimates = tuple(
        CalibratedParameterEstimate(
            name=name,
            value=val,
            std_error=math.sqrt(max(float(cov_phys[idx, idx]), 0.0)),
            unit=unit,
        )
        for idx, (name, val, unit) in enumerate(param_meta)
    )

    state = VALIDATION_LIFECYCLE_MACHINE.require_transition(
        state, ValidationLifecycleState.CALIBRATED, trigger="calibrate_material"
    ).to_state

    if not val_coupons:
        # Calibration succeeded, but no unseen holdout coupons -> stays COUPON_CALIBRATED / REJECTED
        u_exp_cal = float(
            np.sqrt(np.mean([c.measurement_std_mpa**2 for c in cal_coupons]))
        )
        state = VALIDATION_LIFECYCLE_MACHINE.require_transition(
            state, ValidationLifecycleState.REJECTED, trigger="reject_no_holdout"
        ).to_state
        return ExperimentalValidationDossier(
            model_family=cfg.model_family,
            numerical_claim_level=cfg.numerical_claim_level,
            experimental_validation_tier=ExperimentalValidationTier.COUPON_CALIBRATED,
            state=state,
            calibrated_parameters=param_estimates,
            calibration_rmse_mpa=cal_rmse,
            holdout_comparison_error_e_mpa=None,
            u_num_mpa=cfg.u_num_mpa,
            u_input_mpa=0.0,
            u_exp_mpa=u_exp_cal,
            u_val_mpa=math.hypot(cfg.u_num_mpa, u_exp_cal),
            validation_ratio=None,
            passed=False,
            calibration_specimen_ids=tuple(sorted(cal_ids)),
            holdout_specimen_ids=(),
            reason="calibrated on CALIBRATION coupons; unseen VALIDATION_HOLDOUT coupons required for ASME V&V 10 validation",
        )

    # Evaluate blind VALIDATION_HOLDOUT coupons & propagate parameter covariance u_input
    val_errs: list[float] = []
    val_u_in_sq: list[float] = []
    val_u_exp_sq: list[float] = []
    for c in val_coupons:
        x_v = torch.tensor(c.stretches_or_strains, dtype=torch.float64)
        y_v = np.asarray(c.measured_stress_mpa, dtype=np.float64)
        pred_v, _ = predict_coupon_stress_torch(
            cfg.model_family, raw_fixed, c.test_mode, x_v
        )
        p_np = pred_v.detach().numpy()
        val_errs.extend(np.abs(p_np - y_v).tolist())
        val_u_exp_sq.append(c.measurement_std_mpa**2)
        for row_i in range(int(x_v.shape[0])):
            g_v = (
                torch.autograd.grad(pred_v[row_i], raw_fixed, retain_graph=True)[0]
                .detach()
                .numpy()
            )
            val_u_in_sq.append(max(float(g_v @ cov_raw @ g_v), 0.0))

    state = VALIDATION_LIFECYCLE_MACHINE.require_transition(
        state, ValidationLifecycleState.HOLDOUT_EVALUATED, trigger="evaluate_holdout"
    ).to_state

    e_val = float(np.sqrt(np.mean(np.square(val_errs))))
    u_input = float(np.sqrt(np.mean(val_u_in_sq))) if val_u_in_sq else 0.0
    u_exp = float(np.sqrt(np.mean(val_u_exp_sq)))
    u_val = math.sqrt(cfg.u_num_mpa**2 + u_input**2 + u_exp**2)
    budget = cfg.coverage_factor_k * u_val
    ratio = e_val / max(budget, 1.0e-12)
    passed = e_val <= budget

    if passed:
        state = VALIDATION_LIFECYCLE_MACHINE.require_transition(
            state, ValidationLifecycleState.VALIDATED, trigger="accept_vv10"
        ).to_state
        tier = ExperimentalValidationTier.HOLDOUT_EXPERIMENTALLY_VALIDATED
        reason = f"ASME V&V 10 passed: E={e_val:.4g} <= {cfg.coverage_factor_k:.2g}*u_val={budget:.4g} MPa"
    else:
        state = VALIDATION_LIFECYCLE_MACHINE.require_transition(
            state, ValidationLifecycleState.REJECTED, trigger="reject_vv10"
        ).to_state
        tier = ExperimentalValidationTier.COUPON_CALIBRATED
        reason = f"ASME V&V 10 rejected: E={e_val:.4g} > {cfg.coverage_factor_k:.2g}*u_val={budget:.4g} MPa"

    return ExperimentalValidationDossier(
        model_family=cfg.model_family,
        numerical_claim_level=cfg.numerical_claim_level,
        experimental_validation_tier=tier,
        state=state,
        calibrated_parameters=param_estimates,
        calibration_rmse_mpa=cal_rmse,
        holdout_comparison_error_e_mpa=e_val,
        u_num_mpa=cfg.u_num_mpa,
        u_input_mpa=u_input,
        u_exp_mpa=u_exp,
        u_val_mpa=u_val,
        validation_ratio=ratio,
        passed=passed,
        calibration_specimen_ids=tuple(sorted(cal_ids)),
        holdout_specimen_ids=tuple(sorted(val_ids)),
        reason=reason,
    )


__all__ = (
    "CalibrationModelFamily",
    "InverseCalibrationConfig",
    "calibrate_material_from_experiments",
    "predict_coupon_stress_torch",
)
