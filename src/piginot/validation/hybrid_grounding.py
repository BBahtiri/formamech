"""Component-level hybrid simulation + physical-test grounding (V5-2).

Fuses numerical simulation priors (3D/2D neural operators, FEM, or external
physics solvers) with real physical lab measurements (DIC fields, strain rosettes,
load cells, thermocouples, deflection profiles, degradation curves) via a
physics-constrained Kennedy-O'Hagan discrepancy model::

    y_grounded(x, c) = rho * y_sim(x, c) + delta_GP(x, c)

with exact closed-form GP posterior uncertainty ``u_discrepancy(x, c)`` and
strict ASME V&V 10 blind ``VALIDATION_HOLDOUT`` verification.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve  # type: ignore[import-untyped]
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from piginot.validation.component_test_contracts import (
    HYBRID_GROUNDING_MACHINE,
    ComponentTestRecord,
    HybridGroundingDossier,
    HybridGroundingState,
)
from piginot.validation.contracts import CouponRole, ExperimentalValidationTier

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SensorMeshAlignmentReport:
    """Deterministic spatial mapping from physical DIC/rosette sensors to mesh nodes."""

    matched_values: FloatArray
    max_registration_distance_mm: float
    mean_registration_distance_mm: float
    sensor_count: int


def align_sensor_observations_to_mesh(
    mesh_nodes: FloatArray,
    mesh_field: FloatArray,
    sensor_locations: FloatArray,
    *,
    k_neighbors: int = 4,
) -> SensorMeshAlignmentReport:
    """Interpolate simulated nodal fields onto arbitrary 2D/3D sensor coordinates."""
    nodes = np.asarray(mesh_nodes, dtype=np.float64)
    field = np.asarray(mesh_field, dtype=np.float64)
    locs = np.asarray(sensor_locations, dtype=np.float64)
    if nodes.ndim != 2 or locs.ndim != 2 or nodes.shape[1] != locs.shape[1]:
        raise ValueError("mesh_nodes and sensor_locations must share spatial dimension")
    if field.shape[0] != nodes.shape[0]:
        raise ValueError("mesh_field first dimension must equal number of mesh_nodes")
    k_eff = max(1, min(int(k_neighbors), int(nodes.shape[0])))
    tree = cKDTree(nodes)
    dists, idxs = tree.query(locs, k=k_eff)
    if k_eff == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    min_d = dists[:, 0]
    weights = 1.0 / np.maximum(dists, 1.0e-12)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    if field.ndim == 1:
        matched = np.sum(weights * field[idxs], axis=1)
    else:
        matched = np.einsum("mk,mkc->mc", weights, field[idxs])
    return SensorMeshAlignmentReport(
        matched_values=np.asarray(matched, dtype=np.float64),
        max_registration_distance_mm=float(np.max(min_d)),
        mean_registration_distance_mm=float(np.mean(min_d)),
        sensor_count=int(locs.shape[0]),
    )


@dataclass(frozen=True, slots=True)
class HybridGroundingConfig:
    """Configuration for Kennedy-O'Hagan discrepancy fitting & ASME V&V 10 gate."""

    fit_prior_scale_rho: bool = True
    rho_prior_strength: float = 1.0e-2
    length_scale: float = 1.0
    signal_std: float | None = None
    jitter: float = 1.0e-6
    coverage_factor_k: float = 2.0
    u_num: float = 0.01
    u_input: float = 0.0
    numerical_claim_level: str = "NOT_VERIFIED"

    def __post_init__(self) -> None:
        if not math.isfinite(self.rho_prior_strength) or self.rho_prior_strength < 0.0:
            raise ValueError("rho_prior_strength must be nonnegative")
        if not math.isfinite(self.length_scale) or self.length_scale <= 0.0:
            raise ValueError("length_scale must be strictly positive")
        if self.signal_std is not None and (
            not math.isfinite(self.signal_std) or self.signal_std <= 0.0
        ):
            raise ValueError("signal_std must be strictly positive when set")
        if not math.isfinite(self.coverage_factor_k) or self.coverage_factor_k <= 0.0:
            raise ValueError("coverage_factor_k must be positive")
        if not math.isfinite(self.u_num) or self.u_num < 0.0:
            raise ValueError("u_num must be nonnegative")
        if not math.isfinite(self.u_input) or self.u_input < 0.0:
            raise ValueError("u_input must be nonnegative")


def _build_feature_rows(
    records: Sequence[ComponentTestRecord],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Flatten component test records into ``(X, y_sim, y_obs, u_exp)`` arrays."""
    x_rows: list[list[float]] = []
    y_sim_list: list[float] = []
    y_obs_list: list[float] = []
    u_exp_list: list[float] = []
    has_locs = all(bool(r.sensor_locations) for r in records)
    for rec in records:
        cond = [float(v) for v in rec.condition_vector]
        n_obs = len(rec.observed_values)
        for idx in range(n_obs):
            if has_locs:
                loc = [float(v) for v in rec.sensor_locations[idx]]
                x_rows.append([*cond, *loc])
            elif n_obs > 1:
                x_rows.append([*cond, float(idx) / float(max(n_obs - 1, 1))])
            else:
                x_rows.append(cond)
            y_sim_list.append(float(rec.simulated_prior_values[idx]))
            y_obs_list.append(float(rec.observed_values[idx]))
            u_exp_list.append(float(rec.measurement_std))
    return (
        np.asarray(x_rows, dtype=np.float64),
        np.asarray(y_sim_list, dtype=np.float64),
        np.asarray(y_obs_list, dtype=np.float64),
        np.asarray(u_exp_list, dtype=np.float64),
    )


def _rbf_kernel(
    x_a: FloatArray, x_b: FloatArray, *, length_scale: float, signal_var: float
) -> FloatArray:
    diff = x_a[:, None, :] - x_b[None, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)
    return np.asarray(
        signal_var * np.exp(-0.5 * sq_dist / (length_scale * length_scale)),
        dtype=np.float64,
    )


class HybridDiscrepancyModel:
    """Trained Kennedy-O'Hagan sim+test discrepancy corrector with GP uncertainty."""

    def __init__(
        self,
        *,
        rho: float,
        x_mean: FloatArray,
        x_std: FloatArray,
        x_train_norm: FloatArray,
        alpha: FloatArray,
        chol_factor: tuple[FloatArray, bool],
        length_scale: float,
        signal_var: float,
        mean_residual: float,
    ) -> None:
        self.rho = float(rho)
        self._x_mean = x_mean
        self._x_std = x_std
        self._x_train = x_train_norm
        self._alpha = alpha
        self._chol = chol_factor
        self._length_scale = float(length_scale)
        self._signal_var = float(signal_var)
        self._mean_residual = float(mean_residual)

    def predict(
        self,
        features: FloatArray,
        simulated_prior: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Return ``(grounded_mean, discrepancy_std)`` at query ``(features, sim_prior)``."""
        x_q = np.asarray(features, dtype=np.float64)
        y_s = np.asarray(simulated_prior, dtype=np.float64).reshape(-1)
        x_norm = (x_q - self._x_mean[None, :]) / self._x_std[None, :]
        k_star = _rbf_kernel(
            x_norm,
            self._x_train,
            length_scale=self._length_scale,
            signal_var=self._signal_var,
        )
        delta_mean = self._mean_residual + k_star @ self._alpha
        v_mat = cho_solve(self._chol, k_star.T)
        post_var = np.maximum(self._signal_var - np.sum(k_star * v_mat.T, axis=1), 0.0)
        y_grounded = self.rho * y_s + delta_mean
        return (
            np.asarray(y_grounded, dtype=np.float64),
            np.asarray(np.sqrt(post_var), dtype=np.float64),
        )


def _r2_score(y_true: FloatArray, y_pred: FloatArray) -> float:
    ss_res = float(np.sum(np.square(y_true - y_pred)))
    ss_tot = float(np.sum(np.square(y_true - float(np.mean(y_true)))))
    if ss_tot <= 1.0e-15:
        return 1.0 if ss_res <= 1.0e-15 else 0.0
    return 1.0 - ss_res / ss_tot


def ground_simulation_with_experiments(
    records: Sequence[ComponentTestRecord],
    *,
    config: HybridGroundingConfig | None = None,
) -> tuple[HybridDiscrepancyModel | None, HybridGroundingDossier]:
    """Fit a multi-fidelity sim+test discrepancy model & validate on blind holdouts."""
    cfg = HybridGroundingConfig() if config is None else config
    state = HybridGroundingState.INGESTED
    if not records:
        raise ValueError("records must be non-empty")

    sensor_kind = records[0].sensor_kind
    unit = records[0].unit
    cond_dim = len(records[0].condition_vector)
    if any(
        r.sensor_kind is not sensor_kind
        or r.unit != unit
        or len(r.condition_vector) != cond_dim
        for r in records
    ):
        state = HYBRID_GROUNDING_MACHINE.require_transition(
            state, HybridGroundingState.REJECTED, trigger="reject_leakage_or_invalid"
        ).to_state
        return None, HybridGroundingDossier(
            sensor_kind=sensor_kind,
            unit=unit,
            numerical_claim_level=cfg.numerical_claim_level,
            experimental_validation_tier=ExperimentalValidationTier.UNVALIDATED,
            state=state,
            fitted_prior_scale_rho=1.0,
            calibration_specimen_ids=(),
            holdout_specimen_ids=(),
            raw_sim_holdout_rmse=None,
            grounded_holdout_rmse=None,
            raw_sim_max_abs_error=None,
            grounded_max_abs_error=None,
            raw_sim_holdout_r2=None,
            grounded_holdout_r2=None,
            bias_reduction_fraction=None,
            u_num=cfg.u_num,
            u_input=cfg.u_input,
            u_discrepancy=0.0,
            u_exp=0.0,
            u_val=math.hypot(cfg.u_num, cfg.u_input),
            validation_ratio=None,
            passed=False,
            reason="all ComponentTestRecords must share sensor_kind, unit, and condition dimension",
        )

    cal_recs = [r for r in records if r.role is CouponRole.CALIBRATION]
    val_recs = [r for r in records if r.role is CouponRole.VALIDATION_HOLDOUT]
    cal_ids = {r.specimen_id for r in cal_recs}
    val_ids = {r.specimen_id for r in val_recs}
    cal_hashes = {r.dataset_hash for r in cal_recs}
    val_hashes = {r.dataset_hash for r in val_recs}
    cal_mhashes = {r.measurement_hash for r in cal_recs}
    val_mhashes = {r.measurement_hash for r in val_recs}

    if (
        not cal_recs
        or (cal_ids & val_ids)
        or (cal_hashes & val_hashes)
        or (cal_mhashes & val_mhashes)
    ):
        state = HYBRID_GROUNDING_MACHINE.require_transition(
            state, HybridGroundingState.REJECTED, trigger="reject_leakage_or_invalid"
        ).to_state
        return None, HybridGroundingDossier(
            sensor_kind=sensor_kind,
            unit=unit,
            numerical_claim_level=cfg.numerical_claim_level,
            experimental_validation_tier=ExperimentalValidationTier.UNVALIDATED,
            state=state,
            fitted_prior_scale_rho=1.0,
            calibration_specimen_ids=tuple(sorted(cal_ids)),
            holdout_specimen_ids=tuple(sorted(val_ids)),
            raw_sim_holdout_rmse=None,
            grounded_holdout_rmse=None,
            raw_sim_max_abs_error=None,
            grounded_max_abs_error=None,
            raw_sim_holdout_r2=None,
            grounded_holdout_r2=None,
            bias_reduction_fraction=None,
            u_num=cfg.u_num,
            u_input=cfg.u_input,
            u_discrepancy=0.0,
            u_exp=0.0,
            u_val=math.hypot(cfg.u_num, cfg.u_input),
            validation_ratio=None,
            passed=False,
            reason="missing CALIBRATION specimens or detected calibration/holdout leakage",
        )

    state = HYBRID_GROUNDING_MACHINE.require_transition(
        state, HybridGroundingState.ALIGNED, trigger="align_sensors"
    ).to_state

    x_cal, y_sim_cal, y_obs_cal, u_exp_cal = _build_feature_rows(cal_recs)
    x_mean = x_cal.mean(axis=0)
    x_std = np.maximum(x_cal.std(axis=0), 1.0e-3)
    x_cal_norm = (x_cal - x_mean[None, :]) / x_std[None, :]

    # Fit ridge-regularized prior scale rho shrunk toward 1.0
    if cfg.fit_prior_scale_rho:
        w = 1.0 / np.square(u_exp_cal)
        denom = float(np.sum(w * y_sim_cal * y_sim_cal)) + cfg.rho_prior_strength
        numer = float(np.sum(w * y_sim_cal * y_obs_cal)) + cfg.rho_prior_strength
        rho = float(numer / max(denom, 1.0e-12))
    else:
        rho = 1.0

    residuals = y_obs_cal - rho * y_sim_cal
    mean_res = float(np.mean(residuals))
    centered = residuals - mean_res
    sig_std = (
        float(cfg.signal_std)
        if cfg.signal_std is not None
        else max(float(np.std(centered)), float(np.mean(u_exp_cal)), 1.0e-4)
    )
    sig_var = sig_std * sig_std

    k_mat = _rbf_kernel(
        x_cal_norm,
        x_cal_norm,
        length_scale=cfg.length_scale,
        signal_var=sig_var,
    )
    noise_diag = np.square(u_exp_cal) + cfg.jitter * sig_var
    k_noisy = k_mat + np.diag(noise_diag)
    chol = cho_factor(k_noisy, lower=True)
    alpha = cho_solve(chol, centered)

    model = HybridDiscrepancyModel(
        rho=rho,
        x_mean=x_mean,
        x_std=x_std,
        x_train_norm=x_cal_norm,
        alpha=alpha,
        chol_factor=chol,
        length_scale=cfg.length_scale,
        signal_var=sig_var,
        mean_residual=mean_res,
    )
    state = HYBRID_GROUNDING_MACHINE.require_transition(
        state, HybridGroundingState.GROUNDED, trigger="fit_discrepancy"
    ).to_state

    if not val_recs:
        u_exp_m = float(np.sqrt(np.mean(np.square(u_exp_cal))))
        state = HYBRID_GROUNDING_MACHINE.require_transition(
            state, HybridGroundingState.REJECTED, trigger="reject_no_holdout"
        ).to_state
        return model, HybridGroundingDossier(
            sensor_kind=sensor_kind,
            unit=unit,
            numerical_claim_level=cfg.numerical_claim_level,
            experimental_validation_tier=ExperimentalValidationTier.COUPON_CALIBRATED,
            state=state,
            fitted_prior_scale_rho=rho,
            calibration_specimen_ids=tuple(sorted(cal_ids)),
            holdout_specimen_ids=(),
            raw_sim_holdout_rmse=None,
            grounded_holdout_rmse=None,
            raw_sim_max_abs_error=None,
            grounded_max_abs_error=None,
            raw_sim_holdout_r2=None,
            grounded_holdout_r2=None,
            bias_reduction_fraction=None,
            u_num=cfg.u_num,
            u_input=cfg.u_input,
            u_discrepancy=0.0,
            u_exp=u_exp_m,
            u_val=math.sqrt(cfg.u_num**2 + cfg.u_input**2 + u_exp_m**2),
            validation_ratio=None,
            passed=False,
            reason="grounded on CALIBRATION specimens; blind VALIDATION_HOLDOUT required for ASME V&V 10",
        )

    # Evaluate raw simulation vs. grounded hybrid strictly on blind VALIDATION_HOLDOUT
    x_val, y_sim_val, y_obs_val, u_exp_val = _build_feature_rows(val_recs)
    y_ground_val, u_disc_vec = model.predict(x_val, y_sim_val)

    raw_err = np.abs(y_sim_val - y_obs_val)
    grd_err = np.abs(y_ground_val - y_obs_val)
    raw_rmse = float(np.sqrt(np.mean(np.square(raw_err))))
    grd_rmse = float(np.sqrt(np.mean(np.square(grd_err))))
    raw_max = float(np.max(raw_err))
    grd_max = float(np.max(grd_err))
    raw_r2 = _r2_score(y_obs_val, y_sim_val)
    grd_r2 = _r2_score(y_obs_val, y_ground_val)
    bias_red = (raw_rmse - grd_rmse) / max(raw_rmse, 1.0e-12)

    u_disc = float(np.sqrt(np.mean(np.square(u_disc_vec))))
    u_exp_h = float(np.sqrt(np.mean(np.square(u_exp_val))))
    u_val = math.sqrt(cfg.u_num**2 + cfg.u_input**2 + u_disc**2 + u_exp_h**2)
    budget = cfg.coverage_factor_k * u_val
    ratio = grd_rmse / max(budget, 1.0e-12)
    passed = grd_rmse <= budget

    state = HYBRID_GROUNDING_MACHINE.require_transition(
        state, HybridGroundingState.HOLDOUT_EVALUATED, trigger="evaluate_holdout"
    ).to_state

    if passed:
        state = HYBRID_GROUNDING_MACHINE.require_transition(
            state, HybridGroundingState.VALIDATED, trigger="accept_vv10_hybrid"
        ).to_state
        tier = ExperimentalValidationTier.HOLDOUT_EXPERIMENTALLY_VALIDATED
        reason = (
            f"ASME V&V 10 hybrid grounding passed: E={grd_rmse:.4g} <= "
            f"{cfg.coverage_factor_k:.2g}*u_val={budget:.4g} {unit} "
            f"(raw sim RMSE={raw_rmse:.4g} -> {grd_rmse:.4g}, R2={grd_r2:.4f})"
        )
    else:
        state = HYBRID_GROUNDING_MACHINE.require_transition(
            state, HybridGroundingState.REJECTED, trigger="reject_vv10_hybrid"
        ).to_state
        tier = ExperimentalValidationTier.COUPON_CALIBRATED
        reason = (
            f"ASME V&V 10 hybrid grounding rejected: E={grd_rmse:.4g} > "
            f"{cfg.coverage_factor_k:.2g}*u_val={budget:.4g} {unit}"
        )

    return model, HybridGroundingDossier(
        sensor_kind=sensor_kind,
        unit=unit,
        numerical_claim_level=cfg.numerical_claim_level,
        experimental_validation_tier=tier,
        state=state,
        fitted_prior_scale_rho=rho,
        calibration_specimen_ids=tuple(sorted(cal_ids)),
        holdout_specimen_ids=tuple(sorted(val_ids)),
        raw_sim_holdout_rmse=raw_rmse,
        grounded_holdout_rmse=grd_rmse,
        raw_sim_max_abs_error=raw_max,
        grounded_max_abs_error=grd_max,
        raw_sim_holdout_r2=raw_r2,
        grounded_holdout_r2=grd_r2,
        bias_reduction_fraction=bias_red,
        u_num=cfg.u_num,
        u_input=cfg.u_input,
        u_discrepancy=u_disc,
        u_exp=u_exp_h,
        u_val=u_val,
        validation_ratio=ratio,
        passed=passed,
        reason=reason,
    )


__all__ = (
    "HybridDiscrepancyModel",
    "HybridGroundingConfig",
    "SensorMeshAlignmentReport",
    "align_sensor_observations_to_mesh",
    "ground_simulation_with_experiments",
)
