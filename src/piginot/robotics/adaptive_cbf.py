"""Online Adaptive Conformal Quantile Control Barrier Function (ACP-CQR, V7-R2).

Upgrades static split-conformal :mod:`piginot.robotics.cbf` with:

1. **Conformalized Quantile Regression (CQR, Romano et al. 2019):** Fits
   state-dependent asymmetric lower/upper safety quantiles ``(q_lo(x), q_hi(x))``
   with pinball loss and finite-sample CQR calibration so safety margins shrink
   tightly in low-uncertainty operating regimes.
2. **Adaptive Conformal Prediction (ACP, Gibbs & Candes 2021):** Updates the
   effective miscoverage level ``alpha_t`` online from time-lagged MPC / sensor
   prediction residuals::

       err_t = 1[y_obs > y_upper_t]
       alpha_{t+1} = clip(alpha_t + gamma_acp * (alpha_target - err_t), alpha_min, alpha_max)

   dynamically expanding barrier margins during payload or contact-geometry
   shifts *before* tripping the fail-closed ``SAFE_HOLD_OOD`` fallback.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from piginot.contracts.canonical import canonical_json_bytes
from piginot.robotics.cbf import (
    ControlSafetyState,
    SafetyEvaluatorFn,
)

FloatArray = NDArray[np.float64]
ADAPTIVE_CBF_DECISION_SCHEMA: Final = "formamech.adaptive-conformal-cbf-v1"


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _pinball_grad(residual: FloatArray, tau: float) -> FloatArray:
    """Subgradient of pinball loss wrt prediction y_hat where residual = y - y_hat."""
    return np.where(residual > 0.0, -tau, 1.0 - tau).astype(np.float64)


class ConformalQuantileRegressor:
    """State-dependent asymmetric quantile regressor with CQR calibration.

    Uses standardized polynomial + radial basis features trained via smooth
    pinball-loss gradient descent (deterministic, zero-dependency NumPy), then
    calibrates the finite-sample one-sided CQR score ``s_i = y_i - q_hi(x_i)``
    on held-out ``DEV`` samples.
    """

    def __init__(
        self,
        *,
        alpha_target: float = 0.05,
        n_centers: int = 8,
        l2_reg: float = 1.0e-3,
        max_iters: int = 250,
    ) -> None:
        if not (0.0 < alpha_target < 0.5):
            raise ValueError("alpha_target must lie strictly in (0, 0.5)")
        self.alpha_target = float(alpha_target)
        self.n_centers = int(max(2, n_centers))
        self.l2_reg = float(max(0.0, l2_reg))
        self.max_iters = int(max(20, max_iters))

        self._x_mean: FloatArray | None = None
        self._x_std: FloatArray | None = None
        self._centers: FloatArray | None = None
        self._rbf_scale: float = 1.0
        self._w_lo: FloatArray | None = None
        self._w_hi: FloatArray | None = None
        self._dev_scores: FloatArray = np.zeros(0, dtype=np.float64)
        self._cqr_shift: float = 0.0
        self._static_shift: float = 0.0

    def _featurize(self, x: FloatArray) -> FloatArray:
        assert self._x_mean is not None
        assert self._x_std is not None
        assert self._centers is not None
        xs = (x - self._x_mean) / self._x_std
        # [1, xs, xs^2, RBF(xs, centers)]
        diff = xs[:, None, :] - self._centers[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        rbf = np.exp(-0.5 * dist2 / max(self._rbf_scale, 1.0e-6))
        ones = np.ones((xs.shape[0], 1), dtype=np.float64)
        return np.hstack((ones, xs, xs * xs, rbf))

    def _fit_quantile_weights(
        self, phi: FloatArray, y: FloatArray, tau: float
    ) -> FloatArray:
        n, d = phi.shape
        # Warm-start from ridge regression mean + empirical residual quantile
        gram = phi.T @ phi + (self.l2_reg * n) * np.eye(d, dtype=np.float64)
        w: FloatArray = np.asarray(np.linalg.solve(gram, phi.T @ y), dtype=np.float64)
        res0 = y - phi @ w
        w[0] += float(np.quantile(res0, tau))

        # Smooth pinball optimization via Adam scaled to residual magnitude
        m: FloatArray = np.zeros(d, dtype=np.float64)
        v: FloatArray = np.zeros(d, dtype=np.float64)
        lr = 0.03 * max(1.0e-4, float(np.std(y)))
        for step in range(1, self.max_iters + 1):
            pred = phi @ w
            res = y - pred
            g_pred = _pinball_grad(res, tau) / float(n)
            grad = phi.T @ g_pred + self.l2_reg * w
            grad[0] -= self.l2_reg * w[0]  # do not regularize intercept
            m = np.asarray(0.9 * m + 0.1 * grad, dtype=np.float64)
            v = np.asarray(0.999 * v + 0.001 * (grad * grad), dtype=np.float64)
            m_hat = m / (1.0 - 0.9**step)
            v_hat = v / (1.0 - 0.999**step)
            w = np.asarray(w - lr * m_hat / (np.sqrt(v_hat) + 1.0e-8), dtype=np.float64)
        return np.asarray(w, dtype=np.float64)

    def fit_and_calibrate(
        self,
        x_train: FloatArray,
        y_train_residual: FloatArray,
        x_dev: FloatArray,
        y_dev_residual: FloatArray,
    ) -> dict[str, float]:
        """Fit asymmetric lower/upper residual quantiles on TRAIN and CQR-calibrate on DEV."""
        xt = np.atleast_2d(np.asarray(x_train, dtype=np.float64))
        yt = np.asarray(y_train_residual, dtype=np.float64).reshape(-1)
        xd = np.atleast_2d(np.asarray(x_dev, dtype=np.float64))
        yd = np.asarray(y_dev_residual, dtype=np.float64).reshape(-1)
        if xt.shape[0] < 4 or xd.shape[0] < 2:
            raise ValueError("CQR requires at least 4 train and 2 dev samples")

        self._x_mean = np.mean(xt, axis=0)
        self._x_std = np.maximum(np.std(xt, axis=0), 1.0e-6)
        xs = (xt - self._x_mean) / self._x_std
        # Deterministic evenly spaced quantile centers across sorted norm
        order = np.argsort(np.linalg.norm(xs, axis=1))
        k_c = min(self.n_centers, xt.shape[0])
        idx = np.linspace(0, xt.shape[0] - 1, k_c).astype(int)
        self._centers = xs[order[idx]].copy()
        self._rbf_scale = max(float(xt.shape[1]), 1.0)

        phi_t = self._featurize(xt)
        tau_lo = 0.5 * self.alpha_target
        tau_hi = 1.0 - self.alpha_target
        self._w_lo = self._fit_quantile_weights(phi_t, yt, tau_lo)
        self._w_hi = self._fit_quantile_weights(phi_t, yt, tau_hi)

        # Finite-sample CQR one-sided upper conformity scores on DEV
        _, q_hi_dev = self.predict_raw_quantiles(xd)
        scores = yd - q_hi_dev
        self._dev_scores = np.sort(scores.astype(np.float64))
        n_d = int(scores.size)
        q_level = min(
            1.0, math.ceil((n_d + 1) * (1.0 - self.alpha_target)) / float(n_d)
        )
        self._cqr_shift = float(np.quantile(self._dev_scores, q_level))
        # Static split-conformal baseline margin for direct comparison
        self._static_shift = float(np.quantile(yd, q_level))
        return {
            "cqr_shift": self._cqr_shift,
            "static_shift": self._static_shift,
            "dev_count": float(n_d),
        }

    def predict_raw_quantiles(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return uncalibrated ``(q_lo(x), q_hi(x))`` residual quantiles."""
        if self._w_lo is None or self._w_hi is None:
            raise RuntimeError("ConformalQuantileRegressor has not been fitted")
        xa = np.atleast_2d(np.asarray(x, dtype=np.float64))
        phi = self._featurize(xa)
        q_lo = phi @ self._w_lo
        q_hi = phi @ self._w_hi
        q_hi = np.maximum(q_hi, q_lo)
        return q_lo.astype(np.float64), q_hi.astype(np.float64)

    def predict_upper_margin(
        self,
        x: FloatArray,
        *,
        alpha_eff: float | None = None,
        online_scores: Sequence[float] = (),
        min_margin: float = 0.0,
    ) -> float:
        """Predict adaptive CQR one-sided upper safety margin for a single state ``x``."""
        xa = np.asarray(x, dtype=np.float64).reshape(1, -1)
        _, q_hi_arr = self.predict_raw_quantiles(xa)
        q_hi = float(q_hi_arr[0])

        alpha_use = self.alpha_target if alpha_eff is None else float(alpha_eff)
        alpha_use = max(1.0e-4, min(0.95, alpha_use))

        if online_scores:
            pool = np.concatenate(
                (self._dev_scores, np.asarray(online_scores, dtype=np.float64))
            )
            recent = np.asarray(online_scores, dtype=np.float64)
            # Blend finite-sample global quantile with recent local shift quantile
            n_p = int(pool.size)
            lvl_p = min(1.0, math.ceil((n_p + 1) * (1.0 - alpha_use)) / float(n_p))
            shift_pool = float(np.quantile(pool, lvl_p))
            n_r = int(recent.size)
            lvl_r = min(1.0, math.ceil((n_r + 1) * (1.0 - alpha_use)) / float(n_r))
            shift_recent = float(np.quantile(recent, lvl_r))
            cqr_q = max(shift_pool, shift_recent)
        elif self._dev_scores.size > 0:
            n_d = int(self._dev_scores.size)
            lvl = min(1.0, math.ceil((n_d + 1) * (1.0 - alpha_use)) / float(n_d))
            cqr_q = float(np.quantile(self._dev_scores, lvl))
        else:
            cqr_q = self._cqr_shift

        return float(max(min_margin, q_hi + cqr_q))

    @property
    def static_shift(self) -> float:
        return self._static_shift


@dataclass(frozen=True, slots=True)
class AdaptiveConformalBarrierSpec:
    """Adaptive ACP-CQR barrier specification for one safety-critical QoI."""

    name: str
    safety_limit: float
    base_margin: float
    alpha_target: float = 0.05
    gamma_acp: float = 0.08
    alpha_min: float = 0.002
    alpha_max: float = 0.25
    min_margin: float = 0.0
    max_margin: float | None = None
    gamma_cbf: float = 5.0
    window_size: int = 40
    quantile_model: ConformalQuantileRegressor | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("AdaptiveConformalBarrierSpec name must be non-empty")
        if not math.isfinite(self.safety_limit):
            raise ValueError("safety_limit must be finite")
        if not math.isfinite(self.base_margin) or self.base_margin < 0.0:
            raise ValueError("base_margin must be finite and nonnegative")
        if not (0.0 < self.alpha_min <= self.alpha_target <= self.alpha_max < 1.0):
            raise ValueError(
                "must satisfy 0 < alpha_min <= alpha_target <= alpha_max < 1"
            )
        if not math.isfinite(self.gamma_acp) or self.gamma_acp <= 0.0:
            raise ValueError("gamma_acp must be positive")
        if not math.isfinite(self.gamma_cbf) or self.gamma_cbf <= 0.0:
            raise ValueError("gamma_cbf must be positive")
        if self.window_size < 4:
            raise ValueError("window_size must be >= 4")


@dataclass(frozen=True, slots=True)
class AdaptiveCbfState:
    """Snapshot of online ACP miscoverage levels and rolling residual counts."""

    alpha_by_barrier: dict[str, float]
    margin_by_barrier: dict[str, float]
    miscoverage_rate_by_barrier: dict[str, float]
    update_count: int


@dataclass(frozen=True, slots=True)
class AdaptiveCbfResult:
    """Auditable ACP-CQR real-time safety-filter decision with SHA-256 receipt."""

    state: ControlSafetyState
    u_nominal: FloatArray
    u_safe: FloatArray
    barrier_values: dict[str, float]
    adaptive_margins: dict[str, float]
    alpha_effective: dict[str, float]
    min_barrier: float
    ood_distance: float
    modified: bool
    margin_expanded_for_shift: bool
    filter_time_ms: float
    reason: str

    def document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": ADAPTIVE_CBF_DECISION_SCHEMA,
            "state": self.state.value,
            "u_nominal": [float(v) for v in self.u_nominal],
            "u_safe": [float(v) for v in self.u_safe],
            "barrier_values": {
                k: float(v) for k, v in sorted(self.barrier_values.items())
            },
            "adaptive_margins": {
                k: float(v) for k, v in sorted(self.adaptive_margins.items())
            },
            "alpha_effective": {
                k: float(v) for k, v in sorted(self.alpha_effective.items())
            },
            "min_barrier": float(self.min_barrier),
            "ood_distance": float(self.ood_distance),
            "modified": self.modified,
            "margin_expanded_for_shift": self.margin_expanded_for_shift,
            "filter_time_ms": float(self.filter_time_ms),
            "reason": self.reason,
        }
        doc["receipt_sha256"] = _sha256_doc(doc)
        return doc


class AdaptiveConformalCbfFilter:
    """Online ACP-CQR Conformal Control Barrier Function safety filter (V7-R2).

    Unlike static split-conformal CBF, ``AdaptiveConformalCbfFilter``:
    * Uses state-dependent CQR quantiles to tighten margins in low-uncertainty
      regimes (reducing unnecessary braking distance).
    * Adapts ``alpha_t`` online from lagged residuals (Gibbs & Candes ACP),
      expanding margins during payload or contact-geometry shifts in the soft
      OOD zone ``[soft_ood_threshold, hard_ood_threshold]`` before escalating
      fail-closed to ``SAFE_HOLD_OOD`` beyond ``hard_ood_threshold``.
    """

    def __init__(
        self,
        barriers: Sequence[AdaptiveConformalBarrierSpec],
        *,
        u_min: FloatArray,
        u_max: FloatArray,
        safe_hold_action: FloatArray,
        soft_ood_threshold: float = 2.5,
        hard_ood_threshold: float = 5.0,
        ood_margin_gain: float = 0.5,
        dt_s: float = 0.01,
    ) -> None:
        if not barriers:
            raise ValueError("at least one AdaptiveConformalBarrierSpec is required")
        if hard_ood_threshold < soft_ood_threshold:
            raise ValueError("hard_ood_threshold must be >= soft_ood_threshold")
        self.barriers = tuple(barriers)
        self._barrier_map = {b.name: b for b in self.barriers}
        self.u_min = np.asarray(u_min, dtype=np.float64).reshape(-1)
        self.u_max = np.asarray(u_max, dtype=np.float64).reshape(-1)
        self.safe_hold = np.clip(
            np.asarray(safe_hold_action, dtype=np.float64).reshape(-1),
            self.u_min,
            self.u_max,
        )
        self.soft_ood_threshold = float(soft_ood_threshold)
        self.hard_ood_threshold = float(hard_ood_threshold)
        self.ood_margin_gain = float(max(0.0, ood_margin_gain))
        self.dt_s = float(dt_s)

        self._alpha: dict[str, float] = {b.name: b.alpha_target for b in self.barriers}
        self._scores: dict[str, deque[float]] = {
            b.name: deque(maxlen=b.window_size) for b in self.barriers
        }
        self._err_hist: dict[str, deque[float]] = {
            b.name: deque(maxlen=b.window_size) for b in self.barriers
        }
        self._last_margins: dict[str, float] = {
            b.name: b.base_margin for b in self.barriers
        }
        self._update_count: int = 0

    def compute_margin(
        self,
        barrier_name: str,
        x_state: FloatArray,
        *,
        ood_distance: float = 0.0,
    ) -> float:
        """Compute the current state-dependent ACP-CQR upper safety margin."""
        b = self._barrier_map[barrier_name]
        alpha_cur = self._alpha[barrier_name]
        scores = tuple(self._scores[barrier_name])

        if b.quantile_model is not None:
            margin = b.quantile_model.predict_upper_margin(
                x_state,
                alpha_eff=alpha_cur,
                online_scores=scores,
                min_margin=b.min_margin,
            )
        else:
            # Pure online ACP quantile adaptation around base_margin
            if scores:
                arr = np.asarray(scores, dtype=np.float64)
                n_s = int(arr.size)
                lvl = min(1.0, math.ceil((n_s + 1) * (1.0 - alpha_cur)) / float(n_s))
                emp_q = float(np.quantile(arr, lvl))
                # Shrink gently below base_margin when recent residuals are small
                alpha_scale = max(
                    0.35, min(2.5, math.sqrt(b.alpha_target / max(alpha_cur, 1.0e-4)))
                )
                margin = max(b.min_margin, emp_q * alpha_scale)
            else:
                margin = b.base_margin

        # Smooth inflation in soft OOD transition zone before hard SAFE_HOLD_OOD
        if math.isfinite(ood_distance) and ood_distance > self.soft_ood_threshold:
            span = max(self.hard_ood_threshold - self.soft_ood_threshold, 1.0e-6)
            frac = min(1.0, (ood_distance - self.soft_ood_threshold) / span)
            margin = margin * (1.0 + self.ood_margin_gain * frac)

        if b.max_margin is not None:
            margin = min(margin, b.max_margin)
        return float(max(b.min_margin, margin))

    def update_online(
        self,
        x_prev: FloatArray,
        y_pred_prev: Mapping[str, float],
        y_observed: Mapping[str, float],
        *,
        ood_prev: float = 0.0,
    ) -> AdaptiveCbfState:
        """Update ACP miscoverage ``alpha_t`` and CQR residual queues from lagged data."""
        self._update_count += 1
        x_vec = np.asarray(x_prev, dtype=np.float64).reshape(-1)
        for b in self.barriers:
            if b.name not in y_observed or b.name not in y_pred_prev:
                continue
            yp = float(y_pred_prev[b.name])
            yo = float(y_observed[b.name])
            residual = yo - yp
            # Evaluate whether the previous adaptive upper bound covered y_observed
            prev_margin = self.compute_margin(b.name, x_vec, ood_distance=ood_prev)
            err_t = 1.0 if yo > (yp + prev_margin) else 0.0
            self._err_hist[b.name].append(err_t)

            # Store CQR score (residual - raw q_hi) or raw residual
            if b.quantile_model is not None:
                _, q_hi_arr = b.quantile_model.predict_raw_quantiles(
                    x_vec.reshape(1, -1)
                )
                score_t = residual - float(q_hi_arr[0])
            else:
                score_t = max(0.0, residual)
            self._scores[b.name].append(score_t)

            # Gibbs & Candes ACP update: tighten if covered, rapidly expand on miss
            alpha_next = self._alpha[b.name] + b.gamma_acp * (b.alpha_target - err_t)
            self._alpha[b.name] = float(max(b.alpha_min, min(b.alpha_max, alpha_next)))
            self._last_margins[b.name] = self.compute_margin(
                b.name, x_vec, ood_distance=ood_prev
            )

        return self.state_snapshot()

    def state_snapshot(self) -> AdaptiveCbfState:
        """Return immutable snapshot of current online ACP state."""
        rates = {
            b.name: (
                float(np.mean(self._err_hist[b.name]))
                if self._err_hist[b.name]
                else 0.0
            )
            for b in self.barriers
        }
        return AdaptiveCbfState(
            alpha_by_barrier=dict(self._alpha),
            margin_by_barrier=dict(self._last_margins),
            miscoverage_rate_by_barrier=rates,
            update_count=self._update_count,
        )

    def filter_action(
        self,
        x_current: FloatArray,
        u_nominal: FloatArray,
        *,
        eval_current_qois: Callable[[FloatArray], tuple[dict[str, float], float]],
        predict_next_qois: SafetyEvaluatorFn,
    ) -> AdaptiveCbfResult:
        """Filter ``u_nominal`` using live state-dependent ACP-CQR barrier margins."""
        t0 = time.perf_counter()
        x_vec = np.asarray(x_current, dtype=np.float64).reshape(-1)
        u_nom = np.clip(
            np.asarray(u_nominal, dtype=np.float64).reshape(-1),
            self.u_min,
            self.u_max,
        )
        qois_cur, ood_cur = eval_current_qois(x_vec)
        margins_cur = {
            b.name: self.compute_margin(b.name, x_vec, ood_distance=ood_cur)
            for b in self.barriers
        }
        self._last_margins.update(margins_cur)
        h_cur = {
            b.name: b.safety_limit - (float(qois_cur[b.name]) + margins_cur[b.name])
            for b in self.barriers
        }
        expanded = math.isfinite(ood_cur) and ood_cur > self.soft_ood_threshold

        if not math.isfinite(ood_cur) or ood_cur > self.hard_ood_threshold:
            return AdaptiveCbfResult(
                state=ControlSafetyState.SAFE_HOLD_OOD,
                u_nominal=u_nom,
                u_safe=self.safe_hold.copy(),
                barrier_values=h_cur,
                adaptive_margins=margins_cur,
                alpha_effective=dict(self._alpha),
                min_barrier=min(h_cur.values()),
                ood_distance=float(ood_cur),
                modified=True,
                margin_expanded_for_shift=True,
                filter_time_ms=(time.perf_counter() - t0) * 1000.0,
                reason=(
                    f"OOD distance {ood_cur:.3f} exceeds hard threshold "
                    f"{self.hard_ood_threshold:.3f}; fail-closed SAFE_HOLD_OOD"
                ),
            )

        qois_next, ood_next = predict_next_qois(x_vec, u_nom)
        ood_eff = float(max(ood_cur, ood_next))
        if not math.isfinite(ood_next) or ood_next > self.hard_ood_threshold:
            return AdaptiveCbfResult(
                state=ControlSafetyState.SAFE_HOLD_OOD,
                u_nominal=u_nom,
                u_safe=self.safe_hold.copy(),
                barrier_values=h_cur,
                adaptive_margins=margins_cur,
                alpha_effective=dict(self._alpha),
                min_barrier=min(h_cur.values()),
                ood_distance=float(ood_next),
                modified=True,
                margin_expanded_for_shift=True,
                filter_time_ms=(time.perf_counter() - t0) * 1000.0,
                reason="predicted next-step OOD exceeds hard threshold",
            )

        expanded = expanded or (ood_eff > self.soft_ood_threshold)
        margins_eff = {
            b.name: self.compute_margin(b.name, x_vec, ood_distance=ood_eff)
            for b in self.barriers
        }
        self._last_margins.update(margins_eff)

        def _cbf_slack(q_map: dict[str, float]) -> dict[str, float]:
            out: dict[str, float] = {}
            for b in self.barriers:
                h_n = b.safety_limit - (float(q_map[b.name]) + margins_eff[b.name])
                decay = max(0.0, 1.0 - b.gamma_cbf * self.dt_s)
                out[b.name] = h_n - decay * h_cur[b.name]
            return out

        slack_nom = _cbf_slack(qois_next)
        if all(sv >= 0.0 for sv in slack_nom.values()):
            h_next = {
                b.name: b.safety_limit
                - (float(qois_next[b.name]) + margins_eff[b.name])
                for b in self.barriers
            }
            return AdaptiveCbfResult(
                state=ControlSafetyState.NOMINAL,
                u_nominal=u_nom,
                u_safe=u_nom.copy(),
                barrier_values=h_next,
                adaptive_margins=margins_eff,
                alpha_effective=dict(self._alpha),
                min_barrier=min(h_next.values()),
                ood_distance=ood_eff,
                modified=False,
                margin_expanded_for_shift=expanded,
                filter_time_ms=(time.perf_counter() - t0) * 1000.0,
                reason="nominal action satisfies adaptive ACP-CQR conformal barriers",
            )

        # Iterative minimum-norm half-space projection onto adaptive barrier set
        m_dim = int(u_nom.size)
        u_cur = u_nom.copy()
        eps = 1.0e-3
        for _ in range(14):
            q_u, _ = predict_next_qois(x_vec, u_cur)
            sl_u = _cbf_slack(q_u)
            worst_name = min(sl_u, key=lambda k: sl_u[k])
            if sl_u[worst_name] >= -1.0e-7:
                break
            grad_m = np.zeros(m_dim, dtype=np.float64)
            for j in range(m_dim):
                du = np.zeros(m_dim, dtype=np.float64)
                du[j] = eps
                qp, _ = predict_next_qois(
                    x_vec, np.clip(u_cur + du, self.u_min, self.u_max)
                )
                qm, _ = predict_next_qois(
                    x_vec, np.clip(u_cur - du, self.u_min, self.u_max)
                )
                grad_m[j] = (
                    _cbf_slack(qp)[worst_name] - _cbf_slack(qm)[worst_name]
                ) / (2.0 * eps)
            gnorm2 = float(np.dot(grad_m, grad_m))
            if gnorm2 <= 1.0e-14:
                break
            step = (-sl_u[worst_name] + 1.0e-5) / gnorm2
            u_cur = np.clip(u_cur + step * grad_m, self.u_min, self.u_max)

        q_safe, ood_safe = predict_next_qois(x_vec, u_cur)
        sl_safe = _cbf_slack(q_safe)
        h_safe = {
            b.name: b.safety_limit - (float(q_safe[b.name]) + margins_eff[b.name])
            for b in self.barriers
        }
        if (
            all(sv >= -1.0e-4 for sv in sl_safe.values())
            and math.isfinite(ood_safe)
            and ood_safe <= self.hard_ood_threshold
        ):
            return AdaptiveCbfResult(
                state=ControlSafetyState.CBF_MODIFIED,
                u_nominal=u_nom,
                u_safe=u_cur,
                barrier_values=h_safe,
                adaptive_margins=margins_eff,
                alpha_effective=dict(self._alpha),
                min_barrier=min(h_safe.values()),
                ood_distance=float(max(ood_cur, ood_safe)),
                modified=True,
                margin_expanded_for_shift=expanded,
                filter_time_ms=(time.perf_counter() - t0) * 1000.0,
                reason="projected onto adaptive ACP-CQR conformal barrier half-space",
            )

        return AdaptiveCbfResult(
            state=ControlSafetyState.SAFE_HOLD_OOD,
            u_nominal=u_nom,
            u_safe=self.safe_hold.copy(),
            barrier_values=h_safe,
            adaptive_margins=margins_eff,
            alpha_effective=dict(self._alpha),
            min_barrier=min(h_safe.values()),
            ood_distance=float(max(ood_cur, ood_safe)),
            modified=True,
            margin_expanded_for_shift=True,
            filter_time_ms=(time.perf_counter() - t0) * 1000.0,
            reason="adaptive barrier infeasible under actuator limits; engaging SAFE_HOLD_OOD",
        )


__all__ = (
    "ADAPTIVE_CBF_DECISION_SCHEMA",
    "AdaptiveCbfResult",
    "AdaptiveCbfState",
    "AdaptiveConformalBarrierSpec",
    "AdaptiveConformalCbfFilter",
    "ConformalQuantileRegressor",
)
