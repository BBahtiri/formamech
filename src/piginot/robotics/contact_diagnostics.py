"""Energy-Residual Contact State & Jamming Estimator (V7-R4).

Combines three physics invariants to cleanly separate mechanical jamming /
incipient slip from benign high-force nominal snap-fit spikes:

1. **Port-Hamiltonian Power-Balance Residual:** Fits a passive
   :class:`~piginot.robotics.port_hamiltonian.PortHamiltonianSurrogate` on nominal
   motion frames and monitors excess unmodeled energy dissipation::

       r_P(t) = dH/dt - u(t)^T y(t) + q_dot(t)^T D q_dot(t)

2. **Multi-Scale Causal Wrench-Velocity Temporal Encoding:** Extracts causal
   rolling statistics across short/medium/long windows of mechanical impedance
   ``||F|| / (||v|| + v_0)``, torque-to-force lever arm ``||tau|| / (||F|| + f_0)``,
   signed mechanical work rate ``F . v``, and sustained stall-under-load duration.
3. **Coulomb Friction-Cone Slip Invariants:** Computes tangential-to-normal
   load ratio ``||F_t|| / (mu * |F_n| + eps)``, shear-rate escalation, and
   calibrated jamming/slip probabilities via leak-free episode splitting.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from piginot.contracts.canonical import canonical_json_bytes
from piginot.robotics.port_hamiltonian import (
    PortHamiltonianConfig,
    PortHamiltonianSurrogate,
)

FloatArray = NDArray[np.float64]
CONTACT_DIAGNOSTICS_SCHEMA: Final = "formamech.contact-diagnostics-v1"


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


@dataclass(frozen=True, slots=True)
class ContactDiagnosticsConfig:
    """Configuration for :class:`EnergyResidualContactEstimator`."""

    dt_s: float = 0.01
    friction_mu: float = 0.35
    short_window: int = 10
    medium_window: int = 35
    long_window: int = 90
    velocity_eps: float = 1.0e-3
    force_eps: float = 0.25
    target_max_fpr: float = 0.10
    hidden_dim: int = 48
    train_epochs: int = 120
    phnn_epochs: int = 60
    seed: int = 42

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0 or not math.isfinite(self.dt_s):
            raise ValueError("dt_s must be positive")
        if self.friction_mu <= 0.0 or not math.isfinite(self.friction_mu):
            raise ValueError("friction_mu must be positive")
        if not (1 <= self.short_window <= self.medium_window <= self.long_window):
            raise ValueError("windows must satisfy 1 <= short <= medium <= long")
        if not (0.0 < self.target_max_fpr < 1.0):
            raise ValueError("target_max_fpr must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class ContactDiagnosticResult:
    """Calibrated per-frame jamming / slip probabilities and physics residuals."""

    probabilities: FloatArray  # (T,) calibrated probability in [0, 1]
    predictions: NDArray[np.bool_]  # (T,) thresholded binary alarm
    power_balance_residual: FloatArray  # (T,) Port-Hamiltonian excess dissipation
    friction_cone_ratio: FloatArray  # (T,) ||F_t|| / (mu * |F_n| + eps)
    mechanical_impedance: FloatArray  # (T,) ||F|| / (||v|| + v_0)
    decision_threshold: float
    inference_ms: float

    def document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": CONTACT_DIAGNOSTICS_SCHEMA,
            "frame_count": int(self.probabilities.size),
            "alarm_fraction": float(np.mean(self.predictions))
            if self.predictions.size > 0
            else 0.0,
            "max_probability": float(np.max(self.probabilities))
            if self.probabilities.size > 0
            else 0.0,
            "mean_power_residual": float(np.mean(np.abs(self.power_balance_residual)))
            if self.power_balance_residual.size > 0
            else 0.0,
            "max_friction_cone_ratio": float(np.max(self.friction_cone_ratio))
            if self.friction_cone_ratio.size > 0
            else 0.0,
            "decision_threshold": float(self.decision_threshold),
            "inference_ms": float(self.inference_ms),
        }
        doc["receipt_sha256"] = _sha256_doc(doc)
        return doc


def _causal_rolling_mean(x: FloatArray, win: int) -> FloatArray:
    """Exact O(T) causal rolling mean over the last ``win`` samples."""
    n = int(x.size)
    if n == 0 or win <= 1:
        return x.copy()
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    idx = np.arange(1, n + 1)
    starts = np.maximum(0, idx - win)
    counts = np.maximum(1, idx - starts).astype(np.float64)
    return np.asarray((csum[idx] - csum[starts]) / counts, dtype=np.float64)


def _causal_rolling_std(x: FloatArray, win: int) -> FloatArray:
    m1 = _causal_rolling_mean(x, win)
    m2 = _causal_rolling_mean(x * x, win)
    return np.asarray(np.sqrt(np.maximum(m2 - m1 * m1, 0.0)), dtype=np.float64)


def _causal_ema(x: FloatArray, alpha: float) -> FloatArray:
    out = np.empty_like(x, dtype=np.float64)
    if x.size == 0:
        return out
    acc = float(x[0])
    out[0] = acc
    one_m = 1.0 - alpha
    for i in range(1, x.size):
        acc = alpha * float(x[i]) + one_m * acc
        out[i] = acc
    return out


def compute_roc_auroc(y_true: FloatArray, scores: FloatArray) -> float:
    """Exact rank-based AUROC (Mann-Whitney U) with tie handling."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1) > 0.5
    sc = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int(np.sum(yt))
    n_neg = int(yt.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(sc, kind="mergesort")
    sc_s = sc[order]
    yt_s = yt[order]
    # Average ranks for tied scores
    ranks = np.arange(1, sc.size + 1, dtype=np.float64)
    i = 0
    while i < sc_s.size:
        j = i + 1
        while j < sc_s.size and sc_s[j] == sc_s[i]:
            j += 1
        if j - i > 1:
            ranks[i:j] = 0.5 * (ranks[i] + ranks[j - 1])
        i = j
    sum_pos_ranks = float(np.sum(ranks[yt_s]))
    return float((sum_pos_ranks - 0.5 * n_pos * (n_pos + 1)) / float(n_pos * n_neg))


def calibrate_threshold_for_max_fpr(
    y_dev: FloatArray, probs_dev: FloatArray, max_fpr: float = 0.10
) -> tuple[float, float, float]:
    """Select decision threshold on DEV maximizing recall subject to ``FPR <= max_fpr``."""
    yt = np.asarray(y_dev, dtype=np.float64).reshape(-1) > 0.5
    pr = np.asarray(probs_dev, dtype=np.float64).reshape(-1)
    neg = pr[~yt]
    pos = pr[yt]
    if neg.size == 0 or pos.size == 0:
        return 0.5, 0.0, 0.0
    # Candidate threshold at the (1 - max_fpr) quantile of negative DEV frames
    thr = float(np.quantile(neg, max(0.0, min(1.0, 1.0 - max_fpr))))
    # Tiny nudge so strictly > thr meets FPR <= max_fpr
    fpr = float(np.mean(neg >= thr))
    if fpr > max_fpr:
        uniq = np.sort(np.unique(neg))
        idx = int(math.ceil((1.0 - max_fpr) * uniq.size))
        thr = float(uniq[min(idx, uniq.size - 1)]) + 1.0e-7
    fpr_final = float(np.mean(neg >= thr))
    rec_final = float(np.mean(pos >= thr))
    return thr, rec_final, fpr_final


class _ContactClassifierNet(nn.Module):
    """Compact residual MLP mapping multi-scale physics invariants to jamming logits."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x).squeeze(-1)
        return out


class EnergyResidualContactEstimator:
    """Port-Hamiltonian + temporal wrench-velocity jamming & slip estimator (V7-R4)."""

    def __init__(self, config: ContactDiagnosticsConfig | None = None) -> None:
        self.config = ContactDiagnosticsConfig() if config is None else config
        self._phnn: PortHamiltonianSurrogate | None = None
        self._phnn_q_scale: FloatArray = np.ones(3, dtype=np.float64)
        self._phnn_v_scale: FloatArray = np.ones(3, dtype=np.float64)
        self._phnn_f_scale: FloatArray = np.ones(3, dtype=np.float64)
        self._clf: _ContactClassifierNet | None = None
        self._feat_mean: FloatArray | None = None
        self._feat_std: FloatArray | None = None
        self._platt_a: float = 1.0
        self._platt_b: float = 0.0
        self._threshold: float = 0.5

    def _fit_nominal_phnn(
        self,
        positions_list: Sequence[FloatArray],
        velocities_list: Sequence[FloatArray],
        forces_list: Sequence[FloatArray],
        labels_list: Sequence[FloatArray],
    ) -> None:
        """Fit a 3-DOF Port-Hamiltonian energy model on nominal (non-jamming) frames."""
        q_nom_parts: list[FloatArray] = []
        v_nom_parts: list[FloatArray] = []
        f_nom_parts: list[FloatArray] = []
        v_next_parts: list[FloatArray] = []
        for pos, vel, frc, lbl in zip(
            positions_list, velocities_list, forces_list, labels_list, strict=True
        ):
            if pos.shape[0] < 3:
                continue
            nom = lbl[:-1] < 0.5
            if not np.any(nom):
                continue
            # Subsample to keep PHNN fit fast (< 0.5s) even on 150k-frame logs
            idx = np.nonzero(nom)[0]
            if idx.size > 400:
                idx = idx[np.linspace(0, idx.size - 1, 400).astype(int)]
            q_nom_parts.append(pos[idx] - pos[0:1])
            v_nom_parts.append(vel[idx])
            f_nom_parts.append(frc[idx])
            v_next_parts.append(vel[idx + 1])

        if not q_nom_parts:
            return

        q_all = np.vstack(q_nom_parts)
        v_all = np.vstack(v_nom_parts)
        f_all = np.vstack(f_nom_parts)
        vn_all = np.vstack(v_next_parts)
        self._phnn_q_scale = np.maximum(np.std(q_all, axis=0), 1.0e-4)
        self._phnn_v_scale = np.maximum(np.std(v_all, axis=0), 1.0e-4)
        self._phnn_f_scale = np.maximum(np.std(f_all, axis=0), 1.0e-4)

        q_t = torch.tensor(q_all / self._phnn_q_scale, dtype=torch.float32)
        p_t = torch.tensor(v_all / self._phnn_v_scale, dtype=torch.float32)
        u_t = torch.tensor(f_all / self._phnn_f_scale, dtype=torch.float32)
        dp_tgt = torch.tensor(
            (vn_all - v_all) / (self._phnn_v_scale * self.config.dt_s),
            dtype=torch.float32,
        )

        torch.manual_seed(self.config.seed)
        phnn = PortHamiltonianSurrogate(
            PortHamiltonianConfig(state_dim=3, control_dim=3, hidden_dim=32)
        )
        opt = torch.optim.Adam(phnn.parameters(), lr=8.0e-3)
        for _ in range(self.config.phnn_epochs):
            opt.zero_grad()
            _dq, dp_pred, _diss = phnn.vector_field(q_t, p_t, u_t, create_graph=True)
            loss = ((dp_pred - dp_tgt) ** 2).mean()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(phnn.parameters(), 5.0)
            opt.step()
        phnn.eval()
        self._phnn = phnn

    def compute_power_balance_residual(
        self,
        positions: FloatArray,
        velocities: FloatArray,
        forces: FloatArray,
    ) -> FloatArray:
        """Compute Port-Hamiltonian power-balance residual ``dH/dt - u^T y + diss``."""
        t_len = int(forces.shape[0])
        if t_len == 0:
            return np.zeros(0, dtype=np.float64)
        if self._phnn is None:
            # Fallback physical power imbalance: ||F||^2 - (F . v)^2 / (||v||^2 + eps)
            f_norm = np.linalg.norm(forces, axis=1)
            power = np.sum(forces * velocities, axis=1)
            return np.asarray(
                f_norm - np.abs(power) / (np.linalg.norm(velocities, axis=1) + 0.05),
                dtype=np.float64,
            )

        q_rel = (positions - positions[0:1]) / self._phnn_q_scale
        p_norm = velocities / self._phnn_v_scale
        u_norm = forces / self._phnn_f_scale

        # Evaluate in chunks to stay fast & bounded-memory on long episodes
        h_vals = np.zeros(t_len, dtype=np.float64)
        diss_vals = np.zeros(t_len, dtype=np.float64)
        port_in = np.zeros(t_len, dtype=np.float64)
        chunk = 2048
        for s in range(0, t_len, chunk):
            e = min(t_len, s + chunk)
            qt = torch.tensor(q_rel[s:e], dtype=torch.float32)
            pt = torch.tensor(p_norm[s:e], dtype=torch.float32)
            ut = torch.tensor(u_norm[s:e], dtype=torch.float32)
            with torch.no_grad():
                h_vals[s:e] = self._phnn.hamiltonian(qt, pt).numpy()
                m_inv = self._phnn.mass_inv_matrix(qt)
                d_mat = self._phnn.damping_matrix(qt)
                g_mat = self._phnn.input_matrix(qt)
                dh_dp = torch.einsum("bij,bj->bi", m_inv, pt)
                diss = torch.einsum("bi,bij,bj->b", dh_dp, d_mat, dh_dp)
                gu = torch.einsum("bij,bj->bi", g_mat, ut)
                pin = torch.einsum("bi,bi->b", dh_dp, gu)
                diss_vals[s:e] = diss.numpy()
                port_in[s:e] = pin.numpy()

        dh_dt = np.zeros(t_len, dtype=np.float64)
        if t_len > 1:
            dh_dt[1:] = np.diff(h_vals) / self.config.dt_s
            dh_dt[0] = dh_dt[1]
        # Exact Port-Hamiltonian energy balance residual:
        # In nominal passive dynamics: dH/dt = u^T y - diss  => residual ~ 0
        # Under jamming/wedging: unmodeled reaction force injects huge power discrepancy!
        return (dh_dt - port_in + diss_vals).astype(np.float64)

    def extract_episode_features(
        self,
        forces: FloatArray,  # (T, 3) N
        torques: FloatArray | None = None,  # (T, 3) N*m or N*mm
        positions: FloatArray | None = None,  # (T, 3) m or mm
        velocities: FloatArray | None = None,  # (T, 3)
        extra_channels: FloatArray | None = None,  # (T, C) optional auxiliary signals
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Extract causal physics-invariant multi-scale feature matrix ``(T, F)``."""
        cfg = self.config
        f = np.asarray(forces, dtype=np.float64)
        t_len = int(f.shape[0])
        tau = (
            np.zeros((t_len, 3), dtype=np.float64)
            if torques is None
            else np.asarray(torques, dtype=np.float64)
        )
        pos = (
            np.zeros((t_len, 3), dtype=np.float64)
            if positions is None
            else np.asarray(positions, dtype=np.float64)
        )
        if velocities is not None:
            vel = np.asarray(velocities, dtype=np.float64)
        else:
            vel = np.zeros((t_len, 3), dtype=np.float64)
            if t_len > 1:
                vel[1:] = np.diff(pos, axis=0) / cfg.dt_s
                vel[0] = vel[1]

        # 1. Core instantaneous physical invariants
        f_mag = np.linalg.norm(f, axis=1)
        f_xy = np.sqrt(f[:, 0] ** 2 + f[:, 1] ** 2)
        f_z_abs = np.abs(f[:, 2])
        tau_mag = np.linalg.norm(tau, axis=1)
        v_mag = np.linalg.norm(vel, axis=1)

        # Coulomb friction-cone slip invariant: ||F_t|| / (mu * |F_n| + eps)
        slip_ratio = f_xy / (cfg.friction_mu * f_z_abs + cfg.force_eps)
        # Mechanical impedance: high force while stalled (v ~ 0) == jamming!
        impedance = f_mag / (v_mag + cfg.velocity_eps)
        # Off-axis wedging lever-arm ratio: torque per unit force
        lever_arm = tau_mag / (f_mag + cfg.force_eps)
        # Mechanical power F . v (positive = motion assisted, negative/near-zero under load = wedged)
        mech_power = np.sum(f * vel, axis=1)
        # Relative directional cosine cos(F, v)
        cos_fv = mech_power / (f_mag * v_mag + 1.0e-5)

        # Wrench rate-of-change (chatter / snap-fit impulse vs sustained plateau)
        df_dt = np.zeros(t_len, dtype=np.float64)
        dtau_dt = np.zeros(t_len, dtype=np.float64)
        if t_len > 1:
            df_dt[1:] = np.linalg.norm(np.diff(f, axis=0), axis=1) / cfg.dt_s
            dtau_dt[1:] = np.linalg.norm(np.diff(tau, axis=0), axis=1) / cfg.dt_s

        # 2. Port-Hamiltonian power-balance residual
        r_power = self.compute_power_balance_residual(pos, vel, f)
        r_power_abs = np.abs(r_power)

        # 3. Causal multi-scale temporal window features (distinguishes brief snap-fit vs sustained jam!)
        ws, wm, wl = cfg.short_window, cfg.medium_window, cfg.long_window
        f_short = _causal_rolling_mean(f_mag, ws)
        f_med = _causal_rolling_mean(f_mag, wm)
        f_long = _causal_rolling_mean(f_mag, wl)
        f_std_med = _causal_rolling_std(f_mag, wm)

        tau_short = _causal_rolling_mean(tau_mag, ws)
        tau_med = _causal_rolling_mean(tau_mag, wm)
        tau_long = _causal_rolling_mean(tau_mag, wl)

        v_short = _causal_rolling_mean(v_mag, ws)
        v_med = _causal_rolling_mean(v_mag, wm)

        imp_med = _causal_rolling_mean(np.log1p(impedance), wm)
        imp_long = _causal_rolling_mean(np.log1p(impedance), wl)
        rp_short = _causal_rolling_mean(np.log1p(r_power_abs), ws)
        rp_med = _causal_rolling_mean(np.log1p(r_power_abs), wm)
        rp_long = _causal_rolling_mean(np.log1p(r_power_abs), wl)
        slip_short = _causal_rolling_mean(slip_ratio, ws)
        slip_med = _causal_rolling_mean(slip_ratio, wm)
        df_med = _causal_rolling_mean(np.log1p(df_dt), wm)
        dtau_med = _causal_rolling_mean(np.log1p(dtau_dt), wm)

        # Cumulative causal stall-under-load indicator (EMA of high force + low velocity)
        load_stall = (f_mag / (f_long + 1.0)) * np.exp(
            -v_mag / max(cfg.velocity_eps * 5.0, 1.0e-4)
        )
        stall_ema_fast = _causal_ema(load_stall, 0.15)
        stall_ema_slow = _causal_ema(load_stall, 0.03)

        # Causal normalized position progress from episode start
        disp_from_start = np.linalg.norm(pos - pos[0:1], axis=1)
        rel_time = np.linspace(0.0, 1.0, t_len, dtype=np.float64)

        cols = [
            f[:, 0],
            f[:, 1],
            f[:, 2],
            f_mag,
            f_xy,
            tau_mag,
            v_mag,
            slip_ratio,
            np.log1p(impedance),
            lever_arm,
            mech_power,
            cos_fv,
            np.log1p(df_dt),
            np.log1p(dtau_dt),
            np.log1p(r_power_abs),
            r_power / (np.std(r_power) + 1.0),
            f_short,
            f_med,
            f_long,
            f_std_med,
            f_short - f_long,
            tau_short,
            tau_med,
            tau_long,
            v_short,
            v_med,
            imp_med,
            imp_long,
            rp_short,
            rp_med,
            rp_long,
            slip_short,
            slip_med,
            df_med,
            dtau_med,
            stall_ema_fast,
            stall_ema_slow,
            disp_from_start,
            rel_time,
        ]
        feat = np.column_stack(cols).astype(np.float64)
        if extra_channels is not None:
            ex = np.asarray(extra_channels, dtype=np.float64)
            if ex.ndim == 1:
                ex = ex[:, None]
            feat = np.hstack((feat, ex))
        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        return feat, r_power, slip_ratio, impedance

    def fit(
        self,
        train_episodes: Sequence[dict[str, Any]],
        dev_episodes: Sequence[dict[str, Any]],
    ) -> dict[str, float]:
        """Train PHNN + contact classifier on TRAIN episodes and calibrate strictly on DEV.

        Each episode dict holds ``"forces"`` ``(T, 3)``, ``"labels"`` ``(T,)``,
        and optional ``"torques"``, ``"positions"``, ``"velocities"``, ``"extra"``.
        """
        if not train_episodes or not dev_episodes:
            raise ValueError("train_episodes and dev_episodes must be non-empty")

        # 1. Fit Port-Hamiltonian energy balance model strictly on nominal TRAIN frames
        self._fit_nominal_phnn(
            [
                np.asarray(
                    ep.get("positions", np.zeros_like(ep["forces"])), dtype=np.float64
                )
                for ep in train_episodes
            ],
            [
                np.asarray(
                    ep.get("velocities", np.zeros_like(ep["forces"])), dtype=np.float64
                )
                for ep in train_episodes
            ],
            [np.asarray(ep["forces"], dtype=np.float64) for ep in train_episodes],
            [np.asarray(ep["labels"], dtype=np.float64) for ep in train_episodes],
        )

        # 2. Extract multi-scale causal features for TRAIN and DEV episodes
        x_tr_list: list[FloatArray] = []
        y_tr_list: list[FloatArray] = []
        for ep in train_episodes:
            fx, _, _, _ = self.extract_episode_features(
                ep["forces"],
                torques=ep.get("torques"),
                positions=ep.get("positions"),
                velocities=ep.get("velocities"),
                extra_channels=ep.get("extra"),
            )
            x_tr_list.append(fx)
            y_tr_list.append(np.asarray(ep["labels"], dtype=np.float64).reshape(-1))
        x_tr = np.vstack(x_tr_list)
        y_tr = np.concatenate(y_tr_list)

        x_dev_list: list[FloatArray] = []
        y_dev_list: list[FloatArray] = []
        for ep in dev_episodes:
            fx, _, _, _ = self.extract_episode_features(
                ep["forces"],
                torques=ep.get("torques"),
                positions=ep.get("positions"),
                velocities=ep.get("velocities"),
                extra_channels=ep.get("extra"),
            )
            x_dev_list.append(fx)
            y_dev_list.append(np.asarray(ep["labels"], dtype=np.float64).reshape(-1))
        x_dev = np.vstack(x_dev_list)
        y_dev = np.concatenate(y_dev_list)

        # Standardize strictly using TRAIN moments (zero DEV/TEST leakage!)
        self._feat_mean = np.mean(x_tr, axis=0)
        self._feat_std = np.maximum(np.std(x_tr, axis=0), 1.0e-6)
        x_tr_s = np.clip((x_tr - self._feat_mean) / self._feat_std, -8.0, 8.0)
        x_dev_s = np.clip((x_dev - self._feat_mean) / self._feat_std, -8.0, 8.0)

        # Stratified subsampling for fast deterministic training if > 24k frames
        rng = np.random.default_rng(self.config.seed)
        if x_tr_s.shape[0] > 24_000:
            pos_i = np.nonzero(y_tr > 0.5)[0]
            neg_i = np.nonzero(y_tr <= 0.5)[0]
            n_pos_take = min(pos_i.size, 12_000)
            n_neg_take = min(neg_i.size, 12_000)
            sel = np.concatenate(
                (
                    rng.choice(pos_i, size=n_pos_take, replace=False),
                    rng.choice(neg_i, size=n_neg_take, replace=False),
                )
            )
            rng.shuffle(sel)
            x_fit = x_tr_s[sel]
            y_fit = y_tr[sel]
        else:
            x_fit = x_tr_s
            y_fit = y_tr

        torch.manual_seed(self.config.seed)
        clf = _ContactClassifierNet(x_tr_s.shape[1], self.config.hidden_dim)
        pos_cnt = float(np.sum(y_fit > 0.5))
        neg_cnt = float(y_fit.size - pos_cnt)
        pos_weight = torch.tensor(
            [min(8.0, max(0.25, neg_cnt / max(pos_cnt, 1.0)))], dtype=torch.float32
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(clf.parameters(), lr=4.0e-3, weight_decay=2.0e-3)

        xt_t = torch.tensor(x_fit, dtype=torch.float32)
        yt_t = torch.tensor(y_fit, dtype=torch.float32)
        xd_t = torch.tensor(x_dev_s, dtype=torch.float32)

        batch_sz = min(1024, xt_t.shape[0])
        best_dev_auc = -1.0
        best_state = {k: v.detach().clone() for k, v in clf.state_dict().items()}
        for epoch in range(self.config.train_epochs):
            clf.train()
            perm = torch.randperm(xt_t.shape[0])
            for s in range(0, xt_t.shape[0], batch_sz):
                idx = perm[s : s + batch_sz]
                opt.zero_grad()
                logits = clf(xt_t[idx])
                loss = loss_fn(logits, yt_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clf.parameters(), 3.0)
                opt.step()
            if (epoch + 1) % 10 == 0 or epoch == self.config.train_epochs - 1:
                clf.eval()
                with torch.no_grad():
                    dev_logits = clf(xd_t).numpy().astype(np.float64)
                auc = compute_roc_auroc(y_dev, dev_logits)
                if auc > best_dev_auc:
                    best_dev_auc = auc
                    best_state = {
                        k: v.detach().clone() for k, v in clf.state_dict().items()
                    }

        clf.load_state_dict(best_state)
        clf.eval()
        self._clf = clf

        # 3. Platt / temperature calibration + FPR-governed threshold strictly on DEV
        with torch.no_grad():
            dev_logits_t = clf(xd_t)
        a_param = torch.ones((), dtype=torch.float32, requires_grad=True)
        b_param = torch.zeros((), dtype=torch.float32, requires_grad=True)
        cal_opt = torch.optim.Adam([a_param, b_param], lr=0.05)
        y_dev_t = torch.tensor(y_dev, dtype=torch.float32)
        for _ in range(80):
            cal_opt.zero_grad()
            cal_loss = nn.functional.binary_cross_entropy_with_logits(
                a_param * dev_logits_t + b_param, y_dev_t
            )
            cal_loss.backward()  # type: ignore[no-untyped-call]
            cal_opt.step()
        self._platt_a = float(max(0.1, a_param.item()))
        self._platt_b = float(b_param.item())

        dev_probs = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    self._platt_a * dev_logits_t.numpy().astype(np.float64)
                    + self._platt_b,
                    -30.0,
                    30.0,
                )
            )
        )
        thr, dev_rec, dev_fpr = calibrate_threshold_for_max_fpr(
            y_dev, dev_probs, max_fpr=self.config.target_max_fpr
        )
        self._threshold = thr
        return {
            "dev_auroc": float(compute_roc_auroc(y_dev, dev_probs)),
            "dev_recall_at_target_fpr": float(dev_rec),
            "dev_fpr": float(dev_fpr),
            "decision_threshold": float(thr),
        }

    def predict_episode(
        self,
        forces: FloatArray,
        *,
        torques: FloatArray | None = None,
        positions: FloatArray | None = None,
        velocities: FloatArray | None = None,
        extra_channels: FloatArray | None = None,
    ) -> ContactDiagnosticResult:
        """Run calibrated jamming / slip inference on one unseen episode."""
        if self._clf is None or self._feat_mean is None or self._feat_std is None:
            raise RuntimeError("EnergyResidualContactEstimator must be fitted first")
        t0 = time.perf_counter()
        feat, r_power, slip_ratio, impedance = self.extract_episode_features(
            forces,
            torques=torques,
            positions=positions,
            velocities=velocities,
            extra_channels=extra_channels,
        )
        x_s = np.clip((feat - self._feat_mean) / self._feat_std, -8.0, 8.0)
        self._clf.eval()
        with torch.no_grad():
            logits = (
                self._clf(torch.tensor(x_s, dtype=torch.float32))
                .numpy()
                .astype(np.float64)
            )
        z = np.clip(self._platt_a * logits + self._platt_b, -30.0, 30.0)
        probs = (1.0 / (1.0 + np.exp(-z))).astype(np.float64)
        preds = (probs >= self._threshold).astype(np.bool_)
        return ContactDiagnosticResult(
            probabilities=probs,
            predictions=preds,
            power_balance_residual=r_power,
            friction_cone_ratio=slip_ratio,
            mechanical_impedance=impedance,
            decision_threshold=self._threshold,
            inference_ms=(time.perf_counter() - t0) * 1000.0,
        )


__all__ = (
    "CONTACT_DIAGNOSTICS_SCHEMA",
    "ContactDiagnosticResult",
    "ContactDiagnosticsConfig",
    "EnergyResidualContactEstimator",
    "calibrate_threshold_for_max_fpr",
    "compute_roc_auroc",
)
