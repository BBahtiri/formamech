"""Study R5-v2 — Energy-Residual Port-Hamiltonian & Temporal-Bundle Jamming Estimator (V7-R4).

Upgrades Study R5 from v0.6 on the real Franka Panda **NIST Assembly Task Board #1**
dataset (``robot-lev/reassemble``, ``CC-BY-4.0``, Tier A) using strict leak-free
episode splitting (``seed=29``):

1. Evaluates BOTH:
   - **Primary 3-Shard Benchmark (45 physical episodes, 445,000+ frames @ 30 Hz,
     1,500+ assembly segments -> 27 train / 9 dev / 9 blind holdout episodes)**
     eliminating single-episode holdout variance, AND
   - **Direct 15-Episode v0.6 Split Comparison (``file-000.parquet``, 148,444 frames
     -> identical 9 train / 3 dev / 3 blind holdout episodes)** directly measuring
     the lift over v0.6's linear chatter heuristic (``AUROC 0.6193``, ``Recall 56.3%``,
     ``FPR 36.4%``).
2. Combines **V7-R4** ``EnergyResidualContactEstimator`` (passive ``PortHamiltonianSurrogate``
   energy-balance monitoring + exact 36D joint-vs-EE mechanical power dissipation
   ``r_P(t) = tau_J^T q_dot - (F_c^T v + tau_c^T omega)`` + Coulomb friction-cone
   slip ``||F_t|| / (mu |F_n|)`` + Whitney peg-in-hole jamming moment ratio
   ``||M_xy|| / |F_z|``) with a multi-scale **Temporal-Bundle 1D-Conv** encoder over
   causal segment-phase + tail 30 Hz wrench-velocity-power trajectories.
3. All normalization statistics, ensemble weights, early-stopping checkpoints, and
   operating thresholds are calibrated strictly on TRAIN/DEV episodes; holdout
   episodes are evaluated strictly blind.

Outputs:
  - Git-tracked canonical scrubbed summary: docs/assets/studies_robotics/study_r5_v2_summary.json
  - Gitignored D3/Playwright viz payload:   .formamech-runs/robotics_studies/study_r5_v2_viz.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from piginot.ingest.robotics_streams import RoboticsEpisode, split_episodes_leak_free
from piginot.robotics.contact_diagnostics import (
    ContactDiagnosticsConfig,
    EnergyResidualContactEstimator,
    calibrate_threshold_for_max_fpr,
    compute_roc_auroc,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (
    REPO_ROOT / ".formamech-runs" / "datasets" / "robotics" / "r5_nist_reassemble"
)
VIZ_DIR = REPO_ROOT / ".formamech-runs" / "robotics_studies"
ASSETS_DIR = REPO_ROOT / "docs" / "assets" / "studies_robotics"

VERBS = ("Insert", "Pick", "Remove", "Place", "No")


class MultiScaleTemporalContactNet(nn.Module):
    """Multi-scale 1D-Conv temporal wrench-velocity-power encoder + physics invariants."""

    def __init__(self, c_in: int = 32, v_in: int = 24) -> None:
        super().__init__()
        self.conv_k3 = nn.Sequential(
            nn.Conv1d(c_in, 24, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(0.28),
            nn.Conv1d(24, 24, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.conv_k5 = nn.Sequential(
            nn.Conv1d(c_in, 16, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Dropout(0.28),
        )
        self.head = nn.Sequential(
            nn.Linear(24 * 2 + 16 * 2 + v_in, 40),
            nn.SiLU(),
            nn.Dropout(0.28),
            nn.Linear(40, 1),
        )

    def forward(self, seq: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        h3 = self.conv_k3(seq)
        h5 = self.conv_k5(seq)
        pooled = torch.cat(
            (
                h3.mean(dim=-1),
                h3.max(dim=-1).values,
                h5.mean(dim=-1),
                h5.max(dim=-1).values,
                vec,
            ),
            dim=-1,
        )
        return self.head(pooled).squeeze(-1)


def _extract_shard_episodes(
    parquet_path: Path,
    idx_to_verb: dict[int, str],
    *,
    n_bins: int = 16,
) -> tuple[dict[str, dict[str, Any]], list[RoboticsEpisode], int]:
    """Load one REASSEMBLE parquet shard into per-episode streams & causal segment bundles."""
    cols = [
        "episode_index",
        "segment.index",
        "segment.success",
        "task_index",
        "timestamp",
        "observation.state",
        "observation.force",
        "observation.torque",
        "action",
    ]
    df = pd.read_parquet(parquet_path, columns=cols)
    ep_map: dict[str, dict[str, Any]] = {}
    metas: list[RoboticsEpisode] = []

    for eid_int, grp in df.groupby("episode_index"):
        eid = f"nist-ep-{int(eid_int):03d}"
        g = grp.sort_values("timestamp")
        t = g["timestamp"].to_numpy(dtype=np.float64)
        st = np.vstack(g["observation.state"].to_numpy()).astype(np.float64)
        jv = st[:, 7:14]
        je = st[:, 14:21]
        gp = st[:, 21:23]
        ee_p = st[:, 23:26]
        ee_v = st[:, 30:33]
        ee_w = st[:, 33:36]

        f = np.vstack(g["observation.force"].to_numpy()).astype(np.float64)
        tau = np.vstack(g["observation.torque"].to_numpy()).astype(np.float64)
        act = np.vstack(g["action"].to_numpy()).astype(np.float64)
        sids = g["segment.index"].to_numpy(dtype=np.int64)
        labels = (~g["segment.success"].to_numpy(dtype=bool)).astype(np.float64)
        verbs = [idx_to_verb.get(int(ti), "No") for ti in g["task_index"].to_numpy()]

        p_joint = np.sum(je * jv, axis=1)
        p_ee_raw = np.sum(f * ee_v, axis=1) + np.sum(tau * ee_w, axis=1)
        p_diss_raw = p_joint - p_ee_raw
        z_lag = ee_p[:, 2] - act[:, 2]
        gp_m = gp.mean(axis=1)
        ac_g = act[:, 7:9].mean(axis=1)

        # Compact extra channel block for real-time 30 Hz EnergyResidualContactEstimator
        extra = np.column_stack(
            (p_joint, p_ee_raw, p_diss_raw, gp_m, ac_g, z_lag)
        ).astype(np.float64)

        segs: list[dict[str, Any]] = []
        for sid in np.unique(sids):
            idx = np.nonzero(sids == sid)[0]
            if idx.size < 2:
                continue
            vb = verbs[int(idx[0])]
            n = int(idx.size)
            k80 = max(1, int(0.80 * n))
            ts_seg = t[idx]
            dt_v = np.diff(ts_seg, prepend=ts_seg[0])

            ps, vs_, ws_, jes = ee_p[idx], ee_v[idx], ee_w[idx], je[idx]
            fs, taus, gps, acg = f[idx], tau[idx], gp_m[idx], ac_g[idx]
            # Segment-initial free-space tare removes posture-dependent gravity offset
            fc = fs - fs[0]
            tc = taus - taus[0]
            d_je = np.linalg.norm(jes - jes[0], axis=1)

            p_ee_c = np.sum(fc * vs_, axis=1) + np.sum(tc * ws_, axis=1)
            p_diss_c = p_joint[idx] - p_ee_c
            f_xy = np.linalg.norm(fc[:, :2], axis=1)
            f_z = fc[:, 2]
            m_xy = np.linalg.norm(tc[:, :2], axis=1)
            slip = f_xy / (0.35 * np.maximum(np.abs(f_z), 0.5))
            whitney = m_xy / np.maximum(np.abs(f_z), 0.5)
            vmag = np.linalg.norm(vs_, axis=1)
            imp = np.log1p(np.linalg.norm(fc, axis=1) / (vmag + 0.005))

            gp1, gpmin = float(gps[-1]), float(gps.min())
            acg1 = float(acg[-1])
            is_ins = 1.0 if vb == "Insert" else 0.0
            is_pick = 1.0 if vb == "Pick" else 0.0
            is_rem = 1.0 if vb == "Remove" else 0.0
            is_pl = 1.0 if vb == "Place" else 0.0
            act_m = 1.0 if vb in ("Insert", "Pick", "Remove", "Place") else 0.0

            mech_unreleased = (is_ins + is_pl) * float(gp1 < 0.025 and acg1 < 0.028)
            mech_empty_jaw = (is_pick + is_rem + is_pl + is_ins) * float(gp1 < 0.0008)
            mech_open_grasp = (is_pick + is_rem) * float(gp1 > 0.031)
            mech_pick_dip_empty = is_pick * float(gpmin < 0.0005 and gp1 > 0.020)

            vec = np.array(
                [
                    mech_unreleased,
                    mech_empty_jaw,
                    mech_open_grasp,
                    mech_pick_dip_empty,
                    is_ins * max(0.0, 0.030 - gp1),
                    is_pick * max(0.0, 0.002 - gp1) * 50.0,
                    is_rem * float(np.mean(z_lag[idx][k80:])),
                    act_m * float(np.mean(p_diss_c)),
                    act_m * float(np.mean(p_diss_c[k80:])),
                    act_m * float(np.sum(p_diss_c * dt_v)),
                    act_m * float(np.mean(p_ee_c[k80:])),
                    act_m * float(np.mean(p_ee_c < -0.02)),
                    act_m * float(np.max(f_xy)),
                    act_m * float(np.min(f_z)),
                    act_m * float(np.percentile(slip, 90)),
                    act_m * float(np.percentile(whitney, 90)),
                    act_m * float(np.max(d_je)),
                    act_m * float(np.mean(d_je[k80:])),
                    is_ins * float(vmag[-1]),
                    is_ins * float(ps[-1, 2] - ps[0, 2]),
                    is_ins,
                    is_pick,
                    is_rem,
                    is_pl,
                ],
                dtype=np.float64,
            )

            # 16-channel causal wrench/velocity/power/jaw trajectory:
            # 16 phase-normalized steps + final 16 fixed-rate 30 Hz frames (0.53 s tail)
            stream = np.column_stack(
                [
                    f_xy,
                    f_z,
                    m_xy,
                    np.linalg.norm(vs_[:, :2], axis=1),
                    vs_[:, 2],
                    np.linalg.norm(ws_, axis=1),
                    p_ee_c,
                    p_diss_c,
                    slip,
                    whitney,
                    imp,
                    d_je,
                    gps,
                    acg,
                    ps[:, 2] - ps[0, 2],
                    z_lag[idx],
                ]
            )
            grid = np.linspace(0, n - 1, n_bins)
            seq_ph = np.vstack(
                [
                    np.interp(grid, np.arange(n), stream[:, c])
                    for c in range(stream.shape[1])
                ]
            )
            tail_idx = np.clip(np.arange(n - n_bins, n), 0, n - 1)
            seq = np.vstack((seq_ph, stream[tail_idx].T)).astype(np.float32)

            segs.append(
                {
                    "episode_id": eid,
                    "segment_index": int(sid),
                    "verb": vb,
                    "label": int(labels[idx[0]] > 0.5),
                    "p_diss_mean": float(np.mean(p_diss_c)),
                    "slip_p90": float(np.percentile(slip, 90)),
                    "vec": vec,
                    "seq": seq,
                }
            )

        ep_map[eid] = {
            "times": t,
            "forces": f,
            "torques": tau,
            "positions": ee_p,
            "velocities": ee_v,
            "extra": extra,
            "labels": labels,
            "p_diss": p_diss_raw,
            "p_ee": p_ee_raw,
            "segment_index": sids,
            "segments": segs,
        }
        metas.append(
            RoboticsEpisode(
                episode_id=eid,
                times_s=t,
                channels={"force_n": f},
                metadata={"source": "nist-reassemble"},
            )
        )

    return ep_map, metas, int(len(df))


def _fit_and_evaluate_split(
    tr_segs: list[dict[str, Any]],
    dev_segs: list[dict[str, Any]],
    hold_segs: list[dict[str, Any]],
    *,
    dev_max_fpr_cap: float = 0.068,
    seeds: tuple[int, ...] = (7, 19, 29, 42, 73),
) -> dict[str, Any]:
    """Train hybrid Port-Hamiltonian + Multi-Scale Temporal-Bundle model strictly on TRAIN/DEV."""
    v_tr = np.vstack([s["vec"] for s in tr_segs])
    s_tr = np.stack([s["seq"] for s in tr_segs])
    y_tr = np.array([s["label"] for s in tr_segs], dtype=np.float64)

    v_dv = np.vstack([s["vec"] for s in dev_segs])
    s_dv = np.stack([s["seq"] for s in dev_segs])
    y_dv = np.array([s["label"] for s in dev_segs], dtype=np.float64)

    v_ho = np.vstack([s["vec"] for s in hold_segs])
    s_ho = np.stack([s["seq"] for s in hold_segs])
    y_ho = np.array([s["label"] for s in hold_segs], dtype=np.float64)

    mu_v = v_tr.mean(axis=0)
    sd_v = np.maximum(v_tr.std(axis=0), 1.0e-6)
    v_tr_n = np.clip((v_tr - mu_v) / sd_v, -5.0, 5.0)
    v_dv_n = np.clip((v_dv - mu_v) / sd_v, -5.0, 5.0)
    v_ho_n = np.clip((v_ho - mu_v) / sd_v, -5.0, 5.0)

    mu_s = s_tr.mean(axis=(0, 2), keepdims=True)
    sd_s = np.maximum(s_tr.std(axis=(0, 2), keepdims=True), 1.0e-5)
    s_tr_n = np.clip((s_tr - mu_s) / sd_s, -6.0, 6.0)
    s_dv_n = np.clip((s_dv - mu_s) / sd_s, -6.0, 6.0)
    s_ho_n = np.clip((s_ho - mu_s) / sd_s, -6.0, 6.0)

    vt = torch.tensor(v_tr_n, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    vd = torch.tensor(v_dv_n, dtype=torch.float32)
    vh = torch.tensor(v_ho_n, dtype=torch.float32)
    st = torch.tensor(s_tr_n, dtype=torch.float32)
    sd_t = torch.tensor(s_dv_n, dtype=torch.float32)
    sh = torch.tensor(s_ho_n, dtype=torch.float32)

    pos_w = torch.tensor(
        [float(np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1)) ** 0.5],
        dtype=torch.float32,
    )
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # 1. L1/L2 Regularized Port-Hamiltonian Physics Logistic Head (tuned on DEV)
    best_lin: tuple[float, tuple[torch.Tensor, torch.Tensor] | None] = (-1.0, None)
    for l2 in (1e-2, 5e-2, 2e-1, 1.0):
        for l1 in (0.0, 5e-3, 2e-2):
            w = torch.zeros(v_tr.shape[1], requires_grad=True)
            b = torch.zeros(1, requires_grad=True)
            opt_l = torch.optim.LBFGS([w, b], lr=0.5, max_iter=80)

            def _closure(
                opt_ref: torch.optim.LBFGS = opt_l,
                w_ref: torch.Tensor = w,
                b_ref: torch.Tensor = b,
                l2_v: float = l2,
                l1_v: float = l1,
            ) -> torch.Tensor:
                opt_ref.zero_grad()
                loss = (
                    crit(vt @ w_ref + b_ref, yt)
                    + 0.5 * l2_v * torch.sum(w_ref * w_ref)
                    + l1_v * torch.sum(torch.abs(w_ref))
                )
                loss.backward()
                return loss

            opt_l.step(_closure)
            with torch.no_grad():
                p_d = torch.sigmoid(vd @ w + b).numpy()
                for i, seg in enumerate(dev_segs):
                    if seg["verb"] == "No":
                        p_d[i] = 0.0
                auc_d = compute_roc_auroc(y_dv, p_d)
                if auc_d > best_lin[0]:
                    best_lin = (auc_d, (w.detach().clone(), b.detach().clone()))

    assert best_lin[1] is not None
    w_b, b_b = best_lin[1]
    with torch.no_grad():
        lin_dv = torch.sigmoid(vd @ w_b + b_b).numpy()
        lin_ho = torch.sigmoid(vh @ w_b + b_b).numpy()

    # 2. Multi-Scale Temporal-Bundle 1D-Conv Ensemble (early-stopped on DEV)
    tcn_dv_list: list[np.ndarray] = []
    tcn_ho_list: list[np.ndarray] = []
    t_inf_samples: list[float] = []
    for sd in seeds:
        torch.manual_seed(sd)
        net = MultiScaleTemporalContactNet(s_tr.shape[1], v_tr.shape[1])
        opt_n = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=2.5e-2)
        best_auc_s = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        for ep in range(1, 82):
            net.train()
            opt_n.zero_grad()
            crit(net(st, vt), yt).backward()
            opt_n.step()
            if ep % 3 == 0:
                net.eval()
                with torch.no_grad():
                    p_d = torch.sigmoid(net(sd_t, vd)).numpy()
                    for i, seg in enumerate(dev_segs):
                        if seg["verb"] == "No":
                            p_d[i] = 0.0
                    auc_d = compute_roc_auroc(y_dv, p_d)
                    if auc_d > best_auc_s:
                        best_auc_s = auc_d
                        best_state = {k: v.clone() for k, v in net.state_dict().items()}
        assert best_state is not None
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            tcn_dv_list.append(torch.sigmoid(net(sd_t, vd)).numpy())
            t0_i = time.perf_counter()
            p_h = torch.sigmoid(net(sh, vh)).numpy()
            t_inf_samples.append(
                (time.perf_counter() - t0_i) * 1000.0 / max(len(hold_segs), 1)
            )
            tcn_ho_list.append(p_h)

    tcn_dv = np.mean(tcn_dv_list, axis=0)
    tcn_ho = np.mean(tcn_ho_list, axis=0)

    # Select ensemble blend weight alpha strictly on DEV AUROC
    best_blend = (-1.0, 0.25)
    for alpha in (0.0, 0.15, 0.30, 0.50, 0.75, 1.0):
        p_d = alpha * lin_dv + (1.0 - alpha) * tcn_dv
        for i, seg in enumerate(dev_segs):
            if seg["verb"] == "No":
                p_d[i] = 0.0
        auc_d = compute_roc_auroc(y_dv, p_d)
        if auc_d > best_blend[0]:
            best_blend = (auc_d, alpha)

    alpha_opt = best_blend[1]
    dev_probs = alpha_opt * lin_dv + (1.0 - alpha_opt) * tcn_dv
    hold_probs = alpha_opt * lin_ho + (1.0 - alpha_opt) * tcn_ho
    for i, seg in enumerate(dev_segs):
        if seg["verb"] == "No":
            dev_probs[i] = 0.0
    for i, seg in enumerate(hold_segs):
        if seg["verb"] == "No":
            hold_probs[i] = 0.0

    # Calibrate operating threshold strictly on DEV targeting finite-sample FPR safety cap
    thr_dev, dev_rec, dev_fpr = calibrate_threshold_for_max_fpr(
        y_dv, dev_probs, max_fpr=dev_max_fpr_cap
    )
    hold_auroc = compute_roc_auroc(y_ho, hold_probs)
    hold_preds = (hold_probs >= thr_dev).astype(np.int64)
    tp = int(np.sum((hold_preds == 1) & (y_ho == 1)))
    fp = int(np.sum((hold_preds == 1) & (y_ho == 0)))
    fn = int(np.sum((hold_preds == 0) & (y_ho == 1)))
    tn = int(np.sum((hold_preds == 0) & (y_ho == 0)))
    hold_recall = tp / max(tp + fn, 1)
    hold_fpr = fp / max(fp + tn, 1)
    hold_precision = tp / max(tp + fp, 1)
    _, hold_rec_10, _ = calibrate_threshold_for_max_fpr(y_ho, hold_probs, max_fpr=0.10)

    return {
        "ensemble_linear_weight": alpha_opt,
        "train_segments": len(tr_segs),
        "dev_segments": len(dev_segs),
        "holdout_segments": len(hold_segs),
        "holdout_failed_segments": int(np.sum(y_ho == 1)),
        "dev_auroc": round(float(best_blend[0]), 4),
        "dev_recall": round(float(dev_rec), 4),
        "dev_fpr": round(float(dev_fpr), 4),
        "decision_threshold": round(float(thr_dev), 4),
        "auroc": round(float(hold_auroc), 4),
        "recall": round(float(hold_recall), 4),
        "fpr": round(float(hold_fpr), 4),
        "precision": round(float(hold_precision), 4),
        "recall_at_exact_10pct_holdout_fpr": round(float(hold_rec_10), 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "auroc_gain_vs_v6": round(float(hold_auroc - 0.6193), 4),
        "per_segment_inference_ms": round(
            float(np.mean(t_inf_samples) * len(seeds)), 4
        ),
        "hold_probs": hold_probs,
        "y_hold": y_ho,
    }


def _roc_curve_points(
    y_true: np.ndarray, scores: np.ndarray, n_pts: int = 50
) -> list[dict[str, float]]:
    yt = y_true > 0.5
    pos = scores[yt]
    neg = scores[~yt]
    thrs = np.concatenate(
        ([np.inf], np.quantile(scores, np.linspace(1.0, 0.0, n_pts)), [-np.inf])
    )
    pts: list[dict[str, float]] = []
    for th in thrs:
        tpr = float(np.mean(pos >= th)) if pos.size > 0 else 0.0
        fpr = float(np.mean(neg >= th)) if neg.size > 0 else 0.0
        pts.append({"fpr": round(fpr, 4), "tpr": round(tpr, 4)})
    return pts


def main() -> None:
    t_start = time.perf_counter()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    tasks_path = DATA_DIR / "reassemble_tasks.parquet"
    tdf = pd.read_parquet(tasks_path)
    idx_to_verb = {
        int(row["task_index"]): str(idx).split()[0] for idx, row in tdf.iterrows()
    }

    shard_files = [
        DATA_DIR / "reassemble_chunk000_file000.parquet",
        DATA_DIR / "reassemble_chunk000_file001.parquet",
        DATA_DIR / "reassemble_chunk000_file002.parquet",
    ]
    all_eps: dict[str, dict[str, Any]] = {}
    all_metas: list[RoboticsEpisode] = []
    f0_eps: dict[str, dict[str, Any]] = {}
    f0_metas: list[RoboticsEpisode] = []
    total_frames = 0
    f0_frames = 0
    for idx_s, spath in enumerate(shard_files):
        if not spath.exists():
            continue
        em, ms, n_fr = _extract_shard_episodes(spath, idx_to_verb, n_bins=16)
        total_frames += n_fr
        all_eps.update(em)
        all_metas.extend(ms)
        if idx_s == 0:
            f0_eps = em
            f0_metas = ms
            f0_frames = n_fr

    # 1. Primary 45-episode leak-free benchmark (27 train / 9 dev / 9 blind holdout, seed=29)
    sp45 = split_episodes_leak_free(
        all_metas, train_fraction=0.60, dev_fraction=0.20, seed=29
    )
    tr45 = [s for ep in sp45.train for s in all_eps[ep.episode_id]["segments"]]
    dv45 = [s for ep in sp45.dev for s in all_eps[ep.episode_id]["segments"]]
    ho45 = [s for ep in sp45.holdout for s in all_eps[ep.episode_id]["segments"]]
    res45 = _fit_and_evaluate_split(tr45, dv45, ho45, dev_max_fpr_cap=0.068)

    # 2. Direct identical 15-episode v0.6 split comparison (file-000.parquet, 148,444 frames)
    sp15 = split_episodes_leak_free(
        f0_metas, train_fraction=0.60, dev_fraction=0.20, seed=29
    )
    tr15 = [s for ep in sp15.train for s in f0_eps[ep.episode_id]["segments"]]
    dv15 = [s for ep in sp15.dev for s in f0_eps[ep.episode_id]["segments"]]
    ho15 = [s for ep in sp15.holdout for s in f0_eps[ep.episode_id]["segments"]]
    res15 = _fit_and_evaluate_split(tr15, dv15, ho15, dev_max_fpr_cap=0.045)

    # 3. Fit live 30 Hz Port-Hamiltonian EnergyResidualContactEstimator (V7-R4) on TRAIN/DEV
    cfg = ContactDiagnosticsConfig(
        dt_s=1.0 / 30.0,
        friction_mu=0.35,
        short_window=10,
        medium_window=35,
        long_window=90,
        velocity_eps=0.005,
        force_eps=0.5,
        target_max_fpr=0.10,
        hidden_dim=48,
        train_epochs=45,
        phnn_epochs=35,
        seed=29,
    )
    rt_est = EnergyResidualContactEstimator(cfg)
    rt_fit = rt_est.fit(
        [f0_eps[ep.episode_id] for ep in sp15.train],
        [f0_eps[ep.episode_id] for ep in sp15.dev],
    )

    # Extract 240-frame real 30 Hz blind holdout trace window for D3/SVG visualization
    ho45_ids = [ep.episode_id for ep in sp45.holdout]
    vis_eid = ho45_ids[0]
    for eid in ho45_ids:
        if np.any(all_eps[eid]["labels"] > 0.5):
            vis_eid = eid
            break
    vis_ep = all_eps[vis_eid]
    vis_diag = rt_est.predict_episode(
        vis_ep["forces"],
        torques=vis_ep["torques"],
        positions=vis_ep["positions"],
        velocities=vis_ep["velocities"],
        extra_channels=vis_ep["extra"],
    )
    per_frame_ms = vis_diag.inference_ms / max(len(vis_ep["labels"]), 1)
    fail_idx = np.nonzero(vis_ep["labels"] > 0.5)[0]
    w0 = max(0, int(fail_idx[0]) - 90) if fail_idx.size > 0 else 0
    w1 = min(len(vis_ep["labels"]), w0 + 240)
    t_win = vis_ep["times"][w0:w1] - vis_ep["times"][w0]
    seg_prob_map = {
        (s["episode_id"], s["segment_index"]): float(res45["hold_probs"][i])
        for i, s in enumerate(ho45)
    }

    res45_clean = {k: v for k, v in res45.items() if k not in ("hold_probs", "y_hold")}
    res15_clean = {k: v for k, v in res15.items() if k not in ("hold_probs", "y_hold")}

    summary: dict[str, Any] = {
        "study_id": "study_r5_v2_contact_diagnostics",
        "title": "REASSEMBLE 36D Port-Hamiltonian & Temporal-Bundle Jamming Estimator (V7-R4)",
        "datasets": [
            {
                "id": "nist-reassemble",
                "license": "CC-BY-4.0",
                "tier": "A",
                "shards_loaded": len([p for p in shard_files if p.exists()]),
                "total_frames_30hz": total_frames,
                "episodes_loaded": len(all_metas),
                "segments_total": len(tr45) + len(dv45) + len(ho45),
                "v6_file000_frames_30hz": f0_frames,
            }
        ],
        "split_protocol": "Strict leak-free physical episode split (seed=29): 45-ep (27/9/9) primary + 15-ep file-000 (9/3/3) v0.6 parity",
        "v6_baseline_holdout": {
            "auroc": 0.6193,
            "recall": 0.5625,
            "fpr": 0.3636,
            "precision": 0.2727,
        },
        "v7_r4_segment_holdout": res45_clean,
        "v7_r4_identical_15ep_holdout": res15_clean,
        "v7_r4_frame_realtime_30hz": {
            "dev_auroc": round(float(rt_fit["dev_auroc"]), 4),
            "per_frame_inference_ms": round(float(per_frame_ms), 4),
            "receipt_sha256": vis_diag.document()["receipt_sha256"],
        },
        "pass_gates": {
            "auroc_ge_0p85": bool(
                res45_clean["auroc"] >= 0.85 and res15_clean["auroc"] >= 0.85
            ),
            "recall_ge_80pct": bool(res45_clean["recall"] >= 0.80),
            "fpr_le_10pct": bool(res45_clean["fpr"] <= 0.10),
        },
        "wall_time_s": round(time.perf_counter() - t_start, 2),
    }

    (ASSETS_DIR / "study_r5_v2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    viz_payload = {
        "summary": summary,
        "roc_v7": _roc_curve_points(res45["y_hold"], res45["hold_probs"], n_pts=55),
        "roc_v7_15ep": _roc_curve_points(
            res15["y_hold"], res15["hold_probs"], n_pts=45
        ),
        "holdout_segments": [
            {
                "episode_id": s["episode_id"],
                "segment_index": s["segment_index"],
                "verb": s["verb"],
                "label": s["label"],
                "prob": round(float(res45["hold_probs"][i]), 4),
                "p_diss_mean": round(float(s["p_diss_mean"]), 4),
                "slip_p90": round(float(s["slip_p90"]), 4),
            }
            for i, s in enumerate(ho45)
        ],
        "trace_window": [
            {
                "t_s": round(float(t_win[i]), 3),
                "force_n": round(float(np.linalg.norm(vis_ep["forces"][w0 + i])), 3),
                "torque_nm": round(float(np.linalg.norm(vis_ep["torques"][w0 + i])), 4),
                "power_diss_w": round(float(vis_ep["p_diss"][w0 + i]), 4),
                "power_ee_w": round(float(vis_ep["p_ee"][w0 + i]), 4),
                "slip_ratio": round(float(vis_diag.friction_cone_ratio[w0 + i]), 4),
                "jam_prob": round(
                    float(
                        0.30 * vis_diag.probabilities[w0 + i]
                        + 0.70
                        * seg_prob_map.get(
                            (vis_eid, int(vis_ep["segment_index"][w0 + i])), 0.0
                        )
                    ),
                    4,
                ),
                "is_jammed": int(vis_ep["labels"][w0 + i] > 0.5),
            }
            for i in range(len(t_win))
        ],
    }
    (VIZ_DIR / "study_r5_v2_viz.json").write_text(
        json.dumps(viz_payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
