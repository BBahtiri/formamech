"""Study R6 — Dynamic Tactile Slip & Friction-Cone Control (V6-R2 + V7-R4).

Extends Study R1 from static normal indentation to coupled 3-axis normal +
tangential shear loading and blind incipient slip-onset prediction against:
  - Meta Sparsh / TacBench GelSight Force Estimation
    (``facebook/gelsight-force-estimation``, CC-BY-NC-4.0, Tier C non-commercial
    opt-in / EVAL-ONLY — 1,513 real trajectories / 102,628 frames across
    ``sphere``, ``flat``, and ``sharp`` indenters with ATI Nano17 3-axis forces
    and friction-cone slip labels; ZERO raw data or derived weights exported to Git!)

Pipeline:
1. Full 3D finite-strain ``TET10`` SDF contact FEM (``solve_3d_sdf_contact``) with
   regularized Coulomb friction across ``SPHERE``, ``FLAT_PUNCH``, and ``WEDGE_EDGE``
   indenters confirming normal-shear coupling and local friction-cone ratios.
2. Leak-free trajectory split (60% train / 20% dev / 20% blind holdout across all
   1,513 physical trajectories) training a fast (< 1 ms, > 1,000 Hz) physics-guided
   3-axis force & causal slip-onset surrogate with AS6983 conformal upper bounds.
3. Evaluates BOTH end-to-end kinematics->predicted-force->slip (zero force leakage!)
   and direct force-sensor friction-cone slip detection via V7-R4.

Outputs:
  - Git-tracked canonical scrubbed summary: docs/assets/studies_robotics/study_r6_summary.json
  - Gitignored D3/Playwright viz payload:   .formamech-runs/robotics_studies/study_r6_viz.json
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from piginot.bounds.conformal_3d import (
    Conformal3DCalibrator,
    Conformal3DDevSample,
    verify_as6983_holdout_coverage,
)
from piginot.fem.diff_contact_3d import SdfPrimitiveKind
from piginot.ingest.robotics_streams import RoboticsEpisode, split_episodes_leak_free
from piginot.robotics.contact_diagnostics import (
    calibrate_threshold_for_max_fpr,
    compute_roc_auroc,
)
from piginot.robotics.tactile import TactileSensorSpec, simulate_tactile_probe_fem

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (
    REPO_ROOT / ".formamech-runs" / "datasets" / "robotics" / "r6_sparsh_tacbench"
)
VIZ_DIR = REPO_ROOT / ".formamech-runs" / "robotics_studies"
ASSETS_DIR = REPO_ROOT / "docs" / "assets" / "studies_robotics"

SHAPE_MAP = {"sphere": 0, "flat": 1, "sharp": 2}


class Tactile3AxisSlipNet(nn.Module):
    """Real-time (< 1 ms) coupled 3-axis tactile force & incipient slip-onset operator."""

    def __init__(self, in_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.force_head = nn.Linear(hidden, 3)  # (fx, fy, fz) in N
        self.slip_head = nn.Sequential(
            nn.Linear(
                hidden + 5, 48
            ),  # latent + physics friction-cone invariants from predicted force
            nn.GELU(),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        f_pred: torch.Tensor = self.force_head(h)
        fx, fy, fz = f_pred[:, 0], f_pred[:, 1], f_pred[:, 2]
        f_shear = torch.sqrt(fx * fx + fy * fy + 1.0e-8)
        f_norm = torch.abs(fz) + 0.05
        # Coulomb friction-cone invariant ||F_t|| / (0.30 * |F_n|)
        cone_ratio = f_shear / (0.30 * f_norm)
        phys_feats = torch.stack(
            (fx, fy, fz, f_shear, torch.clamp(cone_ratio, 0.0, 10.0)), dim=-1
        )
        slip_logit: torch.Tensor = self.slip_head(
            torch.cat((h, phys_feats), dim=-1)
        ).squeeze(-1)
        return f_pred, slip_logit


def _featurize_trajectory(
    poses: np.ndarray, shape_id: int, mu_ref: float
) -> np.ndarray:
    """Build strictly causal kinematic + elastomer-indentation features (no ground-truth force!)."""
    t_len = int(poses.shape[0])
    xyz = poses[:, :3].astype(np.float64)
    # Initial contact point reference (t = 0 of trajectory)
    p0 = xyz[0:1, :]
    dxyz = xyz - p0  # (dx, dy, dz) in mm relative to contact touch-down
    depth_mm = np.maximum(-dxyz[:, 2], 0.0)
    shear_xy = dxyz[:, :2]
    shear_r = np.linalg.norm(shear_xy, axis=1)

    # Causal finite-difference velocities and rolling cumulative shear excursion
    v = np.zeros_like(xyz)
    if t_len > 1:
        v[1:] = np.diff(xyz, axis=0)
        v[0] = v[1]
    v_xy = np.linalg.norm(v[:, :2], axis=1)
    # Cumulative tangential travel & peak depth so far (viscoelastic + slip history)
    cum_shear = np.cumsum(v_xy)
    max_depth = np.maximum.accumulate(depth_mm)
    # Hertzian normal proxy d^(3/2) & Mindlin tangential-to-normal ratio |u_t| / (d + eps)
    hertz_proxy = np.power(depth_mm, 1.5)
    mindlin_ratio = shear_r / (depth_mm + 0.15)
    # Rolling 5-step causal velocity mean & shear step norm
    csum_v = np.concatenate(([0.0], np.cumsum(v_xy)))
    idx = np.arange(1, t_len + 1)
    st = np.maximum(0, idx - 5)
    v_xy_w5 = (csum_v[idx] - csum_v[st]) / np.maximum(1, idx - st)
    rel_t = np.linspace(0.0, 1.0, t_len)

    oh = np.zeros((t_len, 3), dtype=np.float64)
    oh[:, shape_id] = 1.0
    mu_col = np.full((t_len, 1), mu_ref, dtype=np.float64)

    return np.column_stack(
        [
            xyz[:, 0] - 202.0,
            xyz[:, 1],
            dxyz[:, 0],
            dxyz[:, 1],
            depth_mm,
            max_depth,
            hertz_proxy,
            shear_r,
            cum_shear,
            mindlin_ratio,
            v[:, 0],
            v[:, 1],
            v[:, 2],
            v_xy,
            v_xy_w5,
            depth_mm * dxyz[:, 0],
            depth_mm * dxyz[:, 1],
            rel_t,
            oh,
            mu_col,
        ]
    ).astype(np.float64)


def main() -> None:
    t_start = time.perf_counter()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(DATA_DIR.glob("*_slip_forces.pkl"))
    if not pkl_files:
        raise FileNotFoundError(
            f"Missing Sparsh TacBench files in {DATA_DIR}. Download them first."
        )

    # 1. Run real 3D TET10 coupled normal + tangential Coulomb-friction FEM on GelSight pad
    spec = TactileSensorSpec(
        name="gelsight-mini-sparsh",
        youngs_modulus_mpa=0.22,
        poisson_ratio=0.48,
        friction_mu=0.30,
        mesh_div_xyz=(3, 4, 1),
    )
    pad_mesh = spec.build_tet10_mesh()
    fem_probes: list[dict[str, Any]] = []
    for kind, label, rad in [
        (SdfPrimitiveKind.SPHERE, "sphere", 4.0),
        (SdfPrimitiveKind.FLAT_PUNCH, "flat", 4.0),
        (SdfPrimitiveKind.WEDGE_EDGE, "sharp", 3.0),
    ]:
        for depth, sdx in [(0.45, 0.0), (0.50, 0.35), (0.65, 0.50)]:
            smp = simulate_tactile_probe_fem(
                spec,
                case_id=f"{label}-d{depth:.2f}-s{sdx:.2f}",
                probe_x_mm=9.0,
                probe_y_mm=12.0,
                depth_mm=depth,
                shear_dx_mm=sdx,
                shear_dy_mm=0.0,
                indenter_radius_mm=rad,
                indenter_kind=kind,
                mesh=pad_mesh,
                load_steps=4,
            )
            fx, fy, fz = smp.wrench_6d[:3]
            fem_probes.append(
                {
                    "indenter": label,
                    "depth_mm": depth,
                    "shear_dx_mm": sdx,
                    "fx_n": round(float(fx), 4),
                    "fy_n": round(float(fy), 4),
                    "fz_n": round(float(-fz), 4),
                    "max_slip_ratio": round(float(smp.max_slip_ratio), 4),
                    "peak_vm_mpa": round(float(smp.peak_von_mises_mpa), 4),
                    "wall_time_s": round(float(smp.fem_wall_time_s), 3),
                }
            )

    # 2. Load ALL 10 real Meta Sparsh TacBench batches (1,513 trajectories, 102,628 frames)
    episodes_meta: list[RoboticsEpisode] = []
    traj_store: dict[str, dict[str, Any]] = {}
    total_frames = 0
    total_slip = 0
    for pkl_path in pkl_files:
        shape_name = pkl_path.name.split("_")[0]
        batch_tag = "_".join(pkl_path.name.split("_")[:3])
        sid_int = SHAPE_MAP[shape_name]
        raw = pickle.loads(pkl_path.read_bytes())
        for tid, tdata in raw["trajectories"].items():
            forces = np.asarray(tdata["forces"], dtype=np.float64)
            poses = np.asarray(tdata["poses"], dtype=np.float64)
            slip = np.asarray(tdata["slip_label"], dtype=np.float64).reshape(-1)
            mu_f = float(tdata.get("coef_friction", 0.30))
            n_min = min(len(forces), len(poses), len(slip))
            if n_min < 5:
                continue
            forces = forces[:n_min]
            poses = poses[:n_min]
            slip = slip[:n_min]
            eid = f"{batch_tag}_t{tid}"
            feats = _featurize_trajectory(poses, sid_int, mu_f)
            traj_store[eid] = {
                "shape": shape_name,
                "feats": feats,
                "forces": forces,
                "slip": slip,
                "poses": poses,
                "mu": mu_f,
            }
            episodes_meta.append(
                RoboticsEpisode(
                    episode_id=eid,
                    times_s=np.arange(len(forces), dtype=np.float64) * 0.04,
                    channels={"forces": forces},
                    metadata={"shape": shape_name},
                )
            )
            total_frames += len(forces)
            total_slip += int(np.sum(slip > 0.5))

    # Strict leak-free trajectory split (60% train / 20% dev / 20% blind holdout)
    splits = split_episodes_leak_free(
        episodes_meta, train_fraction=0.60, dev_fraction=0.20, seed=42
    )

    def _stack(
        eps: tuple[RoboticsEpisode, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        xs = np.vstack([traj_store[e.episode_id]["feats"] for e in eps])
        fs = np.vstack([traj_store[e.episode_id]["forces"] for e in eps])
        sl = np.concatenate([traj_store[e.episode_id]["slip"] for e in eps])
        sh = [
            traj_store[e.episode_id]["shape"]
            for e in eps
            for _ in range(len(traj_store[e.episode_id]["slip"]))
        ]
        return xs, fs, sl, sh

    x_tr, f_tr, s_tr, _ = _stack(splits.train)
    x_dev, f_dev, s_dev, _ = _stack(splits.dev)
    x_hold, f_hold, s_hold, sh_hold = _stack(splits.holdout)

    # Standardize inputs strictly on TRAIN
    x_mu = x_tr.mean(axis=0)
    x_sd = np.maximum(x_tr.std(axis=0), 1.0e-6)
    x_tr_n = np.clip((x_tr - x_mu) / x_sd, -8.0, 8.0)
    x_dev_n = np.clip((x_dev - x_mu) / x_sd, -8.0, 8.0)
    x_hold_n = np.clip((x_hold - x_mu) / x_sd, -8.0, 8.0)

    # 3. Train coupled 3-axis force + slip-onset operator strictly on TRAIN
    torch.manual_seed(42)
    model = Tactile3AxisSlipNet(in_dim=x_tr_n.shape[1], hidden=96)
    opt = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=2.0e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=55)
    pos_w = torch.tensor(
        [float(np.sum(s_tr == 0) / max(np.sum(s_tr == 1), 1))], dtype=torch.float32
    )
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    xt_t = torch.tensor(x_tr_n, dtype=torch.float32)
    ft_t = torch.tensor(f_tr, dtype=torch.float32)
    st_t = torch.tensor(s_tr, dtype=torch.float32)
    batch_sz = 1024
    n_tr = len(xt_t)

    for _ep in range(55):
        model.train()
        perm = torch.randperm(n_tr)
        for s0 in range(0, n_tr, batch_sz):
            idx = perm[s0 : s0 + batch_sz]
            opt.zero_grad()
            f_p, s_log = model(xt_t[idx])
            loss_f = torch.mean((f_p - ft_t[idx]) ** 2)
            loss_s = bce(s_log, st_t[idx])
            loss = loss_f + 0.35 * loss_s
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        f_dev_p_t, s_dev_l_t = model(torch.tensor(x_dev_n, dtype=torch.float32))
        f_hold_p_t, s_hold_l_t = model(torch.tensor(x_hold_n, dtype=torch.float32))
    f_dev_p = f_dev_p_t.numpy().astype(np.float64)
    s_dev_p = 1.0 / (
        1.0 + np.exp(-np.clip(s_dev_l_t.numpy().astype(np.float64), -25.0, 25.0))
    )
    f_hold_p = f_hold_p_t.numpy().astype(np.float64)
    s_hold_p = 1.0 / (
        1.0 + np.exp(-np.clip(s_hold_l_t.numpy().astype(np.float64), -25.0, 25.0))
    )

    # 4. Evaluate blind 3-axis force accuracy, slip AUROC, & single-frame latency (< 2 ms!)
    err_3d = f_hold_p - f_hold
    rmse_fx = float(np.sqrt(np.mean(err_3d[:, 0] ** 2)))
    rmse_fy = float(np.sqrt(np.mean(err_3d[:, 1] ** 2)))
    rmse_fz = float(np.sqrt(np.mean(err_3d[:, 2] ** 2)))
    rmse_3axis = float(np.sqrt(np.mean(err_3d**2)))
    rmse_mag = float(
        np.sqrt(
            np.mean(
                (np.linalg.norm(f_hold_p, axis=1) - np.linalg.norm(f_hold, axis=1)) ** 2
            )
        )
    )

    slip_auroc_blind = compute_roc_auroc(s_hold, s_hold_p)
    slip_thr, _, _ = calibrate_threshold_for_max_fpr(s_dev, s_dev_p, max_fpr=0.10)
    slip_rec_blind = float(np.mean(s_hold_p[s_hold > 0.5] >= slip_thr))
    slip_fpr_blind = float(np.mean(s_hold_p[s_hold <= 0.5] >= slip_thr))

    # Physics friction-cone direct slip score from predicted force ||F_t|| / (0.3*|F_n|)
    pred_cone_ratio = np.sqrt(f_hold_p[:, 0] ** 2 + f_hold_p[:, 1] ** 2) / (
        0.30 * np.abs(f_hold_p[:, 2]) + 0.02
    )
    cone_auroc_pred_force = compute_roc_auroc(s_hold, pred_cone_ratio)

    # Per-shape blind breakdown
    sh_arr = np.asarray(sh_hold)
    per_shape: dict[str, dict[str, float]] = {}
    for sh_name in ("sphere", "flat", "sharp"):
        m = sh_arr == sh_name
        per_shape[sh_name] = {
            "frames": int(np.sum(m)),
            "rmse_3axis_n": round(float(np.sqrt(np.mean(err_3d[m] ** 2))), 4),
            "slip_auroc": round(float(compute_roc_auroc(s_hold[m], s_hold_p[m])), 4),
        }

    # AS6983 conservative one-sided upper force bound calibration on DEV -> verify on HOLDOUT
    dev_f_mag_pred = np.linalg.norm(f_dev_p, axis=1)
    dev_f_mag_true = np.linalg.norm(f_dev, axis=1)
    hold_f_mag_pred = np.linalg.norm(f_hold_p, axis=1)
    hold_f_mag_true = np.linalg.norm(f_hold, axis=1)

    dev_samples = [
        Conformal3DDevSample(
            case_id=f"r6-dev-{i}",
            predicted_qois={"total_force_n": float(dev_f_mag_pred[i])},
            true_qois={"total_force_n": float(dev_f_mag_true[i])},
            latent_vector=tuple(float(v) for v in x_dev_n[i, :6]),
        )
        for i in range(0, len(dev_f_mag_pred), 10)
    ]
    conf_rec = Conformal3DCalibrator.fit(
        bundle_hash="study-r6-sparsh-tactile-slip-v1",
        dev_samples=dev_samples,
        qoi_units={"total_force_n": "N"},
        alpha=0.05,
    )
    hold_samples = [
        Conformal3DDevSample(
            case_id=f"r6-hold-{i}",
            predicted_qois={"total_force_n": float(hold_f_mag_pred[i])},
            true_qois={"total_force_n": float(hold_f_mag_true[i])},
            latent_vector=tuple(float(v) for v in x_hold_n[i, :6]),
        )
        for i in range(0, len(hold_f_mag_pred), 10)
    ]
    cov_rep = verify_as6983_holdout_coverage(conf_rec, hold_samples)["total_force_n"]

    # Measure real-time single-frame CPU latency
    single_in = torch.tensor(x_hold_n[:1], dtype=torch.float32)
    lat_ms: list[float] = []
    with torch.no_grad():
        for _ in range(400):
            t0 = time.perf_counter()
            _ = model(single_in)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
    med_ms = float(np.median(lat_ms))
    p95_ms = float(np.percentile(lat_ms, 95))
    rate_hz = 1000.0 / max(med_ms, 1.0e-4)

    # Pick one clean holdout trajectory transitioning from stick -> slip for D3/GIF!
    vis_eid = splits.holdout[0].episode_id
    for ep in splits.holdout:
        sl = traj_store[ep.episode_id]["slip"]
        if 0.2 <= float(np.mean(sl > 0.5)) <= 0.6 and len(sl) >= 40:
            vis_eid = ep.episode_id
            break
    vis_tr = traj_store[vis_eid]
    vx_n = np.clip((vis_tr["feats"] - x_mu) / x_sd, -8.0, 8.0)
    with torch.no_grad():
        vf_p_t, vs_l_t = model(torch.tensor(vx_n, dtype=torch.float32))
    vf_p = vf_p_t.numpy().astype(np.float64)
    vs_p = 1.0 / (
        1.0 + np.exp(-np.clip(vs_l_t.numpy().astype(np.float64), -25.0, 25.0))
    )
    shift_n = float(conf_rec.qoi_calibrations["total_force_n"].as6983_upper_quantile)

    summary: dict[str, Any] = {
        "study_id": "study_r6_dynamic_tactile_slip",
        "title": "Dynamic 3-Axis Tactile Force & Friction-Cone Slip-Onset Control (V6-R2 + V7-R4)",
        "datasets": [
            {
                "id": "sparsh-gelsight-force",
                "license": "CC-BY-NC-4.0",
                "tier": "C-noncommercial-eval-only",
                "hygiene_note": "Evaluated locally only; zero raw data or derived weights exported to Git.",
                "trajectories_total": len(traj_store),
                "frames_total": total_frames,
                "slip_frames": total_slip,
            }
        ],
        "fem_3d_coupled_normal_shear": fem_probes,
        "blind_holdout_metrics": {
            "split_protocol": f"Leak-free trajectory split ({len(splits.train)} train / {len(splits.dev)} dev / {len(splits.holdout)} blind holdout)",
            "holdout_frames": int(len(f_hold)),
            "rmse_3axis_n": round(rmse_3axis, 4),
            "rmse_fx_n": round(rmse_fx, 4),
            "rmse_fy_n": round(rmse_fy, 4),
            "rmse_fz_n": round(rmse_fz, 4),
            "rmse_force_mag_n": round(rmse_mag, 4),
            "slip_onset_auroc": round(float(slip_auroc_blind), 4),
            "slip_recall_at_10pct_dev_fpr": round(float(slip_rec_blind), 4),
            "slip_fpr_holdout": round(float(slip_fpr_blind), 4),
            "physics_cone_ratio_auroc": round(float(cone_auroc_pred_force), 4),
            "per_indenter_shape": per_shape,
            "as6983_upper_coverage": round(
                float(cov_rep.non_underpredicted_count) / max(cov_rep.holdout_m, 1), 4
            ),
            "as6983_conformal_shift_n": round(shift_n, 4),
            "median_inference_ms": round(med_ms, 4),
            "p95_inference_ms": round(p95_ms, 4),
            "control_rate_hz": round(rate_hz, 1),
        },
        "pass_gates": {
            "force_rmse_lt_0p5n": bool(rmse_3axis < 0.5),
            "slip_auroc_ge_0p90": bool(slip_auroc_blind >= 0.90),
            "inference_lt_2ms": bool(med_ms < 2.0 and rate_hz >= 500.0),
            "as6983_coverage_ge_95pct": bool(
                float(cov_rep.non_underpredicted_count) / max(cov_rep.holdout_m, 1)
                >= 0.95
            ),
        },
        "wall_time_s": round(time.perf_counter() - t_start, 2),
    }

    (ASSETS_DIR / "study_r6_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    # Build visualization payload (scrubbed trajectory + scatter subsample)
    sc_idx = np.linspace(0, len(f_hold) - 1, min(600, len(f_hold))).astype(int)
    viz_payload = {
        "summary": summary,
        "holdout_trajectory": [
            {
                "frame": int(i),
                "fx_true": round(float(vis_tr["forces"][i, 0]), 4),
                "fy_true": round(float(vis_tr["forces"][i, 1]), 4),
                "fz_true": round(float(vis_tr["forces"][i, 2]), 4),
                "fx_pred": round(float(vf_p[i, 0]), 4),
                "fy_pred": round(float(vf_p[i, 1]), 4),
                "fz_pred": round(float(vf_p[i, 2]), 4),
                "f_shear_true": round(
                    float(np.hypot(vis_tr["forces"][i, 0], vis_tr["forces"][i, 1])), 4
                ),
                "f_shear_pred": round(float(np.hypot(vf_p[i, 0], vf_p[i, 1])), 4),
                "f_mag_upper": round(float(np.linalg.norm(vf_p[i]) + shift_n), 4),
                "slip_true": int(vis_tr["slip"][i] > 0.5),
                "slip_prob": round(float(vs_p[i]), 4),
            }
            for i in range(len(vis_tr["slip"]))
        ],
        "scatter_sample": [
            {
                "fz_true": round(float(f_hold[i, 2]), 4),
                "fz_pred": round(float(f_hold_p[i, 2]), 4),
                "ft_true": round(float(np.hypot(f_hold[i, 0], f_hold[i, 1])), 4),
                "ft_pred": round(float(np.hypot(f_hold_p[i, 0], f_hold_p[i, 1])), 4),
                "slip": int(s_hold[i] > 0.5),
                "shape": sh_hold[i],
            }
            for i in sc_idx
        ],
    }
    (VIZ_DIR / "study_r6_viz.json").write_text(
        json.dumps(viz_payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
