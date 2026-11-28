"""Study R2 — Continuum & Tendon-Driven Robot Digital Twin (V6-R1).

Validates FormaMech's exact finite-rotation SE(3) Cosserat rod 9-disk 4-tendon backbone
FEM, 3D solid routed-tendon FEM with Capstan friction, inverse FBG contact-force
estimation, leak-free episode splitting, and >500 Hz AS6983-certified continuum surrogate
against:
  - CRL-Dataset-CTCR-Pose (100,000 real physical EM-tracked SE(3) poses of a 3-tube
    Nitinol concentric-tube continuum robot, MIT, Tier A)
  - Gotelli et al. 9-disk 4-tendon Nitinol continuum robot hardware parameters & STLs
    (L = 540 mm, 9 disks, d_c = 37.5 mm, CC-BY-4.0, Tier A)

Outputs:
  - Git-tracked canonical scrubbed summary: docs/assets/studies_robotics/study_r2_summary.json
  - Gitignored D3/Playwright viz payload:   .formamech-runs/robotics_studies/study_r2_viz.json
"""

from __future__ import annotations

import json
import math
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from piginot.bounds.conformal_3d import (
    Conformal3DCalibrator,
    Conformal3DDevSample,
    verify_as6983_holdout_coverage,
)
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import DirichletBC3D
from piginot.fem.tendon_3d import (
    CosseratBackboneSpec,
    TendonRoute3D,
    solve_3d_tendon_continuum_backbone,
    solve_3d_tendon_solid,
)
from piginot.geometry.mesh3d import make_brick_tet4_mesh
from piginot.ingest.robotics_streams import RoboticsEpisode, split_episodes_leak_free

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".formamech-runs" / "datasets" / "robotics" / "r2_continuum"
VIZ_DIR = REPO_ROOT / ".formamech-runs" / "robotics_studies"
ASSETS_DIR = REPO_ROOT / "docs" / "assets" / "studies_robotics"


class ContinuumPoseSurrogateNet(nn.Module):
    """Lightweight residual MLP mapping 12D joint/delta state -> 3D tip & mid-tube coords."""

    def __init__(self, in_dim: int = 12, out_dim: int = 6, hidden: int = 128) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.b1 = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.b2 = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = h + self.b1(h)
        h = h + self.b2(h)
        return self.out_proj(h)


def main() -> None:
    t_start = time.perf_counter()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    ctcr_csv = DATA_DIR / "CRL-Dataset-CTCR-Pose.csv"
    gotelli_stl_zip = DATA_DIR / "gotelli_custom_components.zip"
    if not ctcr_csv.exists():
        raise FileNotFoundError(
            "Missing CRL-Dataset-CTCR-Pose.csv. Run fetch_robotics_datasets.py first."
        )

    # 1. Inspect Gotelli et al. hardware STL components (CC-BY-4.0)
    gotelli_stls: list[str] = []
    if gotelli_stl_zip.exists():
        with zipfile.ZipFile(gotelli_stl_zip) as zf:
            gotelli_stls = sorted(
                n for n in zf.namelist() if n.lower().endswith(".stl")
            )

    # 2. Load real CRL-Dataset-CTCR-Pose (100,000 physical EM-tracked SE(3) poses, MIT)
    #    Cols 0..5: joint angles/translations q1..q6
    #    Cols 6..11: hysteresis/velocity deltas dq1..dq6
    #    Cols 12..39: 4 consecutive SE(3) poses (base, tube1, tube2, tip) each (x,y,z,qw,qx,qy,qz)
    df = pd.read_csv(ctcr_csv, header=None)
    total_rows = int(len(df))
    raw = df.to_numpy(dtype=np.float64)

    # Partition contiguous trajectory into 50 non-overlapping physical episodes (2,000 poses each)
    # to guarantee zero temporal/hysteresis leakage via split_episodes_leak_free
    n_eps = 50
    ep_len = total_rows // n_eps
    episodes: list[RoboticsEpisode] = []
    for ei in range(n_eps):
        s0, s1 = ei * ep_len, (ei + 1) * ep_len
        t_ep = np.arange(s1 - s0, dtype=np.float64) * 0.02
        episodes.append(
            RoboticsEpisode(
                episode_id=f"ctcr-ep-{ei:02d}",
                times_s=t_ep,
                channels={
                    "q_dq": raw[s0:s1, :12],
                    "tip_xyz": raw[s0:s1, 33:36],
                    "mid_xyz": raw[s0:s1, 26:29],
                    "tip_quat": raw[s0:s1, 36:40],
                },
                metadata={"source": "crl-ctcr-pose"},
            )
        )
    splits = split_episodes_leak_free(
        episodes, train_fraction=0.60, dev_fraction=0.20, seed=11
    )

    def stack_split(
        eps: tuple[RoboticsEpisode, ...], stride: int = 2
    ) -> tuple[np.ndarray, np.ndarray]:
        xs = np.vstack([ep.channels["q_dq"][::stride] for ep in eps])
        ys = np.vstack(
            [
                np.hstack(
                    [ep.channels["tip_xyz"][::stride], ep.channels["mid_xyz"][::stride]]
                )
                for ep in eps
            ]
        )
        return xs, ys

    x_tr, y_tr = stack_split(splits.train, stride=2)
    x_dev, y_dev = stack_split(splits.dev, stride=4)
    x_hold, y_hold = stack_split(splits.holdout, stride=4)

    # Normalize inputs/outputs using TRAIN statistics strictly
    x_mu = x_tr.mean(axis=0)
    x_sd = np.maximum(x_tr.std(axis=0), 1e-6)
    y_mu = y_tr.mean(axis=0)
    y_sd = np.maximum(y_tr.std(axis=0), 1e-6)

    torch.manual_seed(42)
    net = ContinuumPoseSurrogateNet(in_dim=12, out_dim=6, hidden=128)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)

    x_tr_t = torch.tensor((x_tr - x_mu) / x_sd, dtype=torch.float32)
    y_tr_t = torch.tensor((y_tr - y_mu) / y_sd, dtype=torch.float32)
    batch_sz = 512
    n_tr = len(x_tr_t)
    for _epoch in range(80):
        idx = torch.randperm(n_tr)
        for start in range(0, n_tr, batch_sz):
            b = idx[start : start + batch_sz]
            opt.zero_grad()
            loss = torch.mean((net(x_tr_t[b]) - y_tr_t[b]) ** 2)
            loss.backward()
            opt.step()
        sched.step()

    net.eval()

    def predict_batch(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xn = torch.tensor((x_np - x_mu) / x_sd, dtype=torch.float32)
            yn = net(xn).numpy().astype(np.float64)
        return yn * y_sd + y_mu

    # Calibrate AS6983 conservative tip position error bound on DEV episodes
    pred_dev = predict_batch(x_dev)
    err_dev_mm = np.linalg.norm(pred_dev[:, :3] - y_dev[:, :3], axis=1)
    dev_samples = [
        Conformal3DDevSample(
            case_id=f"dev-{i}",
            predicted_qois={"tip_position_error_mm": 0.0},
            true_qois={"tip_position_error_mm": float(err_dev_mm[i])},
            latent_vector=tuple(
                float(v) for v in ((x_dev[i, :6] - x_mu[:6]) / x_sd[:6])
            ),
        )
        for i in range(0, len(err_dev_mm), 4)
    ]
    conf_rec = Conformal3DCalibrator.fit(
        bundle_hash="study-r2-ctcr-pose-v1",
        dev_samples=dev_samples,
        qoi_units={"tip_position_error_mm": "mm"},
        alpha=0.10,
    )

    # Evaluate strictly blind on HOLDOUT episodes + single-pose CPU latency
    pred_hold = predict_batch(x_hold)
    tip_err_hold_mm = np.linalg.norm(pred_hold[:, :3] - y_hold[:, :3], axis=1)
    mid_err_hold_mm = np.linalg.norm(pred_hold[:, 3:6] - y_hold[:, 3:6], axis=1)
    hold_samples = [
        Conformal3DDevSample(
            case_id=f"hold-{i}",
            predicted_qois={"tip_position_error_mm": 0.0},
            true_qois={"tip_position_error_mm": float(tip_err_hold_mm[i])},
            latent_vector=tuple(
                float(v) for v in ((x_hold[i, :6] - x_mu[:6]) / x_sd[:6])
            ),
        )
        for i in range(0, len(tip_err_hold_mm), 4)
    ]
    cov_rep = verify_as6983_holdout_coverage(conf_rec, hold_samples)[
        "tip_position_error_mm"
    ]

    # Measure single-pose CPU inference latency (for > 500 Hz surgical control check)
    single_x = torch.tensor((x_hold[:1] - x_mu) / x_sd, dtype=torch.float32)
    lat_ms: list[float] = []
    with torch.no_grad():
        for _ in range(400):
            t0 = time.perf_counter()
            _ = net(single_x)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
    median_lat_ms = float(np.median(lat_ms))
    p95_lat_ms = float(np.percentile(lat_ms, 95))

    # OOD / snap-through regime check: compare in-distribution vs high-jump (||dq|| > p95)
    dq_norm_hold = np.linalg.norm(x_hold[:, 6:12], axis=1)
    p95_dq = float(np.percentile(dq_norm_hold, 95))
    in_dist_err_mm = float(np.mean(tip_err_hold_mm[dq_norm_hold <= p95_dq]))
    high_jump_err_mm = float(np.mean(tip_err_hold_mm[dq_norm_hold > p95_dq]))

    # 3. Exact Finite-Rotation SE(3) Cosserat 9-Disk 4-Tendon Nitinol Backbone (Gotelli et al. geometry)
    #    L = 540 mm, 9 spacer disks, 4 tendons at radius 18.75 mm (d_c = 37.5 mm), E = 58 GPa Nitinol
    spec_gotelli = CosseratBackboneSpec(
        length_mm=540.0,
        num_disks=9,
        youngs_modulus_mpa=58000.0,
        poisson_ratio=0.33,
        outer_radius_mm=1.25,
        tendon_offsets_mm=(
            (18.75, 0.0),
            (0.0, 18.75),
            (-18.75, 0.0),
            (0.0, -18.75),
        ),
        fiber_core_radius_mm=0.95,
    )

    # Sweep tendon tension 0 -> 14 N (producing large finite rotation) + external tip contact force
    cosserat_sweep: list[dict[str, Any]] = []
    anim_frames: list[dict[str, Any]] = []
    n_anim = 28
    cosserat_times: list[float] = []
    for fi in range(n_anim):
        ang = 2.0 * math.pi * (fi / float(n_anim))
        ramp = min(1.0, (fi + 1) / 8.0)
        t_mag = 8.0 * ramp
        t1 = max(0.0, t_mag * math.cos(ang))
        t2 = max(0.0, t_mag * math.sin(ang))
        t3 = max(0.0, -t_mag * math.cos(ang))
        t4 = max(0.0, -t_mag * math.sin(ang))
        f_ext_tip = (0.0, -0.25, 0.0) if 14 <= fi <= 22 else (0.0, 0.0, 0.0)
        sol = solve_3d_tendon_continuum_backbone(
            spec_gotelli,
            tendon_tensions_n=(t1, t2, t3, t4),
            external_forces_n={9: f_ext_tip},
            tolerance=1.0e-8,
        )
        assert sol.converged, f"Cosserat frame {fi} failed (grad={sol.grad_norm:.3e})"
        cosserat_times.append(sol.wall_time_s)
        tip_p = sol.disk_positions_mm[-1]
        tip_dir = sol.disk_rotations[-1, :, 2]
        bend_deg = float(
            math.degrees(math.acos(max(-1.0, min(1.0, float(tip_dir[2])))))
        )
        base_wrench = list(sol.base_force_n) + list(sol.base_moment_n_mm)
        if fi in (0, 3, 6, 9, 14, 18, 24):
            cosserat_sweep.append(
                {
                    "frame": fi,
                    "tensions_n": [
                        round(t1, 2),
                        round(t2, 2),
                        round(t3, 2),
                        round(t4, 2),
                    ],
                    "tip_force_n": list(f_ext_tip),
                    "tip_xyz_mm": [round(float(v), 2) for v in tip_p],
                    "bending_angle_deg": round(bend_deg, 2),
                    "max_fbg_microstrain": round(
                        float(np.max(np.abs(sol.fbg_microstrain))), 1
                    ),
                    "wall_time_s": round(float(sol.wall_time_s), 4),
                }
            )
        anim_frames.append(
            {
                "frame": fi,
                "tensions_n": [round(t1, 2), round(t2, 2), round(t3, 2), round(t4, 2)],
                "tip_force_n": list(f_ext_tip),
                "backbone_xyz": np.round(sol.disk_positions_mm, 2).tolist(),
                "disk_xyz": np.round(sol.disk_positions_mm[1:], 2).tolist(),
                "bending_angle_deg": round(bend_deg, 2),
                "fbg_strain_x": np.round(sol.fbg_microstrain[:, 0], 1).tolist(),
                "fbg_strain_y": np.round(sol.fbg_microstrain[:, 1], 1).tolist(),
                "base_wrench": [round(float(v), 2) for v in base_wrench],
            }
        )

    # 4. Inverse External Tip-Force Estimation from noisy FBG microstrains (Gauss-Newton)
    true_tensions = (5.0, 2.5, 0.0, 0.0)
    true_tip_force = (0.18, -0.24, 0.0)
    sol_true = solve_3d_tendon_continuum_backbone(
        spec_gotelli,
        tendon_tensions_n=true_tensions,
        external_forces_n={9: true_tip_force},
        tolerance=1.0e-8,
    )
    rng = np.random.default_rng(123)
    fbg_meas = sol_true.fbg_microstrain + rng.normal(
        0.0, 4.0, size=sol_true.fbg_microstrain.shape
    )
    f_est = np.array([0.0, 0.0], dtype=np.float64)
    for _gn in range(6):
        s0 = solve_3d_tendon_continuum_backbone(
            spec_gotelli,
            tendon_tensions_n=true_tensions,
            external_forces_n={9: (float(f_est[0]), float(f_est[1]), 0.0)},
            tolerance=1.0e-8,
        )
        r_vec = (s0.fbg_microstrain - fbg_meas).reshape(-1)
        jac = np.zeros((r_vec.size, 2), dtype=np.float64)
        eps_f = 1e-2
        for j in range(2):
            df_v = np.zeros(2)
            df_v[j] = eps_f
            sp = solve_3d_tendon_continuum_backbone(
                spec_gotelli,
                tendon_tensions_n=true_tensions,
                external_forces_n={
                    9: (float(f_est[0] + df_v[0]), float(f_est[1] + df_v[1]), 0.0)
                },
                tolerance=1.0e-8,
            )
            jac[:, j] = (sp.fbg_microstrain - s0.fbg_microstrain).reshape(-1) / eps_f
        step_f, *_ = np.linalg.lstsq(jac, -r_vec, rcond=None)
        f_est += step_f
        if float(np.linalg.norm(step_f)) < 1e-3:
            break
    inv_force_err_n = float(np.linalg.norm(f_est - np.array(true_tip_force[:2])))

    # 5. Full 3D Solid Routed-Tendon FEM with Capstan friction verification
    beam_mesh = make_brick_tet4_mesh(80.0, 12.0, 12.0, nx=8, ny=2, nz=2)
    nodes_b = np.asarray(beam_mesh.nodes, dtype=np.float64)
    top_chain = tuple(
        sorted(
            [
                i
                for i, p in enumerate(nodes_b)
                if abs(p[1] - 6.0) < 0.5 and abs(p[2] - 12.0) < 1e-5
            ],
            key=lambda i: float(nodes_b[i, 0]),
        )
    )
    solid_tendon_res = solve_3d_tendon_solid(
        beam_mesh,
        MaterialSpec(
            name="silicone-arm",
            model="compressible_neo_hookean",
            properties={
                "elastic_modulus": Quantity(2.5, Unit.MPA).to_canonical(),
                "poisson_ratio": Quantity(0.38, Unit.ONE).to_canonical(),
            },
        ),
        dirichlet_bcs=(
            DirichletBC3D("root", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        tendons=(
            TendonRoute3D("upper-cable", top_chain, tension_n=12.0, capstan_mu=0.15),
        ),
        load_steps=3,
    )
    assert solid_tendon_res.converged, (
        f"3D solid tendon failed: {solid_tendon_res.reason}"
    )

    # Sample 300 blind holdout CTCR points for D3 scatter & residual distribution
    sample_idx = np.linspace(0, len(y_hold) - 1, 300, dtype=int)
    ctcr_holdout_viz = [
        {
            "true_x": round(float(y_hold[i, 0]), 2),
            "true_y": round(float(y_hold[i, 1]), 2),
            "true_z": round(float(y_hold[i, 2]), 2),
            "pred_x": round(float(pred_hold[i, 0]), 2),
            "pred_y": round(float(pred_hold[i, 1]), 2),
            "pred_z": round(float(pred_hold[i, 2]), 2),
            "err_mm": round(float(tip_err_hold_mm[i]), 3),
            "dq_norm": round(float(dq_norm_hold[i]), 2),
        }
        for i in sample_idx
    ]

    summary: dict[str, Any] = {
        "study_id": "study_r2_continuum_tendon_twin",
        "title": "Continuum & Tendon-Driven Robot Digital Twin (CRL-CTCR + Gotelli 9-Disk Backbone)",
        "datasets": [
            {
                "id": "crl-ctcr-pose",
                "license": "MIT",
                "tier": "A",
                "total_poses": total_rows,
            },
            {
                "id": "gotelli-continuum-tdcr",
                "license": "CC-BY-4.0",
                "tier": "A",
                "hardware_stls": gotelli_stls,
            },
        ],
        "ctcr_real_data_holdout": {
            "split_protocol": "50 leak-free contiguous trajectory episodes (30 train / 10 dev / 10 blind holdout)",
            "holdout_poses_evaluated": int(len(y_hold)),
            "tip_mean_error_mm": round(float(np.mean(tip_err_hold_mm)), 3),
            "tip_rmse_mm": round(float(np.sqrt(np.mean(tip_err_hold_mm**2))), 3),
            "tip_p95_error_mm": round(float(np.percentile(tip_err_hold_mm, 95)), 3),
            "mid_tube_mean_error_mm": round(float(np.mean(mid_err_hold_mm)), 3),
            "as6983_target_coverage": 0.90,
            "as6983_upper_coverage": round(
                float(cov_rep.non_underpredicted_count) / max(cov_rep.holdout_m, 1), 4
            ),
            "as6983_quantile_mm": round(
                float(
                    conf_rec.qoi_calibrations[
                        "tip_position_error_mm"
                    ].as6983_upper_quantile
                ),
                3,
            ),
            "as6983_certified": bool(cov_rep.as6983_passed),
            "median_inference_ms": round(median_lat_ms, 4),
            "p95_inference_ms": round(p95_lat_ms, 4),
            "inference_hz": round(1000.0 / max(median_lat_ms, 1e-4), 1),
            "in_dist_mean_error_mm": round(in_dist_err_mm, 3),
            "high_jump_ood_error_mm": round(high_jump_err_mm, 3),
        },
        "cosserat_9disk_backbone": {
            "length_mm": 540.0,
            "num_disks": 9,
            "pitch_diameter_mm": 37.5,
            "max_bending_angle_deg": round(
                max(f["bending_angle_deg"] for f in anim_frames), 2
            ),
            "mean_fem_solve_ms": round(float(np.mean(cosserat_times)) * 1000.0, 2),
            "fbg_inverse_force_true_n": [true_tip_force[0], true_tip_force[1]],
            "fbg_inverse_force_est_n": [
                round(float(f_est[0]), 3),
                round(float(f_est[1]), 3),
            ],
            "fbg_inverse_force_error_n": round(inv_force_err_n, 4),
            "fbg_noise_microstrain": 4.0,
        },
        "solid_3d_tendon_capstan": {
            "converged": solid_tendon_res.converged,
            "max_displacement_mm": round(
                float(solid_tendon_res.max_displacement_mm), 3
            ),
            "tendon_stroke_mm": round(
                float(solid_tendon_res.tendon_strokes_mm["upper-cable"]), 3
            ),
            "capstan_tip_tension_n": round(
                float(solid_tendon_res.tip_tensions_n["upper-cable"]), 3
            ),
            "min_det_f": round(float(solid_tendon_res.min_det_f), 4),
        },
        "wall_time_s": round(time.perf_counter() - t_start, 2),
    }

    (ASSETS_DIR / "study_r2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    viz_payload = {
        "summary": summary,
        "ctcr_holdout": ctcr_holdout_viz,
        "ctcr_err_hist": np.histogram(tip_err_hold_mm, bins=24, range=(0.0, 12.0))[
            0
        ].tolist(),
        "ctcr_err_edges": np.round(np.linspace(0.0, 12.0, 25), 2).tolist(),
        "cosserat_sweep": cosserat_sweep,
        "anim_frames": anim_frames,
    }
    (VIZ_DIR / "study_r2_viz.json").write_text(
        json.dumps(viz_payload), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
