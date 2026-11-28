"""Unit tests for FormaMech v0.7 Robotics extensions (V7-R2..V7-R6 & Harness)."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import torch

from piginot.benchmarks.external_data import DATASET_REGISTRY
from piginot.benchmarks.scaling_3d import (
    Scaling3DConfig,
    run_surface_3d_scaling_benchmark,
)
from piginot.export.onnx_export_3d import FrozenSurface3DRuntimeModule
from piginot.fem.tendon_3d import CosseratBackboneSpec
from piginot.harness import FormaMechAgentHarness
from piginot.operators.constitutive_op import (
    MODE_COMPRESSION,
    MODE_SIMPLE_SHEAR,
    MODE_TENSION,
    PalpationProbe,
    PanoConstitutiveIdentifier,
    localize_shallow_inclusion,
    ogden_mode_stress,
)
from piginot.operators.surface_operator_3d import (
    Surface3DConfig,
    Surface3DEnergyOperator,
)
from piginot.robotics.adaptive_cbf import (
    AdaptiveConformalBarrierSpec,
    AdaptiveConformalCbfFilter,
    ConformalQuantileRegressor,
)
from piginot.robotics.cbf import ControlSafetyState
from piginot.robotics.clik_3d import (
    OperatorClikConfig,
    OperatorClikController,
    SphereObstacle3D,
    make_cosserat_backbone_shape_model,
    make_surface_operator_shape_model,
)
from piginot.robotics.contact_diagnostics import (
    ContactDiagnosticsConfig,
    EnergyResidualContactEstimator,
    compute_roc_auroc,
)
from piginot.robotics.live_coupling import (
    ContactGeomState,
    ContactResidualNet,
    LiveSimulatorCoupler,
    compare_oneway_vs_twoway_rollout,
)


def test_v7_r2_acp_cqr_tightens_calm_and_expands_on_shift() -> None:
    rng = np.random.default_rng(42)
    # Heteroscedastic residuals: tiny noise when x < 0.5, large noise when x >= 0.5
    x_tr = rng.uniform(0.0, 1.0, size=(120, 2))
    scale_tr = np.where(x_tr[:, 0] < 0.5, 0.04, 0.35)
    y_tr = rng.normal(0.0, scale_tr)

    x_dev = rng.uniform(0.0, 1.0, size=(60, 2))
    scale_dev = np.where(x_dev[:, 0] < 0.5, 0.04, 0.35)
    y_dev = rng.normal(0.0, scale_dev)

    cqr = ConformalQuantileRegressor(alpha_target=0.05, n_centers=6)
    cal = cqr.fit_and_calibrate(x_tr, y_tr, x_dev, y_dev)
    m_calm = cqr.predict_upper_margin(np.array([0.1, 0.5]))
    m_noisy = cqr.predict_upper_margin(np.array([0.9, 0.5]))
    assert m_calm < m_noisy
    assert m_calm < cal["static_shift"]

    spec = AdaptiveConformalBarrierSpec(
        name="force_n",
        safety_limit=10.0,
        base_margin=cal["static_shift"],
        quantile_model=cqr,
        gamma_acp=0.12,
    )
    filt = AdaptiveConformalCbfFilter(
        (spec,),
        u_min=np.array([0.0]),
        u_max=np.array([1.0]),
        safe_hold_action=np.array([0.0]),
        soft_ood_threshold=2.0,
        hard_ood_threshold=5.0,
    )
    alpha0 = filt.state_snapshot().alpha_by_barrier["force_n"]
    # Inject repeated payload-shift under-coverage -> alpha drops & margin expands
    for _ in range(5):
        filt.update_online(
            np.array([0.2, 0.5]),
            {"force_n": 6.0},
            {"force_n": 8.5},
            ood_prev=2.5,
        )
    snap = filt.state_snapshot()
    assert snap.alpha_by_barrier["force_n"] < alpha0
    assert (
        filt.compute_margin("force_n", np.array([0.2, 0.5]), ood_distance=3.0) > m_calm
    )

    # Filter projects unsafe command and transitions fail-closed on hard OOD
    res_proj = filt.filter_action(
        np.array([0.2, 0.5]),
        np.array([0.95]),
        eval_current_qois=lambda _x: ({"force_n": 7.0}, 1.0),
        predict_next_qois=lambda _x, u: ({"force_n": 5.0 + 6.0 * float(u[0])}, 1.0),
    )
    assert res_proj.state is ControlSafetyState.CBF_MODIFIED
    assert float(res_proj.u_safe[0]) < 0.95
    assert len(res_proj.document()["receipt_sha256"]) == 64

    res_ood = filt.filter_action(
        np.array([0.2, 0.5]),
        np.array([0.5]),
        eval_current_qois=lambda _x: ({"force_n": 5.0}, 6.0),
        predict_next_qois=lambda _x, u: ({"force_n": 5.0}, 6.0),
    )
    assert res_ood.state is ControlSafetyState.SAFE_HOLD_OOD
    assert np.allclose(res_ood.u_safe, np.array([0.0]))


def test_v7_r3_operator_clik_surface_and_cosserat_with_obstacle_cbf() -> None:
    # 1. Surface3DEnergyOperator CLIK wrapper
    torch.manual_seed(11)
    op = Surface3DEnergyOperator(
        Surface3DConfig(
            parameter_dim=3, hidden_dim=16, slice_count=4, num_heads=2, num_blocks=1
        )
    )
    pts = torch.rand((1, 12, 3), dtype=torch.float32) * 20.0
    nrm = torch.zeros_like(pts)
    nrm[..., 2] = 1.0
    wt = torch.full((1, 12), 1.0 / 12.0)
    ft = torch.zeros((1, 12, 6))
    with torch.no_grad():
        enc = op.encode_surface(pts, nrm, wt, ft)
    rt = FrozenSurface3DRuntimeModule(op, enc, pts)
    surf_model = make_surface_operator_shape_model(
        rt, pts[0].numpy().astype(np.float64)
    )
    clik_s = OperatorClikController(
        surf_model,
        u_min=np.zeros(3),
        u_max=np.ones(3),
        config=OperatorClikConfig(gain_kp=0.5),
    )
    step_s = clik_s.step(
        np.array([0.5, 0.5, 0.5]), pts[0, -1].numpy() + np.array([0.2, 0.0, 0.0])
    )
    assert np.isfinite(step_s.tip_error_mm)
    assert len(step_s.document()["receipt_sha256"]) == 64

    # 2. Cosserat continuum backbone CLIK reduces tip error & avoids sphere obstacle
    spec = CosseratBackboneSpec(
        length_mm=180.0,
        num_disks=4,
        youngs_modulus_mpa=58000.0,
        poisson_ratio=0.33,
        outer_radius_mm=1.1,
    )
    coss_model = make_cosserat_backbone_shape_model(spec)
    pts_target, _ = coss_model(np.array([5.0, 2.0, 0.0, 0.0]))
    target_tip = pts_target[-1]
    # Place obstacle near workspace flank so CLIK routes safely around it to target
    obs = SphereObstacle3D(name="vessel", center_mm=(25.0, -28.0, 90.0), radius_mm=10.0)
    cbf = AdaptiveConformalCbfFilter(
        (
            AdaptiveConformalBarrierSpec(
                name="vessel", safety_limit=0.0, base_margin=0.5
            ),
        ),
        u_min=np.zeros(4),
        u_max=np.full(4, 15.0),
        safe_hold_action=np.zeros(4),
    )
    ctrl = OperatorClikController(
        coss_model,
        u_min=np.zeros(4),
        u_max=np.full(4, 15.0),
        config=OperatorClikConfig(gain_kp=0.75, damping_lambda=0.4, max_step_norm=4.0),
    )
    q = np.array([0.5, 0.5, 0.0, 0.0])
    e0 = float(np.linalg.norm(target_tip - coss_model(q)[0][-1]))
    for _ in range(5):
        out = ctrl.step(q, target_tip, obstacles=(obs,), adaptive_cbf=cbf)
        q = out.q_next
        assert out.min_obstacle_clearance_mm > 0.0
    assert out.tip_error_mm < 0.35 * e0


def test_v7_r3_pano_constitutive_fingerprinting_and_inclusion() -> None:
    # Build synthetic heterogeneous tissue specimens spanning 80-450 Pa
    rng = np.random.default_rng(7)
    train_specs = []
    eps_t = np.linspace(0.0, 0.12, 13)
    for mu_true, alpha_true in [
        (90.0, -3.0),
        (180.0, -4.0),
        (320.0, -2.0),
        (420.0, -5.0),
    ]:
        sig = ogden_mode_stress(eps_t, MODE_TENSION, mu_true, alpha_true)
        probes = (
            PalpationProbe(MODE_TENSION, float(eps_t[4]), float(sig[4])),
            PalpationProbe(MODE_TENSION, float(eps_t[10]), float(sig[10])),
        )
        train_specs.append(
            {
                "probes": probes,
                "target_strains": eps_t,
                "target_stresses_pa": sig,
                "mode": MODE_TENSION,
            }
        )

    pano = PanoConstitutiveIdentifier()
    pano.fit_on_specimens(train_specs, epochs=60, seed=42)

    # Blind unseen tissue specimen (mu=275 Pa, alpha=-4.0) with 2 probes
    sig_unseen = ogden_mode_stress(eps_t, MODE_TENSION, 275.0, -4.0)
    probes_u = (
        PalpationProbe(MODE_TENSION, float(eps_t[4]), float(sig_unseen[4])),
        PalpationProbe(MODE_TENSION, float(eps_t[10]), float(sig_unseen[10])),
    )
    fp = pano.identify(probes_u)
    assert fp.inference_ms < 50.0  # fast CPU path (typically < 1 ms)
    pred_curve = pano.predict_stress_curve(eps_t, MODE_TENSION, fp)
    rmse = float(np.sqrt(np.mean((pred_curve - sig_unseen) ** 2)))
    assert rmse < 15.0
    assert len(fp.document()["receipt_sha256"]) == 64

    # Verify compression & shear modes are finite & sign-consistent
    assert (
        float(ogden_mode_stress(np.array([-0.1]), MODE_COMPRESSION, 200.0, -4.0)[0])
        < 0.0
    )
    assert (
        float(ogden_mode_stress(np.array([0.2]), MODE_SIMPLE_SHEAR, 200.0, 2.0)[0])
        == np.testing.assert_allclose(
            ogden_mode_stress(np.array([0.2]), MODE_SIMPLE_SHEAR, 200.0, 2.0)[0],
            40.0,
            rtol=1e-4,
        )
        or True
    )

    # Shallow inclusion localization
    grid = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0]])
    forces = np.array([0.2, 0.2, 0.21, 0.19, 0.75]) + rng.normal(0.0, 0.002, size=5)
    depths = np.full(5, 1.5)
    loc = localize_shallow_inclusion(grid, forces, depths)
    assert (
        np.linalg.norm(np.array(loc["estimated_center_xy_mm"]) - np.array([5.0, 5.0]))
        < 2.0
    )
    assert loc["stiffness_contrast_ratio"] > 3.0


def test_v7_r4_energy_residual_contact_estimator_separates_jamming_from_snap() -> None:
    def _make_episode(seed_offset: int) -> dict[str, np.ndarray]:
        r = np.random.default_rng(100 + seed_offset)
        t_len = 220
        dt = 0.02
        pos = np.zeros((t_len, 3))
        vel = np.zeros((t_len, 3))
        frc = np.zeros((t_len, 3))
        tau = np.zeros((t_len, 3))
        lbl = np.zeros(t_len)
        for i in range(1, t_len):
            # Nominal smooth downward insertion with brief snap-fit spike at i=50..56
            if 50 <= i <= 56:
                vel[i] = np.array([0.0, 0.0, -0.05])
                frc[i] = np.array([0.5, -0.4, 18.0]) + r.normal(0, 0.4, 3)
                tau[i] = np.array([0.05, 0.05, 0.02])
            # Sustained mechanical jam at i=130..185: stalled velocity + high opposing force & off-axis torque
            elif 130 <= i <= 185:
                lbl[i] = 1.0
                vel[i] = r.normal(0, 0.0005, 3)
                frc[i] = np.array([9.0, -7.5, 15.0]) + r.normal(0, 0.5, 3)
                tau[i] = np.array([1.2, 0.9, 0.6]) + r.normal(0, 0.05, 3)
            else:
                vel[i] = np.array([0.0, 0.0, -0.04]) + r.normal(0, 0.002, 3)
                frc[i] = np.array([0.4, 0.3, 2.5]) + r.normal(0, 0.25, 3)
                tau[i] = r.normal(0, 0.03, 3)
            pos[i] = pos[i - 1] + dt * vel[i]
        return {
            "positions": pos,
            "velocities": vel,
            "forces": frc,
            "torques": tau,
            "labels": lbl,
        }

    tr_eps = [_make_episode(i) for i in range(4)]
    dev_eps = [_make_episode(10), _make_episode(11)]
    test_ep = _make_episode(20)

    est = EnergyResidualContactEstimator(
        ContactDiagnosticsConfig(
            dt_s=0.02,
            short_window=5,
            medium_window=15,
            long_window=30,
            train_epochs=45,
            phnn_epochs=25,
        )
    )
    fit_rep = est.fit(tr_eps, dev_eps)
    assert fit_rep["dev_auroc"] > 0.90

    pred = est.predict_episode(
        test_ep["forces"],
        torques=test_ep["torques"],
        positions=test_ep["positions"],
        velocities=test_ep["velocities"],
    )
    auc = compute_roc_auroc(test_ep["labels"], pred.probabilities)
    assert auc > 0.92
    assert len(pred.document()["receipt_sha256"]) == 64


def test_v7_r5_live_twoway_coupling_stop_grad_and_c_abi() -> None:
    nodes = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [5.0, 5.0, 0.0]],
        dtype=np.float64,
    )
    res_net = ContactResidualNet(hidden_dim=16)
    # Give residual network strong weights so stop_grad difference is unmistakable
    with torch.no_grad():
        for p in res_net.parameters():
            p.fill_(0.5)

    coupler = LiveSimulatorCoupler(
        nodes, elastic_stiffness_n_per_mm=1.2, residual_model=res_net
    )
    geom = ContactGeomState(
        name="tool", position_mm=(5.0, 5.0, 6.0), radius_mm=12.0, stiffness_n_per_mm=2.0
    )

    out_full = coupler.step_coupled(
        np.zeros(6), np.zeros(6), (geom,), stop_grad_residual=False
    )
    out_sg = coupler.step_coupled(
        np.zeros(6), np.zeros(6), (geom,), stop_grad_residual=True
    )
    # Values match identically while linearized Jacobian excludes residual gradients!
    assert np.allclose(out_full.wrench_6d, out_sg.wrench_6d, atol=1e-5)
    assert not np.allclose(out_full.tangent_stiffness_6x6, out_sg.tangent_stiffness_6x6)
    assert out_sg.active_contacts > 0
    assert len(out_sg.document()["receipt_sha256"]) == 64

    # Test zero-copy C-ABI callback
    cb = coupler.make_c_abi_callback((geom,), stop_grad_residual=True)
    q_buf = (ctypes.c_double * 6)(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    qd_buf = (ctypes.c_double * 6)(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    w_buf = (ctypes.c_double * 6)()
    k_buf = (ctypes.c_double * 36)()
    rc = cb(q_buf, qd_buf, w_buf, k_buf)
    assert rc == 0
    assert np.allclose(
        np.ctypeslib.as_array(w_buf, shape=(6,)), out_sg.wrench_6d, atol=1e-6
    )

    # Compare one-way vs two-way rollout divergence
    cmd = np.column_stack(
        (np.full(15, 5.0), np.full(15, 5.0), np.linspace(14.0, 4.0, 15))
    )
    cmp = compare_oneway_vs_twoway_rollout(
        nodes, cmd, dt_s=0.02, contact_radius_mm=12.0
    )
    assert cmp.max_tool_divergence_mm > 0.01
    assert len(cmp.document()["receipt_sha256"]) == 64


def test_v7_r6_scaling_3d_benchmark_and_harness_capabilities(tmp_path: Path) -> None:
    rep = run_surface_3d_scaling_benchmark(
        Scaling3DConfig(
            node_counts=(1024, 4096),
            anchor_count=128,
            decode_chunk_size=1024,
            timed_runs=2,
            verify_export_parity=True,
            export_parity_nodes=512,
        ),
        export_dir=tmp_path,
    )
    assert rep.all_parity_passed
    assert rep.max_export_parity_diff <= 1.0e-5
    assert rep.max_chunk_drift <= 1.0e-5
    assert len(rep.document()["report_sha256"]) == 64
    assert "soft-pneumatic-hysteresis-15584329" in DATASET_REGISTRY

    # Verify v0.7 capabilities are live in FormaMechAgentHarness
    harness = FormaMechAgentHarness(tmp_path / "ws")
    env_pano = harness.invoke(
        "validate.identify_constitutive_pano",
        {"probes": [{"mode": 0, "strain": 0.1, "stress_pa": 60.0}]},
    )
    assert env_pano.ok
    assert env_pano.result is not None
    assert env_pano.result["shear_modulus_pa"] > 100.0

    env_scale = harness.invoke(
        "verify.surface_3d_scaling_benchmark",
        {
            "node_counts": [512, 1024],
            "anchor_count": 64,
            "decode_chunk_size": 512,
            "timed_runs": 1,
        },
    )
    assert env_scale.ok
    assert env_scale.result is not None
    assert env_scale.result["all_parity_passed"] is True
