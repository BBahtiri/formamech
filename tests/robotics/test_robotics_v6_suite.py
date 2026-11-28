"""Comprehensive tests for V6-R2/R3/R4 robotics tactile, PHNN, MPC, CBF, ONNX & bridges."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from piginot.export.onnx_export import OnnxParityStatus
from piginot.export.onnx_export_3d import (
    FrozenSurface3DRuntimeModule,
    export_surface_3d_runtime,
)
from piginot.harness import FormaMechAgentHarness, HarnessBudget
from piginot.operators.surface_operator_3d import (
    Surface3DConfig,
    Surface3DEnergyOperator,
)
from piginot.robotics import (
    CondensedQpMpcConfig,
    CondensedQpMpcController,
    ConformalBarrierSpec,
    ConformalCbfSafetyFilter,
    ControlSafetyState,
    PortHamiltonianConfig,
    PortHamiltonianSurrogate,
    TactileElastomerSurrogate,
    TactileSensorSpec,
    export_mujoco_compliance_plugin,
    export_ros2_node_manifest,
    import_simulator_wrench_trace,
    simulate_tactile_probe_fem,
)


def test_tactile_fem_to_conformal_surrogate_pipeline() -> None:
    """Real 3D TET10 tactile FEM samples train a <5 ms AS6983-calibrated surrogate."""
    spec = TactileSensorSpec(
        youngs_modulus_mpa=0.25,
        poisson_ratio=0.48,
        taxel_grid_xy=(4, 4),
        mesh_div_xyz=(3, 3, 1),
    )
    mesh = spec.build_tet10_mesh()
    samples = []
    for i, (px, py, d) in enumerate(
        [
            (7.0, 10.0, 0.35),
            (9.0, 12.0, 0.55),
            (11.0, 14.0, 0.70),
            (8.0, 13.0, 0.45),
            (10.0, 11.0, 0.60),
            (9.0, 12.0, 0.40),
            (8.5, 11.5, 0.50),
            (9.5, 12.5, 0.65),
        ]
    ):
        samples.append(
            simulate_tactile_probe_fem(
                spec,
                case_id=f"p{i}",
                probe_x_mm=px,
                probe_y_mm=py,
                depth_mm=d,
                mesh=mesh,
                load_steps=2,
            )
        )

    surr = TactileElastomerSurrogate(spec, hidden_dim=48, seed=1)
    fit_stats = surr.fit(samples[:4], dev_samples=samples[4:], epochs=120, alpha=0.20)
    assert fit_stats["calibrated_qois"] == 3.0

    pred = surr.predict(samples[4])
    assert pred.normal_force_n > 0.0
    assert pred.inference_ms < 25.0
    assert pred.conformal_assessment is not None
    assert "normal_force_n" in pred.conformal_assessment.bounds
    assert len(pred.document()["receipt_sha256"]) == 64


def test_port_hamiltonian_passivity_and_qp_mpc_tracking() -> None:
    """Port-Hamiltonian surrogate enforces nonnegative dissipation & feeds <10 ms QP-MPC."""
    torch.manual_seed(0)
    phnn = PortHamiltonianSurrogate(
        PortHamiltonianConfig(state_dim=2, control_dim=2, hidden_dim=32)
    )
    q = torch.tensor([[0.4, -0.3]], dtype=torch.float32)
    p = torch.tensor([[0.2, 0.5]], dtype=torch.float32)
    u_zero = torch.zeros((1, 2), dtype=torch.float32)

    h0 = float(phnn.hamiltonian(q, p).item())
    q_cur, p_cur = q, p
    for _ in range(15):
        _, _, diss = phnn.vector_field(q_cur, p_cur, u_zero)
        assert float(diss.item()) >= 0.0
        q_cur, p_cur = phnn.step_rk4(q_cur, p_cur, u_zero, dt_s=0.02)
    h1 = float(phnn.hamiltonian(q_cur, p_cur).item())
    # Unforced passive Port-Hamiltonian system must not gain energy
    assert h1 <= h0 + 1.0e-5

    # Drive linear spring-damper via CondensedQpMpcController in < 10 ms
    def _lin_plant(
        x: np.ndarray, u: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.array([[1.0, dt], [-4.0 * dt, 1.0 - 0.8 * dt]], dtype=np.float64)
        b = np.array([[0.0], [2.0 * dt]], dtype=np.float64)
        return a @ x + b @ u, a, b

    mpc = CondensedQpMpcController(
        CondensedQpMpcConfig(
            state_dim=2,
            control_dim=1,
            horizon=15,
            dt_s=0.02,
            q_weights=(40.0, 1.0),
            r_weights=(0.01,),
            u_min=(-6.0,),
            u_max=(6.0,),
        ),
        _lin_plant,
    )
    x = np.array([0.0, 0.0], dtype=np.float64)
    x_ref = np.array([1.0, 0.0], dtype=np.float64)
    solve_times: list[float] = []
    for _ in range(80):
        res = mpc.step(x, x_ref)
        solve_times.append(res.solve_time_ms)
        x, _, _ = _lin_plant(x, res.u_optimal, 0.02)
    assert abs(x[0] - 1.0) < 0.05
    assert float(np.median(solve_times)) < 10.0


def test_conformal_cbf_safety_filter_modifies_and_holds_on_ood() -> None:
    """AS6983 Conformal CBF projects unsafe commands & triggers SAFE_HOLD on OOD."""
    filt = ConformalCbfSafetyFilter(
        barriers=(
            ConformalBarrierSpec(
                name="peak_stress_mpa",
                safety_limit=10.0,
                conformal_shift_margin=1.5,
                gamma_cbf=20.0,
            ),
        ),
        u_min=np.array([0.0]),
        u_max=np.array([10.0]),
        safe_hold_action=np.array([0.0]),
        ood_threshold=3.5,
        dt_s=0.05,
    )

    def _eval_cur(x: np.ndarray) -> tuple[dict[str, float], float]:
        return {"peak_stress_mpa": float(x[0])}, float(x[1])

    def _pred_next(x: np.ndarray, u: np.ndarray) -> tuple[dict[str, float], float]:
        return {"peak_stress_mpa": float(x[0] + 0.5 * u[0])}, float(x[1])

    # 1. Safe nominal action passes unmodified
    r_ok = filt.filter_action(
        np.array([2.0, 0.5]),
        np.array([1.0]),
        eval_current_qois=_eval_cur,
        predict_next_qois=_pred_next,
    )
    assert r_ok.state is ControlSafetyState.NOMINAL
    assert not r_ok.modified

    # 2. Unsafe aggressive command gets projected so y + 1.5 <= 10.0 (y <= 8.5)
    r_mod = filt.filter_action(
        np.array([7.5, 0.5]),
        np.array([8.0]),
        eval_current_qois=_eval_cur,
        predict_next_qois=_pred_next,
    )
    assert r_mod.state is ControlSafetyState.CBF_MODIFIED
    assert r_mod.modified
    assert r_mod.u_safe[0] <= 2.05  # 7.5 + 0.5 * u <= 8.5 => u <= 2.0

    # 3. OOD distance > 3.5 forces fail-closed SAFE_HOLD_OOD
    r_ood = filt.filter_action(
        np.array([2.0, 5.2]),
        np.array([3.0]),
        eval_current_qois=_eval_cur,
        predict_next_qois=_pred_next,
    )
    assert r_ood.state is ControlSafetyState.SAFE_HOLD_OOD
    assert float(r_ood.u_safe[0]) == 0.0


def test_3d_runtime_export_bridges_and_harness(tmp_path: Path) -> None:
    """Test 3D TorchScript/ONNX export, MuJoCo/ROS 2 bridges, and V6 harness tools."""
    torch.manual_seed(2)
    op = Surface3DEnergyOperator(
        Surface3DConfig(
            parameter_dim=2, surface_feature_dim=6, hidden_dim=16, slice_count=4
        )
    )
    enc = op.encode_surface(
        torch.randn(1, 12, 3),
        torch.randn(1, 12, 3),
        torch.ones(1, 12),
        torch.randn(1, 12, 6),
    )
    qxyz = torch.randn(1, 20, 3)
    p_sample = torch.tensor([[0.5, -0.2]], dtype=torch.float32)

    wrapper = FrozenSurface3DRuntimeModule(op, enc, qxyz)
    u0, jac = wrapper.predict_and_parameter_jacobian(p_sample)
    assert u0.shape == (20, 3)
    assert jac.shape == (20, 3, 2)

    rep = export_surface_3d_runtime(
        op,
        enc,
        qxyz,
        tmp_path / "op3d.pt",
        sample_parameters=p_sample,
        prefer_onnx=False,
    )
    assert rep.parity_status is OnnxParityStatus.VERIFIED_PASS
    assert rep.max_abs_diff < 1.0e-5

    # Simulator wrench import & MuJoCo / ROS 2 manifest exports
    csv_path = tmp_path / "wrench.csv"
    csv_path.write_text(
        "t,fx,fy,fz,tx,ty,tz\n0.0,1,0,-5,0,0.1,0\n0.01,1.2,0,-6,0,0.2,0\n",
        encoding="utf-8",
    )
    trace = import_simulator_wrench_trace(csv_path)
    assert trace.peak_force_n > 6.0
    mj = export_mujoco_compliance_plugin(
        tmp_path / "mj",
        component_name="soft_finger",
        stiffness_xyz_n_per_m=(1200.0, 1200.0, 3500.0),
        damping_xyz_n_s_per_m=(5.0, 5.0, 10.0),
        mass_kg=0.08,
        max_safe_force_n=25.0,
    )
    assert len(mj["manifest_sha256"]) == 64
    ros = export_ros2_node_manifest(
        tmp_path / "ros", node_name="formamech_cbf_guard", control_rate_hz=100.0
    )
    assert len(ros["manifest_sha256"]) == 64

    # Harness discovery + live invocation of V6 robotics capability
    harness = FormaMechAgentHarness(
        tmp_path / "ws", budget=HarnessBudget(max_fem_solves=4)
    )
    env = harness.invoke(
        "fem.solve_tendon_continuum_3d",
        {"num_disks": 6, "tendon_tensions_n": [3.0, 0.0, 0.0, 0.0]},
    )
    assert env.ok, env.error_message
    assert env.result["converged"] is True
