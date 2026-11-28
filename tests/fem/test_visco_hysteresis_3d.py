"""Unit tests for V7-R1 3D Prony Viscoelastic + Mullins Hysteresis FEM & Operator."""

from __future__ import annotations

import numpy as np
import pytest

from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import DirichletBC3D
from piginot.fem.follower_3d import FollowerPressureBC3D
from piginot.fem.visco_hysteresis_3d import (
    MullinsDamageSpec,
    PronyBranch,
    VelocityResidualHysteresisOperator,
    ViscoHysteresis3DConfig,
    solve_3d_visco_hysteresis,
)
from piginot.geometry.mesh3d import make_brick_tet10_mesh


def _elastomer_material(e_mpa: float = 2.0, nu: float = 0.45) -> MaterialSpec:
    return MaterialSpec(
        name="visco-elastomer",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(e_mpa, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(nu, Unit.ONE).to_canonical(),
        },
    )


def test_visco_hysteresis_3d_produces_admissible_loading_unloading_loop() -> None:
    mesh = make_brick_tet10_mesh(20.0, 8.0, 4.0, nx=2, ny=2, nz=1)
    mat = _elastomer_material(2.0, 0.45)
    # Full loading -> hold -> unloading cycle
    load_hist = (0.0, 0.35, 0.7, 1.0, 1.0, 0.7, 0.35, 0.0)
    cfg = ViscoHysteresis3DConfig(
        prony_branches=(
            PronyBranch(weight_gamma=0.25, relaxation_time_s=0.35),
            PronyBranch(weight_gamma=0.15, relaxation_time_s=1.5),
        ),
        mullins=MullinsDamageSpec(enabled=True, r_param=2.5, m_energy_density=0.002),
        dt_s=0.1,
    )
    res = solve_3d_visco_hysteresis(
        mesh,
        mat,
        dirichlet_bcs=(
            DirichletBC3D("clamp", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        load_history=load_hist,
        config=cfg,
        follower_bcs=(
            FollowerPressureBC3D("top_pressure", ("z_max",), pressure_mpa=0.0025),
        ),
    )
    assert res.converged
    assert res.thermodynamically_admissible
    assert res.step_dissipation_nonnegative
    assert float(np.min(res.min_det_f_history)) > 0.0
    assert res.cumulative_dissipation_n_mm > 0.0
    # Hysteresis asymmetry: unloading midpoint (idx 5, lam=0.7) differs from loading midpoint (idx 2, lam=0.7)
    u_load_mid = float(np.linalg.norm(res.displacement_history_mm[2]))
    u_unload_mid = float(np.linalg.norm(res.displacement_history_mm[5]))
    assert abs(u_unload_mid - u_load_mid) > 1.0e-3
    # Mullins softening activated during unloading (eta < 1)
    assert float(np.min(res.min_mullins_eta_history)) < 1.0

    doc = res.document()
    assert doc["thermodynamically_admissible"] is True
    assert len(doc["result_hash"]) == 64


def test_velocity_residual_hysteresis_operator_fits_asymmetric_loop() -> None:
    # Synthetic rate-dependent + Bouc-Wen hysteresis loop
    t = np.linspace(0.0, 4.0 * np.pi, 240)
    u = 0.5 * (1.0 - np.cos(t))
    du = np.gradient(u, t)
    y_true = 1.2 * u + 0.25 * u**2 - 0.12 * np.tanh(du * 3.0) - 0.05 * du

    op = VelocityResidualHysteresisOperator(
        n_prony_branches=2, hidden_dim=24, dt_s=float(t[1] - t[0])
    )
    fit_info = op.fit_sequence(u, y_true, epochs=120, lr=2.0e-2, seed=7)
    assert fit_info["train_rmse"] < 0.04

    out = op.predict_certified(u)
    preds = out["predictions"]
    receipt = out["receipt"]
    rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))
    assert rmse < 0.04
    assert receipt["dissipation_nonnegative"] is True
    assert receipt["cumulative_dissipation"] > 0.0
    assert len(receipt["receipt_sha256"]) == 64


def test_visco_hysteresis_config_validation() -> None:
    with pytest.raises(ValueError, match="sum of Prony"):
        ViscoHysteresis3DConfig(
            prony_branches=(
                PronyBranch(0.6, 0.5),
                PronyBranch(0.5, 1.0),
            )
        )
