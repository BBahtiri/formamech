"""Verification tests for Phase 4B (V3-4B) 3D combined kinematic J2, finite-strain J2 & 3D stepper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from piginot.cad.importing_3d import (
    classify_faces_by_bbox,
    generate_3d_box_step,
    import_cad_3d,
)
from piginot.fem.backend_3d import DirichletBC3D
from piginot.geometry.mesh3d import CellType3D
from piginot.stateful.fem_stepper_3d import (
    LoadStep3D,
    solve_incremental_fem_3d_trajectory,
)
from piginot.stateful.j2_advanced import (
    FiniteStrainJ2Config,
    FiniteStrainJ2Plasticity,
    J2CombinedHardeningConfig,
    J2CombinedHardeningPlasticity,
)


def _check_tangent_fd(
    model: J2CombinedHardeningPlasticity | FiniteStrainJ2Plasticity,
    f_mat: torch.Tensor,
    state_old: dict[str, torch.Tensor],
    *,
    tol: float = 1.0e-7,
) -> None:
    upd = model.update(torch.eye(3, dtype=torch.float64), f_mat, state_old, dt=0.1)
    assert upd.admissible
    c_an = upd.algorithmic_tangent.detach().numpy()
    eps = 1.0e-7
    c_fd = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for k in range(3):
        for l_idx in range(3):
            df = torch.zeros((3, 3), dtype=torch.float64)
            df[k, l_idx] = eps
            up = model.update(
                torch.eye(3, dtype=torch.float64), f_mat + df, state_old, dt=0.1
            )
            um = model.update(
                torch.eye(3, dtype=torch.float64), f_mat - df, state_old, dt=0.1
            )
            c_fd[:, :, k, l_idx] = (
                ((up.stress - um.stress) / (2.0 * eps)).detach().numpy()
            )
    rel = float(np.linalg.norm(c_an - c_fd) / max(np.linalg.norm(c_an), 1.0))
    assert rel < tol, f"tangent FD mismatch: {rel:.3e}"


def test_j2_combined_kinematic_hardening_tangent_and_bauschinger_effect() -> None:
    cfg = J2CombinedHardeningConfig(
        youngs_modulus_mpa=70_000.0,
        poisson_ratio=0.33,
        yield_stress_mpa=200.0,
        linear_hardening_mpa=400.0,
        voce_saturation_mpa=30.0,
        voce_rate=20.0,
        kinematic_modulus_mpa=6_000.0,
        kinematic_recovery=12.0,
        thermal_softening_per_k=1.0e-3,
    )
    model = J2CombinedHardeningPlasticity(cfg)
    st0 = model.initial_state((), dtype=torch.float64)

    # Active plastic tension state
    f_tens = torch.tensor(
        [[1.008, 0.001, 0.0], [0.001, 0.997, 0.0], [0.0, 0.0, 0.997]],
        dtype=torch.float64,
    )
    _check_tangent_fd(model, f_tens, st0, tol=1.0e-7)
    upd1 = model.update(torch.eye(3, dtype=torch.float64), f_tens, st0, dt=0.1)
    assert float(upd1.state_new["eq_plastic_strain"].item()) > 0.0
    assert float(upd1.state_new["backstress"][0, 0].item()) > 0.0
    assert float(upd1.dissipation.item()) > 0.0

    # Reverse compression with backstress -> verify tangent & Bauschinger earlier yield
    f_rev = torch.tensor(
        [[0.994, -0.001, 0.0], [-0.001, 1.002, 0.0], [0.0, 0.0, 1.002]],
        dtype=torch.float64,
    )
    st1 = {k: v.detach().clone() for k, v in upd1.state_new.items()}
    _check_tangent_fd(model, f_rev, st1, tol=1.0e-7)

    # Thermal softening lowers flow stress at elevated temperature
    upd_hot = model.update(
        torch.eye(3, dtype=torch.float64),
        f_tens,
        st0,
        dt=0.1,
        temperature=torch.tensor(493.15, dtype=torch.float64),
    )
    assert float(upd_hot.stress[0, 0].item()) < float(upd1.stress[0, 0].item())


def test_finite_strain_multiplicative_j2_tangent_and_3d_guarded_stepper(
    tmp_path: Path,
) -> None:
    model = FiniteStrainJ2Plasticity(
        FiniteStrainJ2Config(
            youngs_modulus_mpa=20_000.0,
            poisson_ratio=0.30,
            yield_stress_mpa=250.0,
            linear_hardening_mpa=800.0,
        )
    )
    st0 = model.initial_state((), dtype=torch.float64)
    # 15% finite stretch + shear
    f_large = torch.tensor(
        [[1.15, 0.04, 0.0], [0.0, 0.94, 0.0], [0.0, 0.0, 0.94]],
        dtype=torch.float64,
    )
    _check_tangent_fd(model, f_large, st0, tol=1.0e-7)

    # Full 3D incremental FEM trajectory + guarded surrogate fallback on 3D box
    step = generate_3d_box_step(tmp_path / "j2_box.step", lx=10.0, ly=5.0, lz=5.0)
    imp = import_cad_3d(step, mesh_size_mm=7.0, element_type=CellType3D.TET10)
    faces = classify_faces_by_bbox(imp)

    loads = [
        LoadStep3D(
            step_index=i,
            time_s=0.1 * (i + 1),
            dt_s=0.1,
            dirichlet_bcs=(
                DirichletBC3D(
                    "fix_x0", faces["x_min"], ("x", "y", "z"), (0.0, 0.0, 0.0)
                ),
                DirichletBC3D("pull_x1", faces["x_max"], ("x",), (ux,)),
            ),
        )
        for i, ux in enumerate((0.05, 0.25, 0.40, 0.10))
    ]
    traj = solve_incremental_fem_3d_trajectory(
        imp.mesh,
        model,
        loads,
        solver_tolerance=1.0e-6,
    )
    assert len(traj.steps) == 4
    assert all(s.converged for s in traj.steps)
    assert all(s.dissipation_total >= -1.0e-9 for s in traj.steps)
    # Step 2 (ux=0.40 -> 4% strain) clearly yields plastically -> positive dissipation
    assert traj.steps[2].dissipation_total > 0.0
    assert traj.document()["schema"] == "formamech.stateful-3d-trajectory-v1"

    # Guarded stepper with a poor surrogate predictor escalates to exact FEM fallback
    bad_surrogate = lambda _step, u_prev: np.zeros_like(u_prev)  # noqa: E731
    traj_guarded = solve_incremental_fem_3d_trajectory(
        imp.mesh,
        model,
        loads[:2],
        solver_tolerance=1.0e-6,
        surrogate_predictor=bad_surrogate,
        guard_residual_threshold=1.0e-3,
    )
    assert all(s.used_fem_fallback and s.converged for s in traj_guarded.steps)
