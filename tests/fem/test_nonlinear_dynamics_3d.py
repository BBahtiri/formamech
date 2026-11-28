"""Full FEM tests for V6-R3: Large-deformation 3D nonlinear transient dynamics."""

from __future__ import annotations

import numpy as np

from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import DirichletBC3D, TractionBC3D
from piginot.fem.nonlinear_dynamics_3d import (
    NonlinearDynamics3DConfig,
    NonlinearDynamicsScheme,
    solve_3d_nonlinear_dynamics,
)
from piginot.geometry.mesh3d import make_brick_tet10_mesh


def _elastomer() -> MaterialSpec:
    return MaterialSpec(
        name="dyn-elastomer",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(5.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.40, Unit.ONE).to_canonical(),
        },
    )


def test_nonlinear_newmark_undamped_free_vibration_conserves_energy() -> None:
    """Undamped finite-strain 3D TET10 Newmark-beta preserves total mechanical energy."""
    mesh = make_brick_tet10_mesh(20.0, 6.0, 6.0, nx=3, ny=1, nz=1)
    n_nodes = len(mesh.nodes)
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    # Smooth initial transverse velocity along cantilever length
    v0 = np.zeros((n_nodes, 3), dtype=np.float64)
    v0[:, 2] = 400.0 * (nodes[:, 0] / 20.0)

    cfg = NonlinearDynamics3DConfig(
        scheme=NonlinearDynamicsScheme.NEWMARK_BETA,
        dt_s=2.5e-4,
        num_steps=12,
        density_t_per_mm3=1.2e-9,
        rayleigh_alpha=0.0,
        rayleigh_beta=0.0,
    )
    res = solve_3d_nonlinear_dynamics(
        mesh,
        _elastomer(),
        dirichlet_bcs=(
            DirichletBC3D("clamp", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        config=cfg,
        initial_velocity_mm_s=v0,
    )
    assert res.converged
    assert float(np.min(res.min_det_f_history)) > 0.7
    assert res.total_energy_mj[0] > 0.0
    assert res.energy_relative_drift < 0.05
    assert len(res.document()["result_hash"]) == 64


def test_nonlinear_hht_alpha_with_damping_and_dynamic_traction() -> None:
    """Nonlinear HHT-alpha integrates forced + damped finite-strain 3D response."""
    mesh = make_brick_tet10_mesh(16.0, 6.0, 6.0, nx=2, ny=1, nz=1)
    cfg = NonlinearDynamics3DConfig(
        scheme=NonlinearDynamicsScheme.HHT_ALPHA,
        dt_s=5.0e-4,
        num_steps=10,
        density_t_per_mm3=1.2e-9,
        rayleigh_alpha=5.0,
        rayleigh_beta=1.0e-4,
        hht_alpha=-0.05,
    )
    res = solve_3d_nonlinear_dynamics(
        mesh,
        _elastomer(),
        dirichlet_bcs=(
            DirichletBC3D("clamp", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        traction_bcs=(TractionBC3D("tip", ("x_max",), (0.0, 0.0, -0.05)),),
        load_multiplier=lambda t: 1.0 if t <= 2.5e-3 else 0.0,
        config=cfg,
    )
    assert res.converged
    assert float(np.max(np.abs(res.displacement_history_mm[:, :, 2]))) > 0.02
    # After load release (step 5 -> step 10), Rayleigh + HHT damping reduces total energy
    assert res.total_energy_mj[-1] < res.total_energy_mj[5]
