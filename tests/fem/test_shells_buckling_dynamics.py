"""Analytical & fail-closed verification tests for Phase 3 (V4-3) shells, beams, buckling, Riks & dynamics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import DirichletBC3D, TractionBC3D
from piginot.fem.buckling_3d import (
    RiksConfig,
    RiksSolverState,
    solve_3d_linear_buckling,
    solve_3d_truss_riks,
)
from piginot.fem.dynamics_3d import (
    CflStabilityError,
    DynamicsScheme,
    TransientDynamicsConfig,
    solve_3d_transient_dynamics,
)
from piginot.fem.shells_beams_3d import (
    BeamSection3D,
    ShellDirichletBC,
    make_rectangular_plate_shell_mesh,
    solve_3d_dkt_shell,
    solve_3d_timoshenko_beam,
)
from piginot.geometry.mesh3d import make_brick_tet4_mesh, make_brick_tet10_mesh


def test_dkt_shell_thin_clamped_and_ss_plate_no_shear_locking() -> None:
    """DKT-CST thin square plate (t/L = 1/50) matches Kirchhoff/Roark within < 2%."""
    length = 100.0
    thickness = 2.0  # t/L = 1/50 (thin plate regime)
    e_mpa = 210_000.0
    nu = 0.3
    pressure = 0.01  # MPa
    d_flex = e_mpa * (thickness**3) / (12.0 * (1.0 - nu * nu))

    mesh = make_rectangular_plate_shell_mesh(
        length, length, nx=12, ny=12, thickness_mm=thickness
    )
    # 1) All-edges clamped: w_max = 0.0012653 * q * L^4 / D
    w_exact_clamped = 0.0012653 * pressure * (length**4) / d_flex
    res_c = solve_3d_dkt_shell(
        mesh,
        youngs_modulus_mpa=e_mpa,
        poisson_ratio=nu,
        dirichlet_bcs=(ShellDirichletBC("clamp", "perimeter"),),
        uniform_pressure_mpa=pressure,
    )
    rel_err_c = abs(res_c.max_deflection_z_mm - w_exact_clamped) / w_exact_clamped
    assert rel_err_c < 0.02, f"clamped DKT error {rel_err_c:.4f} >= 2%"

    # 2) All-edges simply supported: w_max = 0.0040624 * q * L^4 / D
    w_exact_ss = 0.0040624 * pressure * (length**4) / d_flex
    res_ss = solve_3d_dkt_shell(
        mesh,
        youngs_modulus_mpa=e_mpa,
        poisson_ratio=nu,
        dirichlet_bcs=(
            ShellDirichletBC(
                "ss", "perimeter", components=(0, 1, 2), values=(0.0, 0.0, 0.0)
            ),
        ),
        uniform_pressure_mpa=pressure,
    )
    rel_err_ss = abs(res_ss.max_deflection_z_mm - w_exact_ss) / w_exact_ss
    assert rel_err_ss < 0.02, f"simply-supported DKT error {rel_err_ss:.4f} >= 2%"


def test_timoshenko_beam_matches_exact_analytical_deflection() -> None:
    """3D Timoshenko cantilever beam matches bending + shear deflection to < 1e-10."""
    length = 60.0
    b, h = 12.0, 18.0  # stocky beam where shear deflection is significant
    e_mpa = 200_000.0
    nu = 0.3
    g_mpa = e_mpa / (2.0 * (1.0 + nu))
    sec = BeamSection3D.rectangular(b, h)
    tip_fy = 1500.0
    tip_fz = -800.0

    nodes = np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]], dtype=np.float64)
    elems = np.array([[0, 1]], dtype=np.int64)
    loads = np.zeros((2, 6), dtype=np.float64)
    loads[1, 1] = tip_fy
    loads[1, 2] = tip_fz

    res = solve_3d_timoshenko_beam(
        nodes,
        elems,
        section=sec,
        youngs_modulus_mpa=e_mpa,
        poisson_ratio=nu,
        constrained_dofs={c: 0.0 for c in range(6)},
        nodal_loads_n=loads,
    )
    v_exact = (tip_fy * (length**3)) / (3.0 * e_mpa * sec.iz_mm4) + (
        tip_fy * length
    ) / (sec.shear_factor_ky * g_mpa * sec.area_mm2)
    w_exact = (tip_fz * (length**3)) / (3.0 * e_mpa * sec.iy_mm4) + (
        tip_fz * length
    ) / (sec.shear_factor_kz * g_mpa * sec.area_mm2)

    assert abs(res.nodal_dofs[1, 1] - v_exact) / abs(v_exact) < 1.0e-10
    assert abs(res.nodal_dofs[1, 2] - w_exact) / abs(w_exact) < 1.0e-10


def test_3d_linear_buckling_matches_euler_column() -> None:
    """Slender clamped-free 3D TET10 column matches Euler P_cr = pi^2 E I / (4 L^2) within < 3%."""
    length = 100.0
    b = 6.0
    h = 6.0
    e_mpa = 5000.0
    nu = 0.0  # pure uniaxial comparison without Poisson anticlastic stiffening
    area = b * h
    inertia = b * (h**3) / 12.0
    p_euler = (math.pi**2) * e_mpa * inertia / (4.0 * length * length)

    mesh = make_brick_tet10_mesh(length, b, h, nx=8, ny=1, nz=1)
    mat = MaterialSpec(
        name="euler-col",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(e_mpa, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(nu, Unit.ONE).to_canonical(),
        },
    )
    sigma_ref = 1.0  # MPa => P_ref = area * sigma_ref = 36 N
    res = solve_3d_linear_buckling(
        mesh,
        mat,
        dirichlet_bcs=(
            DirichletBC3D("clamp", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        traction_bcs=(TractionBC3D("compress", ("x_max",), (-sigma_ref, 0.0, 0.0)),),
        num_modes=2,
    )
    p_cr_fem = res.load_factors[0] * (sigma_ref * area)
    rel_err = abs(p_cr_fem - p_euler) / p_euler
    assert rel_err < 0.03, (
        f"Euler buckling error {rel_err:.4f} >= 3% (FEM={p_cr_fem:.2f}, Euler={p_euler:.2f})"
    )


def test_crisfield_riks_tracks_snap_through_past_limit_point() -> None:
    """Crisfield spherical arc-length passes shallow-arch snap-through limit point with d_lambda < 0."""
    # Symmetric 2-bar shallow arch (Von Mises truss) in 3D:
    # Base pins at (-L, 0, 0) and (+L, 0, 0), apex at (0, H, 0), loaded downward at apex.
    half_span = 10.0
    rise = 1.2
    ea = 1000.0
    nodes = np.array(
        [[-half_span, 0.0, 0.0], [0.0, rise, 0.0], [half_span, 0.0, 0.0]],
        dtype=np.float64,
    )
    bars = np.array([[0, 1], [1, 2]], dtype=np.int64)
    # Constrain base nodes (0, 2) completely and apex x, z so apex moves vertically in y
    constrained = [0, 1, 2, 3, 5, 6, 7, 8]
    f_ref = np.zeros(9, dtype=np.float64)
    f_ref[4] = -1.0  # downward unit load at apex

    res = solve_3d_truss_riks(
        nodes,
        bars,
        ea_n=ea,
        constrained_dofs=constrained,
        f_ref_n=f_ref,
        config=RiksConfig(arc_length=0.2, psi=0.5, max_steps=28, tolerance=1.0e-10),
    )
    assert res.state is RiksSolverState.CONVERGED
    assert res.limit_point_passed is True
    assert max(s.residual_norm for s in res.steps) < 1.0e-9
    # Verify equilibrium path rises to first snap-through limit point, descends
    # through unstable branch (d_lambda < 0) into negative load, then recovers
    lam_h = res.load_factor_history
    d_lam = np.diff(lam_h)
    first_peak_idx = int(np.nonzero(d_lam < 0.0)[0][0])
    first_limit_load = float(lam_h[first_peak_idx])
    assert float(np.min(lam_h)) < -0.5  # full snap-through inversion captured
    assert float(lam_h[-1]) > first_limit_load  # secondary stable ascending branch
    # Exact analytical Green-Lagrange snap-through peak load:
    # P(w) = 2 * (EA / L0^3) * (H - w) * (H*w - 0.5*w^2), peak at w = H*(1 - 1/sqrt(3)):
    # P_max = (2 / (3 sqrt(3))) * EA * H^3 / L0^3
    l0 = math.hypot(half_span, rise)
    p_exact_peak = (2.0 / (3.0 * math.sqrt(3.0))) * ea * (rise**3) / (l0**3)
    assert abs(first_limit_load - p_exact_peak) / p_exact_peak < 0.02


def test_transient_dynamics_newmark_energy_conservation_hht_and_cfl_guard() -> None:
    """Newmark conserves free-vibration energy; HHT damps cleanly; Explicit refuses CFL > 1."""
    mesh = make_brick_tet4_mesh(20.0, 5.0, 5.0, nx=3, ny=1, nz=1)
    mat = MaterialSpec(
        name="dyn-steel",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(2000.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.3, Unit.ONE).to_canonical(),
        },
    )
    clamp = (DirichletBC3D("clamp", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),)
    # Smooth initial axial displacement u_x(x) = 0.02 * (x / 20)
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    u0 = np.zeros_like(nodes)
    u0[:, 0] = 0.02 * (nodes[:, 0] / 20.0)

    # 1) Undamped Newmark average acceleration: exact mechanical energy conservation
    res_nm = solve_3d_transient_dynamics(
        mesh,
        mat,
        dirichlet_bcs=clamp,
        initial_displacement_mm=u0,
        config=TransientDynamicsConfig(
            scheme=DynamicsScheme.NEWMARK_BETA,
            dt_s=2.0e-5,
            num_steps=30,
            density_t_per_mm3=7.85e-9,
        ),
    )
    assert res_nm.total_energy_mj[0] > 0.0
    assert res_nm.energy_relative_drift < 1.0e-5

    # 2) HHT-alpha dissipates high-frequency energy stably (E_final <= E_initial)
    res_hht = solve_3d_transient_dynamics(
        mesh,
        mat,
        dirichlet_bcs=clamp,
        initial_displacement_mm=u0,
        config=TransientDynamicsConfig(
            scheme=DynamicsScheme.HHT_ALPHA,
            dt_s=2.0e-5,
            num_steps=30,
            hht_alpha=-0.1,
            density_t_per_mm3=7.85e-9,
        ),
    )
    assert res_hht.total_energy_mj[-1] <= res_hht.total_energy_mj[0] * (1.0 + 1.0e-9)

    # 3) Explicit central-difference: refuses dt above CFL limit, succeeds well below
    with pytest.raises(CflStabilityError):
        solve_3d_transient_dynamics(
            mesh,
            mat,
            dirichlet_bcs=clamp,
            initial_displacement_mm=u0,
            config=TransientDynamicsConfig(
                scheme=DynamicsScheme.EXPLICIT_CENTRAL_DIFF,
                dt_s=1.0e-2,  # way above CFL limit
                num_steps=5,
                density_t_per_mm3=7.85e-9,
            ),
        )

    res_exp = solve_3d_transient_dynamics(
        mesh,
        mat,
        dirichlet_bcs=clamp,
        initial_displacement_mm=u0,
        config=TransientDynamicsConfig(
            scheme=DynamicsScheme.EXPLICIT_CENTRAL_DIFF,
            dt_s=1.0e-6,
            num_steps=20,
            density_t_per_mm3=7.85e-9,
        ),
    )
    assert res_exp.dt_crit_s is not None and res_exp.dt_crit_s > 1.0e-6
    assert math.isfinite(res_exp.total_energy_mj[-1])
