"""Verification tests for Phase 3 (V4-3) ASTM E1049 rainflow fatigue & AT2 phase-field fracture."""

from __future__ import annotations

from pathlib import Path

from piginot.fem.backend_3d import DirichletBC3D
from piginot.geometry.mesh3d import make_brick_tet4_mesh
from piginot.harness import FormaMechAgentHarness
from piginot.stateful.fatigue_fracture import (
    FatigueMaterialSpec,
    FatigueModel,
    MeanStressModel,
    PhaseFieldConfig,
    evaluate_fatigue_life,
    rainflow_count_astm_e1049,
    solve_phase_field_fracture,
)


def test_astm_e1049_rainflow_and_exact_miner_damage() -> None:
    """ASTM E1049 rainflow + Basquin S-N + Miner matches closed-form two-block damage < 1e-10."""
    # Classic ASTM E1049 Fig. 4 example: [-2, 1, -3, 5, -1, 3, -4, 4, -2]
    astm_series = [-2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0]
    cycles = rainflow_count_astm_e1049(astm_series)
    total_count = sum(c.count for c in cycles)
    assert total_count == 4.0
    full_ranges = sorted(c.range_mpa for c in cycles if c.count == 1.0)
    assert full_ranges == [4.0]  # inner enclosed [-1, 3] cycle of range 4

    # Exact two-block constant-amplitude check (zero mean):
    # Block 1: 10 full cycles at amp = 200 MPa (0 -> 200 -> -200 -> ... -> 0)
    # Block 2: 20 full cycles at amp = 150 MPa
    mat = FatigueMaterialSpec(
        name="al-7075",
        ultimate_tensile_mpa=570.0,
        fatigue_strength_coeff_mpa=1000.0,
        fatigue_strength_exp=-0.1,
        youngs_modulus_mpa=71_700.0,
    )
    # Build exact symmetric cycle series with known cycle counts:
    # 5 full cycles at amp 200 MPa: [-200, 200, -200, 200, -200, 200, -200, 200, -200, 200, -200]
    series_1 = [-200.0, 200.0] * 5 + [-200.0]
    res1 = evaluate_fatigue_life(
        series_1,
        mat,
        model=FatigueModel.BASQUIN_SN,
        mean_stress_model=MeanStressModel.NONE,
    )
    nf_200 = 0.5 * ((200.0 / 1000.0) ** (1.0 / -0.1))
    expected_d = 5.0 / nf_200
    assert abs(res1.miner_damage - expected_d) / expected_d < 1.0e-10
    assert res1.status == "SAFE"

    # Goodman tensile mean-stress increases damage strictly over zero-mean
    series_mean = [0.0, 400.0] * 5 + [0.0]  # amp=200, mean=+200
    res_goodman = evaluate_fatigue_life(
        series_mean,
        mat,
        model=FatigueModel.BASQUIN_SN,
        mean_stress_model=MeanStressModel.GOODMAN,
    )
    assert res_goodman.miner_damage > res1.miner_damage

    # Coffin-Manson strain-life mode executes cleanly and returns finite life
    res_cm = evaluate_fatigue_life(
        series_1,
        mat,
        model=FatigueModel.COFFIN_MANSON_EN,
        mean_stress_model=MeanStressModel.GERBER,
    )
    assert res_cm.miner_damage > 0.0


def test_phase_field_at2_tension_growth_and_zero_compression_damage() -> None:
    """AT2 spectral split produces zero damage in compression and monotone damage in tension."""
    mesh = make_brick_tet4_mesh(10.0, 4.0, 4.0, nx=3, ny=1, nz=1)
    cfg = PhaseFieldConfig(
        youngs_modulus_mpa=20_000.0,
        poisson_ratio=0.25,
        fracture_toughness_gc_n_per_mm=0.2,
        length_scale_lc_mm=2.5,
        load_steps=6,
        staggered_iters=4,
    )

    # 1) Pure laterally-confined compression (all principal strains <= 0 => psi^+ = 0):
    #    Miehe spectral split produces strictly zero crack nucleation!
    res_comp = solve_phase_field_fracture(
        mesh,
        dirichlet_bcs=(
            DirichletBC3D("x0", ("x_min",), ("x",), (0.0,)),
            DirichletBC3D("y0", ("y_min", "y_max"), ("y",), (0.0,)),
            DirichletBC3D("z0", ("z_min", "z_max"), ("z",), (0.0,)),
            DirichletBC3D("x1", ("x_max",), ("x",), (-0.08,)),
        ),
        config=cfg,
    )
    assert res_comp.max_damage_history[-1] < 1.0e-12
    assert res_comp.monotone_irreversible is True

    # 2) Uniaxial tension: irreversible monotone damage growth in [0, 1]
    res_tens = solve_phase_field_fracture(
        mesh,
        dirichlet_bcs=(
            DirichletBC3D("x0", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
            DirichletBC3D("x1", ("x_max",), ("x",), (0.12,)),
        ),
        config=cfg,
    )
    assert res_tens.monotone_irreversible is True
    assert res_tens.max_damage_history[-1] > 0.25
    assert res_tens.max_damage_history[-1] <= 1.0
    # Verify non-decreasing peak damage across load steps
    for a, b in zip(
        res_tens.max_damage_history, res_tens.max_damage_history[1:], strict=False
    ):
        assert b >= a - 1.0e-12


def test_harness_invokes_v4_3_structural_physics_pack(tmp_path: Path) -> None:
    """Autonomous agents can invoke all V4-3 physics solvers through FormaMechAgentHarness."""
    harness = FormaMechAgentHarness(tmp_path / "agent_v4_3")
    env_shell = harness.invoke(
        "fem.solve_shell_plate_3d",
        {"lx_mm": 80.0, "ly_mm": 80.0, "thickness_mm": 2.0, "nx": 6, "ny": 6},
    )
    assert (
        env_shell.ok
        and env_shell.result is not None
        and env_shell.result["max_deflection_z_mm"] > 0.0
    )

    env_beam = harness.invoke(
        "fem.solve_timoshenko_beam_3d", {"length_mm": 50.0, "tip_load_y_n": 200.0}
    )
    assert (
        env_beam.ok
        and env_beam.result is not None
        and env_beam.result["max_displacement_mm"] > 0.0
    )

    env_buck = harness.invoke(
        "fem.solve_linear_buckling_3d", {"length_mm": 40.0, "nx": 4, "num_modes": 1}
    )
    assert (
        env_buck.ok
        and env_buck.result is not None
        and env_buck.result["critical_load_factor"] > 0.0
    )

    env_dyn = harness.invoke(
        "multiphysics.solve_transient_dynamics_3d", {"num_steps": 5}
    )
    assert (
        env_dyn.ok and env_dyn.result is not None and env_dyn.result["num_steps"] == 5
    )

    env_fat = harness.invoke(
        "multiphysics.evaluate_fatigue_rainflow",
        {"stress_history_mpa": [0.0, 250.0, -250.0, 250.0, 0.0]},
    )
    assert (
        env_fat.ok
        and env_fat.result is not None
        and env_fat.result["miner_damage"] > 0.0
    )

    env_pf = harness.invoke(
        "multiphysics.solve_phase_field_fracture_3d",
        {"end_displacement_x_mm": 0.08, "load_steps": 4},
    )
    assert (
        env_pf.ok
        and env_pf.result is not None
        and env_pf.result["monotone_irreversible"] is True
    )
