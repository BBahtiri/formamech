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


def test_v4_3_audit_regressions_uts_neuber_periodic_rainflow_and_miehe_compression() -> (
    None
):
    """Regressions for UTS rupture guard, Neuber Coffin-Manson, periodic rainflow, and Miehe compression."""
    import math

    import numpy as np

    mat = FatigueMaterialSpec(
        name="steel-4340",
        ultimate_tensile_mpa=600.0,
        fatigue_strength_coeff_mpa=1000.0,
        fatigue_strength_exp=-0.1,
        youngs_modulus_mpa=210_000.0,
        fatigue_ductility_coeff=0.5,
        fatigue_ductility_exp=-0.6,
    )
    # 1) Static UTS rupture triggers infinite damage even with mean_stress_model=NONE
    res_uts = evaluate_fatigue_life(
        [0.0, 650.0, 0.0],
        mat,
        model=FatigueModel.BASQUIN_SN,
        mean_stress_model=MeanStressModel.NONE,
    )
    assert not math.isfinite(res_uts.miner_damage)
    assert res_uts.status == "FATIGUE_FAILURE_PREDICTED"
    assert res_uts.document()["miner_damage"] is None

    # 2) Periodic repeating block closes all cycles to full count=1.0
    cycles_rep = rainflow_count_astm_e1049(
        [0.0, 100.0, -50.0, 200.0, 50.0, 150.0, 0.0], repeating_block=True
    )
    assert all(c.count == 1.0 for c in cycles_rep)
    assert any(abs(c.range_mpa - 250.0) < 1.0e-12 for c in cycles_rep)

    # 3) Neuber+Ramberg-Osgood Coffin-Manson predicts strictly higher LCF damage than pure elastic strain
    from piginot.stateful.fatigue_fracture import (
        _coffin_manson_cycles,
        _neuber_elastoplastic_strain_amp,
    )

    sig_lcf = 520.0
    eps_neuber = _neuber_elastoplastic_strain_amp(sig_lcf, mat)
    eps_elastic = sig_lcf / mat.youngs_modulus_mpa
    assert eps_neuber > eps_elastic * 1.1
    assert _coffin_manson_cycles(eps_neuber, mat) < _coffin_manson_cycles(
        eps_elastic, mat
    )

    # 4) Exact Miehe spectral split preserves compressive stiffness on a pre-cracked (d=0.99) solid
    mesh = make_brick_tet4_mesh(10.0, 4.0, 4.0, nx=2, ny=1, nz=1)
    n_nodes = len(mesh.nodes)
    d_precrack = np.full(n_nodes, 0.99, dtype=np.float64)
    cfg = PhaseFieldConfig(
        youngs_modulus_mpa=20_000.0,
        poisson_ratio=0.25,
        fracture_toughness_gc_n_per_mm=1.0,
        length_scale_lc_mm=2.0,
        load_steps=2,
        staggered_iters=2,
    )
    # Laterally confined compression: intact vs d=0.99 pre-cracked must yield identical displacement
    bcs_comp = (
        DirichletBC3D("x0", ("x_min",), ("x",), (0.0,)),
        DirichletBC3D("y0", ("y_min", "y_max"), ("y",), (0.0,)),
        DirichletBC3D("z0", ("z_min", "z_max"), ("z",), (0.0,)),
        DirichletBC3D("x1", ("x_max",), ("x",), (-0.05,)),
    )
    res_intact = solve_phase_field_fracture(mesh, dirichlet_bcs=bcs_comp, config=cfg)
    res_cracked = solve_phase_field_fracture(
        mesh,
        dirichlet_bcs=bcs_comp,
        config=cfg,
        initial_damage_nodal=d_precrack,
    )
    assert np.allclose(
        res_cracked.final_displacement_mm,
        res_intact.final_displacement_mm,
        atol=1.0e-9,
    )


def test_phase_field_warm_start_preserves_exact_fem_solution_and_cuts_iterations() -> (
    None
):
    """Warm-starting 3D phase-field FEM matches exact FEM equilibrium and cuts iterations."""
    import numpy as np

    mesh = make_brick_tet4_mesh(12.0, 6.0, 2.0, nx=8, ny=4, nz=1)
    nodes = np.asarray(mesh.nodes)
    d0 = np.zeros(len(nodes), dtype=np.float64)
    d0[(np.abs(nodes[:, 0] - 6.0) <= 0.8) & (nodes[:, 1] <= 2.0)] = 1.0
    cfg = PhaseFieldConfig(
        youngs_modulus_mpa=20_000.0,
        poisson_ratio=0.25,
        fracture_toughness_gc_n_per_mm=0.5,
        length_scale_lc_mm=1.2,
        load_steps=6,
        staggered_iters=50,
        tolerance=1.0e-4,
    )
    bcs = (
        DirichletBC3D("x0", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        DirichletBC3D("x1", ("x_max",), ("x",), (0.08,)),
    )
    res_cold = solve_phase_field_fracture(
        mesh,
        dirichlet_bcs=bcs,
        config=cfg,
        initial_damage_nodal=d0,
        record_history=True,
    )
    assert res_cold.total_newton_iterations > 0
    assert res_cold.total_staggered_iterations >= cfg.load_steps
    assert len(res_cold.damage_snapshots) == cfg.load_steps
    assert len(res_cold.displacement_snapshots) == cfg.load_steps
    assert len(res_cold.reaction_force_history_n) == cfg.load_steps
    assert max(res_cold.reaction_force_history_n) > 0.0

    # Slightly noisy oracle warm-start predictor (~1% noise) must cut iterations while
    # converging to the exact same physical crack & displacement solution
    rng = np.random.default_rng(7)
    step_map = {
        round(lam, 6): (u_s, d_s)
        for lam, u_s, d_s in zip(
            res_cold.load_factors,
            res_cold.displacement_snapshots,
            res_cold.damage_snapshots,
            strict=True,
        )
    }

    def _noisy_predictor(
        lam: float, _u_prev: np.ndarray, _d_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        u_ref, d_ref = step_map[round(lam, 6)]
        u_noisy = u_ref + 0.01 * np.max(np.abs(u_ref)) * rng.normal(size=u_ref.shape)
        d_noisy = np.clip(d_ref + 0.01 * rng.normal(size=d_ref.shape), 0.0, 1.0)
        return u_noisy, d_noisy

    res_warm = solve_phase_field_fracture(
        mesh,
        dirichlet_bcs=bcs,
        config=cfg,
        initial_damage_nodal=d0,
        warm_start_predictor=_noisy_predictor,
    )
    assert res_warm.monotone_irreversible is True
    assert res_warm.total_staggered_iterations <= res_cold.total_staggered_iterations
    assert res_warm.total_newton_iterations < res_cold.total_newton_iterations
    assert np.allclose(
        res_warm.final_damage_nodal, res_cold.final_damage_nodal, atol=5.0e-3
    )
    assert np.allclose(
        res_warm.final_displacement_mm, res_cold.final_displacement_mm, atol=5.0e-4
    )

    # Deliberate massive overshoot warm-start guess must NOT permanently poison irreversibility
    def _bad_overshoot(
        _lam: float, _u_prev: np.ndarray, _d_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros_like(nodes), np.full(len(nodes), 0.95, dtype=np.float64)

    res_guarded = solve_phase_field_fracture(
        mesh,
        dirichlet_bcs=(
            DirichletBC3D("x0", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
            DirichletBC3D("x1", ("x_max",), ("x",), (1.0e-5,)),
        ),
        config=PhaseFieldConfig(
            youngs_modulus_mpa=20_000.0,
            poisson_ratio=0.25,
            fracture_toughness_gc_n_per_mm=10.0,
            length_scale_lc_mm=1.2,
            load_steps=2,
            staggered_iters=6,
            tolerance=1.0e-5,
        ),
        warm_start_predictor=_bad_overshoot,
    )
    assert res_guarded.monotone_irreversible is True
    assert float(np.max(res_guarded.final_damage_nodal)) < 1.0e-3
