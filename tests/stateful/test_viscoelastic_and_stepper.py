"""Tests for Feature F0/F1: finite-strain viscoelasticity, FEM stepper, and guards."""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.contracts.expressions import DerivedExpression
from piginot.contracts.stateful import (
    BoundTrajectoryState,
    LoadSegment,
    StatefulVerificationPolicy,
    TimeGrid,
)
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh
from piginot.physics import consistent_tangent, energy_density, first_piola
from piginot.stateful import (
    STEP_CONTROLLER_VERSION,
    CandidateStepOutput,
    DirichletHistoryBC,
    GeneralizedMaxwellConfig,
    GeneralizedMaxwellViscoelastic,
    GuardedStatefulStepper,
    MaxwellBranch,
    StatelessHyperelasticModel,
    StepStatus,
    hash_state_tensors,
    load_trajectory_checkpoint,
    save_trajectory_checkpoint,
    solve_incremental_fem_trajectory,
    verify_algorithmic_tangent,
)
from tests.platform_support import windows_safe_environment

ROOT = Path(__file__).resolve().parents[2]


def _visco_model() -> GeneralizedMaxwellViscoelastic:
    cfg = GeneralizedMaxwellConfig(
        equilibrium_modulus_mpa=100.0,
        poisson_ratio=0.3,
        branches=(
            MaxwellBranch(stiffness_ratio=0.6, relaxation_time=0.5),
            MaxwellBranch(stiffness_ratio=0.4, relaxation_time=2.5),
        ),
    )
    return GeneralizedMaxwellViscoelastic(cfg)


def _rect_mesh() -> object:
    return canonicalize_mesh(
        MeshPayload(
            ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)),
            ((0, 1, 2), (0, 2, 3)),
            CellType.TRIANGLE,
            ((0, 1), (1, 2), (2, 3), (3, 0)),
            {"bottom": (0,), "right": (1,), "top": (2,), "left": (3,)},
            (1, 1),
            Unit.MM,
            {"kind": "visco-plate"},
        )
    )


def test_stateful_runtime_imports_without_fem_packages() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in ("scipy", "skfem", "meshio", "sympy", "gmsh"):
                    raise ImportError("blocked " + name)
        sys.meta_path.insert(0, Block())
        from formamech.stateful import (
            GeneralizedMaxwellConfig,
            GeneralizedMaxwellViscoelastic,
            GuardedStatefulStepper,
            MaxwellBranch,
        )
        m = GeneralizedMaxwellViscoelastic(
            GeneralizedMaxwellConfig(100.0, 0.3, (MaxwellBranch(0.5, 1.0),))
        )
        assert m.state_schema.model_id == "generalized-maxwell-finite-strain-v1"
        """
    )
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=windows_safe_environment({"PYTHONPATH": str(ROOT / "src")}),
    )
    assert res.returncode == 0, res.stderr


def test_stateless_neo_hookean_adapter_is_bit_identical() -> None:
    mat = MaterialSpec(
        name="nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(100.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.3, Unit.ONE).to_canonical(),
        },
    )
    adapter = StatelessHyperelasticModel(mat)
    f = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(4, 1, 1)
    f[:, 0, 0] = torch.tensor([1.05, 1.15, 0.95, 1.2], dtype=torch.float64)
    f[:, 0, 1] = torch.tensor([0.02, -0.05, 0.03, 0.08], dtype=torch.float64)
    upd = adapter.update(f, f, adapter.initial_state((4,)), dt=0.1)
    assert upd.admissible is True
    assert torch.equal(upd.stress, first_piola(f, mat))
    assert torch.equal(upd.algorithmic_tangent, consistent_tangent(f, mat))
    assert torch.equal(upd.free_energy, energy_density(f, mat))
    assert float(upd.dissipation.abs().max()) == 0.0


def test_viscoelastic_frame_indifference_variational_potential_and_tangent() -> None:
    model = _visco_model()
    gen = torch.Generator().manual_seed(42)
    f_old = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(5, 1, 1)
    f_step1 = f_old + 0.15 * torch.randn((5, 3, 3), generator=gen, dtype=torch.float64)
    f_step2 = f_step1 + 0.10 * torch.randn(
        (5, 3, 3), generator=gen, dtype=torch.float64
    )

    s0 = model.initial_state((5,))
    u1 = model.update(f_old, f_step1, s0, dt=0.2)
    assert u1.admissible is True
    assert bool((u1.dissipation >= 0.0).all().item())

    # 1. Exact algorithmic tangent check against central finite differences (< 1e-6 << 1e-4).
    tcheck = verify_algorithmic_tangent(
        model, f_step1, f_step2, u1.state_new, dt=0.25, eps=1e-6, rel_tol=1e-4
    )
    assert tcheck.passed is True
    assert tcheck.relative_error < 1e-6

    # 2. Exact incremental variational potential gradient: dW_inc / dF_new == P_new.
    f_var = f_step2.clone().requires_grad_(True)
    u2_var = model.update(f_step1, f_var, u1.state_new, dt=0.25)
    (grad_w,) = torch.autograd.grad(u2_var.incremental_potential.sum(), f_var)
    assert torch.allclose(grad_w, u2_var.stress.detach(), atol=1e-11, rtol=1e-11)

    # 3. Exact finite-strain frame indifference under arbitrary 3D rotation Q in SO(3).
    theta = 0.73
    q_rot = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    f_rot = torch.matmul(q_rot, f_step2)
    u2 = model.update(f_step1, f_step2, u1.state_new, dt=0.25)
    u2_rot = model.update(f_step1, f_rot, u1.state_new, dt=0.25)
    assert torch.allclose(u2_rot.free_energy, u2.free_energy, atol=1e-12, rtol=1e-12)
    assert torch.allclose(u2_rot.dissipation, u2.dissipation, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        u2_rot.state_new["viscous_strain"],
        u2.state_new["viscous_strain"],
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        u2_rot.stress, torch.matmul(q_rot, u2.stress), atol=1e-12, rtol=1e-12
    )


def test_viscoelastic_small_strain_relaxation_and_cyclic_dissipation() -> None:
    # Single-branch Prony series in small-strain uniaxial plane stress.
    e_inf = 100.0
    gamma = 0.5
    tau = 1.0
    model = GeneralizedMaxwellViscoelastic(
        GeneralizedMaxwellConfig(
            equilibrium_modulus_mpa=e_inf,
            poisson_ratio=0.3,
            branches=(MaxwellBranch(stiffness_ratio=gamma, relaxation_time=tau),),
        )
    )
    eps0 = 1e-4
    f2d_hold = torch.tensor(
        [[[1.0 + eps0, 0.0], [0.0, 1.0 - 0.3 * eps0]]], dtype=torch.float64
    )
    f2d_id = torch.eye(2, dtype=torch.float64).unsqueeze(0)

    # Instantaneous (dt -> 0) and relaxed (dt -> inf) modulus limits.
    ps_inst = model.plane_stress_update(
        f2d_id, f2d_hold, model.initial_state((1,)), dt=1e-8
    )
    assert ps_inst.closure_converged and ps_inst.closure_residual_max < 1e-11
    sig_inst = float(ps_inst.update_3d.stress[0, 0, 0].item())
    assert sig_inst / eps0 == pytest.approx(e_inf * (1.0 + gamma), rel=1e-3)

    ps_eq = model.plane_stress_update(
        f2d_id, f2d_hold, model.initial_state((1,)), dt=1e8
    )
    sig_eq = float(ps_eq.update_3d.stress[0, 0, 0].item())
    assert sig_eq / eps0 == pytest.approx(e_inf, rel=1e-3)

    # Hold relaxation at t = 1.0s converges to analytic E_inf * (1 + gamma * exp(-t/tau)) * eps0.
    errs = []
    for n_steps in (20, 80):
        dt = 1.0 / n_steps
        state = model.initial_state((1,))
        f_prev = f2d_id
        stress_final = 0.0
        for _ in range(n_steps):
            ps = model.plane_stress_update(f_prev, f2d_hold, state, dt=dt)
            assert ps.closure_converged
            assert float(ps.update_3d.dissipation.min().item()) >= 0.0
            state = dict(ps.update_3d.state_new)
            f_prev = f2d_hold
            stress_final = float(ps.update_3d.stress[0, 0, 0].item())
        analytic = e_inf * (1.0 + gamma * math.exp(-1.0 / tau)) * eps0
        errs.append(abs(stress_final - analytic) / analytic)
    assert errs[1] < errs[0]
    assert errs[1] < 0.01

    # Cyclic loading produces strictly positive hysteresis dissipation at every nonzero step.
    state = model.initial_state((1,))
    f_prev = f2d_id
    total_diss = 0.0
    for step in range(1, 21):
        strain = 0.08 * math.sin(2.0 * math.pi * step / 20.0)
        f_curr = torch.tensor(
            [[[1.0 + strain, 0.0], [0.0, 1.0 - 0.3 * strain]]], dtype=torch.float64
        )
        ps = model.plane_stress_update(f_prev, f_curr, state, dt=0.1)
        assert ps.closure_converged
        d_val = float(ps.update_3d.dissipation.item())
        assert d_val >= 0.0
        total_diss += d_val
        state = dict(ps.update_3d.state_new)
        f_prev = f_curr
    assert total_diss > 0.0


def test_incremental_2d_fem_viscoelastic_relaxation_convergence_and_replay() -> None:
    mesh = _rect_mesh()
    model = _visco_model()
    bcs = (
        DirichletHistoryBC("left_x", "left", "x", load_key=None),
        DirichletHistoryBC("bottom_y", "bottom", "y", load_key=None),
        DirichletHistoryBC("right_x", "right", "x", load_key="ux"),
    )
    segments = (
        LoadSegment(0.0, 0.5, "ramp", {"ux": DerivedExpression.parse("0.2 * s")}),
        LoadSegment(0.5, 1.5, "hold", {"ux": DerivedExpression.parse("0.2")}),
        LoadSegment(1.5, 2.0, "ramp", {"ux": DerivedExpression.parse("0.2 * (1 - s)")}),
    )

    grid_coarse = TimeGrid(0.0, 2.0, dt_initial=0.25, dt_min=0.05, dt_max=0.25)
    grid_fine = TimeGrid(0.0, 2.0, dt_initial=0.125, dt_min=0.025, dt_max=0.125)

    traj_a = solve_incremental_fem_trajectory(
        mesh,
        model,
        bcs,
        grid_coarse,
        segments,
        problem_spec_hash="p" * 64,
        load_path_hash="l" * 64,
        element_order=1,
    )
    traj_b = solve_incremental_fem_trajectory(
        mesh,
        model,
        bcs,
        grid_coarse,
        segments,
        problem_spec_hash="p" * 64,
        load_path_hash="l" * 64,
        element_order=1,
    )
    traj_fine = solve_incremental_fem_trajectory(
        mesh,
        model,
        bcs,
        grid_fine,
        segments,
        problem_spec_hash="p" * 64,
        load_path_hash="l" * 64,
        element_order=1,
    )

    assert traj_a.converged and traj_fine.converged
    # Exact deterministic CPU replay: identical state hashes at every step.
    assert [s.bound_state.state_hash for s in traj_a.steps] == [
        s.bound_state.state_hash for s in traj_b.steps
    ]
    for step in traj_a.steps:
        assert step.residual_norm < 1e-8
        assert step.step_dissipation >= 0.0
        assert step.min_det_f > 0.0

    # Stress relaxation during hold [0.5, 1.5]: reaction strictly decreases at constant ux=0.2.
    hold_steps = [s for s in traj_a.steps if 0.5 <= s.time_value <= 1.5]
    hold_rxns = [s.reactions["right_x"] for s in hold_steps]
    assert len(hold_rxns) >= 4
    for earlier, later in zip(hold_rxns, hold_rxns[1:], strict=False):
        assert later < earlier

    # Halving dt meets the F1-GATE <= 2% time-refinement tolerance on peak hold reaction.
    peak_coarse = max(s.reactions["right_x"] for s in traj_a.steps)
    peak_fine = max(s.reactions["right_x"] for s in traj_fine.steps)
    rel_dt_change = abs(peak_coarse - peak_fine) / abs(peak_fine)
    assert rel_dt_change <= 0.02


def test_guarded_stepper_refusal_fallback_and_checkpoint_tamper(tmp_path: Path) -> None:
    state0 = {"viscous_strain": torch.zeros((2, 1, 3, 3), dtype=torch.float64)}
    u0 = torch.zeros((4, 2), dtype=torch.float64)
    bound0 = BoundTrajectoryState(
        problem_spec_hash="p" * 64,
        mesh_hash="m" * 64,
        material_schema_hash="s" * 64,
        load_path_hash="l" * 64,
        time_index=0,
        time_value=0.0,
        step_controller_version=STEP_CONTROLLER_VERSION,
        state_hash=hash_state_tensors(state0),
    )

    # Save & reload checkpoint cleanly, then verify byte tamper is refused.
    ckpt_dir = tmp_path / "step-0"
    save_trajectory_checkpoint(
        ckpt_dir, bound_state=bound0, displacement=u0, state_tensors=state0
    )
    loaded_bound, loaded_u, loaded_state = load_trajectory_checkpoint(ckpt_dir)
    assert loaded_bound == bound0
    assert torch.equal(loaded_u, u0)
    assert torch.equal(loaded_state["viscous_strain"], state0["viscous_strain"])

    tensor_file = ckpt_dir / "state.safetensors"
    raw = bytearray(tensor_file.read_bytes())
    raw[-1] ^= 0xFF
    tensor_file.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="SHA-256"):
        load_trajectory_checkpoint(ckpt_dir)

    # Predictor fails residual guard on large dt, succeeds on halved dt (substep=2).
    def adaptive_predictor(u_old, s_old, load_next, dt):
        del load_next
        ok = dt <= 0.15
        return CandidateStepOutput(
            displacement=u_old + 0.01 * dt,
            state_new={"viscous_strain": s_old["viscous_strain"] + 1e-3 * dt},
            min_det_f=0.98,
            min_dissipation=1e-4,
            total_dissipation=1e-3,
            residual_norm=1e-8 if ok else 1e-2,
            admissible=True,
        )

    stepper = GuardedStatefulStepper(
        problem_spec_hash="p" * 64,
        mesh_hash="m" * 64,
        material_schema_hash="s" * 64,
        load_path_hash="l" * 64,
        predictor=adaptive_predictor,
        verification=StatefulVerificationPolicy(allow_fem_fallback=True),
        residual_tolerance=1e-6,
        max_substep_halvings=2,
    )
    res_ok = stepper.advance(
        bound_state=bound0,
        displacement_old=u0,
        state_old=state0,
        load_next={"ux": 0.1},
        dt=0.25,
    )
    assert res_ok.status is StepStatus.ACCEPTED_OPERATOR
    assert res_ok.substeps_taken == 2
    assert res_ok.bound_state is not None

    # Tampered state_old refused fail-closed.
    bad_state = {"viscous_strain": torch.ones((2, 1, 3, 3), dtype=torch.float64)}
    res_bad = stepper.advance(
        bound_state=bound0,
        displacement_old=u0,
        state_old=bad_state,
        load_next={"ux": 0.1},
        dt=0.1,
    )
    assert res_bad.status is StepStatus.REFUSED_STATE_MISMATCH

    # Inadmissible operator escalates to FEM fallback cleanly.
    def bad_operator(u_old, s_old, load_next, dt):
        del load_next, dt
        return CandidateStepOutput(
            displacement=u_old,
            state_new=s_old,
            min_det_f=-0.1,
            min_dissipation=-1.0,
            total_dissipation=-1.0,
            residual_norm=1.0,
            admissible=False,
            reason="folded deformation",
        )

    def fem_fallback(u_old, s_old, load_next, dt):
        del load_next
        return CandidateStepOutput(
            displacement=u_old + 0.02 * dt,
            state_new={"viscous_strain": s_old["viscous_strain"] + 2e-3 * dt},
            min_det_f=0.99,
            min_dissipation=1e-4,
            total_dissipation=2e-3,
            residual_norm=1e-11,
            admissible=True,
        )

    esc_stepper = GuardedStatefulStepper(
        problem_spec_hash="p" * 64,
        mesh_hash="m" * 64,
        material_schema_hash="s" * 64,
        load_path_hash="l" * 64,
        predictor=bad_operator,
        fem_fallback=fem_fallback,
        verification=StatefulVerificationPolicy(allow_fem_fallback=True),
        max_substep_halvings=1,
    )
    res_esc = esc_stepper.advance(
        bound_state=bound0,
        displacement_old=u0,
        state_old=state0,
        load_next={"ux": 0.1},
        dt=0.1,
    )
    assert res_esc.status is StepStatus.ACCEPTED_FEM_FALLBACK
    assert res_esc.accepted is True

    # Without fallback, inadmissible operator fails closed.
    no_fb_stepper = GuardedStatefulStepper(
        problem_spec_hash="p" * 64,
        mesh_hash="m" * 64,
        material_schema_hash="s" * 64,
        load_path_hash="l" * 64,
        predictor=bad_operator,
        fem_fallback=None,
        verification=StatefulVerificationPolicy(allow_fem_fallback=False),
        max_substep_halvings=1,
    )
    res_refused = no_fb_stepper.advance(
        bound_state=bound0,
        displacement_old=u0,
        state_old=state0,
        load_next={"ux": 0.1},
        dt=0.1,
    )
    assert res_refused.status is StepStatus.REFUSED_ADMISSIBILITY
    assert res_refused.accepted is False

    # Both operator and FEM fallback failing -> explicit SOLVE_FAILED.
    both_fail = GuardedStatefulStepper(
        problem_spec_hash="p" * 64,
        mesh_hash="m" * 64,
        material_schema_hash="s" * 64,
        load_path_hash="l" * 64,
        predictor=bad_operator,
        fem_fallback=bad_operator,
        verification=StatefulVerificationPolicy(allow_fem_fallback=True),
        max_substep_halvings=0,
    )
    res_fail = both_fail.advance(
        bound_state=bound0,
        displacement_old=u0,
        state_old=state0,
        load_next={"ux": 0.1},
        dt=0.1,
    )
    assert res_fail.status is StepStatus.SOLVE_FAILED

    # Non-positive det(F) in viscoelastic update returns inadmissible cleanly.
    vmodel = _visco_model()
    assert vmodel.config.instantaneous_modulus_mpa == pytest.approx(200.0)
    assert len(vmodel.config.config_hash) == 64
    f_bad = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    f_bad[0, 0, 0] = -1.0
    upd_bad = vmodel.update(f_bad, f_bad, vmodel.initial_state((1,)), dt=0.1)
    assert upd_bad.admissible is False
    assert upd_bad.guards["det_f_positive"] is False
    with pytest.raises(ValueError, match="poisson_ratio"):
        GeneralizedMaxwellConfig(100.0, 0.5, (MaxwellBranch(0.5, 1.0),))
    with pytest.raises(ValueError, match="stiffness_ratio"):
        MaxwellBranch(0.0, 1.0)


def test_viscoelastic_artifact_rollout_checkpoint_and_fallback_guards(
    tmp_path: Path,
) -> None:
    from piginot.operators.energy_operator import QueryFrame
    from piginot.stateful import (
        HistoryAwareViscoelasticOperator,
        ViscoelasticOperatorConfig,
        load_viscoelastic_artifact,
        run_viscoelastic_rollout,
        save_viscoelastic_artifact,
    )

    mesh = _rect_mesh()
    visco = _visco_model()
    op = HistoryAwareViscoelasticOperator(
        ViscoelasticOperatorConfig(
            load_dim=1, n_branches=2, hidden_dim=16, num_layers=2
        )
    ).to(torch.float64)
    frame = QueryFrame(
        origin=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        extent=torch.tensor([[2.0, 1.0]], dtype=torch.float64),
    )

    with pytest.raises(ValueError):
        save_viscoelastic_artifact(
            tmp_path / "bad-claim",
            operator=op,
            material=visco,
            mesh=mesh,
            frame=frame,
            claim_level="INVALID_CLAIM",
        )

    art_dir = tmp_path / "visco-art"
    save_viscoelastic_artifact(
        art_dir, operator=op, material=visco, mesh=mesh, frame=frame
    )
    artifact = load_viscoelastic_artifact(art_dir)
    segments = (
        LoadSegment(0.0, 0.5, "ramp", {"ux": DerivedExpression.parse("0.1 * s")}),
    )
    grid = TimeGrid(0.0, 0.5, dt_initial=0.25, dt_min=0.05, dt_max=0.25)

    # allow_fem_fallback=True without fem_fallback must fail closed
    with pytest.raises(ValueError, match="explicit fem_fallback"):
        run_viscoelastic_rollout(
            artifact,
            time_grid=grid,
            segments=segments,
            parameters={},
            allow_fem_fallback=True,
        )

    # Save checkpoint at t=0.5 (time_index=2), then verify resuming against a mismatched
    # or shorter time grid raises ValueError instead of silently succeeding with 0 steps
    ckpt_dir = tmp_path / "ckpt-end"
    res = run_viscoelastic_rollout(
        artifact,
        time_grid=grid,
        segments=segments,
        parameters={},
        save_checkpoint_to=ckpt_dir,
    )
    assert res["accepted"] is True
    short_grid = TimeGrid(0.0, 0.25, dt_initial=0.25, dt_min=0.05, dt_max=0.25)
    with pytest.raises(ValueError, match="outside time_grid"):
        run_viscoelastic_rollout(
            artifact,
            time_grid=short_grid,
            segments=segments,
            parameters={},
            resume_checkpoint=ckpt_dir,
        )
    shifted_grid = TimeGrid(0.0, 0.6, dt_initial=0.2, dt_min=0.05, dt_max=0.2)
    with pytest.raises(ValueError, match="time_value"):
        run_viscoelastic_rollout(
            artifact,
            time_grid=shifted_grid,
            segments=(
                LoadSegment(
                    0.0, 0.6, "ramp", {"ux": DerivedExpression.parse("0.1 * s")}
                ),
            ),
            parameters={},
            resume_checkpoint=ckpt_dir,
        )
