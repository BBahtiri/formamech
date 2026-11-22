"""Comprehensive verification suite for M3 small-strain J2 elastoplasticity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from formamech.stateful import (
    J2_PLASTICITY_MODEL_ID,
    PLASTICITY_OPERATOR_ARCH_ID,
    CandidateStepOutput,
    DirichletHistoryBC,
    GuardedStatefulStepper,
    HistoryAwarePlasticityOperator,
    J2IsotropicPlasticity,
    J2PlasticityConfig,
    PlasticityOperatorConfig,
    StepStatus,
    analytical_uniaxial_j2_response,
    evaluate_plasticity_discrete_residual,
    hash_state_tensors,
    solve_incremental_fem_trajectory,
    summarize_plastic_state,
    train_plasticity_operator_variational,
    verify_algorithmic_tangent,
    verify_plane_stress_tangent,
)

from piginot.cli import main as cli_main
from piginot.contracts import Unit
from piginot.contracts.stateful import (
    BoundTrajectoryState,
    IntegratorSpec,
    LoadSegment,
    StatefulVerificationPolicy,
    TimeGrid,
)
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh
from piginot.lifting.base import AffineDirichletLifting, DirichletLifting
from piginot.objectives.energy import TriangleDomain
from piginot.operators.energy_operator import QueryFrame
from piginot.stateful.stepper import STEP_CONTROLLER_VERSION


def _rect_mesh(lx: float = 10.0, ly: float = 4.0, nx: int = 4, ny: int = 2):
    nodes = [(lx * i / nx, ly * j / ny) for j in range(ny + 1) for i in range(nx + 1)]

    def _nid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    cells = []
    for j in range(ny):
        for i in range(nx):
            n00, n10 = _nid(i, j), _nid(i + 1, j)
            n01, n11 = _nid(i, j + 1), _nid(i + 1, j + 1)
            cells.append((n00, n10, n11))
            cells.append((n00, n11, n01))

    facets: list[tuple[int, int]] = []
    tags: dict[str, list[int]] = {"left": [], "right": [], "bottom": [], "top": []}
    for i in range(nx):
        tags["bottom"].append(len(facets))
        facets.append((_nid(i, 0), _nid(i + 1, 0)))
    for j in range(ny):
        tags["right"].append(len(facets))
        facets.append((_nid(nx, j), _nid(nx, j + 1)))
    for i in range(nx, 0, -1):
        tags["top"].append(len(facets))
        facets.append((_nid(i, ny), _nid(i - 1, ny)))
    for j in range(ny, 0, -1):
        tags["left"].append(len(facets))
        facets.append((_nid(0, j), _nid(0, j - 1)))

    return canonicalize_mesh(
        MeshPayload(
            nodes=tuple(nodes),
            cells=tuple(cells),
            cell_type=CellType.TRIANGLE,
            boundary_facets=tuple(facets),
            boundary_tags={k: tuple(v) for k, v in tags.items()},
            region_ids=tuple(1 for _ in cells),
            unit=Unit.MM,
            source_provenance={"source": "unit-test-rect"},
        )
    )


def test_j2_config_state_validation_and_tangent_fd_all_modes() -> None:
    cfg_lin = J2PlasticityConfig(
        youngs_modulus_mpa=200_000.0,
        poisson_ratio=0.3,
        yield_stress_mpa=250.0,
        hardening_modulus_mpa=10_000.0,
    )
    assert cfg_lin.document()["model_id"] == J2_PLASTICITY_MODEL_ID
    assert J2PlasticityConfig.from_document(cfg_lin.document()) == cfg_lin
    assert len(cfg_lin.config_hash) == 64
    assert cfg_lin.mu > 0.0 and cfg_lin.bulk_modulus > 0.0 and cfg_lin.lame_lambda > 0.0

    cfg_voce = J2PlasticityConfig(
        youngs_modulus_mpa=70_000.0,
        poisson_ratio=0.33,
        yield_stress_mpa=180.0,
        hardening_modulus_mpa=1_500.0,
        voce_saturation_mpa=120.0,
        voce_exponent=18.0,
    )

    # Invalid config & state guards
    with pytest.raises(ValueError, match="strictly positive"):
        J2PlasticityConfig(
            youngs_modulus_mpa=0.0, poisson_ratio=0.3, yield_stress_mpa=100.0
        )
    with pytest.raises(ValueError, match=r"\[0, 0.5\)"):
        J2PlasticityConfig(
            youngs_modulus_mpa=100.0, poisson_ratio=0.5, yield_stress_mpa=10.0
        )
    with pytest.raises(ValueError, match="finite number"):
        J2PlasticityConfig(
            youngs_modulus_mpa=True, poisson_ratio=0.3, yield_stress_mpa=10.0
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model_id"):
        J2PlasticityConfig.from_document({"model_id": "wrong"})
    with pytest.raises(TypeError, match="J2PlasticityConfig"):
        J2IsotropicPlasticity("bad")  # type: ignore[arg-type]

    for cfg in (cfg_lin, cfg_voce):
        model = J2IsotropicPlasticity(cfg)
        ey = cfg.yield_stress_mpa / cfg.youngs_modulus_mpa
        eye3 = torch.eye(3, dtype=torch.float64)
        eye2 = torch.eye(2, dtype=torch.float64)

        # Invalid dt / shape / state checks
        st0 = model.initial_state((), dtype=torch.float64)
        with pytest.raises(ValueError, match="dt"):
            model.update(eye3, eye3, st0, dt=0.0)
        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            model.update(eye3, eye2, st0, dt=1.0)
        with pytest.raises(ValueError, match="plastic_strain"):
            model.update(eye3, eye3, {}, dt=1.0)

        # Sequence: elastic -> active plastic -> elastic unloading -> reverse plastic
        st3 = model.initial_state((), dtype=torch.float64)
        st2 = model.initial_state((), dtype=torch.float64)
        f3_prev = eye3.clone()
        f2_prev = eye2.clone()
        steps = (
            (0.4 * ey, 0.2 * ey),  # elastic
            (3.0 * ey, 1.2 * ey),  # active plastic loading
            (1.8 * ey, 0.5 * ey),  # elastic unloading
            (-3.2 * ey, -1.0 * ey),  # reversed plastic loading
        )
        for exx, exy in steps:
            f3 = eye3.clone()
            f3[0, 0] = 1.0 + exx
            f3[1, 1] = 1.0 - 0.25 * exx
            f3[2, 2] = 1.0 - 0.1 * exx
            f3[0, 1] = exy
            f3[1, 0] = exy

            t3 = verify_algorithmic_tangent(
                model, f3_prev, f3, st3, dt=1.0, eps=1e-7, rel_tol=1e-7
            )
            assert t3.passed, f"3D tangent FD err={t3.relative_error:.3e}"
            assert t3.relative_error < 1.0e-8

            # Verify Ortiz-Steinmann variational potential exactness: dW_inc/dF == sigma
            f3_req = f3.detach().clone().requires_grad_(True)
            upd3 = model.update(f3_prev, f3_req, st3, dt=1.0)
            assert upd3.admissible is True
            assert all(upd3.guards.values())
            grad_w = torch.autograd.grad(upd3.incremental_potential.sum(), f3_req)[0]
            rel_pot = float(
                torch.linalg.vector_norm(grad_w - upd3.stress).item()
                / max(float(torch.linalg.vector_norm(upd3.stress).item()), 1e-12)
            )
            assert rel_pot < 1.0e-10
            st3 = {k: v.detach().clone() for k, v in upd3.state_new.items()}
            f3_prev = f3

            # 2D plane-stress Schur-condensed tangent & variational potential exactness
            f2 = eye2.clone()
            f2[0, 0] = 1.0 + exx
            f2[1, 1] = 1.0 - 0.3 * exx
            f2[0, 1] = exy
            f2[1, 0] = exy
            t2 = verify_plane_stress_tangent(
                model, f2_prev, f2, st2, dt=1.0, eps=1e-7, rel_tol=1e-7
            )
            assert t2.passed, f"Plane-stress tangent FD err={t2.relative_error:.3e}"
            assert t2.relative_error < 1.0e-8

            f2_req = f2.detach().clone().requires_grad_(True)
            ps = model.plane_stress_update(f2_prev, f2_req, st2, dt=1.0)
            assert ps.closure_converged is True
            assert ps.closure_residual_max < 1.0e-9
            grad_w2 = torch.autograd.grad(
                ps.update_3d.incremental_potential.sum(), f2_req
            )[0]
            s2 = ps.update_3d.stress[:2, :2]
            rel_pot2 = float(
                torch.linalg.vector_norm(grad_w2 - s2).item()
                / max(float(torch.linalg.vector_norm(s2).item()), 1e-12)
            )
            assert rel_pot2 < 1.0e-10
            st2 = {k: v.detach().clone() for k, v in ps.update_3d.state_new.items()}
            f2_prev = f2


@pytest.mark.parametrize("voce_sat", [0.0, 120.0])
def test_uniaxial_cyclic_fem_matches_analytical_to_machine_precision(
    voce_sat: float,
) -> None:
    """Plane-stress P2 FEM uniaxial bar must match analytical J2 tension-unload-compression."""
    lx, ly = 10.0, 4.0
    mesh = _rect_mesh(lx=lx, ly=ly, nx=3, ny=2)
    cfg = J2PlasticityConfig(
        youngs_modulus_mpa=100_000.0,
        poisson_ratio=0.3,
        yield_stress_mpa=200.0,
        hardening_modulus_mpa=5_000.0,
        voce_saturation_mpa=voce_sat,
        voce_exponent=25.0,
    )
    model = J2IsotropicPlasticity(cfg)

    # Fix x=0 on left edge, constrain one corner/bottom symmetry y=0 so bar contracts freely in y
    bcs = (
        DirichletHistoryBC("left_x", "left", "x", None),
        DirichletHistoryBC("bottom_y", "bottom", "y", None),
        DirichletHistoryBC("right_pull", "right", "x", "ux"),
    )
    # Cyclic path: 0 -> +0.06 mm (0.6% strain, 3x yield) -> 0 -> -0.06 mm -> 0
    u_peak = 0.06
    grid = TimeGrid(t0=0.0, t1=4.0, dt_initial=0.25, dt_min=0.05, dt_max=0.25)
    segments = (
        LoadSegment(
            start=0.0,
            end=1.0,
            kind="ramp",
            boundary_values={"ux": f"{u_peak} * s"},  # type: ignore[dict-item]
        ),
        LoadSegment(
            start=1.0,
            end=2.0,
            kind="ramp",
            boundary_values={"ux": f"{u_peak} * (1.0 - s)"},  # type: ignore[dict-item]
        ),
        LoadSegment(
            start=2.0,
            end=3.0,
            kind="ramp",
            boundary_values={"ux": f"-{u_peak} * s"},  # type: ignore[dict-item]
        ),
        LoadSegment(
            start=3.0,
            end=4.0,
            kind="ramp",
            boundary_values={"ux": f"-{u_peak} * (1.0 - s)"},  # type: ignore[dict-item]
        ),
    )

    traj = solve_incremental_fem_trajectory(
        mesh,
        model,
        bcs,
        grid,
        segments,
        problem_spec_hash="p-uniaxial",
        load_path_hash="l-cycle",
        element_order=2,
        integrator=IntegratorSpec(residual_tolerance=1e-9, max_newton_iterations=15),
        kinematics_mode="plane_stress",
    )
    assert traj.converged is True, traj.reason
    assert len(traj.steps) == 16

    # Compute exact analytical uniaxial stress history at the same times
    times = grid.fixed_times()[1:]
    strains: list[float] = []
    for t in times:
        if t <= 1.0:
            u_val = u_peak * t
        elif t <= 2.0:
            u_val = u_peak * (2.0 - t)
        elif t <= 3.0:
            u_val = -u_peak * (t - 2.0)
        else:
            u_val = -u_peak * (4.0 - t)
        strains.append(u_val / lx)

    ana = analytical_uniaxial_j2_response(cfg, strains)
    for step, sig_ana, eq_p_ana in zip(
        traj.steps, ana["axial_stress_mpa"], ana["eq_plastic_strain"], strict=True
    ):
        # Exact uniaxial force: F_x = sigma_11 * ly (unit thickness 1 mm)
        f_ana = sig_ana * ly
        f_fem = step.reactions["right_pull"]
        assert abs(f_fem - f_ana) <= 1.0e-7 * max(1.0, abs(f_ana))
        assert step.max_eq_plastic_strain is not None
        assert abs(step.max_eq_plastic_strain - eq_p_ana) <= 1.0e-9
        assert step.step_dissipation >= -1.0e-12
        assert step.substeps == 1
        assert (
            step.newton_iterations <= 10
        )  # quadratic Newton convergence without substepping

    # Unloading back to u=0 after tension leaves nonzero compressive residual reaction
    step_at_zero = traj.steps[7]  # t = 2.0
    assert step_at_zero.reactions["right_pull"] < -100.0
    assert (step_at_zero.max_eq_plastic_strain or 0.0) > 0.002


def test_plane_strain_multiaxial_operator_training_and_guarded_stepper() -> None:
    """Verify plane-strain J2 FEM, variational operator training, and GuardedStatefulStepper."""
    lx, ly = 10.0, 4.0
    mesh = _rect_mesh(lx=lx, ly=ly, nx=3, ny=2)
    cfg = J2PlasticityConfig(
        youngs_modulus_mpa=10_000.0,
        poisson_ratio=0.3,
        yield_stress_mpa=100.0,
        hardening_modulus_mpa=1_000.0,
    )
    model = J2IsotropicPlasticity(cfg)
    bcs = (
        DirichletHistoryBC("left_x", "left", "x", None),
        DirichletHistoryBC("left_y", "left", "y", None),
        DirichletHistoryBC("right_pull", "right", "x", "ux"),
    )
    grid = TimeGrid(t0=0.0, t1=2.0, dt_initial=0.5, dt_min=0.1, dt_max=0.5)
    segments = (
        LoadSegment(
            start=0.0,
            end=1.0,
            kind="ramp",
            boundary_values={"ux": "0.25 * s"},  # type: ignore[dict-item]
        ),
        LoadSegment(
            start=1.0,
            end=2.0,
            kind="ramp",
            boundary_values={"ux": "0.25 - 0.20 * s"},  # type: ignore[dict-item]
        ),
    )

    # 1) Plane-strain FEM solve + deterministic state-hash replay
    traj_a = solve_incremental_fem_trajectory(
        mesh,
        model,
        bcs,
        grid,
        segments,
        problem_spec_hash="p-ps",
        load_path_hash="l-ps",
        element_order=1,
        kinematics_mode="plane_strain",
    )
    traj_b = solve_incremental_fem_trajectory(
        mesh,
        model,
        bcs,
        grid,
        segments,
        problem_spec_hash="p-ps",
        load_path_hash="l-ps",
        element_order=1,
        kinematics_mode="plane_strain",
    )
    assert traj_a.converged is True
    assert [s.bound_state.state_hash for s in traj_a.steps] == [
        s.bound_state.state_hash for s in traj_b.steps
    ]
    assert (traj_a.steps[-1].max_eq_plastic_strain or 0.0) > 0.005

    # Invalid kinematics mode raises cleanly
    with pytest.raises(ValueError, match="kinematics_mode"):
        solve_incremental_fem_trajectory(
            mesh,
            model,
            bcs,
            grid,
            segments,
            problem_spec_hash="p",
            load_path_hash="l",
            kinematics_mode="invalid",  # type: ignore[arg-type]
        )

    # 2) Variational Ortiz-Steinmann operator training (label-free)
    domain = TriangleDomain(
        vertices=torch.tensor(mesh.nodes, dtype=torch.float64),
        cells=torch.tensor(mesh.cells, dtype=torch.int64),
    )
    frame = QueryFrame(
        origin=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        extent=torch.tensor([[lx, ly]], dtype=torch.float64),
    )

    class _BarLifting(DirichletLifting):
        def __init__(self, ux: float) -> None:
            self.ux = ux

        def __call__(self, points: torch.Tensor) -> AffineDirichletLifting:
            x = points[..., 0]
            xi = x / lx
            offset = torch.stack((self.ux * xi, torch.zeros_like(xi)), dim=-1)
            scale = torch.stack((xi * (1.0 - xi), xi), dim=-1)
            return AffineDirichletLifting(offset=offset, scale=scale)

    op_cfg = PlasticityOperatorConfig(
        load_dim=1,
        kinematics_mode="plane_strain",
        hidden_dim=16,
        num_layers=2,
        positional_frequencies=2,
    )
    op = HistoryAwarePlasticityOperator(op_cfg).to(dtype=torch.float64)
    assert op.architecture_id == PLASTICITY_OPERATOR_ARCH_ID

    train_traj = [
        (torch.tensor([0.125], dtype=torch.float64), 0.5, _BarLifting(0.125)),
        (torch.tensor([0.250], dtype=torch.float64), 0.5, _BarLifting(0.250)),
        (torch.tensor([0.150], dtype=torch.float64), 0.5, _BarLifting(0.150)),
    ]
    summary = train_plasticity_operator_variational(
        op,
        domain=domain,
        frame=frame,
        model=model,
        trajectories=[train_traj],
        epochs=3,
        learning_rate=5e-3,
        seed=0,
    )
    assert summary.training_mode == "variational-energy-label-free"
    assert summary.total_steps == 9
    assert summary.max_eq_plastic_strain_seen > 0.0
    assert summary.min_dissipation_seen >= -1.0e-9

    # 3) Wire operator + exact FEM fallback into GuardedStatefulStepper
    from piginot.stateful.fem_stepper import (
        _build_mesh_and_basis,
        _constrained_dofs,
        _solve_single_increment,
    )

    skmesh, basis, _ = _build_mesh_and_basis(mesh, element_order=1)
    by_name, _, free_dofs = _constrained_dofs(basis, skmesh, mesh, bcs)
    q_shape = tuple(int(d) for d in basis.dx.shape)
    pts_nodes = torch.tensor(mesh.nodes, dtype=torch.float64).unsqueeze(0)
    x_dofs = np.asarray(basis.nodal_dofs[0], dtype=np.int64)
    y_dofs = np.asarray(basis.nodal_dofs[1], dtype=np.int64)

    def _op_predictor(u_old, state_old, load_next, dt):
        ux = float(load_next["ux"])
        hist = summarize_plastic_state(state_old)
        with torch.no_grad():
            u_nodal = (
                op.displacement(
                    query_xy=pts_nodes,
                    frame=frame,
                    load_vector=torch.tensor([ux], dtype=torch.float64),
                    history_features=hist,
                    lifting=_BarLifting(ux),
                )[0]
                .cpu()
                .numpy()
            )
        u_vec = np.zeros(basis.N, dtype=np.float64)
        u_vec[x_dofs] = u_nodal[:, 0]
        u_vec[y_dofs] = u_nodal[:, 1]
        res, s_new, min_det, min_diss, tot_diss, adm = (
            evaluate_plasticity_discrete_residual(
                basis=basis,
                free_dofs=free_dofs,
                model=model,
                u_vec=u_vec,
                u_old_vec=u_old.cpu().numpy(),
                state_old=state_old,
                dt=dt,
                kinematics_mode="plane_strain",
            )
        )
        return CandidateStepOutput(
            displacement=torch.from_numpy(u_vec),
            state_new=s_new,
            min_det_f=min_det,
            min_dissipation=min_diss,
            total_dissipation=tot_diss,
            residual_norm=res,
            admissible=adm,
        )

    def _fem_fallback(u_old, state_old, load_next, dt):
        target_d: dict[int, float] = {}
        for bc in bcs:
            val = 0.0 if bc.load_key is None else float(load_next[bc.load_key])
            for dof in by_name[bc.name]:
                target_d[int(dof)] = val
        ok, u_next, s_next, _, _, res, _, diss, min_det, _, reason = (
            _solve_single_increment(
                basis=basis,
                model=model,
                u_old=u_old.cpu().numpy(),
                state_old=state_old,
                f33_old=torch.ones(q_shape, dtype=torch.float64),
                target_dirichlet=target_d,
                free_dofs=free_dofs,
                dt=dt,
                integrator=IntegratorSpec(residual_tolerance=1e-9),
                kinematics_mode="plane_strain",
            )
        )
        return CandidateStepOutput(
            displacement=torch.from_numpy(u_next),
            state_new=s_next,
            min_det_f=min_det,
            min_dissipation=0.0,
            total_dissipation=diss,
            residual_norm=res,
            admissible=ok,
            reason=reason,
        )

    state_curr = model.initial_state(q_shape, dtype=torch.float64)
    u_curr = torch.zeros(basis.N, dtype=torch.float64)
    bound = BoundTrajectoryState(
        problem_spec_hash="p-ps",
        mesh_hash=mesh.mesh_id,
        material_schema_hash=model.state_schema.schema_hash,
        load_path_hash="l-ps",
        time_index=0,
        time_value=0.0,
        step_controller_version=STEP_CONTROLLER_VERSION,
        state_hash=hash_state_tensors(state_curr),
    )
    stepper = GuardedStatefulStepper(
        problem_spec_hash="p-ps",
        mesh_hash=mesh.mesh_id,
        material_schema_hash=model.state_schema.schema_hash,
        load_path_hash="l-ps",
        predictor=_op_predictor,
        verification=StatefulVerificationPolicy(),
        residual_tolerance=1.0e-2,  # Tight tolerance triggers exact FEM fallback
        max_substep_halvings=0,
        fem_fallback=_fem_fallback,
    )
    step_res = stepper.advance(
        bound_state=bound,
        displacement_old=u_curr,
        state_old=state_curr,
        load_next={"ux": 0.25},
        dt=0.5,
    )
    assert step_res.accepted is True
    assert step_res.status in (
        StepStatus.ACCEPTED_OPERATOR,
        StepStatus.ACCEPTED_FEM_FALLBACK,
    )
    assert step_res.residual_norm is not None and step_res.residual_norm <= 1.0e-2


def test_cli_stateful_verify_tangent_all_modes(tmp_path: Path) -> None:
    mat_file = tmp_path / "j2.json"
    cfg = J2PlasticityConfig(
        youngs_modulus_mpa=210_000.0,
        poisson_ratio=0.3,
        yield_stress_mpa=250.0,
        hardening_modulus_mpa=4_000.0,
        voce_saturation_mpa=80.0,
        voce_exponent=15.0,
    )
    mat_file.write_text(json.dumps(cfg.document()), encoding="utf-8")

    for mode in ("3d", "plane-strain", "plane-stress"):
        out_file = tmp_path / f"tan_{mode}.json"
        rc = cli_main(
            [
                "stateful",
                "verify-tangent",
                "--material",
                str(mat_file),
                "--mode",
                mode,
                "--output",
                str(out_file),
            ]
        )
        assert rc == 0
        doc = json.loads(out_file.read_text(encoding="utf-8"))
        assert doc["passed"] is True
        for regime in ("elastic", "active_plastic", "unloading"):
            assert doc["regimes"][regime]["passed"] is True
            assert doc["regimes"][regime]["relative_error"] < 1.0e-7


def test_plasticity_operator_plane_stress_multifidelity_and_guards() -> None:
    """Verify plane-stress variational training, multi-fidelity disclosure, and validation guards."""
    lx, ly = 10.0, 4.0
    mesh = _rect_mesh(lx=lx, ly=ly, nx=2, ny=2)
    cfg = J2PlasticityConfig(
        youngs_modulus_mpa=10_000.0,
        poisson_ratio=0.3,
        yield_stress_mpa=100.0,
        hardening_modulus_mpa=1_000.0,
        voce_saturation_mpa=40.0,
        voce_exponent=12.0,
    )
    model = J2IsotropicPlasticity(cfg)

    # Extra J2PlasticityConfig & state validation guards
    with pytest.raises(ValueError, match="nonnegative"):
        J2PlasticityConfig(
            youngs_modulus_mpa=10_000.0,
            poisson_ratio=0.3,
            yield_stress_mpa=100.0,
            hardening_modulus_mpa=-1.0,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        J2PlasticityConfig(
            youngs_modulus_mpa=10_000.0,
            poisson_ratio=0.3,
            yield_stress_mpa=100.0,
            voce_saturation_mpa=50.0,
            voce_exponent=0.0,
        )
    eye3 = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="plastic_strain shape"):
        model.update(
            eye3,
            eye3,
            {
                "plastic_strain": torch.zeros((2, 3, 3), dtype=torch.float64),
                "eq_plastic_strain": torch.zeros((), dtype=torch.float64),
            },
            dt=1.0,
        )
    with pytest.raises(ValueError, match="eq_plastic_strain shape"):
        model.update(
            eye3,
            eye3,
            {
                "plastic_strain": torch.zeros((3, 3), dtype=torch.float64),
                "eq_plastic_strain": torch.zeros((2,), dtype=torch.float64),
            },
            dt=1.0,
        )
    with pytest.raises(ValueError, match="eps"):
        verify_plane_stress_tangent(
            model,
            torch.eye(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
            model.initial_state((), dtype=torch.float64),
            dt=1.0,
            eps=0.0,
        )

    # PlasticityOperatorConfig guards
    with pytest.raises(ValueError, match="kinematics_mode"):
        PlasticityOperatorConfig(load_dim=1, kinematics_mode="3d")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        PlasticityOperatorConfig(load_dim=0)
    with pytest.raises(ValueError, match="output_init_std"):
        PlasticityOperatorConfig(load_dim=1, output_init_std=-0.1)
    with pytest.raises(TypeError, match="boolean"):
        PlasticityOperatorConfig(load_dim=1, include_local_state="yes")  # type: ignore[arg-type]

    domain = TriangleDomain(
        vertices=torch.tensor(mesh.nodes, dtype=torch.float64),
        cells=torch.tensor(mesh.cells, dtype=torch.int64),
    )
    frame = QueryFrame(
        origin=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        extent=torch.tensor([[lx, ly]], dtype=torch.float64),
    )

    class _BarLifting(DirichletLifting):
        def __init__(self, ux: float) -> None:
            self.ux = ux

        def __call__(self, points: torch.Tensor) -> AffineDirichletLifting:
            x = points[..., 0]
            xi = x / lx
            offset = torch.stack((self.ux * xi, torch.zeros_like(xi)), dim=-1)
            scale = torch.stack((xi * (1.0 - xi), xi), dim=-1)
            return AffineDirichletLifting(offset=offset, scale=scale)

    # Plane-stress multi-fidelity training explicitly records disclosure tag
    op_ps = HistoryAwarePlasticityOperator(
        PlasticityOperatorConfig(
            load_dim=1,
            kinematics_mode="plane_stress",
            hidden_dim=12,
            num_layers=2,
            positional_frequencies=2,
            include_local_state=True,
        )
    ).to(dtype=torch.float64)

    pts_nodes = torch.tensor(mesh.nodes, dtype=torch.float64)
    train_traj = [
        (torch.tensor([0.15], dtype=torch.float64), 0.5, _BarLifting(0.15)),
        (torch.tensor([0.25], dtype=torch.float64), 0.5, _BarLifting(0.25)),
    ]
    sup_traj = [
        (
            pts_nodes,
            torch.stack(
                (0.15 * pts_nodes[:, 0] / lx, -0.045 * pts_nodes[:, 1] / lx), dim=-1
            ),
        ),
        (
            pts_nodes,
            torch.stack(
                (0.25 * pts_nodes[:, 0] / lx, -0.08 * pts_nodes[:, 1] / lx), dim=-1
            ),
        ),
    ]
    summary = train_plasticity_operator_variational(
        op_ps,
        domain=domain,
        frame=frame,
        model=model,
        trajectories=[train_traj],
        epochs=2,
        learning_rate=5e-3,
        seed=1,
        supervised_trajectories=[sup_traj],
        supervision_weight=2.0,
    )
    assert summary.training_mode == "multi-fidelity-label-efficient"
    assert summary.total_steps == 4
    assert summary.final_loss >= 0.0

    # Also exercise include_local_state=False and plane-stress discrete residual
    op_nolocal = HistoryAwarePlasticityOperator(
        PlasticityOperatorConfig(
            load_dim=1,
            kinematics_mode="plane_stress",
            hidden_dim=8,
            num_layers=2,
            positional_frequencies=1,
            include_local_state=False,
        )
    ).to(dtype=torch.float64)
    u_out = op_nolocal.displacement(
        query_xy=pts_nodes.unsqueeze(0),
        frame=frame,
        load_vector=torch.tensor([0.15], dtype=torch.float64),
        history_features=torch.zeros(5, dtype=torch.float64),
        lifting=_BarLifting(0.15),
    )
    assert u_out.shape == (1, pts_nodes.shape[0], 2)

    from piginot.stateful.fem_stepper import _build_mesh_and_basis, _constrained_dofs

    bcs = (
        DirichletHistoryBC("left_x", "left", "x", None),
        DirichletHistoryBC("bottom_y", "bottom", "y", None),
        DirichletHistoryBC("right_pull", "right", "x", "ux"),
    )
    skmesh, basis, _ = _build_mesh_and_basis(mesh, element_order=1)
    _, _, free_dofs = _constrained_dofs(basis, skmesh, mesh, bcs)
    q_shape = tuple(int(d) for d in basis.dx.shape)
    u_vec = np.zeros(basis.N, dtype=np.float64)
    u_vec[np.asarray(basis.nodal_dofs[0], dtype=np.int64)] = (
        0.20 * np.asarray(mesh.nodes)[:, 0] / lx
    )
    res, s_new, min_det, min_diss, tot_diss, adm = (
        evaluate_plasticity_discrete_residual(
            basis=basis,
            free_dofs=free_dofs,
            model=model,
            u_vec=u_vec,
            u_old_vec=np.zeros(basis.N, dtype=np.float64),
            state_old=model.initial_state(q_shape, dtype=torch.float64),
            dt=0.5,
            kinematics_mode="plane_stress",
        )
    )
    assert np.isfinite(res)
    assert min_det > 0.0
    assert min_diss >= -1.0e-9
    assert tot_diss >= -1.0e-9
    assert adm is True
    assert float(s_new["eq_plastic_strain"].max().item()) > 0.0

    # Invalid epochs / trajectories guards
    with pytest.raises(ValueError, match="epochs"):
        train_plasticity_operator_variational(
            op_ps,
            domain=domain,
            frame=frame,
            model=model,
            trajectories=[train_traj],
            epochs=0,
        )
    with pytest.raises(ValueError, match="trajectories"):
        train_plasticity_operator_variational(
            op_ps,
            domain=domain,
            frame=frame,
            model=model,
            trajectories=[],
            epochs=1,
        )
