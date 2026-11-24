"""Verification tests for Phase 1 (V3-1) 3D iterative linear solvers & neural deflation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags  # type: ignore[import-untyped]

from piginot.cad.importing_3d import (
    classify_faces_by_bbox,
    generate_3d_box_step,
    import_cad_3d,
)
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import (
    DirichletBC3D,
    FemRequest3D,
    ScikitFemBackend3D,
    TorchFemBackend3D,
    TractionBC3D,
)
from piginot.fem.linear_solvers import (
    LINEAR_SOLVER_MACHINE,
    LinearSolver3D,
    LinearSolverConfig,
    LinearSolverMethod,
    LinearSolverState,
    NeuralCoarsePreconditioner,
    SparseLinearSolver3D,
    is_pyamg_available,
    solve_linear_system,
)
from piginot.geometry.mesh3d import CellType3D


def _spd_3d_laplacian(n_blocks: int = 40, block_size: int = 3) -> csr_matrix:
    """Build a well-conditioned coupled 3x3-block SPD sparse matrix."""
    n = n_blocks * block_size
    main = np.full(n, 6.0, dtype=np.float64)
    off1 = np.full(n - 1, -0.8, dtype=np.float64)
    off3 = np.full(n - block_size, -1.2, dtype=np.float64)
    return diags(
        [off3, off1, main, off1, off3],
        [-block_size, -1, 0, 1, block_size],
        format="csr",
    )


def _neo_hookean_3d() -> MaterialSpec:
    return MaterialSpec(
        name="nh-3d-v31",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(10.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.35, Unit.ONE).to_canonical(),
        },
    )


def test_linear_solver_config_validation_and_fsm() -> None:
    cfg = LinearSolverConfig(method=LinearSolverMethod.PCG_SSOR, rtol=1e-11)
    doc = cfg.document()
    assert doc["method"] == "pcg-ssor"
    assert LINEAR_SOLVER_MACHINE.machine_id == "formamech.fem.linear_solver"

    with pytest.raises(ValueError, match="direct_dof_threshold"):
        LinearSolverConfig(direct_dof_threshold=-1)
    with pytest.raises(ValueError, match="ssor_omega"):
        LinearSolverConfig(ssor_omega=2.0)
    with pytest.raises(ValueError, match="rtol"):
        LinearSolverConfig(rtol=0.0)


@pytest.mark.parametrize(
    "method",
    [
        LinearSolverMethod.DIRECT,
        LinearSolverMethod.PCG_JACOBI,
        LinearSolverMethod.PCG_SSOR,
        LinearSolverMethod.PCG_ILU,
        LinearSolverMethod.GMRES_ILU,
    ],
)
def test_iterative_methods_match_direct_on_spd_matrix(
    method: LinearSolverMethod,
) -> None:
    mat = _spd_3d_laplacian(n_blocks=50)
    rng = np.random.default_rng(7)
    x_true = rng.standard_normal(mat.shape[0])
    rhs = np.asarray(mat @ x_true, dtype=np.float64)

    solver = SparseLinearSolver3D(
        LinearSolverConfig(
            method=method, rtol=1e-12, atol=1e-14, allow_fallback_to_direct=False
        )
    )
    assert isinstance(solver, LinearSolver3D)
    sol, rep = solver.solve(mat, rhs, is_symmetric=True)
    assert rep.converged
    assert rep.state is LinearSolverState.CONVERGED
    assert not rep.used_fallback
    assert rep.state_snapshot().state == "CONVERGED"
    rel_err = float(np.linalg.norm(sol - x_true) / np.linalg.norm(x_true))
    assert rel_err < 1.0e-10
    assert rep.document()["converged"] is True


def test_unsymmetric_matrix_refuses_cg_and_routes_auto_to_gmres() -> None:
    base = _spd_3d_laplacian(n_blocks=25)
    # Add unsymmetric skew perturbation
    n = base.shape[0]
    skew = diags([np.full(n - 1, 0.5), np.full(n - 1, -0.5)], [1, -1], format="csr")
    mat = (base + skew).tocsr()
    rhs = np.ones(n, dtype=np.float64)

    with pytest.raises(ValueError, match="unsymmetric tangent"):
        solve_linear_system(
            mat,
            rhs,
            config=LinearSolverConfig(method=LinearSolverMethod.PCG_SSOR),
        )

    sol_auto, rep_auto = solve_linear_system(
        mat,
        rhs,
        config=LinearSolverConfig(
            method=LinearSolverMethod.AUTO,
            direct_dof_threshold=0,
            rtol=1e-12,
        ),
    )
    sol_dir, _ = solve_linear_system(
        mat,
        rhs,
        config=LinearSolverConfig(method=LinearSolverMethod.DIRECT),
    )
    assert rep_auto.converged
    assert rep_auto.method_used is LinearSolverMethod.GMRES_ILU
    assert float(np.linalg.norm(sol_auto - sol_dir) / np.linalg.norm(sol_dir)) < 1e-10


def test_neural_coarse_preconditioner_reduces_pcg_iterations() -> None:
    mat = _spd_3d_laplacian(n_blocks=80)
    n = mat.shape[0]
    # Low-frequency smooth modes typical of structural deformation
    grid = np.linspace(0.0, np.pi, n)
    snaps = [np.sin((k + 1) * grid) for k in range(6)]
    precond = NeuralCoarsePreconditioner.from_snapshots(snaps, max_modes=6)
    assert precond.n_modes == 6

    # Target solution predominantly in smooth subspace + small high-frequency part
    x_true = (
        1.5 * snaps[0] - 0.7 * snaps[1] + 0.4 * snaps[2] + 0.05 * np.sin(19.0 * grid)
    )
    rhs = np.asarray(mat @ x_true, dtype=np.float64)

    _, rep_plain = solve_linear_system(
        mat,
        rhs,
        config=LinearSolverConfig(
            method=LinearSolverMethod.PCG_JACOBI,
            rtol=1e-11,
            use_neural_coarse_deflation=False,
            allow_fallback_to_direct=False,
        ),
        is_symmetric=True,
    )
    sol_def, rep_def = solve_linear_system(
        mat,
        rhs,
        config=LinearSolverConfig(
            method=LinearSolverMethod.PCG_JACOBI,
            rtol=1e-11,
            use_neural_coarse_deflation=True,
            allow_fallback_to_direct=False,
        ),
        is_symmetric=True,
        neural_preconditioner=precond,
    )
    assert rep_plain.converged and rep_def.converged
    assert rep_def.used_neural_deflation is True
    assert rep_def.iterations < rep_plain.iterations
    assert float(np.linalg.norm(sol_def - x_true) / np.linalg.norm(x_true)) < 1e-10


def test_stagnated_krylov_fallback_and_nonfinite_rhs_handling() -> None:
    mat = _spd_3d_laplacian(n_blocks=30)
    rhs = np.ones(mat.shape[0], dtype=np.float64)

    # Max 1 iter forces stagnation -> direct fallback recovers exact solution
    sol_fb, rep_fb = solve_linear_system(
        mat,
        rhs,
        config=LinearSolverConfig(
            method=LinearSolverMethod.PCG_JACOBI,
            max_iterations=1,
            rtol=1e-13,
            allow_fallback_to_direct=True,
        ),
        is_symmetric=True,
    )
    assert rep_fb.converged
    assert rep_fb.used_fallback is True
    assert rep_fb.method_used is LinearSolverMethod.DIRECT
    assert float(np.linalg.norm(mat @ sol_fb - rhs)) < 1e-10

    # With fallback disabled, stagnation returns converged=False, state=FAILED
    _, rep_fail = solve_linear_system(
        mat,
        rhs,
        config=LinearSolverConfig(
            method=LinearSolverMethod.PCG_JACOBI,
            max_iterations=1,
            rtol=1e-13,
            allow_fallback_to_direct=False,
        ),
        is_symmetric=True,
    )
    assert not rep_fail.converged
    assert rep_fail.state is LinearSolverState.FAILED

    # Non-finite RHS fails cleanly without throwing
    bad_rhs = rhs.copy()
    bad_rhs[0] = np.nan
    _, rep_nan = solve_linear_system(mat, bad_rhs)
    assert not rep_nan.converged
    assert rep_nan.state is LinearSolverState.FAILED


def test_3d_fem_backends_iterative_solver_parity_and_solve_many(tmp_path: Path) -> None:
    step = generate_3d_box_step(tmp_path / "box.step", lx=16.0, ly=8.0, lz=8.0)
    imported = import_cad_3d(step, mesh_size_mm=8.0, element_type=CellType3D.TET10)
    faces = classify_faces_by_bbox(imported)
    mat = _neo_hookean_3d()

    base_bcs = (
        DirichletBC3D("clamp_x0", faces["x_min"], ("x", "y", "z"), (0.0, 0.0, 0.0)),
    )
    tbc = (TractionBC3D("pull_x1", faces["x_max"], (0.6, 0.15, 0.0)),)

    req_direct = FemRequest3D(
        request_id="box-direct",
        mesh=imported.mesh,
        material=mat,
        dirichlet_bcs=base_bcs,
        traction_bcs=tbc,
        load_steps=2,
        linear_solver=LinearSolverConfig(method=LinearSolverMethod.DIRECT),
    )
    req_iter = FemRequest3D(
        request_id="box-iter",
        mesh=imported.mesh,
        material=mat,
        dirichlet_bcs=base_bcs,
        traction_bcs=tbc,
        load_steps=2,
        linear_solver=LinearSolverConfig(
            method=LinearSolverMethod.PCG_SSOR,
            rtol=1e-11,
            allow_fallback_to_direct=False,
        ),
    )
    req_gmres = FemRequest3D(
        request_id="box-gmres",
        mesh=imported.mesh,
        material=mat,
        dirichlet_bcs=base_bcs,
        traction_bcs=tbc,
        load_steps=2,
        linear_solver=LinearSolverConfig(
            method=LinearSolverMethod.GMRES_ILU,
            rtol=1e-11,
            allow_fallback_to_direct=False,
        ),
    )

    torch_backend = TorchFemBackend3D()
    res_dir, res_iter, res_gmres = torch_backend.solve_many(
        (req_direct, req_iter, req_gmres)
    )
    assert res_dir.converged and res_iter.converged and res_gmres.converged
    assert res_iter.linear_solver_stats is not None
    assert "pcg-ssor" in res_iter.linear_solver_stats["methods_used"]
    assert res_gmres.linear_solver_stats is not None
    assert "gmres-ilu" in res_gmres.linear_solver_stats["methods_used"]

    # Machine-level parity between direct LU, PCG-SSOR, and GMRES-ILU
    ref_norm = float(np.linalg.norm(res_dir.nodal_displacement))
    assert (
        float(np.linalg.norm(res_iter.nodal_displacement - res_dir.nodal_displacement))
        / ref_norm
        < 1.0e-8
    )
    assert (
        float(np.linalg.norm(res_gmres.nodal_displacement - res_dir.nodal_displacement))
        / ref_norm
        < 1.0e-8
    )

    # Also verify ScikitFemBackend3D with iterative solver vs direct & cross-backend
    sk_backend = ScikitFemBackend3D()
    sk_dir = sk_backend.solve(req_direct)
    sk_res = sk_backend.solve(req_gmres)
    assert sk_dir.converged and sk_res.converged
    assert (
        float(np.linalg.norm(sk_res.nodal_displacement - sk_dir.nodal_displacement))
        / ref_norm
        < 1.0e-8
    )
    assert (
        float(np.linalg.norm(sk_res.nodal_displacement - res_dir.nodal_displacement))
        / ref_norm
        < 1.0e-4
    )

    # Verify optional pyamg flag check is well-behaved
    if not is_pyamg_available():
        _, rep_amg = solve_linear_system(
            _spd_3d_laplacian(n_blocks=10),
            np.ones(30, dtype=np.float64),
            config=LinearSolverConfig(
                method=LinearSolverMethod.PCG_AMG,
                allow_fallback_to_direct=False,
            ),
        )
        assert not rep_amg.converged
        assert "pyamg" in rep_amg.reason
