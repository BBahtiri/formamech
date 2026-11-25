"""Verification tests for Phase 4 (V4-4) 3D SIMP topology opt & Bernstein FFD shape opt."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import (
    DirichletBC3D,
    TractionBC3D,
    assemble_surface_tractions_3d,
    face_node_indices_3d,
)
from piginot.geometry.mesh3d import make_brick_tet4_mesh
from piginot.harness import FormaMechAgentHarness
from piginot.optimization.ffd_shape_opt_3d import (
    FFDGrid3D,
    FfdMeshInversionError,
    optimize_3d_ffd_shape,
)
from piginot.optimization.topology_opt_3d import (
    TopologyOpt3DConfig,
    TopologyOpt3DState,
    _precompute_tet4_linear_elements,
    optimize_3d_topology_simp,
    simp_3d_compliance_and_sensitivity,
)


def test_simp_3d_adjoint_sensitivity_matches_central_fd() -> None:
    """Exact self-adjoint 3D SIMP sensitivity dC/d(rho_e) matches central FD to < 1e-7."""
    mesh = make_brick_tet4_mesh(12.0, 6.0, 6.0, nx=2, ny=1, nz=1)
    n_dofs = 3 * len(mesh.nodes)
    k_e0, vols, _, elem_u_dofs, u_rows, u_cols = _precompute_tet4_linear_elements(
        mesh, e_mpa=1000.0, nu=0.3
    )
    f_vec = assemble_surface_tractions_3d(
        mesh, (TractionBC3D("shear", ("x_max",), (0.0, -1.0, 0.0)),)
    ).reshape(n_dofs)
    constrained = np.zeros(n_dofs, dtype=bool)
    for nid in face_node_indices_3d(mesh, ("x_min",)):
        constrained[3 * nid : 3 * nid + 3] = True
    free = np.nonzero(~constrained)[0]

    rng = np.random.default_rng(11)
    rho = rng.uniform(0.3, 0.9, size=vols.size)
    _, dc_adj, _ = simp_3d_compliance_and_sensitivity(
        rho,
        k_e0=k_e0,
        elem_u_dofs=elem_u_dofs,
        u_rows=u_rows,
        u_cols=u_cols,
        free=free,
        f_vec=f_vec,
        n_dofs=n_dofs,
        penalization=3.0,
        min_density=1.0e-3,
    )
    eps = 1.0e-6
    for idx in range(min(6, vols.size)):
        rp = rho.copy()
        rm = rho.copy()
        rp[idx] += eps
        rm[idx] -= eps
        cp, _, _ = simp_3d_compliance_and_sensitivity(
            rp,
            k_e0=k_e0,
            elem_u_dofs=elem_u_dofs,
            u_rows=u_rows,
            u_cols=u_cols,
            free=free,
            f_vec=f_vec,
            n_dofs=n_dofs,
            penalization=3.0,
            min_density=1.0e-3,
        )
        cm, _, _ = simp_3d_compliance_and_sensitivity(
            rm,
            k_e0=k_e0,
            elem_u_dofs=elem_u_dofs,
            u_rows=u_rows,
            u_cols=u_cols,
            free=free,
            f_vec=f_vec,
            n_dofs=n_dofs,
            penalization=3.0,
            min_density=1.0e-3,
        )
        fd = (cp - cm) / (2.0 * eps)
        rel_err = abs(dc_adj[idx] - fd) / max(abs(fd), 1.0e-12)
        assert rel_err < 1.0e-7, f"element {idx} adjoint error {rel_err:.2e} >= 1e-7"


def test_verified_3d_simp_topology_opt_certifies_and_refuses_empty() -> None:
    """3D SIMP strictly reduces compliance, certifies thresholded solid via FEM, and refuses empty."""
    mesh = make_brick_tet4_mesh(24.0, 12.0, 6.0, nx=4, ny=2, nz=1)
    clamp = (DirichletBC3D("clamp", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),)
    load = (TractionBC3D("tip", ("x_max",), (0.0, -1.0, 0.0)),)

    res = optimize_3d_topology_simp(
        mesh,
        dirichlet_bcs=clamp,
        traction_bcs=load,
        config=TopologyOpt3DConfig(
            volume_fraction=0.55,
            penalization=3.0,
            filter_radius_mm=5.5,
            max_iterations=15,
            threshold_density=0.35,
        ),
    )
    assert res.state is TopologyOpt3DState.CERTIFIED
    assert res.claim_level == "SINGLE_POINT_VERIFIED"
    assert res.final_simp_compliance_mj < res.initial_compliance_mj
    assert abs(res.achieved_volume_fraction - 0.55) < 0.03
    assert (
        res.certified_fem_compliance_mj is not None
        and res.certified_fem_compliance_mj > 0.0
    )
    assert len(res.receipt_sha256) == 64

    # Empty threshold refuses fail-closed!
    res_ref = optimize_3d_topology_simp(
        mesh,
        dirichlet_bcs=clamp,
        traction_bcs=load,
        config=TopologyOpt3DConfig(
            volume_fraction=0.50,
            max_iterations=3,
            threshold_density=1.01,
        ),
    )
    assert res_ref.state is TopologyOpt3DState.REFUSED
    assert res_ref.claim_level == "NOT_VERIFIED"
    assert res_ref.certified_fem_compliance_mj is None


def test_bernstein_ffd_refuses_inversion_and_reduces_3d_mass() -> None:
    """FFD cage refuses self-intersecting moves and reduces volume under FEM displacement limit."""
    mesh = make_brick_tet4_mesh(20.0, 8.0, 8.0, nx=3, ny=1, nz=1)
    cage = FFDGrid3D.from_mesh(mesh, nx=3, ny=2, nz=2)

    # Deliberately collapse/invert opposite x faces -> FfdMeshInversionError!
    bad_delta = np.zeros((cage.num_control_points, 3), dtype=np.float64)
    bad_delta[:4, 0] = 35.0  # push x=0 control plane past x=20 plane!
    with pytest.raises(FfdMeshInversionError):
        cage.morph_mesh(mesh, bad_delta)

    mat = MaterialSpec(
        name="ffd-nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(100.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.30, Unit.ONE).to_canonical(),
        },
    )
    res = optimize_3d_ffd_shape(
        mesh,
        mat,
        dirichlet_bcs=(
            DirichletBC3D("clamp", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        traction_bcs=(TractionBC3D("load", ("x_max",), (0.3, 0.0, 0.0)),),
        max_displacement_limit_mm=0.25,
        iterations=5,
    )
    assert res.min_det_j > 0.0
    assert res.optimized_volume_mm3 < res.initial_volume_mm3
    assert res.optimized_max_displacement_mm <= 0.25
    assert len(res.receipt_sha256) == 64


def test_harness_invokes_v4_4_generative_design_capabilities(tmp_path: Path) -> None:
    """Autonomous agents can invoke 3D SIMP and FFD shape optimization via harness."""
    harness = FormaMechAgentHarness(tmp_path / "agent_v4_4")
    e1 = harness.invoke(
        "optimize.topology_simp_3d",
        {
            "nx": 3,
            "ny": 2,
            "nz": 1,
            "max_iterations": 6,
            "volume_fraction": 0.6,
            "threshold_density": 0.3,
        },
    )
    assert e1.ok and e1.result is not None and e1.result["state"] == "CERTIFIED"

    e2 = harness.invoke(
        "optimize.ffd_shape_3d",
        {"iterations": 3, "max_displacement_limit_mm": 0.30},
    )
    assert e2.ok and e2.result is not None and e2.result["min_det_j"] > 0.0


def test_v4_4_audit_regressions_tet10_ffd_inversion_and_simp_guards() -> None:
    """FFD catches TET10 midside inversion; SIMP refuses non-zero Dirichlet & floating islands."""
    from piginot.geometry.mesh3d import make_brick_tet10_mesh
    from piginot.optimization.topology_opt_3d import _has_floating_solid_island

    # 1) TET10 midside-node inversion check via full isoparametric quadrature
    mesh10 = make_brick_tet10_mesh(10.0, 4.0, 4.0, nx=1, ny=1, nz=1)
    cage10 = FFDGrid3D.from_mesh(mesh10, nx=3, ny=2, nz=2)
    delta_fold = np.zeros((cage10.num_control_points, 3), dtype=np.float64)
    delta_fold[-4:, 0] = -25.0  # fold right face past left face
    with pytest.raises(FfdMeshInversionError):
        cage10.morph_mesh(mesh10, delta_fold)

    # 2) Floating island detector refuses unsupported disconnected tetrahedron
    cells_two = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    supported = np.array([True, True, False, False, False, False, False, False])
    assert _has_floating_solid_island(cells_two, supported, 8) is True
    supported_both = np.array([True, False, False, False, True, False, False, False])
    assert _has_floating_solid_island(cells_two, supported_both, 8) is False

    # 3) SIMP refuses non-zero Dirichlet BC fail-closed
    mesh4 = make_brick_tet4_mesh(12.0, 4.0, 4.0, nx=2, ny=1, nz=1)
    with pytest.raises(ValueError, match="homogeneous"):
        optimize_3d_topology_simp(
            mesh4,
            dirichlet_bcs=(DirichletBC3D("nz", ("x_min",), ("x",), (0.2,)),),
            traction_bcs=(TractionBC3D("load", ("x_max",), (1.0, 0.0, 0.0)),),
        )
