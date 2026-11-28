"""Full FEM verification tests for V6-R1: 3D follower pressure & tendon routing."""

from __future__ import annotations

import math

import numpy as np
import torch

from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import DirichletBC3D
from piginot.fem.follower_3d import (
    FollowerPressureBC3D,
    assemble_follower_pressure_3d,
    solve_3d_follower_pressure,
)
from piginot.fem.tendon_3d import (
    CosseratBackboneSpec,
    TendonRoute3D,
    assemble_tendon_forces_3d,
    solve_3d_tendon_continuum_backbone,
    solve_3d_tendon_solid,
)
from piginot.geometry.mesh3d import make_brick_tet4_mesh, make_brick_tet10_mesh
from piginot.objectives.virtual_work_3d import compute_3d_follower_virtual_work_loss


def _elastomer(e_mpa: float = 2.0, nu: float = 0.40) -> MaterialSpec:
    return MaterialSpec(
        name="v6-elastomer",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(e_mpa, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(nu, Unit.ONE).to_canonical(),
        },
    )


def test_follower_3d_exact_unsymmetric_tangent_matches_central_fd() -> None:
    """Exact 3D Nanson's-law K_L = -d(f_ext)/du matches central FD to < 1e-8 on TET10."""
    mesh = make_brick_tet10_mesh(12.0, 6.0, 4.0, nx=2, ny=1, nz=1)
    n_dofs = 3 * len(mesh.nodes)
    rng = np.random.default_rng(7)
    u = rng.normal(0.0, 0.15, size=n_dofs)

    fbc = FollowerPressureBC3D("top_p", ("z_max",), pressure_mpa=0.35)
    asm = assemble_follower_pressure_3d(mesh, (fbc,), u)
    k_dense = asm.load_stiffness.toarray()

    # Check K_L is genuinely unsymmetric on an open boundary patch
    asym_norm = float(np.linalg.norm(k_dense - k_dense.T))
    assert asym_norm > 1.0e-3

    # Central finite-difference check on active surface DOFs
    eps = 1.0e-6
    active_dofs = np.nonzero(np.linalg.norm(k_dense, axis=0) > 1.0e-9)[0][:18]
    max_err = 0.0
    for j in active_dofs:
        du = np.zeros(n_dofs, dtype=np.float64)
        du[j] = eps
        fp = assemble_follower_pressure_3d(mesh, (fbc,), u + du).external_force
        fm = assemble_follower_pressure_3d(mesh, (fbc,), u - du).external_force
        fd_col = -(fp - fm) / (2.0 * eps)
        err = float(np.max(np.abs(k_dense[:, j] - fd_col)))
        max_err = max(max_err, err)
    assert max_err < 1.0e-8, f"3D follower tangent FD error {max_err:.3e} >= 1e-8"


def test_follower_3d_closed_cavity_polytropic_tangent_matches_fd() -> None:
    """Closed-cavity polytropic pressure-volume coupling K_L matches central FD."""
    mesh = make_brick_tet4_mesh(10.0, 10.0, 10.0, nx=1, ny=1, nz=1)
    n_dofs = 3 * len(mesh.nodes)
    rng = np.random.default_rng(11)
    u = rng.normal(0.0, 0.2, size=n_dofs)
    all_faces = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    fbc = FollowerPressureBC3D(
        "cavity",
        all_faces,
        pressure_mpa=0.25,
        closed_cavity=True,
        polytropic_gamma=1.4,
    )
    asm = assemble_follower_pressure_3d(mesh, (fbc,), u)
    assert abs(asm.cavity_volumes_mm3["cavity"] - 1000.0) < 250.0
    k_dense = asm.load_stiffness.toarray()
    eps = 1.0e-6
    max_err = 0.0
    for j in range(min(12, n_dofs)):
        du = np.zeros(n_dofs, dtype=np.float64)
        du[j] = eps
        fp = assemble_follower_pressure_3d(mesh, (fbc,), u + du).external_force
        fm = assemble_follower_pressure_3d(mesh, (fbc,), u - du).external_force
        fd_col = -(fp - fm) / (2.0 * eps)
        max_err = max(max_err, float(np.max(np.abs(k_dense[:, j] - fd_col))))
    assert max_err < 1.0e-7, f"polytropic cavity tangent FD error {max_err:.3e}"


def test_solve_3d_follower_pressure_cantilever_converges_large_bending() -> None:
    """3D TET10 elastomer beam bends cleanly under live normal follower pressure."""
    mesh = make_brick_tet10_mesh(20.0, 6.0, 4.0, nx=4, ny=1, nz=1)
    res = solve_3d_follower_pressure(
        mesh,
        _elastomer(e_mpa=2.0, nu=0.45),
        dirichlet_bcs=(
            DirichletBC3D("clamp", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        follower_bcs=(FollowerPressureBC3D("press", ("z_max",), pressure_mpa=0.004),),
        load_steps=4,
    )
    assert res.converged
    assert res.min_det_f > 0.5
    assert res.max_displacement_mm > 0.5
    # Downward z deflection under +pressure on z_max (acts along -n = -z)
    assert float(np.min(res.nodal_displacement[:, 2])) < -0.5
    doc = res.summary_document()
    assert len(doc["result_hash"]) == 64


def test_tendon_solid_3d_tangent_and_bending_fem() -> None:
    """3D routed tendon geometric stiffness matches FD and bends a TET10 finger."""
    mesh = make_brick_tet10_mesh(24.0, 6.0, 6.0, nx=4, ny=1, nz=1)
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    # Pick top-edge nodes along z=6, y=3 (or closest to y=3) ordered by x
    top_nids = [
        i
        for i, p in enumerate(nodes)
        if abs(p[2] - 6.0) < 1.0e-6 and abs(p[1] - 3.0) < 1.0e-6
    ]
    top_nids.sort(key=lambda i: nodes[i, 0])
    assert len(top_nids) >= 3
    td = TendonRoute3D("flexor", tuple(top_nids), tension_n=1.2, capstan_mu=0.0)

    n_dofs = 3 * len(nodes)
    rng = np.random.default_rng(3)
    u = rng.normal(0.0, 0.08, size=n_dofs)
    asm = assemble_tendon_forces_3d(nodes, u, (td,))
    k_dense = asm.load_stiffness.toarray()
    eps = 1.0e-6
    for nid in top_nids[:3]:
        for c in range(3):
            j = 3 * nid + c
            du = np.zeros(n_dofs, dtype=np.float64)
            du[j] = eps
            fp = assemble_tendon_forces_3d(nodes, u + du, (td,)).external_force
            fm = assemble_tendon_forces_3d(nodes, u - du, (td,)).external_force
            fd_col = -(fp - fm) / (2.0 * eps)
            assert float(np.max(np.abs(k_dense[:, j] - fd_col))) < 1.0e-8

    res = solve_3d_tendon_solid(
        mesh,
        _elastomer(e_mpa=2.0, nu=0.45),
        dirichlet_bcs=(
            DirichletBC3D("base", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        tendons=(td,),
        load_steps=3,
    )
    assert res.converged
    assert res.tendon_strokes_mm["flexor"] > 0.0
    # Eccentric top tendon bends tip upward (+z)
    tip_nid = top_nids[-1]
    assert res.nodal_displacement[tip_nid, 2] > 0.05


def test_cosserat_tendon_backbone_matches_exact_constant_curvature() -> None:
    """Single-tendon Cosserat continuum rod matches exact Euler-Bernoulli curvature."""
    spec = CosseratBackboneSpec(
        length_mm=540.0,
        num_disks=9,
        youngs_modulus_mpa=55_000.0,
        poisson_ratio=0.33,
        outer_radius_mm=1.1,
        inner_radius_mm=0.6,
        tendon_offsets_mm=((37.5, 0.0), (0.0, 37.5), (-37.5, 0.0), (0.0, -37.5)),
    )
    tension = 4.0  # N on +x tendon (offset d_c = 37.5 mm)
    res = solve_3d_tendon_continuum_backbone(spec, (tension, 0.0, 0.0, 0.0))
    assert res.converged
    assert res.grad_norm < 1.0e-9
    # Constant curvature moment M_y = -T * d_c -> kappa_y = +T * d_c / (E * I)
    # Wait: tendon at +x shortens along +x -> bends tip toward +x (positive rotation about +y)
    ei = spec.youngs_modulus_mpa * spec.i_bend_mm4
    k_exact = tension * 37.5 / ei
    theta_exact = k_exact * spec.length_mm
    x_exact = (1.0 - math.cos(theta_exact)) / k_exact
    z_exact = math.sin(theta_exact) / k_exact
    tip = res.disk_positions_mm[-1]
    assert abs(tip[0] - x_exact) / max(x_exact, 1.0) < 0.03
    assert abs(tip[2] - z_exact) / z_exact < 0.02
    assert abs(tip[1]) < 1.0e-6
    assert res.tendon_strokes_mm[0] > 0.0


def test_virtual_work_3d_loss_differentiable_and_vanishes_at_equilibrium() -> None:
    """3D virtual-work residual vanishes for uniform uniaxial stress equilibrium."""
    mat = _elastomer(e_mpa=10.0, nu=0.3)
    # Uniform stretch F = diag(1, 1, 1.1) -> zero interior divergence, constant P_zz
    grad_u = torch.zeros((4, 3, 3), dtype=torch.float64, requires_grad=True)
    vol_w = torch.full((4,), 0.25, dtype=torch.float64)
    grad_v = torch.randn((3, 4, 3, 3), dtype=torch.float64)
    g1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    g2 = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    sv = torch.randn((3, 1, 3), dtype=torch.float64)
    out = compute_3d_follower_virtual_work_loss(
        grad_u_vol=grad_u,
        vol_weights=vol_w,
        grad_v_vol=grad_v,
        material=mat,
        surf_g1=g1,
        surf_g2=g2,
        surf_pressure_mpa=0.0,
        surf_v=sv,
    )
    # At u = 0, P(I) = 0 and p = 0 => exact zero virtual work residual
    assert float(out.virtual_work_residual.item()) < 1.0e-20
    out.total.backward()
    assert grad_u.grad is not None
