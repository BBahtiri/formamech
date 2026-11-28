"""Full FEM & autograd tests for V6-R2: 3D SDF contact & DiffMJX gradients."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import DirichletBC3D
from piginot.fem.diff_contact_3d import (
    RigidSdfIndenter3D,
    SdfPrimitiveKind,
    assemble_sdf_contact_3d,
    diffmjx_straight_through_contact_force,
    solve_3d_sdf_contact,
)
from piginot.geometry.mesh3d import make_brick_tet10_mesh


def _gel_material() -> MaterialSpec:
    return MaterialSpec(
        name="gelsight-gel",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(0.30, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.48, Unit.ONE).to_canonical(),
        },
    )


def test_sdf_contact_consistent_tangent_matches_fd_with_friction() -> None:
    """Exact 3D SDF contact consistent tangent matches central FD with Coulomb friction."""
    mesh = make_brick_tet10_mesh(12.0, 12.0, 4.0, nx=2, ny=2, nz=1)
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    top_nids = tuple(i for i, p in enumerate(nodes) if abs(p[2] - 4.0) < 1.0e-6)
    ind = RigidSdfIndenter3D(
        name="punch-fd",
        kind=SdfPrimitiveKind.SPHERE,
        contact_face_tags=("z_max",),
        center_mm=(6.0, 6.0, 6.4),
        radius_mm=3.0,
        tangential_shift_mm=(0.15, -0.10, 0.0),
        penalty_mpa_per_mm=40.0,
        friction_mu=0.5,
        friction_reg_mm=1.0e-2,
    )
    u = np.zeros(3 * len(nodes), dtype=np.float64)
    lam = np.zeros(len(top_nids), dtype=np.float64)
    _, k_c, _, _, _, active, _, _, _ = assemble_sdf_contact_3d(
        nodes, u, top_nids, ind, lam
    )
    assert active > 0
    k_dense = k_c.toarray()
    eps = 1.0e-6
    active_dofs = np.nonzero(np.linalg.norm(k_dense, axis=0) > 1.0e-9)[0][:12]
    max_err = 0.0
    for j in active_dofs:
        du = np.zeros_like(u)
        du[j] = eps
        fp = assemble_sdf_contact_3d(nodes, u + du, top_nids, ind, lam)[0]
        fm = assemble_sdf_contact_3d(nodes, u - du, top_nids, ind, lam)[0]
        fd_col = (fp - fm) / (2.0 * eps)
        max_err = max(max_err, float(np.max(np.abs(k_dense[:, j] - fd_col))))
    assert max_err < 1.0e-6, f"SDF contact tangent FD err {max_err:.3e}"


@pytest.mark.parametrize(
    "kind",
    [
        SdfPrimitiveKind.SPHERE,
        SdfPrimitiveKind.FLAT_PUNCH,
        SdfPrimitiveKind.BOX,
        SdfPrimitiveKind.WEDGE_EDGE,
    ],
)
def test_solve_3d_sdf_contact_primitives_converge_on_tet10(
    kind: SdfPrimitiveKind,
) -> None:
    """All 3D rigid SDF indenter geometries converge cleanly on incompressible TET10 gel."""
    mesh = make_brick_tet10_mesh(16.0, 16.0, 4.0, nx=3, ny=3, nz=1)
    ind = RigidSdfIndenter3D(
        name=f"ind-{kind.value}",
        kind=kind,
        contact_face_tags=("z_max",),
        initial_center_mm=(8.0, 8.0, 7.05),
        center_mm=(8.0, 8.0, 6.45),
        radius_mm=3.0,
        half_extents_mm=(3.0, 3.0, 3.0),
        corner_rounding_mm=0.35,
        penalty_mpa_per_mm=25.0,
        friction_mu=0.35 if kind is SdfPrimitiveKind.SPHERE else 0.0,
        tangential_shift_mm=(0.15, 0.0, 0.0)
        if kind is SdfPrimitiveKind.SPHERE
        else (0.0, 0.0, 0.0),
    )
    res = solve_3d_sdf_contact(
        mesh,
        _gel_material(),
        dirichlet_bcs=(
            DirichletBC3D("base", ("z_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        indenter=ind,
        load_steps=2,
        max_al_iterations=5,
        penetration_tol_mm=5.0e-3,
    )
    assert res.converged, f"{kind.value} failed: {res.reason}"
    assert res.active_contact_nodes > 0
    assert res.total_contact_force_n[2] < -1.0e-3  # downward normal force on gel pad
    assert res.min_det_f > 0.4
    assert len(res.summary_document()["result_hash"]) == 64


def test_diffmjx_straight_through_contact_hard_forward_smooth_backward() -> None:
    """Separated bodies produce exact 0 forward force yet nonzero smooth backward gradient."""
    gap = torch.tensor([0.5, -0.2], dtype=torch.float64, requires_grad=True)
    normals = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=torch.float64)
    forces = diffmjx_straight_through_contact_force(
        gap, normals, penalty_mpa_per_mm=100.0, cfd_temperature_mm=0.5
    )
    # Forward pass is strictly hard contact: separated node (gap=+0.5) has exact zero force
    assert float(forces[0, 2].item()) == 0.0
    assert abs(float(forces[1, 2].item()) - (-20.0)) < 1.0e-12

    # Backward pass yields nonzero smooth gradient even for the separated body (gap=+0.5)!
    loss = forces[:, 2].sum()
    loss.backward()
    assert gap.grad is not None
    assert float(gap.grad[0].item()) > 1.0  # nonzero Contacts-From-Distance signal!
    assert float(gap.grad[1].item()) > 50.0
