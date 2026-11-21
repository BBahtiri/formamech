"""Phase 3 tests: dead tractions, non-conservative follower pressure, arcs, VVUQ L1-L3."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from campaigns.vvuq_suite import (
    rectangular_strip_mesh,
    run_level2_cooks_membrane,
    run_level3_follower_pressure_cantilever,
    run_level3_hgo_arterial_strip,
)
from piginot.cad.settings import SCHEMA_V2, _arc_piece
from piginot.certify import (
    CertificateVerdict,
    ReactionForceQoi,
    coercive_inertia,
    estimate_qoi_dwr,
)
from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalQuantity,
    LoadSpec,
    MaterialSpec,
    Unit,
)
from piginot.fem import FemRequest, FemSolverConfig, ScikitFemBackend
from piginot.fem.follower import is_closed_boundary_loop
from piginot.fem.mms_runner import verification_execution_profile
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh
from piginot.objectives import (
    PotentialEnergyConfig,
    TriangleDomain,
    check_energy_quadrature_overshoot,
    dead_traction_potential_energy,
    discrete_fem_residual_loss,
    enclosed_cavity_pressure_energy,
    follower_pressure_traction_residual,
)
from piginot.objectives.free_traction import boundary_quadrature


def _follower_request(element_order: int = 1, pressure: float = 0.25) -> FemRequest:
    mesh = rectangular_strip_mesh(4.0, 1.0, 3, 2, label="follower-test")
    material = MaterialSpec(
        "nh-elastomer",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(10.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.35, Unit.ONE),
        },
    )
    return FemRequest(
        request_id=f"follower-fd-p{element_order}",
        mesh=mesh,
        material=material,
        boundary_conditions=(
            BoundaryConditionSpec("clamp_left", "left", "dirichlet", ("x", "y")),
            BoundaryConditionSpec("free_right", "right", "free"),
            BoundaryConditionSpec("free_bottom", "bottom", "free"),
        ),
        loads=(
            LoadSpec(
                "top_pressure",
                "follower_pressure",
                "top",
                ("p",),
                (CanonicalQuantity(pressure, Unit.MPA),),
            ),
        ),
        qois=(),
        config=FemSolverConfig(
            element_order=element_order,
            solver_tolerance=1.0e-10,
            max_newton_iterations=25,
            plane_stress_max_iterations=32,
            load_steps=4,
            max_dofs=20_000,
        ),
        execution_profile=verification_execution_profile(),
    )


@pytest.mark.parametrize("order", [1, 2])
def test_follower_pressure_unsymmetric_tangent_matches_finite_differences(
    order: int,
) -> None:
    req = _follower_request(element_order=order, pressure=0.5)
    backend = ScikitFemBackend()
    lin = backend.linearize(req)
    assert lin.is_tangent_symmetric is False

    # Construct an admissible non-zero displacement state with free top-right corner.
    xy = lin.dof_coordinates
    u_nodal = np.column_stack((0.02 * xy[:, 0], -0.03 * xy[:, 0] ** 2 / 4.0))
    vec = lin.with_exact_dirichlet(lin.vector(u_nodal))
    st = lin.state(vec)
    k_free = lin.free_tangent(st)

    # Open boundary with free right tip MUST yield an unsymmetric tangent!
    coercive = coercive_inertia(k_free)
    assert coercive.tangent_symmetric is False
    assert coercive.relative_asymmetry > 1.0e-3
    assert coercive.positive_definite is True

    # Central finite difference directional derivative of R_free(u) matches K_free d!
    rng = np.random.default_rng(100 + order)
    d_free = rng.standard_normal(lin.constraints.free.size)
    d_free /= np.linalg.norm(d_free)
    d_full = np.zeros(lin.dof_count, dtype=np.float64)
    d_full[lin.constraints.free] = d_free
    eps = 1.0e-7
    r_plus = lin.state(vec + eps * d_full).residual[lin.constraints.free]
    r_minus = lin.state(vec - eps * d_full).residual[lin.constraints.free]
    fd_dir = (r_plus - r_minus) / (2.0 * eps)
    analytic_dir = k_free @ d_free
    assert np.allclose(analytic_dir, fd_dir, rtol=1e-6, atol=1e-8)


def test_follower_pressure_dwr_adjoint_works_with_unsymmetric_tangent() -> None:
    req = _follower_request(element_order=1, pressure=0.2)
    backend = ScikitFemBackend()
    attempt = backend.solve_many((req,))[0]
    assert attempt.success and attempt.result is not None
    lin = backend.linearize(req)
    u_fem = np.asarray(attempt.result.displacement, dtype=np.float64)
    xy = lin.dof_coordinates
    u_surr = u_fem + 1e-3 * np.column_stack((xy[:, 0] * 0.1, -xy[:, 0] * 0.2))

    qoi = ReactionForceQoi("clamp_rxn", "clamp_left")
    est = estimate_qoi_dwr(lin, u_surr, qoi)
    assert est.verdict is CertificateVerdict.OK
    assert est.guards.get("tangent_symmetric") is False
    assert est.adjoint_residual_norm is not None and est.adjoint_residual_norm < 1e-10


def test_dead_traction_follower_residual_cavity_energy_and_overshoot() -> None:
    mesh = rectangular_strip_mesh(2.0, 1.0, 2, 1, label="obj-test")
    mat = MaterialSpec(
        "nh",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(100.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
        },
    )
    verts = torch.tensor(mesh.nodes, dtype=torch.float64)
    cells = torch.tensor(mesh.cells, dtype=torch.int64)
    domain = TriangleDomain(verts, cells)

    grad_u = torch.zeros((4, 2, 2), dtype=torch.float64)
    grad_u[:, 0, 0] = 0.05
    vol_w = torch.full((4,), domain.area / 4.0, dtype=torch.float64)
    b_u = torch.tensor([[0.1, 0.0], [0.1, 0.0]], dtype=torch.float64)
    t0 = torch.tensor([[5.0, 0.0], [5.0, 0.0]], dtype=torch.float64)
    b_w = torch.tensor([0.5, 0.5], dtype=torch.float64)

    terms = dead_traction_potential_energy(
        grad_u,
        vol_w,
        mat,
        config=PotentialEnergyConfig(),
        reference_area=domain.area,
        boundary_displacement=b_u,
        nominal_traction=t0,
        boundary_weights=b_w,
    )
    assert math.isfinite(float(terms.energy.item()))

    # Zero-measure point load must be refused.
    with pytest.raises(ValueError, match="zero-measure point loads"):
        dead_traction_potential_energy(
            grad_u,
            vol_w,
            mat,
            config=PotentialEnergyConfig(),
            reference_area=domain.area,
            boundary_displacement=b_u,
            nominal_traction=t0,
            boundary_weights=torch.zeros(2, dtype=torch.float64),
        )

    # Constant affine strain has zero quadrature overshoot discrepancy between deg 2 and deg 5.
    diag = check_energy_quadrature_overshoot(
        lambda pts: (
            torch.tensor([[0.05, 0.0], [0.0, -0.015]], dtype=torch.float64)
            .unsqueeze(0)
            .expand(pts.shape[0], 2, 2)
        ),
        domain,
        mat,
    )
    assert diag.passed is True
    assert diag.relative_discrepancy < 1e-12

    # Follower pressure Piola traction residual vanishes when P N = -p cof(F) N.
    rule = boundary_quadrature(mesh, ("top",))
    grad_b = torch.zeros((rule.points.shape[0], 2, 2), dtype=torch.float64)
    # At F=I, cof(F)=I, so P = -p I satisfies P N + p I N = 0 identically.
    p_val = 2.5
    p_tensor = -p_val * torch.eye(2, dtype=torch.float64).unsqueeze(0).expand(
        rule.points.shape[0], 2, 2
    )
    res_zero = follower_pressure_traction_residual(
        p_tensor, grad_b, rule, pressure_mpa=p_val, stress_scale_mpa=100.0
    )
    assert float(res_zero.item()) == pytest.approx(0.0, abs=1e-14)

    # Open boundary refused by enclosed_cavity_pressure_energy.
    assert is_closed_boundary_loop(mesh, "top") is False
    with pytest.raises(ValueError, match="open"):
        enclosed_cavity_pressure_energy(
            mesh,
            "top",
            torch.zeros((len(mesh.nodes), 2), dtype=torch.float64),
            pressure_mpa=1.0,
            reference_area=domain.area,
            young_modulus_mpa=100.0,
        )

    # Closed loop accepted by enclosed_cavity_pressure_energy.
    all_loop_mesh = canonicalize_mesh(
        MeshPayload(
            nodes=mesh.nodes,
            cells=mesh.cells,
            cell_type=CellType.TRIANGLE,
            boundary_facets=mesh.boundary_facets,
            boundary_tags={"outer_loop": tuple(range(len(mesh.boundary_facets)))},
            region_ids=mesh.region_ids,
            unit=Unit.MM,
            source_provenance={"test": "closed-loop"},
        )
    )
    assert is_closed_boundary_loop(all_loop_mesh, "outer_loop") is True
    # Uniform 10% expansion increases enclosed area by 21% -> delta_A = 0.21 * A0.
    u_exp = 0.1 * torch.tensor(all_loop_mesh.nodes, dtype=torch.float64)
    e_cav = enclosed_cavity_pressure_energy(
        all_loop_mesh,
        "outer_loop",
        u_exp,
        pressure_mpa=10.0,
        reference_area=domain.area,
        young_modulus_mpa=100.0,
    )
    assert float(e_cav.item()) == pytest.approx(-10.0 * 0.21 / 100.0, rel=1e-6)

    loss_fem = discrete_fem_residual_loss(
        torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])
    )
    assert float(loss_fem.item()) == pytest.approx(0.0)


def test_cad_settings_v2_extracts_exact_circular_arc_piece() -> None:
    # Build a quarter-circle fan centered at (0, 0) with radius 10 mm.
    angles = [0.0, math.pi / 6.0, math.pi / 3.0, math.pi / 2.0]
    arc_pts = [(10.0 * math.cos(a), 10.0 * math.sin(a)) for a in angles]
    nodes = ((0.0, 0.0), *arc_pts)
    cells = ((0, 1, 2), (0, 2, 3), (0, 3, 4))
    facets = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))
    mesh = canonicalize_mesh(
        MeshPayload(
            nodes=nodes,
            cells=cells,
            cell_type=CellType.TRIANGLE,
            boundary_facets=facets,
            boundary_tags={"bottom": (0,), "arc_outer": (1, 2, 3), "left": (4,)},
            region_ids=(1, 1, 1),
            unit=Unit.MM,
            source_provenance={"test": "arc"},
        )
    )
    piece = _arc_piece(mesh, "arc_outer")
    assert SCHEMA_V2 == "formamech.cad-settings-v2"
    cx, cy = (float(v) for v in piece["arc"]["center"])
    assert cx == pytest.approx(0.0, abs=1e-9)
    assert cy == pytest.approx(0.0, abs=1e-9)


def test_vvuq_nonlinear_benchmarks_levels_2_and_3_execute_cleanly() -> None:
    l2 = run_level2_cooks_membrane(
        mesh_resolutions=(2, 4), element_order=2, load_steps=6
    )
    assert l2["all_jacobians_positive"] is True
    assert l2["finest_tip_uy_mm"] > 1.0
    assert l2["mesh_convergence_tip_rel_change"] is not None

    l3a = run_level3_hgo_arterial_strip(axial_strains=(0.05, 0.20), element_order=1)
    assert l3a["all_jacobians_positive"] is True
    assert l3a["exhibits_strain_stiffening"] is True

    l3b = run_level3_follower_pressure_cantilever(pressure_mpa=0.04, element_order=1)
    assert l3b["tangent_symmetric"] is False
    assert l3b["coercive_positive_definite"] is True
    assert l3b["warm_start_iteration_reduction_ratio"] > 1.0
    assert l3b["min_det_f_2d"] > 0.0
