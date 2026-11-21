"""Unit tests for 2.5D potential energies, dead tractions, follower pressure, and DEM guards."""

from __future__ import annotations

import math

import pytest
import torch

from campaigns.vvuq_suite import rectangular_strip_mesh
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.objectives import (
    PotentialEnergyConfig,
    TriangleDomain,
    axisymmetric_potential_energy,
    check_energy_quadrature_overshoot,
    dead_traction_potential_energy,
    discrete_fem_residual_loss,
    enclosed_cavity_pressure_energy,
    follower_pressure_traction_residual,
    plane_strain_potential_energy,
)
from piginot.objectives.free_traction import boundary_quadrature


def _neo_hookean(e_mpa: float = 10.0, nu: float = 0.35) -> MaterialSpec:
    return MaterialSpec(
        name="nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(e_mpa, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(nu, Unit.ONE).to_canonical(),
        },
    )


def test_plane_strain_and_axisymmetric_potential_energy_and_guards() -> None:
    mat = _neo_hookean(10.0, 0.35)
    grad = torch.zeros((4, 2, 2), dtype=torch.float64)
    grad[:, 0, 0] = 0.1
    w = torch.full((4,), 0.25, dtype=torch.float64)

    ps = plane_strain_potential_energy(grad, w, mat)
    assert ps.inadmissible_points == 0
    assert ps.minimum_determinant == pytest.approx(1.1)
    assert float(ps.energy.item()) > 0.0
    assert float(ps.barrier.item()) == 0.0

    # Inverted deformation triggers admissibility barrier cleanly
    grad_bad = grad.clone()
    grad_bad[0, 0, 0] = -1.5
    ps_bad = plane_strain_potential_energy(grad_bad, w, mat)
    assert ps_bad.inadmissible_points == 1
    assert float(ps_bad.barrier.item()) > 0.0

    # Plane-strain input validation
    with pytest.raises(ValueError, match="displacement_gradient"):
        plane_strain_potential_energy(torch.zeros((4, 2), dtype=torch.float64), w, mat)
    with pytest.raises(ValueError, match="weights"):
        plane_strain_potential_energy(grad, torch.ones((2,), dtype=torch.float64), mat)
    with pytest.raises(ValueError, match="reference_area"):
        plane_strain_potential_energy(grad, w, mat, reference_area=0.0)

    # Axisymmetric potential energy
    u_ax = torch.tensor(
        [[0.05, 0.0], [0.06, 0.01], [0.05, 0.02], [0.04, 0.0]], dtype=torch.float64
    )
    r_ax = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    ax = axisymmetric_potential_energy(u_ax, grad, r_ax, w, mat)
    assert ax.inadmissible_points == 0
    assert float(ax.energy.item()) > 0.0

    # Axisymmetric validation guards
    with pytest.raises(ValueError, match="displacement must"):
        axisymmetric_potential_energy(
            torch.zeros((4, 3), dtype=torch.float64), grad, r_ax, w, mat
        )
    with pytest.raises(ValueError, match="displacement_gradient"):
        axisymmetric_potential_energy(
            u_ax, torch.zeros((4, 2), dtype=torch.float64), r_ax, w, mat
        )
    with pytest.raises(ValueError, match="radii must"):
        axisymmetric_potential_energy(
            u_ax, grad, torch.ones((2,), dtype=torch.float64), w, mat
        )
    with pytest.raises(ValueError, match="weights must"):
        axisymmetric_potential_energy(
            u_ax, grad, r_ax, torch.ones((2,), dtype=torch.float64), mat
        )
    with pytest.raises(ValueError, match="strictly positive"):
        axisymmetric_potential_energy(
            u_ax, grad, torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64), w, mat
        )
    with pytest.raises(ValueError, match="reference_volume"):
        axisymmetric_potential_energy(u_ax, grad, r_ax, w, mat, reference_volume=-1.0)


def test_dead_traction_overshoot_follower_cavity_and_discrete_residual() -> None:
    mat = _neo_hookean(10.0, 0.32)
    cfg = PotentialEnergyConfig()
    mesh = rectangular_strip_mesh(length_mm=4.0, height_mm=2.0, nx=2, ny=1)
    dom = TriangleDomain(
        torch.tensor(mesh.nodes, dtype=torch.float64),
        torch.tensor(mesh.cells, dtype=torch.int64),
    )

    grad = torch.zeros((4, 2, 2), dtype=torch.float64)
    grad[:, 0, 0] = 0.05
    vw = torch.full((4,), 2.0, dtype=torch.float64)
    bu = torch.tensor([[0.2, 0.0], [0.2, 0.0]], dtype=torch.float64)
    bt = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    bw = torch.tensor([1.0, 1.0], dtype=torch.float64)

    dt = dead_traction_potential_energy(
        grad,
        vw,
        mat,
        config=cfg,
        reference_area=dom.area,
        boundary_displacement=bu,
        nominal_traction=bt,
        boundary_weights=bw,
    )
    assert math.isfinite(float(dt.total.item()))

    # Zero-measure point load & shape/thickness refusals
    with pytest.raises(ValueError, match="zero-measure point loads"):
        dead_traction_potential_energy(
            grad,
            vw,
            mat,
            config=cfg,
            reference_area=dom.area,
            boundary_displacement=bu,
            nominal_traction=bt,
            boundary_weights=torch.zeros_like(bw),
        )
    with pytest.raises(ValueError, match="boundary_displacement"):
        dead_traction_potential_energy(
            grad,
            vw,
            mat,
            config=cfg,
            reference_area=dom.area,
            boundary_displacement=torch.zeros((2, 3), dtype=torch.float64),
            nominal_traction=bt,
            boundary_weights=bw,
        )
    with pytest.raises(ValueError, match="nominal_traction"):
        dead_traction_potential_energy(
            grad,
            vw,
            mat,
            config=cfg,
            reference_area=dom.area,
            boundary_displacement=bu,
            nominal_traction=torch.zeros((1, 2), dtype=torch.float64),
            boundary_weights=bw,
        )
    with pytest.raises(ValueError, match="boundary_weights must have shape"):
        dead_traction_potential_energy(
            grad,
            vw,
            mat,
            config=cfg,
            reference_area=dom.area,
            boundary_displacement=bu,
            nominal_traction=bt,
            boundary_weights=torch.ones((1,), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="non-negative"):
        dead_traction_potential_energy(
            grad,
            vw,
            mat,
            config=cfg,
            reference_area=dom.area,
            boundary_displacement=bu,
            nominal_traction=bt,
            boundary_weights=torch.tensor([-1.0, 2.0], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="thickness"):
        dead_traction_potential_energy(
            grad,
            vw,
            mat,
            config=cfg,
            reference_area=dom.area,
            boundary_displacement=bu,
            nominal_traction=bt,
            boundary_weights=bw,
            thickness=0.0,
        )

    # DEM quadrature overshoot check on constant-strain field passes identically
    diag = check_energy_quadrature_overshoot(
        lambda pts: (
            torch.zeros((pts.shape[0], 2, 2), dtype=torch.float64)
            + torch.tensor([[0.05, 0.0], [0.0, -0.02]], dtype=torch.float64)
        ),
        dom,
        mat,
    )
    assert diag.passed is True
    assert diag.relative_discrepancy < 1.0e-12
    with pytest.raises(ValueError, match="max_relative_discrepancy"):
        check_energy_quadrature_overshoot(
            lambda pts: torch.zeros((pts.shape[0], 2, 2), dtype=torch.float64),
            dom,
            mat,
            max_relative_discrepancy=0.0,
        )

    # Non-conservative follower pressure traction residual
    rule = boundary_quadrature(mesh, ("top",))
    n_b = rule.points.shape[0]
    p_val = 0.5
    p_tensor = torch.zeros((n_b, 2, 2), dtype=torch.float64)
    p_tensor[:, 0, 0] = -p_val
    p_tensor[:, 1, 1] = -p_val
    zero_g = torch.zeros((n_b, 2, 2), dtype=torch.float64)
    res_zero = follower_pressure_traction_residual(
        p_tensor, zero_g, rule, pressure_mpa=p_val, stress_scale_mpa=10.0
    )
    assert float(res_zero.item()) == pytest.approx(0.0, abs=1.0e-14)
    res_tensor_p = follower_pressure_traction_residual(
        p_tensor,
        zero_g,
        rule,
        pressure_mpa=torch.full((n_b,), p_val, dtype=torch.float64),
        stress_scale_mpa=10.0,
    )
    assert float(res_tensor_p.item()) == pytest.approx(0.0, abs=1.0e-14)
    with pytest.raises(ValueError, match="first_piola_2d"):
        follower_pressure_traction_residual(
            torch.zeros((n_b, 2), dtype=torch.float64),
            zero_g,
            rule,
            pressure_mpa=p_val,
            stress_scale_mpa=10.0,
        )
    with pytest.raises(ValueError, match="displacement_gradient_2d"):
        follower_pressure_traction_residual(
            p_tensor,
            torch.zeros((1, 2, 2), dtype=torch.float64),
            rule,
            pressure_mpa=p_val,
            stress_scale_mpa=10.0,
        )
    with pytest.raises(ValueError, match="BoundaryQuadrature"):
        follower_pressure_traction_residual(
            torch.zeros((1, 2, 2), dtype=torch.float64),
            torch.zeros((1, 2, 2), dtype=torch.float64),
            rule,
            pressure_mpa=p_val,
            stress_scale_mpa=10.0,
        )
    with pytest.raises(ValueError, match="stress_scale_mpa"):
        follower_pressure_traction_residual(
            p_tensor, zero_g, rule, pressure_mpa=p_val, stress_scale_mpa=0.0
        )

    # Open boundary refused for enclosed_cavity_pressure_energy; closed boundary accepted
    u_nodes = torch.zeros((len(mesh.nodes), 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="unknown or empty"):
        enclosed_cavity_pressure_energy(
            mesh,
            "missing",
            u_nodes,
            pressure_mpa=1.0,
            reference_area=dom.area,
            young_modulus_mpa=10.0,
        )
    with pytest.raises(ValueError, match="is open"):
        enclosed_cavity_pressure_energy(
            mesh,
            "top",
            u_nodes,
            pressure_mpa=1.0,
            reference_area=dom.area,
            young_modulus_mpa=10.0,
        )
    # Build closed loop tag from all outer boundary facets
    from dataclasses import replace

    closed_mesh = replace(
        mesh,
        semantic_tags={"outer_loop": tuple(range(len(mesh.boundary_facets)))},
    )
    u_expand = 0.1 * torch.tensor(closed_mesh.nodes, dtype=torch.float64)
    e_cav = enclosed_cavity_pressure_energy(
        closed_mesh,
        "outer_loop",
        u_expand,
        pressure_mpa=1.0,
        reference_area=dom.area,
        young_modulus_mpa=10.0,
    )
    # Uniform 10% expansion increases area by 21% -> -p * 0.21*A / (E*A) = -0.021
    assert float(e_cav.item()) == pytest.approx(-0.021, rel=1.0e-10)
    with pytest.raises(ValueError, match="nodal_displacement"):
        enclosed_cavity_pressure_energy(
            closed_mesh,
            "outer_loop",
            torch.zeros((1, 2), dtype=torch.float64),
            pressure_mpa=1.0,
            reference_area=dom.area,
            young_modulus_mpa=10.0,
        )

    # Discrete FEM residual loss
    r_loss = discrete_fem_residual_loss(
        torch.tensor([1.0, -1.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        force_scale=2.0,
    )
    assert float(r_loss.item()) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="align"):
        discrete_fem_residual_loss(torch.zeros(2), torch.zeros(3))
    with pytest.raises(ValueError, match="force_scale"):
        discrete_fem_residual_loss(torch.zeros(2), torch.zeros(2), force_scale=-1.0)
