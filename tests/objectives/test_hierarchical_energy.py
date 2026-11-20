"""Tests for Feature A: hierarchical physics-consistency energy & quadrature checks."""

from __future__ import annotations

import pytest
import torch

from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.objectives import (
    HierarchicalEnergyConfig,
    PotentialEnergyConfig,
    TriangleDomain,
    check_quadrature_energy_convergence,
    deterministic_triangle_quadrature,
    hierarchical_plane_stress_energy,
    subdivide_triangle_domain,
)


def _neo_hookean() -> MaterialSpec:
    return MaterialSpec(
        name="test-nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(100.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.3, Unit.ONE).to_canonical(),
        },
    )


def _domain() -> TriangleDomain:
    vertices = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=torch.float64
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    return TriangleDomain(vertices=vertices, cells=cells)


def test_deterministic_quadrature_integrates_polynomials_exactly() -> None:
    domain = _domain()
    # Area exact for all orders; degree 2 integrates quadratic exactly; degree 5 integrates x^4 y.
    for degree, expected_count in ((1, 2), (2, 6), (5, 14)):
        pts, wts = deterministic_triangle_quadrature(domain, degree=degree)
        assert pts.shape == (expected_count, 2)
        assert float(wts.sum()) == pytest.approx(2.0, abs=1e-14)
        # Linear integral int_0^2 int_0^1 (3x + 2y) dy dx = 3*2 + 2*1 = 8
        linear = torch.sum((3.0 * pts[:, 0] + 2.0 * pts[:, 1]) * wts)
        assert float(linear) == pytest.approx(8.0, abs=1e-13)

    pts2, wts2 = deterministic_triangle_quadrature(domain, degree=2)
    # Quadratic: int_0^2 int_0^1 (x^2 + xy) dy dx = 8/3 + 1 = 11/3
    quad = torch.sum((pts2[:, 0] ** 2 + pts2[:, 0] * pts2[:, 1]) * wts2)
    assert float(quad) == pytest.approx(11.0 / 3.0, abs=1e-13)

    pts5, wts5 = deterministic_triangle_quadrature(domain, degree=5)
    # Degree 5: int_0^2 int_0^1 (x^4 * y) dy dx = (32/5) * (1/2) = 3.2
    deg5 = torch.sum((pts5[:, 0] ** 4 * pts5[:, 1]) * wts5)
    assert float(deg5) == pytest.approx(3.2, abs=1e-12)


def test_subdivide_triangle_domain_conserves_area_and_maps_children() -> None:
    domain = _domain()
    fine = subdivide_triangle_domain(domain)
    assert fine.cells.shape == (8, 3)
    assert fine.area == pytest.approx(domain.area, abs=1e-14)
    for parent_idx in range(domain.cells.shape[0]):
        child_sum = float(fine.areas[4 * parent_idx : 4 * parent_idx + 4].sum())
        assert child_sum == pytest.approx(float(domain.areas[parent_idx]), abs=1e-14)


def test_hierarchical_energy_vanishes_on_constant_strain_and_flows_gradients() -> None:
    domain = _domain()
    material = _neo_hookean()

    # Constant strain -> cross-mesh discrepancy and sub-cell variance are identically zero.
    eps = torch.tensor(0.12, dtype=torch.float64, requires_grad=True)

    def constant_grad(pts: torch.Tensor) -> torch.Tensor:
        g = torch.zeros((pts.shape[0], 2, 2), dtype=pts.dtype)
        g[:, 0, 0] = eps
        g[:, 1, 1] = -0.3 * eps
        return g

    cfg = HierarchicalEnergyConfig(
        base=PotentialEnergyConfig(),
        cross_mesh_weight=0.5,
        local_physics_weight=0.25,
        quadrature_degree=2,
        use_deterministic_quadrature=True,
    )
    assert len(cfg.config_hash) > 10
    terms = hierarchical_plane_stress_energy(
        constant_grad, domain, material, config=cfg
    )
    assert float(terms.cross_mesh.detach()) == pytest.approx(0.0, abs=1e-24)
    assert float(terms.local_physics.detach()) == pytest.approx(0.0, abs=1e-24)
    terms.total.backward()
    assert eps.grad is not None and float(eps.grad) > 0.0

    # Non-constant (spatially varying) gradient -> positive cross-mesh & local physics terms
    # with valid autograd gradients into parameters.
    alpha = torch.tensor(0.08, dtype=torch.float64, requires_grad=True)

    def curved_grad(pts: torch.Tensor) -> torch.Tensor:
        x = pts[:, 0]
        y = pts[:, 1]
        g = torch.zeros((pts.shape[0], 2, 2), dtype=pts.dtype)
        g[:, 0, 0] = alpha * (0.2 + x * x)
        g[:, 0, 1] = alpha * 0.15 * x * y
        g[:, 1, 1] = -0.25 * alpha * (0.2 + y * y)
        return g

    focus = torch.tensor([3.0, 1.0], dtype=torch.float64)
    terms_var = hierarchical_plane_stress_energy(
        curved_grad, domain, material, config=cfg, cell_focus_weights=focus
    )
    assert float(terms_var.cross_mesh.detach()) > 0.0
    assert float(terms_var.local_physics.detach()) > 0.0
    assert float(terms_var.total.detach()) > float(terms_var.primary.total.detach())
    terms_var.total.backward()
    assert alpha.grad is not None and torch.isfinite(alpha.grad)


def test_quadrature_energy_convergence_check_passes_for_smooth_field() -> None:
    domain = subdivide_triangle_domain(_domain())
    material = _neo_hookean()

    def smooth_grad(pts: torch.Tensor) -> torch.Tensor:
        x, y = pts[:, 0], pts[:, 1]
        g = torch.zeros((pts.shape[0], 2, 2), dtype=pts.dtype)
        g[:, 0, 0] = 0.05 * (1.0 + 0.2 * torch.sin(x))
        g[:, 1, 1] = -0.015 * (1.0 + 0.1 * torch.cos(y))
        return g

    report = check_quadrature_energy_convergence(
        smooth_grad, domain, material, relative_tolerance=1e-4
    )
    assert report.converged is True
    assert report.degree_2_to_5_relative_diff < 1e-4
    assert report.mesh_subdivision_relative_diff < 1e-4


def test_hierarchical_energy_modes_and_validation_guards() -> None:
    domain = _domain()
    material = _neo_hookean()

    def lin_grad(pts: torch.Tensor) -> torch.Tensor:
        g = torch.zeros((pts.shape[0], 2, 2), dtype=pts.dtype)
        g[:, 0, 0] = 0.05 * (1.0 + 0.1 * pts[:, 0])
        return g

    # Zero-weight baseline arm matches primary potential energy cleanly.
    zero_cfg = HierarchicalEnergyConfig(cross_mesh_weight=0.0, local_physics_weight=0.0)
    t_zero = hierarchical_plane_stress_energy(
        lin_grad, domain, material, config=zero_cfg
    )
    assert float(t_zero.cross_mesh.detach()) == 0.0
    assert float(t_zero.total.detach()) == float(t_zero.primary.total.detach())

    # Stochastic stratified quadrature + default inverse-area local focus weights.
    stoch_cfg = HierarchicalEnergyConfig(
        cross_mesh_weight=0.2,
        local_physics_weight=0.1,
        use_deterministic_quadrature=False,
    )
    gen = torch.Generator().manual_seed(0)
    t_stoch = hierarchical_plane_stress_energy(
        lin_grad, domain, material, config=stoch_cfg, generator=gen
    )
    assert float(t_stoch.total.detach()) > 0.0

    with pytest.raises(ValueError, match="quadrature_degree"):
        HierarchicalEnergyConfig(quadrature_degree=3)
    with pytest.raises(ValueError, match="cross_mesh_weight"):
        HierarchicalEnergyConfig(cross_mesh_weight=-0.1)
    with pytest.raises(ValueError, match="degree"):
        deterministic_triangle_quadrature(domain, degree=4)
    with pytest.raises(ValueError, match="relative_tolerance"):
        check_quadrature_energy_convergence(
            lin_grad, domain, material, relative_tolerance=0.0
        )
