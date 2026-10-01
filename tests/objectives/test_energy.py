from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from piginot.contracts import CanonicalQuantity, MaterialSpec, Unit
from piginot.objectives import (
    POTENTIAL_ENERGY_POLICY_ID,
    PotentialEnergyConfig,
    PotentialEnergyTerms,
    TriangleDomain,
    plane_stress_potential_energy,
    stratified_triangle_quadrature,
)
from piginot.operators import constitutive_in_plane_stress
from piginot.physics import energy_density

_DTYPE = torch.float64
_E = 760.0
_NU = 0.23


def _material() -> MaterialSpec:
    return MaterialSpec(
        "energy-test",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(_E, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(_NU, Unit.ONE),
        },
    )


def _box(
    nx: int = 4, ny: int = 2, width: float = 2.0, height: float = 1.0
) -> TriangleDomain:
    xs = torch.linspace(0.0, width, nx + 1, dtype=_DTYPE)
    ys = torch.linspace(0.0, height, ny + 1, dtype=_DTYPE)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    vertices = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    cells: list[tuple[int, int, int]] = []
    for i in range(nx):
        for j in range(ny):
            a = i * (ny + 1) + j
            b = (i + 1) * (ny + 1) + j
            cells.extend(((a, b, b + 1), (a, b + 1, a + 1)))
    return TriangleDomain(vertices, torch.tensor(cells, dtype=torch.int64))


def _uniform_gradient(
    count: int, values: tuple[float, float, float, float]
) -> torch.Tensor:
    return torch.tensor(values, dtype=_DTYPE).reshape(1, 2, 2).expand(count, 2, 2)


def test_config_is_preregistered_and_validated() -> None:
    config = PotentialEnergyConfig()

    assert config.policy_id == POTENTIAL_ENERGY_POLICY_ID
    assert (config.points_per_cell, config.barrier_weight) == (3, 1.0e3)
    assert len(config.config_hash) == 64
    assert replace(config, points_per_cell=2).config_hash != config.config_hash
    for changes in (
        {"policy_id": "other"},
        {"points_per_cell": 0},
        {"points_per_cell": True},
        {"barrier_weight": -1.0},
        {"barrier_threshold": math.nan},
        {"admissibility_floor": 0.1},
    ):
        with pytest.raises(ValueError):
            replace(config, **changes)


def test_triangle_domain_requires_positive_orientation_and_valid_indices() -> None:
    domain = _box()
    assert domain.area == pytest.approx(2.0, abs=1e-14)
    assert torch.all(domain.areas > 0.0)
    flipped = domain.cells[:, [0, 2, 1]]
    with pytest.raises(ValueError, match="positively oriented"):
        TriangleDomain(domain.vertices, flipped)
    with pytest.raises(ValueError, match="missing vertices"):
        TriangleDomain(domain.vertices, domain.cells + 100)
    with pytest.raises(ValueError, match="int64"):
        TriangleDomain(domain.vertices, domain.cells.to(torch.int32))
    with pytest.raises(ValueError, match="shape"):
        TriangleDomain(domain.vertices, domain.cells[:, :2])
    with pytest.raises(ValueError, match="finite"):
        TriangleDomain(domain.vertices[:2], domain.cells)
    with pytest.raises(TypeError):
        TriangleDomain([[0.0, 0.0]], domain.cells)  # type: ignore[arg-type]


def test_stratified_points_are_inside_their_cells_and_weights_are_exact() -> None:
    domain = _box()
    generator = torch.Generator().manual_seed(0)
    points, weights = stratified_triangle_quadrature(domain, 3, generator=generator)

    assert points.shape == (domain.cells.shape[0] * 3, 2)
    assert float(weights.sum()) == pytest.approx(domain.area, rel=1e-15)
    corners = domain.corners.repeat_interleave(3, dim=0)
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]

    def cross(o: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return (p[:, 0] - o[:, 0]) * (q[:, 1] - o[:, 1]) - (p[:, 1] - o[:, 1]) * (
            q[:, 0] - o[:, 0]
        )

    tolerance = -1e-14
    assert bool((cross(a, b, points) >= tolerance).all())
    assert bool((cross(b, c, points) >= tolerance).all())
    assert bool((cross(c, a, points) >= tolerance).all())


def test_stratified_quadrature_is_reproducible_and_unbiased() -> None:
    domain = _box()
    first = stratified_triangle_quadrature(
        domain, 2, generator=torch.Generator().manual_seed(4)
    )
    second = stratified_triangle_quadrature(
        domain, 2, generator=torch.Generator().manual_seed(4)
    )
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])

    generator = torch.Generator().manual_seed(9)
    estimates = []
    for _ in range(400):
        points, weights = stratified_triangle_quadrature(domain, 1, generator=generator)
        estimates.append(float((weights * (3.0 * points[:, 0] + points[:, 1])).sum()))
    exact = 3.0 * 2.0 + 0.5 * 2.0  # integral of 3x + y over [0, 2] x [0, 1]
    assert sum(estimates) / len(estimates) == pytest.approx(exact, rel=1e-2)
    with pytest.raises(ValueError):
        stratified_triangle_quadrature(domain, 0)
    with pytest.raises(TypeError):
        stratified_triangle_quadrature(object(), 1)  # type: ignore[arg-type]


def test_identity_has_zero_energy_and_no_barrier() -> None:
    weights = torch.full((5,), 0.4, dtype=_DTYPE)
    terms = plane_stress_potential_energy(
        torch.zeros((5, 2, 2), dtype=_DTYPE),
        weights,
        _material(),
        config=PotentialEnergyConfig(),
        reference_area=2.0,
    )

    assert type(terms) is PotentialEnergyTerms
    assert float(terms.energy) == pytest.approx(0.0, abs=1e-15)
    assert float(terms.barrier) == 0.0
    assert terms.minimum_determinant == pytest.approx(1.0)
    assert terms.inadmissible_points == 0


def test_energy_matches_the_s4_density_and_its_gradient_is_the_stress() -> None:
    gradient = torch.tensor(
        [[[0.12, 0.03], [-0.02, -0.04]], [[0.05, 0.0], [0.01, 0.02]]],
        dtype=_DTYPE,
        requires_grad=True,
    )
    weights = torch.tensor([0.7, 0.3], dtype=_DTYPE)
    material = _material()
    terms = plane_stress_potential_energy(
        gradient,
        weights,
        material,
        config=PotentialEnergyConfig(),
        reference_area=1.0,
    )
    (derivative,) = torch.autograd.grad(terms.total, gradient)
    stress = constitutive_in_plane_stress(gradient.detach(), material)

    expected = weights[:, None, None] * stress / _E
    assert torch.allclose(derivative, expected, rtol=1e-9, atol=1e-13)
    assert float(terms.barrier.detach()) == 0.0


def test_finite_differences_confirm_the_energy_gradient() -> None:
    generator = torch.Generator().manual_seed(2)
    gradient = 0.1 * torch.randn((6, 2, 2), generator=generator, dtype=_DTYPE)
    weights = torch.rand(6, generator=generator, dtype=_DTYPE)
    config = PotentialEnergyConfig()

    def energy(values: torch.Tensor) -> torch.Tensor:
        return plane_stress_potential_energy(
            values, weights, _material(), config=config, reference_area=1.0
        ).total

    leaf = gradient.clone().requires_grad_(True)
    (analytic,) = torch.autograd.grad(energy(leaf), leaf)
    step = 1.0e-6
    numeric = torch.zeros_like(gradient)
    for index in range(gradient.numel()):
        shift = torch.zeros(gradient.numel(), dtype=_DTYPE)
        shift[index] = step
        shift = shift.reshape(gradient.shape)
        numeric.view(-1)[index] = (
            energy(gradient + shift) - energy(gradient - shift)
        ) / (2.0 * step)
    assert torch.allclose(analytic, numeric, rtol=1e-6, atol=1e-11)


def test_free_lateral_contraction_minimises_the_uniaxial_energy() -> None:
    material = _material()
    stretch = 0.15
    config = PotentialEnergyConfig()
    weights = torch.ones(1, dtype=_DTYPE)

    def energy(lateral: float) -> float:
        return float(
            plane_stress_potential_energy(
                _uniform_gradient(1, (stretch, 0.0, 0.0, lateral)),
                weights,
                material,
                config=config,
                reference_area=1.0,
            ).energy
        )

    candidates = torch.linspace(-0.08, 0.0, 801, dtype=_DTYPE)
    values = torch.tensor([energy(float(value)) for value in candidates])
    best = float(candidates[int(torch.argmin(values))])
    stress = constitutive_in_plane_stress(
        _uniform_gradient(1, (stretch, 0.0, 0.0, best)), material
    )[0]

    assert -0.08 < best < 0.0
    assert abs(float(stress[1, 1])) < 0.5
    assert float(stress[0, 0]) > 50.0


def test_barrier_acts_only_below_threshold_and_masks_folded_energy() -> None:
    config = PotentialEnergyConfig()
    weights = torch.full((3,), 1.0 / 3.0, dtype=_DTYPE)
    safe = plane_stress_potential_energy(
        _uniform_gradient(3, (-0.5, 0.0, 0.0, 0.0)),
        weights,
        _material(),
        config=config,
        reference_area=1.0,
    )
    near = plane_stress_potential_energy(
        _uniform_gradient(3, (-0.97, 0.0, 0.0, 0.0)),
        weights,
        _material(),
        config=config,
        reference_area=1.0,
    )
    folded_gradient = torch.tensor(
        [[[-1.2, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
        dtype=_DTYPE,
        requires_grad=True,
    )
    folded = plane_stress_potential_energy(
        folded_gradient,
        torch.tensor([0.5, 0.5], dtype=_DTYPE),
        _material(),
        config=config,
        reference_area=1.0,
    )
    (recovery,) = torch.autograd.grad(folded.total, folded_gradient)

    assert float(safe.barrier) == 0.0
    assert float(near.barrier) > 0.0 and near.inadmissible_points == 0
    assert folded.inadmissible_points == 1
    assert math.isfinite(float(folded.total))
    assert folded.minimum_determinant == pytest.approx(-0.2)
    assert float(recovery[0, 0, 0]) < 0.0  # descent raises the folded determinant


def test_energy_inputs_fail_closed() -> None:
    config = PotentialEnergyConfig()
    gradient = torch.zeros((2, 2, 2), dtype=_DTYPE)
    weights = torch.ones(2, dtype=_DTYPE)
    material = _material()
    with pytest.raises(TypeError):
        plane_stress_potential_energy(
            gradient,
            weights,
            material,
            config=object(),
            reference_area=1.0,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="shape"):
        plane_stress_potential_energy(
            gradient[0], weights, material, config=config, reference_area=1.0
        )
    with pytest.raises(ValueError, match="weights"):
        plane_stress_potential_energy(
            gradient, weights[:1], material, config=config, reference_area=1.0
        )
    with pytest.raises(ValueError, match="dtype"):
        plane_stress_potential_energy(
            gradient, weights.float(), material, config=config, reference_area=1.0
        )
    with pytest.raises(ValueError, match="finite"):
        plane_stress_potential_energy(
            gradient * math.nan, weights, material, config=config, reference_area=1.0
        )
    with pytest.raises(ValueError, match="non-negative"):
        plane_stress_potential_energy(
            gradient, -weights, material, config=config, reference_area=1.0
        )
    with pytest.raises(ValueError, match="reference_area"):
        plane_stress_potential_energy(
            gradient, weights, material, config=config, reference_area=0.0
        )
    with pytest.raises(ValueError):
        PotentialEnergyTerms(
            total=torch.zeros(1),
            energy=torch.zeros(()),
            barrier=torch.zeros(()),
            minimum_determinant=1.0,
            inadmissible_points=0,
        )
    with pytest.raises(ValueError):
        PotentialEnergyTerms(
            total=torch.zeros(()),
            energy=torch.zeros(()),
            barrier=torch.zeros(()),
            minimum_determinant=math.inf,
            inadmissible_points=0,
        )
    with pytest.raises(ValueError):
        PotentialEnergyTerms(
            total=torch.zeros(()),
            energy=torch.zeros(()),
            barrier=torch.zeros(()),
            minimum_determinant=1.0,
            inadmissible_points=-1,
        )


def test_s4_density_is_used_unchanged() -> None:
    gradient = torch.tensor([[[0.1, 0.0], [0.0, -0.02]]], dtype=_DTYPE)
    terms = plane_stress_potential_energy(
        gradient,
        torch.ones(1, dtype=_DTYPE),
        _material(),
        config=PotentialEnergyConfig(),
        reference_area=1.0,
    )
    from piginot.operators import embed_plane_stress_deformation
    from piginot.physics import plane_stress_closure

    deformation = torch.eye(2, dtype=_DTYPE) + gradient
    closed = embed_plane_stress_deformation(
        deformation, plane_stress_closure(deformation, _material())
    )
    assert float(terms.energy) == pytest.approx(
        float(energy_density(closed, _material())[0]) / _E, rel=1e-14
    )
