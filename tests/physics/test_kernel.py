"""Machine-precision tests for the compressible Neo-Hookean kernel."""

from __future__ import annotations

import torch

from piginot.physics import consistent_tangent, energy_density, first_piola

from .reference import (
    lame_parameters,
    make_material,
    random_admissible_deformations,
    relative_l2,
)


def test_energy_matches_the_frozen_neo_hookean_equation() -> None:
    deformation = random_admissible_deformations(1_000, 3, seed=4101)
    material = make_material(elastic_modulus=7_500.0, poisson_ratio=0.33)
    shear_modulus, first_lame = lame_parameters(material)
    determinant = torch.linalg.det(deformation)
    log_determinant = torch.log(determinant)
    squared_frobenius = torch.sum(deformation * deformation, dim=(-2, -1))
    expected = (
        0.5 * shear_modulus * (squared_frobenius - 3.0)
        - shear_modulus * log_determinant
        + 0.5 * first_lame * log_determinant.square()
    )

    actual = energy_density(deformation, material)

    assert actual.shape == (1_000,)
    torch.testing.assert_close(actual, expected, rtol=2.0e-13, atol=2.0e-13)


def test_first_piola_matches_energy_autodiff_for_1000_deformations() -> None:
    deformation = random_admissible_deformations(1_000, 3, seed=4102).requires_grad_(
        True
    )
    material = make_material(elastic_modulus=5_000.0, poisson_ratio=0.27)

    energy = energy_density(deformation, material)
    autodiff_stress = torch.autograd.grad(energy.sum(), deformation)[0]
    analytic_stress = first_piola(deformation.detach(), material)

    assert relative_l2(analytic_stress, autodiff_stress) < 1.0e-10


def test_consistent_tangent_matches_central_differences() -> None:
    deformation = random_admissible_deformations(16, 3, seed=4103, scale=0.12)
    material = make_material(elastic_modulus=2_100.0, poisson_ratio=0.31)
    step = 1.0e-6
    basis = torch.eye(9, dtype=torch.float64).reshape(9, 3, 3)
    derivatives = []

    for perturbation in basis:
        stress_plus = first_piola(deformation + step * perturbation, material)
        stress_minus = first_piola(deformation - step * perturbation, material)
        derivatives.append((stress_plus - stress_minus) / (2.0 * step))

    finite_difference = torch.stack(derivatives, dim=-1).reshape(
        deformation.shape[0], 3, 3, 3, 3
    )
    analytic = consistent_tangent(deformation, material)

    assert analytic.shape == (16, 3, 3, 3, 3)
    assert relative_l2(analytic, finite_difference) < 1.0e-6


def test_first_piola_is_objective_for_100_random_rotations() -> None:
    deformation = random_admissible_deformations(100, 3, seed=4104)
    material = make_material(elastic_modulus=9_000.0, poisson_ratio=0.42)
    generator = torch.Generator(device="cpu").manual_seed(4105)
    raw = torch.randn(100, 3, 3, generator=generator, dtype=torch.float64)
    orthogonal, _ = torch.linalg.qr(raw)
    determinants = torch.linalg.det(orthogonal)
    correction_diagonal = torch.stack(
        (
            torch.ones_like(determinants),
            torch.ones_like(determinants),
            determinants,
        ),
        dim=-1,
    )
    rotations = orthogonal @ torch.diag_embed(correction_diagonal)
    rotated_deformation = rotations @ deformation

    rotated_stress = first_piola(rotated_deformation, material)
    expected = rotations @ first_piola(deformation, material)

    difference_norm = torch.linalg.vector_norm(rotated_stress - expected, dim=(-2, -1))
    reference_norm = torch.linalg.vector_norm(expected, dim=(-2, -1))
    nonzero_reference = reference_norm > 0.0
    relative_error = (
        difference_norm[nonzero_reference] / reference_norm[nonzero_reference]
    )

    assert torch.max(relative_error).item() < 1.0e-12
    if torch.any(~nonzero_reference):
        zero_state_error = difference_norm[~nonzero_reference]
        assert (
            torch.max(zero_state_error).item() <= 10.0 * torch.finfo(expected.dtype).eps
        )
