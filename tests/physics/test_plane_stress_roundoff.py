"""Regressions for representable plane-stress roots under compression."""

import pytest
import torch

from piginot.objectives.energy import (
    PotentialEnergyConfig,
    plane_stress_potential_energy,
)
from piginot.physics import first_piola, plane_stress_closure

from .reference import make_material
from .test_plane_stress import _embed_plane_stress_deformation


def failed_trial_deformation() -> torch.Tensor:
    # FIT-000020, frozen training point 51903, first trial of outer block 3.
    return torch.tensor(
        [
            [1.0711956846778217, -0.03986977240683461],
            [0.17134713316963662, 0.062100761857066256],
        ],
        dtype=torch.float64,
    )


def test_saved_s29_trial_reaches_a_roundoff_accurate_root_in_32_iterations():
    material = make_material(760.0, 0.23)
    deformation = failed_trial_deformation()
    f33 = plane_stress_closure(deformation, material, max_iterations=32)
    stress = first_piola(_embed_plane_stress_deformation(deformation, f33), material)
    assert torch.isfinite(f33) and f33 > 0
    assert abs(float(stress[2, 2])) < 1e-10


@pytest.mark.parametrize("poisson_ratio", [0.01, 0.1, 0.23, 0.3, 0.45, 0.49])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compressed_and_expanded_diagonal_batches_resolve_within_float_precision(
    poisson_ratio, dtype
):
    determinant = torch.logspace(-6, 3, 1201, dtype=dtype)
    stretch = determinant.sqrt()
    deformation = torch.diag_embed(torch.stack((stretch, stretch), dim=-1))
    material = make_material(760.0, poisson_ratio)
    f33 = plane_stress_closure(deformation, material, max_iterations=32)
    stress = first_piola(_embed_plane_stress_deformation(deformation, f33), material)
    mu = 760.0 / (2.0 * (1.0 + poisson_ratio))
    lam = 760.0 * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    # P33 contains 1/F33: use the scale of its physical F33*P33 equation.
    scale = (
        760.0
        + mu * (f33.square() + 1.0)
        + lam * (torch.linalg.det(deformation).log().abs() + f33.log().abs())
    )
    normalized = stress[..., 2, 2].abs() * f33 / scale
    assert torch.isfinite(f33).all() and (f33 > 0).all()
    assert float(normalized.max()) < 32 * torch.finfo(dtype).eps


def test_saved_trial_root_has_the_correct_implicit_gradient():
    deformation = failed_trial_deformation().requires_grad_(True)
    material = make_material(760.0, 0.23)
    assert torch.autograd.gradcheck(
        lambda value: plane_stress_closure(value, material),
        (deformation,),
        eps=1e-6,
        atol=1e-7,
        rtol=1e-5,
    )


def test_energy_objective_can_differentiate_the_saved_trial_without_a_solver_error():
    gradient = failed_trial_deformation() - torch.eye(2, dtype=torch.float64)
    gradient = gradient.unsqueeze(0).requires_grad_(True)
    terms = plane_stress_potential_energy(
        gradient,
        torch.ones(1, dtype=torch.float64),
        make_material(760.0, 0.23),
        config=PotentialEnergyConfig(),
        reference_area=1.0,
    )
    terms.total.backward()
    assert torch.isfinite(terms.total) and terms.total > 0
    assert gradient.grad is not None and torch.isfinite(gradient.grad).all()
