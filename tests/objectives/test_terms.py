from __future__ import annotations

import pytest
import torch

from piginot.fem.mms_runner import verification_material
from piginot.objectives import (
    NondimensionalScales,
    angular_momentum_residual,
    constitutive_residual,
    dirichlet_residual,
    supervised_displacement_residual,
    supervised_stress_residual,
    traction_free_residual,
)
from piginot.physics import first_piola, plane_stress_closure

SCALES = NondimensionalScales(
    length_mm=10.0,
    displacement_mm=2.0,
    stress_mpa=100.0,
)


def _plane_stress_state() -> tuple[torch.Tensor, torch.Tensor]:
    material = verification_material()
    deformation_2d = torch.tensor(
        [[[[1.08, 0.03], [-0.01, 0.96]]]], dtype=torch.float64
    )
    f33 = plane_stress_closure(deformation_2d, material)
    deformation_3d = torch.zeros((1, 1, 3, 3), dtype=torch.float64)
    deformation_3d[..., :2, :2] = deformation_2d
    deformation_3d[..., 2, 2] = f33
    return deformation_2d, first_piola(deformation_3d, material)[..., :2, :2]


def test_data_and_dirichlet_terms_use_declared_reference_scales() -> None:
    predicted_u = torch.tensor([[[3.0, -1.0]]], dtype=torch.float64)
    target_u = torch.tensor([[[1.0, 1.0]]], dtype=torch.float64)
    predicted_p = torch.full((1, 1, 2, 2), 110.0, dtype=torch.float64)
    target_p = torch.full((1, 1, 2, 2), 10.0, dtype=torch.float64)

    assert torch.equal(
        supervised_displacement_residual(predicted_u, target_u, SCALES),
        torch.tensor([[[1.0, -1.0]]], dtype=torch.float64),
    )
    assert torch.equal(
        supervised_stress_residual(predicted_p, target_p, SCALES),
        torch.ones((1, 1, 2, 2), dtype=torch.float64),
    )
    mask = torch.tensor([[[True, False]]])
    assert torch.equal(
        dirichlet_residual(predicted_u, target_u, mask, SCALES),
        torch.tensor([1.0], dtype=torch.float64),
    )


def test_constitutive_is_exact_and_in_plane_only() -> None:
    deformation_2d, material_stress = _plane_stress_state()
    gradient = deformation_2d - torch.eye(2, dtype=torch.float64)

    residual = constitutive_residual(
        material_stress,
        gradient,
        verification_material(),
        SCALES,
    )

    assert residual.shape == (1, 1, 2, 2)
    assert residual.abs().max().item() < 1.0e-12
    with pytest.raises(ValueError, match="in-plane"):
        constitutive_residual(
            torch.zeros((1, 1, 3, 3), dtype=torch.float64),
            gradient,
            verification_material(),
            SCALES,
        )


def test_angular_term_uses_served_stress_and_is_separate_from_constitutive() -> None:
    deformation_2d, material_stress = _plane_stress_state()

    exact = angular_momentum_residual(material_stress, deformation_2d, SCALES)
    perturbed = material_stress.clone()
    perturbed[..., 0, 1] += 10.0
    fired = angular_momentum_residual(perturbed, deformation_2d, SCALES)

    assert exact.shape == (1, 1, 1)
    assert exact.abs().max().item() < 1.0e-12
    assert fired.abs().max().item() > 1.0e-3


def test_traction_free_uses_authoritative_in_plane_stress_and_outward_normal() -> None:
    stress = torch.tensor(
        [[[[0.0, 0.0], [0.0, 4.0]], [[3.0, 0.0], [0.0, 0.0]]]],
        dtype=torch.float64,
    )
    normals = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float64)

    residual = traction_free_residual(stress, normals, SCALES)

    assert torch.equal(
        residual,
        torch.tensor([[[0.0, 0.0], [0.03, 0.0]]], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"length_mm": 0.0, "displacement_mm": 1.0, "stress_mpa": 1.0},
        {"length_mm": 1.0, "displacement_mm": float("nan"), "stress_mpa": 1.0},
        {"length_mm": 1.0, "displacement_mm": 1.0, "stress_mpa": -1.0},
    ),
)
def test_scales_fail_closed(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        NondimensionalScales(**kwargs)
