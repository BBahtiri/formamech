"""Executable physical verification checks for the S8 FEM gate."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from piginot.contracts import CanonicalQuantity, MaterialSpec, Unit
from piginot.fem.verification_checks import (
    affine_patch_error,
    cauchy_skew_ratio,
    frame_objectivity_metrics,
    proportional_path_energy_balance,
    reaction_equilibrium_error,
    small_strain_metrics,
)


def _material() -> MaterialSpec:
    return MaterialSpec(
        name="verification-rubber",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": CanonicalQuantity(760.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
        },
    )


def _rotation(angle_degrees: float) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=torch.float64,
    )


def test_affine_patch_error_is_zero_for_exact_dof_values() -> None:
    coordinates = np.array(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5)),
        dtype=np.float64,
    )
    gradient = np.array(((0.08, -0.03), (0.02, 0.05)), dtype=np.float64)
    offset = np.array((0.13, -0.04), dtype=np.float64)
    displacement = coordinates @ gradient.T + offset

    error = affine_patch_error(coordinates, displacement, gradient, offset)

    assert error == 0.0


def test_affine_patch_error_exposes_a_perturbed_dof() -> None:
    coordinates = np.array(((0.0, 0.0), (1.0, 0.0)), dtype=np.float64)
    gradient = np.array(((0.1, 0.0), (0.0, -0.05)), dtype=np.float64)
    displacement = coordinates @ gradient.T
    displacement[1, 1] += 2.0e-6

    error = affine_patch_error(
        coordinates,
        displacement,
        gradient,
        np.zeros(2, dtype=np.float64),
    )

    assert error == pytest.approx(2.0e-5)


def test_affine_patch_error_rejects_nonfinite_or_misaligned_data() -> None:
    with pytest.raises(ValueError, match="same number"):
        affine_patch_error(
            np.zeros((2, 2)),
            np.zeros((3, 2)),
            np.eye(2),
            np.zeros(2),
        )
    with pytest.raises(ValueError, match="finite"):
        affine_patch_error(
            np.array(((0.0, math.nan),)),
            np.zeros((1, 2)),
            np.eye(2),
            np.zeros(2),
        )
    with pytest.raises(ValueError, match="must be nonzero"):
        affine_patch_error(
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.zeros((2, 2)),
            np.zeros(2),
        )


def test_frame_objectivity_holds_for_a_proper_rigid_rotation() -> None:
    deformation = torch.tensor(
        ((1.16, 0.12, 0.0), (-0.04, 0.91, 0.0), (0.0, 0.0, 0.98)),
        dtype=torch.float64,
    )

    metrics = frame_objectivity_metrics(
        deformation,
        _rotation(37.0),
        _material(),
    )

    assert metrics.maximum_relative_error < 1.0e-12


def test_frame_objectivity_rejects_a_nonrigid_transformation() -> None:
    stretch = torch.diag(torch.tensor((1.001, 1.0, 1.0), dtype=torch.float64))

    with pytest.raises(ValueError, match="proper orthogonal"):
        frame_objectivity_metrics(
            torch.eye(3, dtype=torch.float64), stretch, _material()
        )


def test_angular_momentum_check_passes_constitutive_stress() -> None:
    deformation = torch.tensor(
        ((1.12, 0.09, 0.0), (-0.03, 0.94, 0.0), (0.0, 0.0, 1.03)),
        dtype=torch.float64,
    )

    ratio = cauchy_skew_ratio(deformation, _material())

    assert ratio < 1.0e-12


def test_angular_momentum_check_exposes_a_skew_stress_perturbation() -> None:
    deformation = torch.diag(torch.tensor((1.1, 0.95, 1.0), dtype=torch.float64))
    perturbed_stress = torch.tensor(
        ((10.0, 0.2, 0.0), (-0.1, 4.0, 0.0), (0.0, 0.0, 2.0)),
        dtype=torch.float64,
    )

    ratio = cauchy_skew_ratio(
        deformation,
        _material(),
        first_piola_stress=perturbed_stress,
    )

    assert ratio > 1.0e-3


def test_reaction_equilibrium_is_zero_for_balanced_resultants() -> None:
    reactions = {"left": (-2.0, 0.2), "bottom": (0.0, -0.2)}

    error = reaction_equilibrium_error(reactions, (2.0, 0.0))

    assert error == 0.0


def test_reaction_equilibrium_exposes_a_force_imbalance() -> None:
    reactions = {"left": (-1.98, 0.0)}

    error = reaction_equilibrium_error(reactions, (2.0, 0.0))

    assert error > 1.0e-3


def test_reaction_equilibrium_rejects_empty_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        reaction_equilibrium_error({}, (0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        reaction_equilibrium_error({"left": (math.inf, 0.0)}, (0.0, 0.0))


def test_small_strain_limit_is_within_half_percent_at_point_one_percent() -> None:
    strain = torch.tensor(
        ((1.0e-3, 2.0e-4), (2.0e-4, -3.0e-4)),
        dtype=torch.float64,
    )

    metrics = small_strain_metrics(strain, _material())

    assert metrics.strain_amplitude == pytest.approx(1.0e-3)
    assert metrics.relative_error < 5.0e-3


def test_small_strain_limit_exposes_a_stress_scale_perturbation() -> None:
    strain = torch.tensor(((1.0e-3, 0.0), (0.0, 0.0)), dtype=torch.float64)
    baseline = small_strain_metrics(strain, _material())
    perturbed = torch.tensor(baseline.nonlinear_stress, dtype=torch.float64) * 1.02

    metrics = small_strain_metrics(
        strain,
        _material(),
        first_piola_stress=perturbed,
    )

    assert metrics.relative_error > 5.0e-3


def test_small_strain_limit_requires_a_nonzero_symmetric_strain() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        small_strain_metrics(torch.zeros((2, 2), dtype=torch.float64), _material())
    with pytest.raises(ValueError, match="symmetric"):
        small_strain_metrics(
            torch.tensor(((1.0e-3, 1.0e-4), (0.0, 0.0)), dtype=torch.float64),
            _material(),
        )


def test_proportional_path_work_matches_hyperelastic_energy() -> None:
    final_deformation = torch.tensor(
        ((1.18, 0.08), (0.03, 0.92)),
        dtype=torch.float64,
    )

    metrics = proportional_path_energy_balance(
        final_deformation,
        _material(),
        quadrature_order=16,
    )

    assert metrics.strain_energy_density > 0.0
    assert metrics.relative_error < 1.0e-10


def test_proportional_path_work_exposes_underintegration() -> None:
    final_deformation = torch.tensor(
        ((1.35, 0.14), (0.02, 0.78)),
        dtype=torch.float64,
    )

    metrics = proportional_path_energy_balance(
        final_deformation,
        _material(),
        quadrature_order=1,
    )

    assert metrics.relative_error > 1.0e-8


def test_proportional_path_work_rejects_a_path_crossing_zero_jacobian() -> None:
    final_rotation = -torch.eye(2, dtype=torch.float64)

    with pytest.raises(ValueError, match="admissible along the full path"):
        proportional_path_energy_balance(
            final_rotation,
            _material(),
            quadrature_order=4,
        )


@pytest.mark.parametrize("quadrature_order", (True, 0, -1, 1.5))
def test_proportional_path_work_rejects_invalid_quadrature_order(
    quadrature_order: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        proportional_path_energy_balance(
            torch.eye(2, dtype=torch.float64),
            _material(),
            quadrature_order=quadrature_order,  # type: ignore[arg-type]
        )
