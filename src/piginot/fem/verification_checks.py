"""Pure physical-consistency metrics used by the S8 FEM verification gate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from piginot.contracts import MaterialSpec
from piginot.physics import energy_density, first_piola, plane_stress_closure
from piginot.physics._validation import DETERMINANT_MARGIN

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ObjectivityMetrics:
    """Relative energy-invariance and first-Piola covariance residuals."""

    energy_relative_error: float
    first_piola_relative_error: float

    @property
    def maximum_relative_error(self) -> float:
        """Return the conservative scalar objectivity residual."""

        return max(self.energy_relative_error, self.first_piola_relative_error)


@dataclass(frozen=True, slots=True)
class SmallStrainMetrics:
    """Nonlinear-versus-linear plane-stress comparison for one strain state."""

    strain_amplitude: float
    nonlinear_stress: tuple[tuple[float, float], tuple[float, float]]
    linear_stress: tuple[tuple[float, float], tuple[float, float]]
    relative_error: float


@dataclass(frozen=True, slots=True)
class EnergyBalanceMetrics:
    """Energy and explicitly integrated work along a proportional path."""

    strain_energy_density: float
    path_work_density: float
    absolute_error: float
    relative_error: float


def affine_patch_error(
    dof_coordinates: object,
    displacement: object,
    displacement_gradient: object,
    displacement_offset: object,
) -> float:
    """Return the relative maximum DOF error for a nonzero exact affine field.

    Coordinates and displacement values use row layout ``(n_dofs, 2)`` and the
    exact field is ``u(X) = displacement_gradient @ X + displacement_offset``.
    """

    coordinates = _float_array(dof_coordinates, "dof_coordinates", columns=2)
    computed = _float_array(displacement, "displacement", columns=2)
    gradient = _float_matrix(displacement_gradient, "displacement_gradient", 2)
    offset = _float_vector(displacement_offset, "displacement_offset", 2)
    if coordinates.shape[0] != computed.shape[0]:
        raise ValueError(
            "dof_coordinates and displacement must have the same number of rows"
        )

    exact = coordinates @ gradient.T + offset
    scale = float(np.max(np.abs(exact)))
    if scale == 0.0:
        raise ValueError("the exact affine displacement field must be nonzero")
    return float(np.max(np.abs(computed - exact)) / scale)


def frame_objectivity_metrics(
    deformation: torch.Tensor,
    rotation: torch.Tensor,
    material: MaterialSpec,
) -> ObjectivityMetrics:
    """Measure ``W(QF)=W(F)`` and ``P(QF)=Q P(F)`` for a rigid rotation ``Q``."""

    deformation = _float_tensor(deformation, "deformation", (3, 3))
    rotation = _float_tensor(rotation, "rotation", (3, 3))
    _require_same_tensor_context(deformation, rotation, "rotation")
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    tolerance = 100.0 * torch.finfo(rotation.dtype).eps
    determinant = torch.linalg.det(rotation)
    is_orthogonal = torch.allclose(
        rotation.transpose(-2, -1) @ rotation,
        identity,
        rtol=0.0,
        atol=tolerance,
    )
    if not is_orthogonal or not torch.isclose(
        determinant,
        torch.ones((), dtype=rotation.dtype, device=rotation.device),
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("rotation must be a proper orthogonal matrix")

    rotated_deformation = rotation @ deformation
    base_energy = energy_density(deformation, material)
    rotated_energy = energy_density(rotated_deformation, material)
    base_stress = first_piola(deformation, material)
    rotated_stress = first_piola(rotated_deformation, material)
    expected_rotated_stress = rotation @ base_stress
    return ObjectivityMetrics(
        energy_relative_error=_symmetric_relative_error(
            rotated_energy.reshape(1), base_energy.reshape(1)
        ),
        first_piola_relative_error=_symmetric_relative_error(
            rotated_stress, expected_rotated_stress
        ),
    )


def cauchy_skew_ratio(
    deformation: torch.Tensor,
    material: MaterialSpec,
    *,
    first_piola_stress: torch.Tensor | None = None,
) -> float:
    """Return ``||sigma-sigma.T||_F / ||sigma||_F`` for Cauchy stress.

    Supplying a stress field supports independent-solver and mutation checks;
    omitting it evaluates the S4 constitutive kernel directly.
    """

    deformation = _float_tensor(deformation, "deformation", (3, 3))
    constitutive_stress = first_piola(deformation, material)
    if first_piola_stress is None:
        stress = constitutive_stress
    else:
        stress = _float_tensor(first_piola_stress, "first_piola_stress", (3, 3))
        _require_same_tensor_context(deformation, stress, "first_piola_stress")

    determinant = torch.linalg.det(deformation)
    cauchy = stress @ deformation.transpose(-2, -1) / determinant
    skew_residual = torch.linalg.vector_norm(cauchy - cauchy.transpose(-2, -1))
    scale = torch.linalg.vector_norm(cauchy)
    if scale.item() == 0.0:
        return float(skew_residual.item())
    return float((skew_residual / scale).item())


def reaction_equilibrium_error(
    reactions: Mapping[str, Sequence[float]],
    applied_resultant: object,
) -> float:
    """Return the relative norm of total reaction plus applied resultant."""

    if not isinstance(reactions, Mapping):
        raise TypeError("reactions must be a mapping")
    if not reactions:
        raise ValueError("reactions must contain at least one resultant")
    reaction_vectors = tuple(
        _float_vector(value, f"reactions[{name!r}]", 2)
        for name, value in reactions.items()
    )
    applied = _float_vector(applied_resultant, "applied_resultant", 2)
    reaction_sum = np.array(
        tuple(
            math.fsum(vector[index] for vector in reaction_vectors)
            for index in range(2)
        ),
        dtype=np.float64,
    )
    residual = float(np.linalg.norm(reaction_sum + applied))
    scale = max(float(np.linalg.norm(reaction_sum)), float(np.linalg.norm(applied)))
    return residual if scale == 0.0 else residual / scale


def small_strain_metrics(
    infinitesimal_strain: torch.Tensor,
    material: MaterialSpec,
    *,
    first_piola_stress: torch.Tensor | None = None,
) -> SmallStrainMetrics:
    """Compare finite-strain S4 stress with linear isotropic plane stress."""

    strain = _float_tensor(infinitesimal_strain, "infinitesimal_strain", (2, 2))
    symmetry_tolerance = 100.0 * torch.finfo(strain.dtype).eps
    if not torch.allclose(
        strain,
        strain.transpose(-2, -1),
        rtol=0.0,
        atol=symmetry_tolerance,
    ):
        raise ValueError("infinitesimal_strain must be symmetric")
    amplitude = float(torch.max(torch.abs(strain)).item())
    if amplitude == 0.0:
        raise ValueError("infinitesimal_strain must be nonzero")

    identity = torch.eye(2, dtype=strain.dtype, device=strain.device)
    deformation_2d = identity + strain
    deformation_3d = _embed_plane_stress(deformation_2d, material)
    constitutive_stress = first_piola(deformation_3d, material)[:2, :2]
    if first_piola_stress is None:
        nonlinear_stress = constitutive_stress
    else:
        nonlinear_stress = _float_tensor(
            first_piola_stress, "first_piola_stress", (2, 2)
        )
        _require_same_tensor_context(strain, nonlinear_stress, "first_piola_stress")

    young = material.properties["elastic_modulus"].value
    poisson = material.properties["poisson_ratio"].value
    normal_factor = young / (1.0 - poisson * poisson)
    shear_factor = young / (1.0 + poisson)
    linear_stress = torch.stack(
        (
            torch.stack(
                (
                    normal_factor * (strain[0, 0] + poisson * strain[1, 1]),
                    shear_factor * strain[0, 1],
                )
            ),
            torch.stack(
                (
                    shear_factor * strain[1, 0],
                    normal_factor * (strain[1, 1] + poisson * strain[0, 0]),
                )
            ),
        )
    )
    relative_error = _reference_relative_error(nonlinear_stress, linear_stress)
    return SmallStrainMetrics(
        strain_amplitude=amplitude,
        nonlinear_stress=_matrix2_tuple(nonlinear_stress),
        linear_stress=_matrix2_tuple(linear_stress),
        relative_error=relative_error,
    )


def proportional_path_energy_balance(
    final_deformation_2d: torch.Tensor,
    material: MaterialSpec,
    *,
    quadrature_order: int = 16,
) -> EnergyBalanceMetrics:
    """Integrate in-plane ``P:dF`` along ``F(a)=I+a(F_final-I)``.

    The out-of-plane stretch is solved at every Gauss--Legendre point.  Its
    work-conjugate stress is zero by the plane-stress closure, so the boundary
    work is the in-plane contraction integrated over ``a in [0, 1]``.
    """

    if (
        isinstance(quadrature_order, bool)
        or not isinstance(quadrature_order, int)
        or quadrature_order <= 0
    ):
        raise ValueError("quadrature_order must be a positive integer")
    final_deformation = _float_tensor(
        final_deformation_2d, "final_deformation_2d", (2, 2)
    )
    identity_2d = torch.eye(
        2, dtype=final_deformation.dtype, device=final_deformation.device
    )
    increment = final_deformation - identity_2d
    _require_admissible_proportional_path(increment)

    nodes_numpy, weights_numpy = np.polynomial.legendre.leggauss(quadrature_order)
    parameters = 0.5 * (
        torch.as_tensor(
            nodes_numpy, dtype=final_deformation.dtype, device=final_deformation.device
        )
        + 1.0
    )
    weights = 0.5 * torch.as_tensor(
        weights_numpy, dtype=final_deformation.dtype, device=final_deformation.device
    )
    path_deformation_2d = identity_2d + parameters[:, None, None] * increment
    path_deformation_3d = _embed_plane_stress(path_deformation_2d, material)
    path_stress = first_piola(path_deformation_3d, material)[..., :2, :2]
    work_integrand = torch.sum(path_stress * increment, dim=(-2, -1))
    path_work = torch.sum(weights * work_integrand)

    final_deformation_3d = _embed_plane_stress(final_deformation, material)
    reference_deformation_3d = _embed_plane_stress(identity_2d, material)
    stored_energy = energy_density(final_deformation_3d, material) - energy_density(
        reference_deformation_3d, material
    )
    absolute_error = torch.abs(path_work - stored_energy)
    scale = torch.abs(stored_energy)
    relative_error = absolute_error if scale.item() == 0.0 else absolute_error / scale
    return EnergyBalanceMetrics(
        strain_energy_density=float(stored_energy.item()),
        path_work_density=float(path_work.item()),
        absolute_error=float(absolute_error.item()),
        relative_error=float(relative_error.item()),
    )


def _embed_plane_stress(
    deformation_2d: torch.Tensor, material: MaterialSpec
) -> torch.Tensor:
    f33 = plane_stress_closure(deformation_2d, material)
    zero_column = torch.zeros(
        (*deformation_2d.shape[:-2], 2, 1),
        dtype=deformation_2d.dtype,
        device=deformation_2d.device,
    )
    upper = torch.cat((deformation_2d, zero_column), dim=-1)
    bottom = torch.stack(
        (torch.zeros_like(f33), torch.zeros_like(f33), f33), dim=-1
    ).unsqueeze(-2)
    return torch.cat((upper, bottom), dim=-2)


def _require_admissible_proportional_path(increment: torch.Tensor) -> None:
    """Check the exact quadratic determinant on the complete 2D path."""

    increment_64 = increment.detach().to(device="cpu", dtype=torch.float64)
    linear_coefficient = float(torch.trace(increment_64).item())
    quadratic_coefficient = float(torch.linalg.det(increment_64).item())
    candidates = [1.0, 1.0 + linear_coefficient + quadratic_coefficient]
    if quadratic_coefficient != 0.0:
        stationary_point = -linear_coefficient / (2.0 * quadratic_coefficient)
        if 0.0 < stationary_point < 1.0:
            candidates.append(
                1.0
                + linear_coefficient * stationary_point
                + quadratic_coefficient * stationary_point * stationary_point
            )
    if min(candidates) <= DETERMINANT_MARGIN:
        raise ValueError(
            "final_deformation_2d must remain admissible along the full path"
        )


def _float_array(value: object, name: str, *, columns: int) -> FloatArray:
    array = _numeric_array(value, name)
    if array.ndim != 2 or array.shape[1] != columns or array.shape[0] == 0:
        raise ValueError(f"{name} must have shape (n, {columns}) with n > 0")
    return array


def _float_matrix(value: object, name: str, size: int) -> FloatArray:
    array = _numeric_array(value, name)
    if array.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    return array


def _float_vector(value: object, name: str, size: int) -> FloatArray:
    array = _numeric_array(value, name)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return array


def _numeric_array(value: object, name: str) -> FloatArray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    converted = np.asarray(array, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _float_tensor(value: object, name: str, shape: tuple[int, int]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout is not torch.strided:
        raise TypeError(f"{name} must use strided layout")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} dtype must be float32 or float64")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _require_same_tensor_context(
    reference: torch.Tensor, value: torch.Tensor, name: str
) -> None:
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must match the deformation dtype and device")


def _symmetric_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm(actual - expected)
    scale = torch.maximum(
        torch.linalg.vector_norm(actual), torch.linalg.vector_norm(expected)
    )
    return float((difference if scale.item() == 0.0 else difference / scale).item())


def _reference_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm(actual - expected)
    scale = torch.linalg.vector_norm(expected)
    if scale.item() == 0.0:
        raise ValueError("reference norm must be nonzero")
    return float((difference / scale).item())


def _matrix2_tuple(
    value: torch.Tensor,
) -> tuple[tuple[float, float], tuple[float, float]]:
    detached = value.detach().cpu()
    return (
        (float(detached[0, 0].item()), float(detached[0, 1].item())),
        (float(detached[1, 0].item()), float(detached[1, 1].item())),
    )


__all__ = (
    "EnergyBalanceMetrics",
    "ObjectivityMetrics",
    "SmallStrainMetrics",
    "affine_patch_error",
    "cauchy_skew_ratio",
    "frame_objectivity_metrics",
    "proportional_path_energy_balance",
    "reaction_equilibrium_error",
    "small_strain_metrics",
)
