"""scikit-fem assembly backed by the verified S4 constitutive kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import spsolve  # type: ignore[import-untyped]
from skfem import BilinearForm, LinearForm, asm  # type: ignore[import-untyped]
from skfem.helpers import grad  # type: ignore[import-untyped]

from piginot.contracts import MaterialSpec
from piginot.fem.models import FemFailureCode, FemSolveFailure
from piginot.physics import (
    consistent_tangent,
    energy_density,
    first_piola,
    plane_stress_closure,
)
from piginot.physics.errors import InadmissibleDeformationError, PhysicsError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AssemblyState:
    """One assembled nonlinear state in reference coordinates."""

    internal_force: FloatArray
    tangent: csr_matrix
    deformation_gradient: FloatArray
    first_piola: FloatArray
    strain_energy: float


@LinearForm  # type: ignore[untyped-decorator]
def _internal_form(test: Any, data: Any) -> Any:
    return np.einsum("ij...,ij...->...", data.stress, grad(test))


@BilinearForm  # type: ignore[untyped-decorator]
def _tangent_form(increment: Any, test: Any, data: Any) -> Any:
    return np.einsum(
        "ij...,ijkl...,kl...->...",
        grad(test),
        data.modulus,
        grad(increment),
    )


@BilinearForm  # type: ignore[untyped-decorator]
def _mass_form(trial: Any, test: Any, _data: Any) -> Any:
    return trial * test


@LinearForm  # type: ignore[untyped-decorator]
def _projection_form(test: Any, data: Any) -> Any:
    return data["field"] * test


def assemble_state(
    basis: Any,
    displacement: FloatArray,
    material: MaterialSpec,
    *,
    closure_iterations: int,
    thickness: float,
) -> AssemblyState:
    """Assemble residual and exact condensed plane-stress tangent."""

    gradient = np.asarray(basis.interpolate(displacement).grad, dtype=np.float64)
    deformation_2d = np.transpose(gradient, (2, 3, 0, 1))
    deformation_2d = deformation_2d + np.eye(2, dtype=np.float64)
    deformation_tensor = torch.from_numpy(np.ascontiguousarray(deformation_2d))

    try:
        f33 = plane_stress_closure(
            deformation_tensor,
            material,
            max_iterations=closure_iterations,
        )
        zeros = torch.zeros((*deformation_tensor.shape[:-2], 2, 1), dtype=torch.float64)
        upper = torch.cat((deformation_tensor, zeros), dim=-1)
        bottom = torch.cat(
            (
                torch.zeros(
                    (*deformation_tensor.shape[:-2], 1, 2), dtype=torch.float64
                ),
                f33[..., None, None],
            ),
            dim=-1,
        )
        deformation_3d = torch.cat((upper, bottom), dim=-2)
        stress_3d = first_piola(deformation_3d, material)
        tangent_3d = consistent_tangent(deformation_3d, material)
        density = energy_density(deformation_3d, material)
    except InadmissibleDeformationError as exc:
        raise FemSolveFailure(
            FemFailureCode.NEGATIVE_JACOBIAN,
            "a quadrature point has non-positive deformation Jacobian",
        ) from exc
    except PhysicsError as exc:
        raise FemSolveFailure(
            FemFailureCode.CLOSURE_FAILURE,
            "plane-stress closure failed at a quadrature point",
        ) from exc

    denominator = tangent_3d[..., 2, 2, 2, 2]
    left = tangent_3d[..., :2, :2, 2, 2]
    right = tangent_3d[..., 2, 2, :2, :2]
    condensed = tangent_3d[..., :2, :2, :2, :2] - (
        left[..., :, :, None, None]
        * right[..., None, None, :, :]
        / denominator[..., None, None, None, None]
    )

    stress_2d = stress_3d[..., :2, :2].detach().cpu().numpy()
    modulus = condensed.detach().cpu().numpy()
    stress_field = np.transpose(stress_2d, (2, 3, 0, 1))
    modulus_field = np.transpose(modulus, (2, 3, 4, 5, 0, 1))
    internal = np.asarray(
        asm(_internal_form, basis, stress=stress_field), dtype=np.float64
    )
    tangent = asm(_tangent_form, basis, modulus=modulus_field).tocsr()
    strain_energy = float(
        np.sum(density.detach().cpu().numpy() * np.asarray(basis.dx)) * thickness
    )
    return AssemblyState(
        internal_force=internal * thickness,
        tangent=tangent * thickness,
        deformation_gradient=deformation_3d.detach().cpu().numpy(),
        first_piola=stress_3d.detach().cpu().numpy(),
        strain_energy=strain_energy,
    )


def recover_stress(scalar_basis: Any, stress: FloatArray) -> FloatArray:
    """Recover quadrature stress to P1/P2 field DOFs by an L2 projection."""

    mass = asm(_mass_form, scalar_basis).tocsr()
    recovered = np.empty((scalar_basis.N, 3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            right_hand_side = np.asarray(
                asm(
                    _projection_form,
                    scalar_basis,
                    field=stress[:, :, row, column],
                ),
                dtype=np.float64,
            )
            recovered[:, row, column] = np.asarray(
                spsolve(mass, right_hand_side), dtype=np.float64
            )
    return recovered


__all__ = ("AssemblyState", "FloatArray", "assemble_state", "recover_stress")
