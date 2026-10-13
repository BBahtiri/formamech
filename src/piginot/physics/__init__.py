"""Verified finite-strain constitutive physics kernels.

The five public functions take a :class:`~piginot.contracts.MaterialSpec`.
The closed-form compressible Neo-Hookean model runs the hand-written kernels
of :mod:`piginot.physics.neo_hookean` and :mod:`piginot.physics.plane_stress`
unchanged; every model of the S25 material registry (a strain-energy formula)
runs :mod:`piginot.materials` (ADR 0107). An unknown model is refused by the
Neo-Hookean validation, as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from piginot.contracts import MaterialSpec
from piginot.physics import neo_hookean as _closed_form
from piginot.physics import plane_stress as _closed_form_closure
from piginot.physics.errors import (
    InadmissibleDeformationError,
    InvalidMaterialError,
    InvalidTensorError,
    InvalidTensorShapeError,
    NewtonNonConvergenceError,
    NonFiniteTensorError,
    PhysicsError,
    PhysicsValidationError,
    PlaneStressClosureError,
)

if TYPE_CHECKING:
    from piginot.materials import FormulaMaterial


def _formula(material: object) -> FormulaMaterial | None:
    """The registry material of ``material``; ``None`` for the closed form."""

    if not isinstance(material, MaterialSpec):
        return None
    from piginot.materials import is_formula_model, resolve

    return resolve(material) if is_formula_model(material.model) else None


def reference_modulus(material: MaterialSpec) -> float:
    """The stress scale of a material in MPa: Young's modulus ``E`` of the
    closed-form model, or the small-strain ``E0 = 9 K mu / (3 K + mu)`` of a
    formula model's declared moduli."""

    formula = _formula(material)
    if formula is None:
        _closed_form.admissibility(torch.eye(3, dtype=torch.float64), material)
        return float(material.properties["elastic_modulus"].value)
    shear, bulk = formula.moduli
    return 9.0 * bulk * shear / (3.0 * bulk + shear)


def admissibility(deformation: torch.Tensor, material: MaterialSpec) -> bool:
    """Return whether all 3D deformation gradients are admitted."""

    formula = _formula(material)
    if formula is None:
        return _closed_form.admissibility(deformation, material)
    return formula.admissibility(deformation)


def energy_density(deformation: torch.Tensor, material: MaterialSpec) -> torch.Tensor:
    """Strain-energy density in MPa."""

    formula = _formula(material)
    if formula is None:
        return _closed_form.energy_density(deformation, material)
    return formula.energy_density(deformation)


def first_piola(deformation: torch.Tensor, material: MaterialSpec) -> torch.Tensor:
    """First Piola-Kirchhoff stress in MPa."""

    formula = _formula(material)
    if formula is None:
        return _closed_form.first_piola(deformation, material)
    return formula.first_piola(deformation)


def consistent_tangent(
    deformation: torch.Tensor, material: MaterialSpec
) -> torch.Tensor:
    """``dP_iJ/dF_kL`` with layout ``(..., i, J, k, L)``."""

    formula = _formula(material)
    if formula is None:
        return _closed_form.consistent_tangent(deformation, material)
    return formula.consistent_tangent(deformation)


def plane_stress_closure(
    deformation_2d: torch.Tensor,
    material: MaterialSpec,
    *,
    initial_f33: float | torch.Tensor = 1.0,
    max_iterations: int = 32,
) -> torch.Tensor:
    """Solve ``P33 = 0``, differentiable by the implicit-function theorem."""

    formula = _formula(material)
    if formula is None:
        return _closed_form_closure.plane_stress_closure(
            deformation_2d,
            material,
            initial_f33=initial_f33,
            max_iterations=max_iterations,
        )
    if isinstance(initial_f33, torch.Tensor):
        raise PlaneStressClosureError("formula materials take a scalar initial_f33")
    return formula.plane_stress_closure(
        deformation_2d, initial_f33=initial_f33, max_iterations=max_iterations
    )


__all__ = (
    "InadmissibleDeformationError",
    "InvalidMaterialError",
    "InvalidTensorError",
    "InvalidTensorShapeError",
    "NewtonNonConvergenceError",
    "NonFiniteTensorError",
    "PhysicsError",
    "PhysicsValidationError",
    "PlaneStressClosureError",
    "admissibility",
    "consistent_tangent",
    "energy_density",
    "first_piola",
    "plane_stress_closure",
    "reference_modulus",
)
