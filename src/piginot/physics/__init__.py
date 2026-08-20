"""Verified finite-strain constitutive physics kernels."""

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
from piginot.physics.neo_hookean import (
    admissibility,
    consistent_tangent,
    energy_density,
    first_piola,
)
from piginot.physics.plane_stress import plane_stress_closure

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
)
