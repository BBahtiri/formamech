"""Typed failures raised by the finite-strain physics kernel."""

from __future__ import annotations


class PhysicsError(Exception):
    """Base class for all physics-kernel failures."""


class PhysicsValidationError(PhysicsError):
    """Base class for invalid physics inputs."""


class InvalidMaterialError(PhysicsValidationError):
    """The material does not satisfy the Neo-Hookean contract."""


class InvalidTensorError(PhysicsValidationError):
    """A tensor has an unsupported representation or dtype."""


class InvalidTensorShapeError(InvalidTensorError):
    """A tensor does not have the required trailing matrix shape."""


class NonFiniteTensorError(InvalidTensorError):
    """A tensor contains a NaN or infinity."""


class InadmissibleDeformationError(PhysicsValidationError):
    """A deformation has a determinant outside the admitted domain."""


class PlaneStressClosureError(PhysicsError):
    """Base class for plane-stress closure failures."""


class NewtonNonConvergenceError(PlaneStressClosureError):
    """The bounded plane-stress Newton solve did not converge."""


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
)
