"""S7 primary finite-element solver surface."""

from piginot.fem.backend import ScikitFemBackend
from piginot.fem.body_forces import BodyForceProvider, ConstantBodyForce
from piginot.fem.models import (
    FemAttempt,
    FemConvergenceEntry,
    FemFailureCode,
    FemRequest,
    FemResult,
    FemSolveFailure,
    FemSolverConfig,
)
from piginot.fem.protocol import FemBackend

__all__ = (
    "FemAttempt",
    "FemBackend",
    "BodyForceProvider",
    "ConstantBodyForce",
    "FemConvergenceEntry",
    "FemFailureCode",
    "FemRequest",
    "FemResult",
    "FemSolveFailure",
    "FemSolverConfig",
    "ScikitFemBackend",
)
