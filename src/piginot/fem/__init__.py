"""S7 primary finite-element solver surface."""

from piginot.fem.backend import ScikitFemBackend
from piginot.fem.body_forces import BodyForceProvider, ConstantBodyForce
from piginot.fem.field_providers import (
    DirichletFieldProvider,
    TractionFieldProvider,
)
from piginot.fem.models import (
    FemAttempt,
    FemConvergenceEntry,
    FemFailureCode,
    FemRequest,
    FemResult,
    FemSolveFailure,
    FemSolverConfig,
)
from piginot.fem.protocol import FemBackend, IndependentSolverAdapter

__all__ = (
    "FemAttempt",
    "FemBackend",
    "BodyForceProvider",
    "ConstantBodyForce",
    "DirichletFieldProvider",
    "FemConvergenceEntry",
    "FemFailureCode",
    "FemRequest",
    "FemResult",
    "FemSolveFailure",
    "FemSolverConfig",
    "IndependentSolverAdapter",
    "ScikitFemBackend",
    "TractionFieldProvider",
)
