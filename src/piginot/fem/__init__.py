"""S7 primary finite-element solver surface."""

from piginot.fem.backend import ScikitFemBackend
from piginot.fem.models import (
    FemAttempt,
    FemConvergenceEntry,
    FemFailureCode,
    FemRequest,
    FemResult,
    FemSolveFailure,
    FemSolverConfig,
)

__all__ = (
    "FemAttempt",
    "FemConvergenceEntry",
    "FemFailureCode",
    "FemRequest",
    "FemResult",
    "FemSolveFailure",
    "FemSolverConfig",
    "ScikitFemBackend",
)
