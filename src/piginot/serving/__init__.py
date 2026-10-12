"""Verified serving: FEM recipes and the fast, certified and exact modes (S22).

Importing this package needs only the runtime extra; the FEM stack is loaded
when a certified or exact request is served.
"""

from piginot.serving.hybrid import (
    MissingExtraError,
    ServingMode,
    ServingPath,
    SolveAttempt,
    VerifiedPrediction,
    default_request_id,
    operator_displacement,
    serve,
    solve_ladder,
)

__all__ = (
    "MissingExtraError",
    "ServingMode",
    "ServingPath",
    "SolveAttempt",
    "VerifiedPrediction",
    "default_request_id",
    "operator_displacement",
    "serve",
    "solve_ladder",
)
