"""What a :class:`~piginot.contracts.ProblemSpec` builds (S24, ADR 0106).

- :func:`fem_request`: the reference FEM request of one case;
- :func:`query_frame` and :func:`build_lifting`: the operator's frame and its
  exact Dirichlet lifting for a batch of cases;
- :func:`verify_lifting`: the lifting evidence of ADR 0027 over a set of
  parameter vectors.
"""

from piginot.problem.fem import FEM_MATERIAL_MODELS, fem_request
from piginot.problem.lifting import adf_pieces, build_lifting, query_frame
from piginot.problem.verification import (
    LiftingEvidence,
    box_points,
    verify_lifting,
)

__all__ = (
    "FEM_MATERIAL_MODELS",
    "LiftingEvidence",
    "adf_pieces",
    "box_points",
    "build_lifting",
    "fem_request",
    "query_frame",
    "verify_lifting",
)
