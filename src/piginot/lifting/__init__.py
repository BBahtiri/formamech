"""Exact Dirichlet liftings of the energy-form operator (ADR 0018, ADR 0106).

One interface, :class:`AffineDirichletLifting` (``u = offset + scale * phi``),
and two kinds: the quarter-model tension lifting and the approximate distance
function lifting over segments and arcs.
"""

from piginot.lifting.adf import AdfLifting, DirichletPiece
from piginot.lifting.base import AffineDirichletLifting, DirichletLifting
from piginot.lifting.quarter import AxisAlignedTensionLifting
from piginot.lifting.verify import boundary_points, boundary_residual

__all__ = (
    "AdfLifting",
    "AffineDirichletLifting",
    "AxisAlignedTensionLifting",
    "DirichletLifting",
    "DirichletPiece",
    "boundary_points",
    "boundary_residual",
)
