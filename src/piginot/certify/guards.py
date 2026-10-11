"""Definiteness of a symmetric sparse tangent from an LDL^T-style factorization.

SuperLU in symmetric mode with no off-diagonal pivoting factors a symmetric
matrix as ``P^T A P = L U`` with ``U = D L^T``; by Sylvester's law of inertia
the signs of ``diag(U)`` count the matrix's positive and negative eigenvalues.
If SuperLU nevertheless pivots rows, the inertia is unknown and reported as
``None`` (callers treat that conservatively).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import splu  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class Inertia:
    """Counts of non-positive pivots; ``None`` when pivoting hid the inertia."""

    size: int
    non_positive: int | None

    @property
    def positive_definite(self) -> bool | None:
        return None if self.non_positive is None else self.non_positive == 0


def symmetric_inertia(matrix: Any) -> Inertia:
    """Inertia of a symmetric sparse matrix (positive definiteness test)."""

    square = csc_matrix(matrix, dtype=np.float64)
    size = int(square.shape[0])
    if size == 0:
        return Inertia(0, 0)
    try:
        factor = splu(
            square,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )
    except RuntimeError:  # exactly singular
        return Inertia(size, 1)
    if not np.array_equal(factor.perm_r, factor.perm_c):
        return Inertia(size, None)
    pivots = np.asarray(factor.U.diagonal(), dtype=np.float64)
    return Inertia(size, int(np.count_nonzero(~(pivots > 0.0))))
