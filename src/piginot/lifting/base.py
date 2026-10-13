"""The lifting interface of the energy-form operator (ADR 0018, ADR 0106).

A lifting maps query points to ``u = offset + scale * phi``: ``offset``
extends the Dirichlet data into the domain and ``scale`` vanishes on each
component's Dirichlet set, so any finite operator output ``phi`` meets the
essential conditions exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from piginot.lifting._checks import (
    require_same_context as _require_same_context,
)
from piginot.lifting._checks import (
    validate_field as _validate_field,
)


@dataclass(frozen=True, slots=True)
class AffineDirichletLifting:
    """Exact lifting ``u = offset + scale * phi`` evaluated at query points.

    ``offset`` satisfies every prescribed displacement and ``scale`` vanishes on
    each constrained component's boundary, so any finite correction ``phi``
    reproduces the essential conditions exactly.
    """

    offset: torch.Tensor
    scale: torch.Tensor

    def __post_init__(self) -> None:
        _validate_field(self.offset, "offset", trailing=2)
        _validate_field(self.scale, "scale", trailing=2)
        if self.offset.shape != self.scale.shape:
            raise ValueError("offset and scale must have the same shape")
        _require_same_context(self.offset, self.scale, "offset", "scale")


DirichletLifting = Callable[[torch.Tensor], AffineDirichletLifting]


__all__ = ("AffineDirichletLifting", "DirichletLifting")
