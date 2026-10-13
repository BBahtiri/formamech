"""The quarter-model tension lifting (ADR 0018), the ``quarter`` kind of
ADR 0106, moved here from :mod:`piginot.operators.energy_operator` without
numerical change. Bundles name it ``axis-aligned-tension``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piginot.lifting._checks import (
    require_same_context as _require_same_context,
)
from piginot.lifting._checks import (
    validate_field as _validate_field,
)
from piginot.lifting._checks import (
    validate_matrix as _validate_matrix,
)
from piginot.lifting.base import AffineDirichletLifting


@dataclass(frozen=True, slots=True)
class AxisAlignedTensionLifting:
    """Lifting for ``u_x(0, y) = 0``, ``u_x(a, y) = d`` and ``u_y(x, 0) = 0``.

    The domain lies in ``[0, a] x [0, b]`` with ``a`` and ``b`` given per sample by
    ``extent``. Writing ``xi = x / a`` and ``eta = y / b``, the displacement is
    ``u_x = d * (xi + xi * (1 - xi) * phi_x)`` and ``u_y = d * eta * phi_y``, so the
    correction ``phi`` is measured in units of the prescribed displacement.
    """

    extent: torch.Tensor
    displacement: torch.Tensor

    def __post_init__(self) -> None:
        _validate_matrix(self.extent, "extent", trailing=2)
        if bool((self.extent <= 0.0).any().item()):
            raise ValueError("extent must be strictly positive")
        if not isinstance(self.displacement, torch.Tensor):
            raise TypeError("displacement must be a torch.Tensor")
        if self.displacement.shape != (self.extent.shape[0],):
            raise ValueError("displacement must have shape (batch,)")
        _require_same_context(self.extent, self.displacement, "extent", "displacement")
        if not bool(torch.isfinite(self.displacement).all().item()):
            raise ValueError("displacement must be finite")

    def __call__(self, query_xy: torch.Tensor) -> AffineDirichletLifting:
        _validate_field(query_xy, "query_xy", trailing=2)
        if query_xy.shape[0] != self.extent.shape[0]:
            raise ValueError("query_xy and lifting must share a batch dimension")
        _require_same_context(query_xy, self.extent, "query_xy", "extent")
        xi = query_xy[..., 0] / self.extent[:, 0:1]
        eta = query_xy[..., 1] / self.extent[:, 1:2]
        d = self.displacement.unsqueeze(1)
        offset = torch.stack((d * xi, torch.zeros_like(eta)), dim=-1)
        scale = torch.stack((d * xi * (1.0 - xi), d * eta), dim=-1)
        return AffineDirichletLifting(offset, scale)


__all__ = ("AxisAlignedTensionLifting",)
