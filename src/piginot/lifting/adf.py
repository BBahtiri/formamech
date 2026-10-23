"""Exact Dirichlet lifting from approximate distance functions (ADR 0106).

For each displacement component ``c`` the Dirichlet set is a union of pieces:
line segments and minor circular arcs, each traversed with the domain on its
left and carrying a prescribed value. Following Sukumar and Srivastava (2022,
CMAME 389, 114333):

- each piece ``i`` has a distance-like function ``phi_i``: the signed distance
  ``f`` to its supporting line or circle, trimmed to the piece by
  ``t = ((L/2)^2 - |q - m|^2) / L`` (``m`` and ``L`` the chord's middle and
  length) as ``phi_i = sqrt(f^2 + psi^2)``, ``psi = (sqrt(t^2 + f^4) - t) / 2``;
  an untrimmed line keeps ``phi_i = f``;
- the component's function is their R-equivalence of order one,
  ``phi_c = (sum_i 1 / phi_i)^-1``, which vanishes on every piece and has unit
  normal derivative there;
- the extension of the data is the transfinite interpolation
  ``g_c = sum_i w_i g_i`` with ``w_i = (1 / phi_i) / sum_j (1 / phi_j)``.

All of it is evaluated in coordinates ``q = (x - origin) / l`` with ``l`` the
frame's larger extent, so ``t`` and ``f^4`` are commensurate. The lifting is
``offset_c = g_c`` and ``scale_c = d * l * phi_c / extent_c`` with ``d`` the
displacement scale. For untrimmed lines ``x = 0``, ``x = a`` and ``y = 0`` this
is the quarter lifting of :mod:`piginot.lifting.quarter`.

Every branch is written so that values and gradients stay finite on the
pieces, at their end points and at joints between pieces; on a piece the
gradient of ``phi_i`` is the inward unit normal. ``phi_i`` behaves like
``|f|`` across a piece, so a point a rounding error outside it
(within :data:`BOUNDARY_BAND`) is given the domain-side gradient; its value
is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
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

ARC_RADIUS_TOLERANCE = 1e-9
"""Largest relative difference of an arc's start and end radii."""

BOUNDARY_BAND = 1e-9
"""Depth, in units of the frame's larger extent, of the band outside a piece
whose points count as on it: a boundary node a rounding error outside keeps
the domain-side derivative instead of its mirror image."""


def _safe_sqrt(value: torch.Tensor) -> torch.Tensor:
    positive = value > 0
    root = torch.sqrt(torch.where(positive, value, torch.ones_like(value)))
    return torch.where(positive, root, torch.zeros_like(value))


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    nonzero = denominator != 0
    quotient = numerator / torch.where(
        nonzero, denominator, torch.ones_like(denominator)
    )
    return torch.where(nonzero, quotient, torch.zeros_like(quotient))


def _cross(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


@dataclass(frozen=True, slots=True)
class DirichletPiece:
    """A segment (``center is None``) or the minor arc from ``start`` to ``end``
    about ``center``, with the value prescribed on it.

    ``start``, ``end`` and ``center`` have shape ``(batch, 2)`` in physical
    coordinates; ``value`` has shape ``(batch,)``. The domain lies on the left
    of the direction of travel. ``trimmed=False`` uses a segment's whole
    supporting line.
    """

    start: torch.Tensor
    end: torch.Tensor
    value: torch.Tensor
    center: torch.Tensor | None = None
    trimmed: bool = True

    def __post_init__(self) -> None:
        _validate_matrix(self.start, "start", trailing=2)
        _validate_matrix(self.end, "end", trailing=2)
        if self.start.shape != self.end.shape:
            raise ValueError("start and end must have the same shape")
        _require_same_context(self.start, self.end, "start", "end")
        _require_same_context(self.start, self.value, "start", "value")
        if self.value.shape != (self.start.shape[0],):
            raise ValueError("value must have shape (batch,)")
        if not bool(torch.isfinite(self.value).all().item()):
            raise ValueError("value must be finite")
        if not isinstance(self.trimmed, bool):
            raise TypeError("trimmed must be a bool")
        chord = torch.linalg.vector_norm(self.end - self.start, dim=-1)
        if bool((chord <= 0.0).any().item()):
            raise ValueError("a piece needs distinct end points")
        if self.center is None:
            return
        if not self.trimmed:
            raise ValueError("an arc is always trimmed")
        _validate_matrix(self.center, "center", trailing=2)
        if self.center.shape != self.start.shape:
            raise ValueError("center must have the shape of start")
        _require_same_context(self.start, self.center, "start", "center")
        first = torch.linalg.vector_norm(self.start - self.center, dim=-1)
        second = torch.linalg.vector_norm(self.end - self.center, dim=-1)
        if bool(((first - second).abs() > ARC_RADIUS_TOLERANCE * first).any().item()):
            raise ValueError("an arc's end points must lie on one circle")
        turn = _cross(self.start - self.center, self.end - self.center)
        if bool((turn == 0.0).any().item()):
            raise ValueError("an arc must span less than a half circle")

    @property
    def batch(self) -> int:
        return int(self.start.shape[0])

    def signed_and_trim(
        self, query: torch.Tensor, origin: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The signed distance ``f`` (positive on the domain side) and the
        trimming function ``t`` (``None`` when untrimmed), in units of
        ``length``, at ``query`` of shape ``(batch, count, 2)`` in the same
        scaled coordinates ``(x - origin) / length``."""

        def scaled(point: torch.Tensor) -> torch.Tensor:
            return ((point - origin) / length[:, None]).unsqueeze(1)

        start, end = scaled(self.start), scaled(self.end)
        chord = end - start
        size = torch.linalg.vector_norm(chord, dim=-1)
        if self.center is None:
            signed = _cross(chord, query - start) / size
        else:
            center = scaled(self.center)
            radius = torch.linalg.vector_norm(start - center, dim=-1)
            orientation = torch.sign(_cross(start - center, end - center))
            squared = ((query - center) ** 2).sum(dim=-1)
            signed = orientation * (radius * radius - squared) / (2.0 * radius)
        if not self.trimmed:
            return signed, None
        middle = 0.5 * (start + end)
        trim = ((0.5 * size) ** 2 - ((query - middle) ** 2).sum(dim=-1)) / size
        return signed, trim

    def distance(
        self, query: torch.Tensor, origin: torch.Tensor, length: torch.Tensor
    ) -> torch.Tensor:
        """``phi_i`` at ``query`` (``(batch, count, 2)``), in units of ``length``."""

        signed, trim = self.signed_and_trim(query, origin, length)
        if trim is None:
            return signed
        quartic = signed**4
        inside = trim > 0
        trim_in = torch.where(inside, trim, torch.ones_like(trim))
        psi_in = quartic / (2.0 * (torch.sqrt(trim_in * trim_in + quartic) + trim_in))
        trim_out = torch.where(inside, -torch.ones_like(trim), trim)
        psi_out = 0.5 * (_safe_sqrt(trim_out * trim_out + quartic) - trim_out)
        psi = torch.where(inside, psi_in, psi_out)
        squared_psi = psi * psi
        band = (signed < 0) & (signed > -BOUNDARY_BAND)
        reflected = torch.where(band, signed - 2.0 * signed.detach(), signed)
        ahead = reflected >= 0
        signed_ahead = torch.where(ahead, reflected, torch.zeros_like(signed))
        root_ahead = _safe_sqrt(signed_ahead * signed_ahead + squared_psi)
        phi_ahead = signed_ahead + _safe_divide(squared_psi, root_ahead + signed_ahead)
        phi_behind = _safe_sqrt(signed * signed + squared_psi)
        return torch.where(ahead, phi_ahead, phi_behind)


def _combine(
    phis: Sequence[torch.Tensor], values: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """R-equivalence of order one and the transfinite interpolation."""

    if len(phis) == 1:
        return phis[0], values[0][:, None].expand_as(phis[0])
    others = []
    for index in range(len(phis)):
        product = torch.ones_like(phis[0])
        for other, phi in enumerate(phis):
            if other != index:
                product = product * phi
        others.append(product)
    denominator = torch.stack(others).sum(dim=0)
    every = others[0] * phis[0]
    phi = _safe_divide(every, denominator)
    weighted = sum(
        (w * v[:, None] for w, v in zip(others, values, strict=True)),
        torch.zeros_like(phis[0]),
    )
    interpolated = _safe_divide(weighted, denominator)
    zeros = [(phi_i == 0).to(phi_i.dtype) for phi_i in phis]
    count = torch.stack(zeros).sum(dim=0)
    joint = _safe_divide(
        sum(
            (z * v[:, None] for z, v in zip(zeros, values, strict=True)),
            torch.zeros_like(phis[0]),
        ),
        count,
    )
    return phi, torch.where(denominator == 0, joint, interpolated)


@dataclass(frozen=True, slots=True)
class AdfLifting:
    """The ADF lifting of one batch; ``pieces[c]`` lists component ``c``'s pieces.

    ``origin`` and ``extent`` are the query frame, shape ``(batch, 2)``;
    ``displacement`` is the scale ``d`` of the correction, shape ``(batch,)``.
    """

    origin: torch.Tensor
    extent: torch.Tensor
    displacement: torch.Tensor
    pieces: tuple[tuple[DirichletPiece, ...], tuple[DirichletPiece, ...]]

    def __post_init__(self) -> None:
        _validate_matrix(self.origin, "origin", trailing=2)
        _validate_matrix(self.extent, "extent", trailing=2)
        if self.origin.shape != self.extent.shape:
            raise ValueError("origin and extent must have the same shape")
        _require_same_context(self.origin, self.extent, "origin", "extent")
        if bool((self.extent <= 0.0).any().item()):
            raise ValueError("extent must be strictly positive")
        if not isinstance(self.displacement, torch.Tensor):
            raise TypeError("displacement must be a torch.Tensor")
        if self.displacement.shape != (self.extent.shape[0],):
            raise ValueError("displacement must have shape (batch,)")
        _require_same_context(self.extent, self.displacement, "extent", "displacement")
        if not bool(torch.isfinite(self.displacement).all().item()):
            raise ValueError("displacement must be finite")
        if len(self.pieces) != 2 or any(not component for component in self.pieces):
            raise ValueError("each component needs at least one Dirichlet piece")
        for component in self.pieces:
            for piece in component:
                if not isinstance(piece, DirichletPiece):
                    raise TypeError("pieces must be DirichletPiece values")
                if piece.batch != self.extent.shape[0]:
                    raise ValueError("pieces and frame must share a batch dimension")
                _require_same_context(self.extent, piece.start, "extent", "piece")

    @property
    def length(self) -> torch.Tensor:
        """The frame's larger extent per sample, the unit of ``q``."""

        return self.extent.max(dim=-1).values

    def components(
        self, query_xy: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """``(phi_c, g_c)`` per component; ``phi_c`` in units of :attr:`length`."""

        length = self.length
        query = (query_xy - self.origin.unsqueeze(1)) / length[:, None, None]
        result = []
        for component in self.pieces:
            phis = [piece.distance(query, self.origin, length) for piece in component]
            result.append(_combine(phis, [piece.value for piece in component]))
        return result[0], result[1]

    def __call__(self, query_xy: torch.Tensor) -> AffineDirichletLifting:
        _validate_field(query_xy, "query_xy", trailing=2)
        if query_xy.shape[0] != self.extent.shape[0]:
            raise ValueError("query_xy and lifting must share a batch dimension")
        _require_same_context(query_xy, self.extent, "query_xy", "extent")
        (phi_x, g_x), (phi_y, g_y) = self.components(query_xy)
        factor = (self.displacement * self.length)[:, None]
        scale = torch.stack(
            (
                factor * phi_x / self.extent[:, 0:1],
                factor * phi_y / self.extent[:, 1:2],
            ),
            dim=-1,
        )
        return AffineDirichletLifting(torch.stack((g_x, g_y), dim=-1), scale)


__all__ = ("ARC_RADIUS_TOLERANCE", "BOUNDARY_BAND", "AdfLifting", "DirichletPiece")
