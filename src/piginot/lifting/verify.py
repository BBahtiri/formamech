"""Boundary residuals of a lifting: the evidence ADR 0027 asks of a hard
lifting, measured on points along each Dirichlet piece (ADR 0106 decision 4).
"""

from __future__ import annotations

import math

import torch

from piginot.lifting.adf import AdfLifting, DirichletPiece


def boundary_points(piece: DirichletPiece, count: int) -> torch.Tensor:
    """``count`` points from ``start`` to ``end`` along a piece, end points
    included; shape ``(batch, count, 2)``."""

    if count < 2:
        raise ValueError("at least the two end points are needed")
    fraction = torch.linspace(0.0, 1.0, count, dtype=piece.start.dtype)[None, :]
    points: torch.Tensor
    if piece.center is None:
        points = (
            piece.start[:, None, :]
            + fraction[..., None] * (piece.end - piece.start)[:, None, :]
        )
        points[:, 0] = piece.start
        points[:, -1] = piece.end
        return points
    start = piece.start - piece.center
    end = piece.end - piece.center
    radius = torch.linalg.vector_norm(start, dim=-1)
    first = torch.atan2(start[:, 1], start[:, 0])
    turn = torch.atan2(
        start[:, 0] * end[:, 1] - start[:, 1] * end[:, 0],
        (start * end).sum(dim=-1),
    )
    angle = first[:, None] + fraction * turn[:, None]
    circle = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    points = piece.center[:, None, :] + radius[:, None, None] * circle
    points[:, 0] = piece.start
    points[:, -1] = piece.end
    return points


def boundary_residual(
    lifting: AdfLifting,
    count: int,
    *,
    generator: torch.Generator,
    correction_scale: float = 1.0,
) -> float:
    """The largest ``|u_c - value| / |d|`` over ``count`` points of every piece,
    for a random correction ``phi`` of standard deviation ``correction_scale``.
    """

    worst = 0.0
    for component, pieces in enumerate(lifting.pieces):
        for piece in pieces:
            points = boundary_points(piece, count)
            lifted = lifting(points)
            phi = correction_scale * torch.randn(
                lifted.offset.shape, generator=generator, dtype=points.dtype
            )
            u = lifted.offset + lifted.scale * phi
            error = (u[..., component] - piece.value[:, None]).abs()
            scale = lifting.displacement.abs()[:, None]
            residual = float((error / scale).max().item())
            if not math.isfinite(residual):
                return math.inf
            worst = max(worst, residual)
    return worst


__all__ = ("boundary_points", "boundary_residual")
