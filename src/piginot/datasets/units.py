"""Finite S10 vocabulary for physical integration measures."""

from __future__ import annotations

from enum import StrEnum


class AreaUnit(StrEnum):
    """Canonical area unit used by 2D FEM quadrature weights."""

    SQUARE_MILLIMETRE = "mm^2"


__all__ = ("AreaUnit",)
