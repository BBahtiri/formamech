"""End-to-end FormaMech showcase on a CAD-generated viscoelastic elastomer mount bracket."""

from __future__ import annotations

from examples.viscoelastic_cad_showcase.cad_builder import (
    BracketGeometryParams,
    classify_bracket_boundaries,
    generate_bracket_step,
)

__all__ = (
    "BracketGeometryParams",
    "classify_bracket_boundaries",
    "generate_bracket_step",
)
