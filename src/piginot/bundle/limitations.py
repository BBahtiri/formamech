"""The limitation paragraph every bundle report carries (``G15.limitations``)."""

from __future__ import annotations

REQUIRED_LIMITATION_PARAGRAPH = (
    "PI-GINOT performs numerical verification against finite element method (FEM) "
    "solutions. That is not experimental validation, and it does not show that "
    "predictions agree with measurements from physical specimens. Verification for "
    "a bundle establishes nothing about geometries outside that bundle's declared "
    "input envelope. Inputs outside the envelope must not be treated as covered by "
    "the bundle's results. Finite verification sampling is evidence about the "
    "evaluated cases, not a mathematical proof of a uniform error bound over every "
    "admissible input."
)

__all__ = ("REQUIRED_LIMITATION_PARAGRAPH",)
