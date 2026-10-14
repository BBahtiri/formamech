"""Scalar values of a strain-energy formula (S25, ADR 0107 decision 3).

The contract (:mod:`piginot.contracts.energy_formula`) holds a formula as a
syntax tree and does not evaluate it (S2); this module evaluates a tree for
scalar parameter values in float64: the declared small-strain moduli and
checks. :mod:`piginot.materials.hyperelastic` evaluates the energy itself in
torch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

from piginot.contracts.energy_formula import EnergyFormula, Tree, fibre_angle


def _ramp(value: float) -> float:
    return max(value, 0.0)


def fibre_directions(
    formula: EnergyFormula, values: Mapping[str, float]
) -> dict[str, tuple[float, float, float]]:
    """Each family's unit direction ``(cos θ, sin θ, 0)`` for ``values``
    (angles in degrees from the x axis)."""

    directions = {}
    for label, reference in formula.fibres.items():
        sign, name = fibre_angle(reference)
        angle = math.radians(sign * values[name])
        directions[label] = (math.cos(angle), math.sin(angle), 0.0)
    return directions


def formula_value(tree: Tree, values: Mapping[str, float]) -> float:
    """``tree`` in float64 for scalar ``values`` (moduli, checks)."""

    kind = tree[0]
    if kind == "number":
        return float(tree[1])
    if kind == "name":
        return float(values[cast(str, tree[1])])
    if kind == "unary":
        operand = formula_value(cast(Tree, tree[2]), values)
        return operand if tree[1] == "+" else -operand
    if kind == "call":
        argument = formula_value(cast(Tree, tree[2]), values)
        function = {
            "exp": math.exp,
            "log": math.log,
            "sqrt": math.sqrt,
            "ramp": _ramp,
        }[tree[1]]
        return function(argument)
    if kind == "power":
        return float(formula_value(cast(Tree, tree[1]), values) ** float(tree[2]))
    left = formula_value(cast(Tree, tree[2]), values)
    right = formula_value(cast(Tree, tree[3]), values)
    operator = tree[1]
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    return left / right


def formula_moduli(
    formula: EnergyFormula, values: Mapping[str, float]
) -> tuple[float, float]:
    """The declared small-strain shear and bulk moduli for ``values``."""

    return (
        formula_value(formula.shear_modulus.tree, values),
        formula_value(formula.bulk_modulus.tree, values),
    )


__all__ = ("fibre_directions", "formula_moduli", "formula_value")
