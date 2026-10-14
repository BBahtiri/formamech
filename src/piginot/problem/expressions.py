"""The value of a ProblemSpec expression (ADR 0106 decision 2).

Contracts hold expressions as syntax trees and do not evaluate them (S2);
this module does, left to right in float64, so a value is the same on every
platform and equals the bundle's S15 expressions for the shipped families.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from piginot.contracts.problem_spec import Expression

Tree = tuple[object, ...]


def _value(tree: Tree, values: Mapping[str, float]) -> float:
    kind = tree[0]
    if kind == "number":
        return float(cast(float, tree[1]))
    if kind == "name":
        return float(values[cast(str, tree[1])])
    if kind == "unary":
        operand = _value(cast(Tree, tree[2]), values)
        return operand if tree[1] == "+" else -operand
    left = _value(cast(Tree, tree[2]), values)
    right = _value(cast(Tree, tree[3]), values)
    operator = tree[1]
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if right == 0.0:
        raise ZeroDivisionError("expression divides by zero")
    return left / right


def value(expression: Expression, values: Mapping[str, float]) -> float:
    """The value of ``expression`` for the parameter ``values``."""

    return _value(expression.tree, values)


__all__ = ("value",)
