"""Safe syntax contracts for named derived quantities.

This module parses and validates expression structure.  It deliberately contains no
substitution or evaluation operation; DOE and envelope execution belong to S9.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias, cast

from piginot.contracts.errors import (
    ContractValidationError,
    ExpressionSyntaxError,
    UnitMismatchError,
    UnknownSymbolError,
)
from piginot.contracts.units import Dimension

ExpressionTree: TypeAlias = tuple[object, ...]
DimensionVector: TypeAlias = tuple[int, int]

_DIMENSION_VECTORS: dict[Dimension, DimensionVector] = {
    Dimension.DIMENSIONLESS: (0, 0),
    Dimension.LENGTH: (1, 0),
    Dimension.FORCE: (0, 1),
    Dimension.STRESS: (-2, 1),
    Dimension.MOMENT: (1, 1),
}
_VECTOR_DIMENSIONS = {
    vector: dimension for dimension, vector in _DIMENSION_VECTORS.items()
}


def _number(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExpressionSyntaxError("only finite numeric literals are allowed")
    number = float(value)
    if not math.isfinite(number):
        raise ContractValidationError("numeric literals must be finite")
    return int(number) if number.is_integer() else number


def _tree(node: ast.AST) -> ExpressionTree:
    if isinstance(node, ast.Name):
        return ("name", node.id)
    if isinstance(node, ast.Constant):
        return ("number", _number(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operator = "+" if isinstance(node.op, ast.UAdd) else "-"
        return ("unary", operator, _tree(node.operand))
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        operator = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
        }[type(node.op)]
        return ("binary", operator, _tree(node.left), _tree(node.right))
    raise ExpressionSyntaxError(
        "derived expressions allow only names, finite numbers, +, -, *, /, and parentheses"
    )


def _combine(
    tree: ExpressionTree, parameter_dimensions: Mapping[str, Dimension]
) -> DimensionVector:
    kind = tree[0]
    if kind == "number":
        return (0, 0)
    if kind == "name":
        name = cast(str, tree[1])
        try:
            return _DIMENSION_VECTORS[parameter_dimensions[name]]
        except KeyError as exc:
            raise UnknownSymbolError(f"unknown parameter {name!r}") from exc
    if kind == "unary":
        return _combine(cast(ExpressionTree, tree[2]), parameter_dimensions)
    if kind != "binary":
        raise ExpressionSyntaxError("invalid canonical expression tree")
    operator = cast(str, tree[1])
    left = _combine(cast(ExpressionTree, tree[2]), parameter_dimensions)
    right = _combine(cast(ExpressionTree, tree[3]), parameter_dimensions)
    if operator in {"+", "-"}:
        if left != right:
            raise UnitMismatchError(
                "addition and subtraction require matching dimensions"
            )
        return left
    if operator == "*":
        return (left[0] + right[0], left[1] + right[1])
    if operator == "/":
        return (left[0] - right[0], left[1] - right[1])
    raise ExpressionSyntaxError(f"unsupported canonical operator {operator!r}")


@dataclass(frozen=True, slots=True)
class DerivedExpression:
    """A normalized, immutable syntax tree with static unit validation."""

    source: str = field(compare=False, repr=False)
    canonical_ast: ExpressionTree

    @classmethod
    def parse(cls, source: str) -> DerivedExpression:
        if not isinstance(source, str) or not source.strip():
            raise ExpressionSyntaxError("derived expression must be non-empty text")
        try:
            parsed = ast.parse(source.strip(), mode="eval")
        except (SyntaxError, ValueError) as exc:
            raise ExpressionSyntaxError("invalid derived expression syntax") from exc
        return cls(source=source, canonical_ast=_tree(parsed.body))

    def validate(
        self,
        *,
        parameter_dimensions: Mapping[str, Dimension],
        expected_dimension: Dimension,
    ) -> Dimension:
        vector = _combine(self.canonical_ast, parameter_dimensions)
        inferred = _VECTOR_DIMENSIONS.get(vector)
        if inferred is None:
            raise UnitMismatchError(f"unsupported derived dimension vector {vector}")
        if inferred is not expected_dimension:
            raise UnitMismatchError(
                f"expression is {inferred.value}, expected {expected_dimension.value}"
            )
        return inferred
