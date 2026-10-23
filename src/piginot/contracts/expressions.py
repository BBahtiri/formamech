"""Safe syntax contracts for named derived quantities.

This module parses and validates expression structure.  It deliberately contains no
substitution or evaluation operation; DOE and envelope execution belong to S9.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias, cast

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
    Dimension.ENERGY: (1, 1),
    Dimension.ANGLE: (0, 0),
}
_VECTOR_DIMENSIONS: dict[DimensionVector, frozenset[Dimension]] = {}
for _dimension, _vector in _DIMENSION_VECTORS.items():
    _VECTOR_DIMENSIONS[_vector] = _VECTOR_DIMENSIONS.get(_vector, frozenset()) | {
        _dimension
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
) -> tuple[DimensionVector, frozenset[Dimension]]:
    kind = tree[0]
    if kind == "number":
        return (0, 0), frozenset({Dimension.DIMENSIONLESS})
    if kind == "name":
        name = cast(str, tree[1])
        try:
            dimension = parameter_dimensions[name]
        except KeyError as exc:
            raise UnknownSymbolError(f"unknown parameter {name!r}") from exc
        try:
            return _DIMENSION_VECTORS[dimension], frozenset({dimension})
        except KeyError as exc:
            raise UnitMismatchError(
                f"unsupported parameter dimension {dimension!r}"
            ) from exc
    if kind == "unary":
        return _combine(cast(ExpressionTree, tree[2]), parameter_dimensions)
    if kind != "binary":
        raise ExpressionSyntaxError("invalid canonical expression tree")
    operator = cast(str, tree[1])
    left_vector, left_dimensions = _combine(
        cast(ExpressionTree, tree[2]), parameter_dimensions
    )
    right_vector, right_dimensions = _combine(
        cast(ExpressionTree, tree[3]), parameter_dimensions
    )
    if operator in {"+", "-"}:
        shared_dimensions = left_dimensions & right_dimensions
        if left_vector != right_vector or not shared_dimensions:
            raise UnitMismatchError(
                "addition and subtraction require matching dimensions"
            )
        return left_vector, shared_dimensions
    if operator == "*":
        vector = (
            left_vector[0] + right_vector[0],
            left_vector[1] + right_vector[1],
        )
        if left_vector == (0, 0) and right_vector == (0, 0):
            # A plain number scales an angle; anything else stays dimensionless.
            tagged = (left_dimensions | right_dimensions) - {Dimension.DIMENSIONLESS}
            return vector, tagged or frozenset({Dimension.DIMENSIONLESS})
        if left_vector == (0, 0):
            return vector, right_dimensions
        if right_vector == (0, 0):
            return vector, left_dimensions
    elif operator == "/":
        vector = (
            left_vector[0] - right_vector[0],
            left_vector[1] - right_vector[1],
        )
        if right_vector == (0, 0):
            return vector, left_dimensions
    else:
        raise ExpressionSyntaxError(f"unsupported canonical operator {operator!r}")
    return vector, _VECTOR_DIMENSIONS.get(vector, frozenset())


@dataclass(frozen=True, slots=True)
class DerivedExpression:
    """A canonical structural syntax tree with static unit validation.

    Whitespace, redundant parentheses, and numeric spelling normalize.  Algebraic
    equivalence does not: operand order and unary operators remain part of identity.
    """

    __canonical_namespace__: ClassVar[str] = "piginot.contracts"

    source: str = field(compare=False, repr=False, metadata={"canonical": False})
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
        vector, possible_dimensions = _combine(self.canonical_ast, parameter_dimensions)
        if not possible_dimensions:
            raise UnitMismatchError(f"unsupported derived dimension vector {vector}")
        if expected_dimension not in possible_dimensions:
            inferred = ", ".join(
                dimension.value
                for dimension in sorted(
                    possible_dimensions, key=lambda dimension: dimension.value
                )
            )
            raise UnitMismatchError(
                f"expression is {inferred}, expected {expected_dimension.value}"
            )
        return expected_dimension
