"""Strain-energy formulas as data (S25, ADR 0105 decision 4, ADR 0107).

A hyperelastic material is a strain-energy density ``W`` written as a formula
of the invariants of the deformation and of named parameters. The formula is
parsed by a whitelist parser into a syntax tree — numbers, parameters, the
invariants ``I1``, ``I2``, ``J``, ``I1bar``, ``I2bar``, ``+ - * /``, ``**``
with a numeric exponent, and ``exp``, ``log``, ``sqrt`` of one argument — and
never evaluated as code. :mod:`piginot.materials` turns the tree into torch
operations; stress and tangent come from automatic differentiation.

``I1 = tr C``, ``I2 = (I1^2 - tr C^2) / 2`` and ``J = det F`` for
``C = F^T F``; ``I1bar = J^(-2/3) I1`` and ``I2bar = J^(-4/3) I2`` are the
isochoric invariants. Parameters carry a canonical unit (``MPa`` or ``1``);
the formula must be a stress (MPa), and every argument of ``exp``, ``log``
and ``sqrt`` must be dimensionless.

Each formula declares its small-strain shear and bulk moduli as expressions
of its parameters; the admission gate checks them against the tangent at the
reference state.
"""

from __future__ import annotations

import ast
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, cast

from piginot.contracts.canonical import canonical_json_bytes
from piginot.contracts.errors import ContractValidationError
from piginot.contracts.units import Dimension, Unit

SCHEMA: Final = "piginot.energy-formula-v1"
INVARIANTS: Final = ("I1", "I2", "J", "I1bar", "I2bar")
FUNCTIONS: Final = ("exp", "log", "sqrt")
SYMMETRIES: Final = ("isotropic",)
_UNITS: Final = {"MPa": Unit.MPA, "1": Unit.ONE}
_STRESS: Final = {Unit.MPA: 1, Unit.ONE: 0}

Tree = tuple[Any, ...]


class EnergyFormulaError(ContractValidationError):
    """A strain-energy formula broke a rule of the formula language."""


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EnergyFormulaError("only finite numeric literals are allowed")
    number = float(value)
    if not math.isfinite(number):
        raise EnergyFormulaError("numeric literals must be finite")
    return number


def _constant(node: ast.AST) -> float | None:
    """The value of a numeric literal, possibly signed; else ``None``."""

    if isinstance(node, ast.Constant):
        return _number(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _constant(node.operand)
        if inner is None:
            return None
        return inner if isinstance(node.op, ast.UAdd) else -inner
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        top, bottom = _constant(node.left), _constant(node.right)
        if top is None or bottom is None or bottom == 0.0:
            return None
        return top / bottom
    return None


_BINARY = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


def _tree(node: ast.AST) -> Tree:
    if isinstance(node, ast.Constant):
        return ("number", _number(node.value))
    if isinstance(node, ast.Name):
        return ("name", node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = "+" if isinstance(node.op, ast.UAdd) else "-"
        return ("unary", sign, _tree(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return ("binary", _BINARY[type(node.op)], _tree(node.left), _tree(node.right))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        exponent = _constant(node.right)
        if exponent is None:
            raise EnergyFormulaError("an exponent must be a number")
        return ("power", _tree(node.left), exponent)
    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in FUNCTIONS
            or len(node.args) != 1
            or node.keywords
        ):
            raise EnergyFormulaError(
                f"only {', '.join(FUNCTIONS)} of one argument may be called"
            )
        return ("call", node.func.id, _tree(node.args[0]))
    raise EnergyFormulaError(
        "a formula allows numbers, names, + - * /, ** with a numeric exponent, "
        "exp, log, sqrt and parentheses"
    )


def parse(source: object) -> Tree:
    """The syntax tree of ``source``; ``^`` is read as ``**``."""

    if not isinstance(source, str) or not source.strip():
        raise EnergyFormulaError("a formula must be non-empty text")
    text = source.strip().replace("^", "**")
    try:
        parsed = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise EnergyFormulaError("invalid formula syntax") from exc
    return _tree(parsed.body)


def names(tree: Tree) -> set[str]:
    kind = tree[0]
    if kind == "name":
        return {cast(str, tree[1])}
    if kind == "number":
        return set()
    if kind in ("unary", "call"):
        return names(cast(Tree, tree[2]))
    if kind == "power":
        return names(cast(Tree, tree[1]))
    return names(cast(Tree, tree[2])) | names(cast(Tree, tree[3]))


def stress_power(tree: Tree, powers: Mapping[str, int]) -> float:
    """The power of MPa that ``tree`` carries; refuses inconsistent units."""

    kind = tree[0]
    if kind == "number":
        return 0.0
    if kind == "name":
        name = cast(str, tree[1])
        if name not in powers:
            raise EnergyFormulaError(f"unknown name {name!r}")
        return float(powers[name])
    if kind == "unary":
        return stress_power(cast(Tree, tree[2]), powers)
    if kind == "call":
        if stress_power(cast(Tree, tree[2]), powers) != 0.0:
            raise EnergyFormulaError(f"the argument of {tree[1]} must be dimensionless")
        return 0.0
    if kind == "power":
        return stress_power(cast(Tree, tree[1]), powers) * float(tree[2])
    left = stress_power(cast(Tree, tree[2]), powers)
    right = stress_power(cast(Tree, tree[3]), powers)
    operator = tree[1]
    if operator in ("+", "-"):
        if left != right:
            raise EnergyFormulaError("added terms must have the same unit")
        return left
    return left + right if operator == "*" else left - right


def evaluate(tree: Tree, values: Mapping[str, float]) -> float:
    """``tree`` in float64 for scalar ``values`` (moduli, checks)."""

    kind = tree[0]
    if kind == "number":
        return float(tree[1])
    if kind == "name":
        return float(values[cast(str, tree[1])])
    if kind == "unary":
        operand = evaluate(cast(Tree, tree[2]), values)
        return operand if tree[1] == "+" else -operand
    if kind == "call":
        argument = evaluate(cast(Tree, tree[2]), values)
        function = {"exp": math.exp, "log": math.log, "sqrt": math.sqrt}[tree[1]]
        return function(argument)
    if kind == "power":
        return float(evaluate(cast(Tree, tree[1]), values) ** float(tree[2]))
    left = evaluate(cast(Tree, tree[2]), values)
    right = evaluate(cast(Tree, tree[3]), values)
    operator = tree[1]
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    return left / right


def _document(tree: Tree) -> list[Any]:
    return [_document(item) if isinstance(item, tuple) else item for item in tree]


@dataclass(frozen=True, slots=True)
class Formula:
    """A parsed formula; equality and identity follow its syntax tree."""

    source: str = field(compare=False)
    tree: Tree

    @classmethod
    def parse(cls, source: object) -> Formula:
        return cls(cast(str, source), parse(source))

    def canonical(self) -> list[Any]:
        return _document(self.tree)


@dataclass(frozen=True, slots=True)
class EnergyFormula:
    """A strain-energy density, its parameters and declared small-strain moduli.

    ``parameters`` maps each parameter name to its unit (``MPa`` or ``1``);
    ``positive`` lists the parameters that must be strictly positive.
    """

    energy: Formula
    parameters: Mapping[str, Unit]
    shear_modulus: Formula
    bulk_modulus: Formula
    positive: tuple[str, ...] = ()
    symmetry: str = "isotropic"

    def __post_init__(self) -> None:
        if self.symmetry not in SYMMETRIES:
            raise EnergyFormulaError(f"symmetry must be one of {SYMMETRIES}")
        for name, unit in self.parameters.items():
            if not name.isidentifier() or name in INVARIANTS or name in FUNCTIONS:
                raise EnergyFormulaError(f"parameter name {name!r} is not allowed")
            if unit not in _STRESS:
                raise EnergyFormulaError(f"{name}: the unit must be MPa or 1")
        unknown = sorted(set(self.positive) - set(self.parameters))
        if unknown:
            raise EnergyFormulaError(f"positive names unknown parameters {unknown}")
        powers = {name: _STRESS[unit] for name, unit in self.parameters.items()}
        energy_powers = {**powers, **dict.fromkeys(INVARIANTS, 0)}
        if stress_power(self.energy.tree, energy_powers) != 1.0:
            raise EnergyFormulaError("the strain energy must be a stress (MPa)")
        unused = sorted(set(self.parameters) - names(self.energy.tree))
        if unused:
            raise EnergyFormulaError(f"parameters not used by the energy: {unused}")
        for label, modulus in (
            ("shear_modulus", self.shear_modulus),
            ("bulk_modulus", self.bulk_modulus),
        ):
            if names(modulus.tree) & set(INVARIANTS):
                raise EnergyFormulaError(f"{label} may use parameters only")
            if stress_power(modulus.tree, powers) != 1.0:
                raise EnergyFormulaError(f"{label} must be a stress (MPa)")

    def moduli(self, values: Mapping[str, float]) -> tuple[float, float]:
        """The declared small-strain shear and bulk moduli for ``values``."""

        return (
            evaluate(self.shear_modulus.tree, values),
            evaluate(self.bulk_modulus.tree, values),
        )

    def check_values(self, values: Mapping[str, float]) -> None:
        """Refuse missing, extra, non-finite or non-positive parameter values."""

        if set(values) != set(self.parameters):
            raise EnergyFormulaError(
                f"give exactly the parameters {sorted(self.parameters)}"
            )
        for name, value in values.items():
            if not math.isfinite(value):
                raise EnergyFormulaError(f"{name} must be finite")
            if name in self.positive and value <= 0.0:
                raise EnergyFormulaError(f"{name} must be positive")

    def document(self, canonical: bool = False) -> dict[str, Any]:
        def text(formula: Formula) -> Any:
            return formula.canonical() if canonical else formula.source

        return {
            "schema": SCHEMA,
            "energy": text(self.energy),
            "parameters": {
                name: unit.value for name, unit in sorted(self.parameters.items())
            },
            "positive": sorted(self.positive),
            "small_strain": {
                "shear_modulus": text(self.shear_modulus),
                "bulk_modulus": text(self.bulk_modulus),
            },
            "symmetry": self.symmetry,
        }

    @property
    def formula_hash(self) -> str:
        """SHA-256 of the canonical form (expressions as syntax trees)."""

        return hashlib.sha256(canonical_json_bytes(self.document(True))).hexdigest()

    @property
    def model_name(self) -> str:
        """The material model name a user formula is registered under."""

        return f"energy-{self.formula_hash[:16]}"

    @classmethod
    def from_document(cls, document: object) -> EnergyFormula:
        if not isinstance(document, Mapping):
            raise EnergyFormulaError("an energy formula must be an object")
        allowed = {
            "schema",
            "energy",
            "parameters",
            "positive",
            "small_strain",
            "symmetry",
        }
        unknown = sorted(set(document) - allowed)
        if unknown:
            raise EnergyFormulaError(f"unknown key(s) {', '.join(unknown)}")
        missing = [
            k for k in ("energy", "parameters", "small_strain") if k not in document
        ]
        if missing:
            raise EnergyFormulaError(f"missing key(s) {', '.join(missing)}")
        if document.get("schema", SCHEMA) != SCHEMA:
            raise EnergyFormulaError(f"schema must be {SCHEMA!r}")
        parameters = document["parameters"]
        small = document["small_strain"]
        if not isinstance(parameters, Mapping) or not isinstance(small, Mapping):
            raise EnergyFormulaError("parameters and small_strain must be objects")
        if set(small) != {"shear_modulus", "bulk_modulus"}:
            raise EnergyFormulaError(
                "small_strain gives exactly shear_modulus and bulk_modulus"
            )
        units = {}
        for name, unit in parameters.items():
            if unit not in _UNITS:
                raise EnergyFormulaError(f"{name}: the unit must be 'MPa' or '1'")
            units[str(name)] = _UNITS[unit]
        positive = document.get("positive", [])
        if not isinstance(positive, Sequence) or isinstance(positive, str):
            raise EnergyFormulaError("positive must be a list of parameter names")
        return cls(
            energy=Formula.parse(document["energy"]),
            parameters=units,
            shear_modulus=Formula.parse(small["shear_modulus"]),
            bulk_modulus=Formula.parse(small["bulk_modulus"]),
            positive=tuple(str(name) for name in positive),
            symmetry=str(document.get("symmetry", "isotropic")),
        )


def _builtin(
    energy: str,
    parameters: Mapping[str, str],
    positive: Sequence[str],
    shear: str,
    bulk: str,
) -> EnergyFormula:
    return EnergyFormula.from_document(
        {
            "energy": energy,
            "parameters": dict(parameters),
            "positive": list(positive),
            "small_strain": {"shear_modulus": shear, "bulk_modulus": bulk},
        }
    )


BUILTIN_FORMULAS: Final[Mapping[str, EnergyFormula]] = {
    "mooney_rivlin": _builtin(
        "c10 * (I1bar - 3) + c01 * (I2bar - 3) + bulk_modulus / 2 * (J - 1) ** 2",
        {"c10": "MPa", "c01": "MPa", "bulk_modulus": "MPa"},
        ("bulk_modulus",),
        "2 * (c10 + c01)",
        "bulk_modulus",
    ),
    "yeoh": _builtin(
        "c10 * (I1bar - 3) + c20 * (I1bar - 3) ** 2 + c30 * (I1bar - 3) ** 3"
        " + bulk_modulus / 2 * (J - 1) ** 2",
        {"c10": "MPa", "c20": "MPa", "c30": "MPa", "bulk_modulus": "MPa"},
        ("c10", "bulk_modulus"),
        "2 * c10",
        "bulk_modulus",
    ),
    "gent": _builtin(
        "-shear_modulus * jm / 2 * log(1 - (I1bar - 3) / jm)"
        " + bulk_modulus / 2 * (J - 1) ** 2",
        {"shear_modulus": "MPa", "jm": "1", "bulk_modulus": "MPa"},
        ("shear_modulus", "jm", "bulk_modulus"),
        "shear_modulus",
        "bulk_modulus",
    ),
}
"""The built-in formula models (decoupled isochoric and volumetric parts, with
``kappa / 2 (J - 1)^2`` as the volumetric energy). Signs not listed as
positive are left to the admission gate (ellipticity), since published fits
use, for example, negative Yeoh ``c20`` or Mooney-Rivlin ``c01``."""

CLOSED_FORM_MODELS: Final[Mapping[str, Mapping[str, Dimension]]] = {
    "compressible_neo_hookean": {
        "elastic_modulus": Dimension.STRESS,
        "poisson_ratio": Dimension.DIMENSIONLESS,
    },
}
"""Models with a hand-written kernel in :mod:`piginot.physics`."""


def model_dimensions(model: str) -> Mapping[str, Dimension]:
    """Each property's dimension for a built-in model name; KeyError otherwise."""

    if model in CLOSED_FORM_MODELS:
        return CLOSED_FORM_MODELS[model]
    formula = BUILTIN_FORMULAS[model]
    return {name: unit.dimension for name, unit in formula.parameters.items()}


__all__ = (
    "BUILTIN_FORMULAS",
    "CLOSED_FORM_MODELS",
    "FUNCTIONS",
    "INVARIANTS",
    "SCHEMA",
    "EnergyFormula",
    "EnergyFormulaError",
    "Formula",
    "evaluate",
    "model_dimensions",
    "names",
    "parse",
    "stress_power",
)
