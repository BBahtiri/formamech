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

Fibre families (S26, ADR 0108): ``"fibres": {"a": "beta", "b": "-beta"}``
declares at most two in-plane families at the angle of a ``deg`` parameter
(or its negative) from the x axis. Each family ``<label>`` adds the full
invariants ``I4<label> = a.C a`` and ``I5<label> = a.C^2 a``; ``ramp(x) =
max(x, 0)`` writes the tension-only switch. A formula without fibres keeps
its S25 canonical form and identity.
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
FUNCTIONS: Final = ("exp", "log", "sqrt", "ramp")
SYMMETRIES: Final = ("isotropic", "fibre-reinforced")
MAX_FIBRE_FAMILIES: Final = 2
FIBRE_INVARIANTS: Final = ("I4", "I5")
_UNITS: Final = {"MPa": Unit.MPA, "1": Unit.ONE, "deg": Unit.DEG}
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
        "exp, log, sqrt, ramp and parentheses"
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


def fibre_angle(reference: str) -> tuple[int, str]:
    """``(sign, parameter)`` of a family's angle reference (``beta``, ``-beta``)."""

    if isinstance(reference, str):
        sign, name = (
            (-1, reference[1:]) if reference.startswith("-") else (1, reference)
        )
        if name.isidentifier() and name.isascii():
            return sign, name
    raise EnergyFormulaError(
        f"a fibre angle is a parameter or its negative, not {reference!r}"
    )


def _label(label: object) -> bool:
    """A family label: 1-8 lowercase ASCII letters or digits, a letter first."""

    return (
        isinstance(label, str)
        and 1 <= len(label) <= 8
        and label.isascii()
        and label.isalnum()
        and label == label.lower()
        and label[0].isalpha()
    )


def fibre_invariant_names(labels: Sequence[str]) -> tuple[str, ...]:
    """``I4<label>`` and ``I5<label>`` of each family."""

    return tuple(f"{kind}{label}" for label in labels for kind in FIBRE_INVARIANTS)


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
    fibres: Mapping[str, str] = field(default_factory=dict)
    bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.symmetry not in SYMMETRIES:
            raise EnergyFormulaError(f"symmetry must be one of {SYMMETRIES}")
        if (self.symmetry == "fibre-reinforced") != bool(self.fibres):
            raise EnergyFormulaError(
                "symmetry is 'fibre-reinforced' exactly when fibres are declared"
            )
        reserved = (*INVARIANTS, *FUNCTIONS, *self.fibre_invariants)
        for name, unit in self.parameters.items():
            if not name.isidentifier() or name in reserved:
                raise EnergyFormulaError(f"parameter name {name!r} is not allowed")
            if unit not in _STRESS and unit is not Unit.DEG:
                raise EnergyFormulaError(f"{name}: the unit must be MPa, 1 or deg")
        unknown = sorted(set(self.positive) - set(self.parameters))
        if unknown:
            raise EnergyFormulaError(f"positive names unknown parameters {unknown}")
        angles = self._check_fibres()
        self._check_bounds()
        powers = {
            name: _STRESS[unit]
            for name, unit in self.parameters.items()
            if unit is not Unit.DEG
        }
        energy_powers = {
            **powers,
            **dict.fromkeys((*INVARIANTS, *self.fibre_invariants), 0),
        }
        if stress_power(self.energy.tree, energy_powers) != 1.0:
            raise EnergyFormulaError("the strain energy must be a stress (MPa)")
        unused = sorted(set(self.parameters) - names(self.energy.tree) - angles)
        if unused:
            raise EnergyFormulaError(f"parameters not used by the energy: {unused}")
        for label, modulus in (
            ("shear_modulus", self.shear_modulus),
            ("bulk_modulus", self.bulk_modulus),
        ):
            if names(modulus.tree) & {*INVARIANTS, *self.fibre_invariants}:
                raise EnergyFormulaError(f"{label} may use parameters only")
            if stress_power(modulus.tree, powers) != 1.0:
                raise EnergyFormulaError(f"{label} must be a stress (MPa)")

    @property
    def fibre_invariants(self) -> tuple[str, ...]:
        """``I4<label>`` and ``I5<label>`` of every declared family."""

        return fibre_invariant_names(tuple(self.fibres))

    def _check_fibres(self) -> set[str]:
        """Validate the families; the angle parameters they use."""

        if len(self.fibres) > MAX_FIBRE_FAMILIES:
            raise EnergyFormulaError(
                f"at most {MAX_FIBRE_FAMILIES} fibre families may be declared"
            )
        used: set[str] = set()
        for label, reference in self.fibres.items():
            if not _label(label):
                raise EnergyFormulaError(
                    f"fibre label {label!r} must be a lowercase name of 1-8 "
                    "letters or digits"
                )
            _, name = fibre_angle(reference)
            if self.parameters.get(name) is not Unit.DEG:
                raise EnergyFormulaError(
                    f"fibre {label}: {name!r} is not a parameter in deg"
                )
            used.add(name)
        angles = {n for n, unit in self.parameters.items() if unit is Unit.DEG}
        if angles - used:
            raise EnergyFormulaError(
                f"deg parameters must be fibre angles: {sorted(angles - used)}"
            )
        return angles

    def _check_bounds(self) -> None:
        for name, (lo, hi) in self.bounds.items():
            if name not in self.parameters or self.parameters[name] is Unit.DEG:
                raise EnergyFormulaError(f"bounds name an unknown parameter {name!r}")
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise EnergyFormulaError(f"{name}: bounds must be finite, lo <= hi")

    def check_values(self, values: Mapping[str, float]) -> None:
        """Refuse missing, extra, non-finite, non-positive or out-of-bounds
        parameter values."""

        if set(values) != set(self.parameters):
            raise EnergyFormulaError(
                f"give exactly the parameters {sorted(self.parameters)}"
            )
        for name, value in values.items():
            if not math.isfinite(value):
                raise EnergyFormulaError(f"{name} must be finite")
            if name in self.positive and value <= 0.0:
                raise EnergyFormulaError(f"{name} must be positive")
            lo, hi = self.bounds.get(name, (-math.inf, math.inf))
            if not lo <= value <= hi:
                raise EnergyFormulaError(f"{name} must lie in [{lo}, {hi}]")

    def document(self, canonical: bool = False) -> dict[str, Any]:
        def text(formula: Formula) -> Any:
            return formula.canonical() if canonical else formula.source

        document: dict[str, Any] = {
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
        # Additive keys (S26): absent when empty, so S25 identities stay.
        if self.fibres:
            document["fibres"] = dict(sorted(self.fibres.items()))
        if self.bounds:
            document["bounds"] = {
                name: [lo, hi] for name, (lo, hi) in sorted(self.bounds.items())
            }
        return document

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
            "fibres",
            "bounds",
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
                raise EnergyFormulaError(
                    f"{name}: the unit must be 'MPa', '1' or 'deg'"
                )
            units[str(name)] = _UNITS[unit]
        positive = document.get("positive", [])
        if not isinstance(positive, Sequence) or isinstance(positive, str):
            raise EnergyFormulaError("positive must be a list of parameter names")
        fibres = document.get("fibres", {})
        if not isinstance(fibres, Mapping):
            raise EnergyFormulaError("fibres must map labels to angle parameters")
        bounds = document.get("bounds", {})
        if not isinstance(bounds, Mapping):
            raise EnergyFormulaError("bounds must map parameters to [lo, hi]")
        pairs: dict[str, tuple[float, float]] = {}
        for name, pair in bounds.items():
            if (
                not isinstance(pair, Sequence)
                or isinstance(pair, str)
                or len(pair) != 2
            ):
                raise EnergyFormulaError(f"bounds of {name!r} must be [lo, hi]")
            pairs[str(name)] = (_number(pair[0]), _number(pair[1]))
        default = "fibre-reinforced" if fibres else "isotropic"
        return cls(
            energy=Formula.parse(document["energy"]),
            parameters=units,
            shear_modulus=Formula.parse(small["shear_modulus"]),
            bulk_modulus=Formula.parse(small["bulk_modulus"]),
            positive=tuple(str(name) for name in positive),
            symmetry=str(document.get("symmetry", default)),
            fibres={str(k): v for k, v in fibres.items()},
            bounds=pairs,
        )


def _builtin(
    energy: str,
    parameters: Mapping[str, str],
    positive: Sequence[str],
    shear: str,
    bulk: str,
    **extra: Any,
) -> EnergyFormula:
    return EnergyFormula.from_document(
        {
            "energy": energy,
            "parameters": dict(parameters),
            "positive": list(positive),
            "small_strain": {"shear_modulus": shear, "bulk_modulus": bulk},
            **extra,
        }
    )


_MATRIX: Final = "shear_modulus / 2 * (I1bar - 3) + bulk_modulus / 2 * (J - 1) ** 2"


def _hgo_term(strain: str) -> str:
    return f" + k1 / (2 * k2) * (exp(k2 * ramp({strain}) ** 2) - 1)"


def _goh_strain(label: str) -> str:
    return f"kappa_d * (I1 - 3) + (1 - 3 * kappa_d) * (I4{label} - 1)"


_HGO_PARAMETERS: Final = {
    "shear_modulus": "MPa",
    "bulk_modulus": "MPa",
    "k1": "MPa",
    "k2": "1",
    "fibre_angle": "deg",
}


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
    # S26 (ADR 0108 decision 2): neo-Hookean matrix and HGO fibres in the full
    # invariant I4, switched off in compression; k1 is left to admission.
    "hgo": _builtin(
        _MATRIX + _hgo_term("I4a - 1") + _hgo_term("I4b - 1"),
        _HGO_PARAMETERS,
        ("shear_modulus", "bulk_modulus", "k2"),
        "shear_modulus",
        "bulk_modulus",
        fibres={"a": "fibre_angle", "b": "-fibre_angle"},
    ),
    "hgo_single": _builtin(
        _MATRIX + _hgo_term("I4a - 1"),
        _HGO_PARAMETERS,
        ("shear_modulus", "bulk_modulus", "k2"),
        "shear_modulus",
        "bulk_modulus",
        fibres={"a": "fibre_angle"},
    ),
    "goh": _builtin(
        _MATRIX + _hgo_term(_goh_strain("a")) + _hgo_term(_goh_strain("b")),
        {**_HGO_PARAMETERS, "kappa_d": "1"},
        ("shear_modulus", "bulk_modulus", "k2"),
        "shear_modulus",
        "bulk_modulus",
        fibres={"a": "fibre_angle", "b": "-fibre_angle"},
        bounds={"kappa_d": [0.0, 1.0 / 3.0]},
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
    "FIBRE_INVARIANTS",
    "Formula",
    "MAX_FIBRE_FAMILIES",
    "SYMMETRIES",
    "fibre_angle",
    "fibre_invariant_names",
    "model_dimensions",
    "names",
    "parse",
    "stress_power",
)
