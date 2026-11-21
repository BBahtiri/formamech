"""One 2D finite-strain problem described as data (S24, ADR 0106).

A :class:`ProblemSpec` is read from a JSON document and says what a family's
geometry, parameter ranges, material, boundary conditions, exact Dirichlet
lifting, quantities of interest, verification tolerances, DOE budget and FEM
settings are. Reading it validates it; every refusal is a
:class:`ProblemSpecError` naming the rule it broke. :meth:`ProblemSpec.check_mesh`
adds the rules that need the reference mesh.

Expressions (values, frame, lifting geometry, constraints) use the safe S2
grammar of :class:`~piginot.contracts.expressions.DerivedExpression`: names,
finite numbers, unary signs, ``+ - * /`` and parentheses. They are parsed into
a syntax tree and evaluated left to right; nothing is passed to ``eval``.

The identity :attr:`ProblemSpec.problem_hash` is a BLAKE2b-256 hash in the
``problem`` domain over the canonical form, in which every expression is its
syntax tree. Key order, whitespace, number spelling, the spelling of an
expression and the file's Unicode encoding therefore do not change it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

from piginot.contracts.canonical import strict_json_loads
from piginot.contracts.energy_formula import (
    BUILTIN_FORMULAS,
    CLOSED_FORM_MODELS,
    EnergyFormula,
    EnergyFormulaError,
    fibre_angle,
    model_dimensions,
)
from piginot.contracts.errors import ContractValidationError
from piginot.contracts.expressions import DerivedExpression, ExpressionTree
from piginot.contracts.hashing import hash_spec
from piginot.contracts.specs import CanonicalMesh
from piginot.contracts.units import Dimension, Unit

SCHEMA: Final = "piginot.problem-spec-v1"
PHYSICS: Final = ("2d-plane-stress-finite-strain",)
EXTENDED_PHYSICS: Final = (
    "2d-plane-stress-finite-strain",
    "2d-plane-strain-finite-strain",
    "2d-axisymmetric-finite-strain",
)
ROLES: Final = ("geometry", "material", "load")
DISTRIBUTIONS: Final = ("uniform", "fixed")
BOUNDARY_KINDS: Final = ("dirichlet", "symmetry", "traction_free", "neumann")
DIRICHLET_KINDS: Final = ("dirichlet", "symmetry")
COMPONENTS: Final = ("x", "y")
LIFTING_KINDS: Final = ("quarter", "adf")
PIECE_KINDS: Final = ("line", "arc")
QOI_FIELDS: Final = ("P_served", "sigma")
QOI_REDUCTIONS: Final = ("max-frobenius",)
VERIFIED_QUANTITIES: Final = ("absolute_peak_error", "p_relative_l2", "u_relative_l2")
SPLIT_ROLES: Final = ("pilot", "train", "development", "holdout", "reserve")
MATERIAL_MODELS: Final[Mapping[str, Mapping[str, Dimension]]] = {
    name: model_dimensions(name) for name in (*CLOSED_FORM_MODELS, *BUILTIN_FORMULAS)
}
"""Built-in models and the dimension of each property (S25: the closed-form
Neo-Hookean model and the built-in formulas)."""
DEFAULT_STRETCH_RANGE: Final = (0.6, 1.6)
GEOMETRY_TOLERANCE: Final = 1e-9
"""Relative tolerance of the geometric mesh checks (to the mesh's extent)."""

_UNITS: Final = {unit.value: unit for unit in Unit if unit.is_canonical}
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_HEX = re.compile(r"[0-9a-f]{64}\Z")


class ProblemSpecError(ContractValidationError):
    """A problem specification broke the validation rule ``rule``."""

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(f"[{rule}] {message}")
        self.rule = rule


# ------------------------------------------------------------- JSON helpers


def _fields(
    value: object,
    label: str,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProblemSpecError("type", f"{label} must be an object")
    unknown = sorted(set(value) - set(required) - set(optional))
    if unknown:
        raise ProblemSpecError(
            "unknown-key", f"{label}: unknown key(s) {', '.join(unknown)}"
        )
    missing = [key for key in required if key not in value]
    if missing:
        raise ProblemSpecError(
            "missing-key", f"{label}: missing key(s) {', '.join(missing)}"
        )
    return dict(value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise ProblemSpecError("type", f"{label} must be non-empty text")
    return value


def _identifier(value: object, label: str) -> str:
    text = _text(value, label)
    if not text.isidentifier():
        raise ProblemSpecError("name", f"{label} {text!r} must be an identifier")
    return text


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProblemSpecError("type", f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ProblemSpecError("type", f"{label} must be finite")
    return number


def _count(value: object, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProblemSpecError("type", f"{label} must be an integer")
    if value < minimum:
        raise ProblemSpecError("range", f"{label} must be at least {minimum}")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProblemSpecError("type", f"{label} must be a list")
    return value


def _unit(value: object, label: str) -> Unit:
    if not isinstance(value, str):
        raise ProblemSpecError("unit", f"{label} needs a unit")
    if value not in _UNITS:
        allowed = ", ".join(sorted(_UNITS))
        raise ProblemSpecError(
            "unit",
            f"{label}: unit {value!r} is not one of the canonical units {allowed}",
        )
    return _UNITS[value]


def _unique(names: Iterable[str], label: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ProblemSpecError("duplicate", f"{label} {name!r} is declared twice")
        seen.add(name)


# -------------------------------------------------------------- expressions


def _tree_names(tree: ExpressionTree) -> set[str]:
    kind = tree[0]
    if kind == "name":
        return {cast(str, tree[1])}
    if kind == "unary":
        return _tree_names(cast(ExpressionTree, tree[2]))
    if kind == "binary":
        return _tree_names(cast(ExpressionTree, tree[2])) | _tree_names(
            cast(ExpressionTree, tree[3])
        )
    return set()


def _constant_value(tree: ExpressionTree) -> float:
    """Fold an expression without names (validation only; contracts do not
    evaluate expressions of parameters, ``piginot.problem.expressions`` does)."""

    kind = tree[0]
    if kind == "number":
        return float(cast(float, tree[1]))
    if kind == "name":
        raise ValueError("a constant expression has no names")
    if kind == "unary":
        operand = _constant_value(cast(ExpressionTree, tree[2]))
        return operand if tree[1] == "+" else -operand
    left = _constant_value(cast(ExpressionTree, tree[2]))
    right = _constant_value(cast(ExpressionTree, tree[3]))
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


def _tree_document(tree: ExpressionTree) -> list[Any]:
    return [
        _tree_document(cast(ExpressionTree, item)) if isinstance(item, tuple) else item
        for item in tree
    ]


@dataclass(frozen=True, slots=True)
class Expression:
    """A parsed expression; equality and identity follow its syntax tree."""

    source: str = field(compare=False)
    tree: ExpressionTree

    @classmethod
    def parse(cls, source: object, label: str) -> Expression:
        if not isinstance(source, str):
            raise ProblemSpecError("type", f"{label} must be an expression string")
        try:
            parsed = DerivedExpression.parse(source)
        except ContractValidationError as exc:
            raise ProblemSpecError("expression", f"{label}: {exc}") from exc
        return cls(source, parsed.canonical_ast)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(_tree_names(self.tree))

    @property
    def constant(self) -> float | None:
        """The value of an expression without names, else ``None``."""

        return None if self.names else _constant_value(self.tree)

    def check(
        self,
        dimensions: Mapping[str, Dimension],
        expected: Dimension,
        label: str,
    ) -> None:
        """Refuse unknown names and a dimension other than ``expected``.

        An expression without names is a constant in the expected unit.
        """

        unknown = sorted(self.names - set(dimensions))
        if unknown:
            raise ProblemSpecError(
                "expression", f"{label}: unknown parameter(s) {', '.join(unknown)}"
            )
        if not self.names:
            return
        try:
            DerivedExpression(source=self.source, canonical_ast=self.tree).validate(
                parameter_dimensions=dimensions, expected_dimension=expected
            )
        except ContractValidationError as exc:
            raise ProblemSpecError("expression", f"{label}: {exc}") from exc

    def canonical(self) -> list[Any]:
        return _tree_document(self.tree)


def _pair(value: object, label: str) -> tuple[Expression, Expression]:
    items = _list(value, label)
    if len(items) != 2:
        raise ProblemSpecError("type", f"{label} must hold two expressions")
    return (
        Expression.parse(items[0], f"{label}[0]"),
        Expression.parse(items[1], f"{label}[1]"),
    )


def _pair_document(pair: tuple[Expression, Expression]) -> list[str]:
    return [pair[0].source, pair[1].source]


def _pair_canonical(pair: tuple[Expression, Expression]) -> list[Any]:
    return [pair[0].canonical(), pair[1].canonical()]


# ----------------------------------------------------------------- sections


@dataclass(frozen=True, slots=True)
class ParameterRange:
    """One request input: its role, range, canonical unit and distribution."""

    name: str
    role: str
    lo: float
    hi: float
    unit: Unit
    distribution: str

    def __post_init__(self) -> None:
        _identifier(self.name, "parameter name")
        if self.role not in ROLES:
            raise ProblemSpecError("role", f"{self.name}: role must be one of {ROLES}")
        if not self.unit.is_canonical:
            raise ProblemSpecError("unit", f"{self.name}: the unit must be canonical")
        if self.lo > self.hi:
            raise ProblemSpecError("range", f"{self.name}: lo exceeds hi")
        if self.distribution not in DISTRIBUTIONS:
            raise ProblemSpecError(
                "distribution",
                f"{self.name}: distribution must be one of {DISTRIBUTIONS}",
            )
        if (self.distribution == "fixed") != (self.lo == self.hi):
            raise ProblemSpecError(
                "distribution",
                f"{self.name}: 'fixed' needs lo = hi and 'uniform' needs lo < hi",
            )

    @property
    def dimension(self) -> Dimension:
        return self.unit.dimension

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "lo": self.lo,
            "hi": self.hi,
            "unit": self.unit.value,
            "distribution": self.distribution,
        }


@dataclass(frozen=True, slots=True)
class Constant:
    """A named value with a unit that expressions may use; not a request input."""

    name: str
    value: float
    unit: Unit

    def __post_init__(self) -> None:
        _identifier(self.name, "constant name")
        if not self.unit.is_canonical:
            raise ProblemSpecError("unit", f"{self.name}: the unit must be canonical")

    def document(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "unit": self.unit.value}


@dataclass(frozen=True, slots=True)
class Constraint:
    """A feasibility bound ``lo <= expression <= hi`` on the parameters."""

    name: str
    expression: Expression
    lo: float
    hi: float
    unit: Unit

    def __post_init__(self) -> None:
        _identifier(self.name, "constraint name")
        if self.lo > self.hi:
            raise ProblemSpecError("range", f"{self.name}: lo exceeds hi")

    def document(self, canonical: bool = False) -> dict[str, Any]:
        expression = (
            self.expression.canonical() if canonical else self.expression.source
        )
        return {
            "name": self.name,
            "expression": expression,
            "lo": self.lo,
            "hi": self.hi,
            "unit": self.unit.value,
        }


@dataclass(frozen=True, slots=True)
class Geometry:
    """The reference mesh and morph map by content, its regions and frame.

    The frame is the domain's bounding box, ``origin + [0, extent]`` per axis;
    the operator scales query points by it.
    """

    reference_mesh_id: str
    morph_map_sha256: str
    regions: tuple[str, ...]
    frame_origin: tuple[Expression, Expression]
    frame_extent: tuple[Expression, Expression]

    def __post_init__(self) -> None:
        for label, value in (
            ("reference_mesh_id", self.reference_mesh_id),
            ("morph_map_sha256", self.morph_map_sha256),
        ):
            if not isinstance(value, str) or _HEX.fullmatch(value) is None:
                raise ProblemSpecError("hash", f"{label} must be 64 lowercase hex")
        if not self.regions:
            raise ProblemSpecError("region", "the geometry declares no region")
        for region in self.regions:
            _text(region, "region")
        _unique(self.regions, "region")

    def document(self, canonical: bool = False) -> dict[str, Any]:
        pair = _pair_canonical if canonical else _pair_document
        return {
            "reference_mesh_id": self.reference_mesh_id,
            "morph_map_sha256": self.morph_map_sha256,
            "regions": list(self.regions),
            "frame": {
                "origin": pair(self.frame_origin),
                "extent": pair(self.frame_extent),
            },
        }


@dataclass(frozen=True, slots=True)
class Material:
    """The constitutive model and the parameter that sets each property.

    ``model`` names a built-in model; a user's strain-energy formula is given
    as ``energy`` and named by its content (``energy-<16 hex>``).
    ``stretch_range`` is the range of principal stretches the admission gate
    checks the model over.
    """

    model: str
    parameters: Mapping[str, str]
    energy: EnergyFormula | None = None
    stretch_range: tuple[float, float] = DEFAULT_STRETCH_RANGE

    def __post_init__(self) -> None:
        if self.energy is not None and self.model != self.energy.model_name:
            raise ProblemSpecError(
                "material", f"a user formula is named {self.energy.model_name!r}"
            )
        if self.energy is None and self.model not in MATERIAL_MODELS:
            raise ProblemSpecError(
                "material",
                f"model {self.model!r} is not supported; "
                f"available: {', '.join(MATERIAL_MODELS)}, or an energy formula",
            )
        expected = set(self.dimensions)
        if set(self.parameters) != expected:
            raise ProblemSpecError(
                "material",
                f"{self.model} needs exactly the properties {sorted(expected)}",
            )
        low, high = self.stretch_range
        if not 0.0 < low < 1.0 < high:
            raise ProblemSpecError(
                "range", "the stretch range must be positive and contain 1"
            )

    @property
    def dimensions(self) -> Mapping[str, Dimension]:
        """The dimension of each property of the model."""

        if self.energy is not None:
            return {
                name: unit.dimension for name, unit in self.energy.parameters.items()
            }
        return MATERIAL_MODELS[self.model]

    def document(self, canonical: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": self.model,
            "parameters": dict(sorted(self.parameters.items())),
        }
        if self.energy is not None:
            result["energy"] = self.energy.document(canonical)
        if self.stretch_range != DEFAULT_STRETCH_RANGE:
            result["stretch_range"] = list(self.stretch_range)
        return result


@dataclass(frozen=True, slots=True)
class Value:
    """A boundary value: an expression with a unit, or a nominal strain.

    A nominal strain ``e`` on component ``c`` prescribes ``u_c = e * x_c`` with
    ``x_c`` the region's coordinate along ``c``; the region must be
    perpendicular to ``c`` (:meth:`ProblemSpec.check_mesh`).
    """

    expression: Expression | None = None
    unit: Unit | None = None
    nominal_strain: str | None = None

    def __post_init__(self) -> None:
        if (self.expression is None) == (self.nominal_strain is None):
            raise ProblemSpecError(
                "values", "a value is either an expression or a nominal strain"
            )
        if (self.expression is None) != (self.unit is None):
            raise ProblemSpecError("unit", "an expression value needs a unit")

    @property
    def is_zero(self) -> bool:
        return self.expression is not None and self.expression.constant == 0.0

    def document(self, canonical: bool = False) -> dict[str, Any]:
        if self.expression is None:
            return {"nominal_strain": self.nominal_strain}
        assert self.unit is not None
        expression = (
            self.expression.canonical() if canonical else self.expression.source
        )
        return {"expression": expression, "unit": self.unit.value}


@dataclass(frozen=True, slots=True)
class BoundaryCondition:
    """A condition on one boundary region.

    ``dirichlet`` prescribes ``values`` (lengths) on ``components``;
    ``symmetry`` prescribes zero on them; ``traction_free`` names a free region;
    ``neumann`` prescribes tractions (stresses), which validate but are not
    supported by the builders yet. ``load_path`` is reserved and must be null.
    """

    name: str
    region: str
    kind: str
    components: tuple[str, ...] = ()
    values: tuple[Value, ...] = ()
    load_path: None = None

    def __post_init__(self) -> None:
        _text(self.name, "boundary condition name")
        _text(self.region, f"{self.name}: region")
        if self.kind not in BOUNDARY_KINDS:
            raise ProblemSpecError(
                "kind", f"{self.name}: kind must be one of {BOUNDARY_KINDS}"
            )
        if self.load_path is not None:
            raise ProblemSpecError(
                "load-path", f"{self.name}: load_path is reserved and must be null"
            )
        if any(c not in COMPONENTS for c in self.components):
            raise ProblemSpecError(
                "components", f"{self.name}: components must be among {COMPONENTS}"
            )
        _unique(self.components, f"{self.name}: component")
        if self.kind == "traction_free":
            if self.components or self.values:
                raise ProblemSpecError(
                    "components", f"{self.name}: a free region constrains nothing"
                )
            return
        if not self.components:
            raise ProblemSpecError(
                "components", f"{self.name}: name the constrained components"
            )
        if self.kind == "symmetry":
            if self.values:
                raise ProblemSpecError(
                    "values", f"{self.name}: symmetry prescribes zero; give no values"
                )
            return
        if len(self.values) != len(self.components):
            raise ProblemSpecError(
                "values", f"{self.name}: give one value per component"
            )
        if self.kind == "neumann" and any(
            v.nominal_strain is not None for v in self.values
        ):
            raise ProblemSpecError(
                "values", f"{self.name}: a traction is an expression in MPa"
            )

    @property
    def is_dirichlet(self) -> bool:
        return self.kind in DIRICHLET_KINDS

    def value(self, component: str) -> Value | None:
        """The prescribed value of ``component``; ``None`` means zero."""

        if self.kind == "symmetry":
            return None
        return self.values[self.components.index(component)]

    def is_zero(self, component: str) -> bool:
        value = self.value(component)
        return value is None or value.is_zero

    def document(self, canonical: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "region": self.region,
            "kind": self.kind,
        }
        if self.components:
            result["components"] = list(self.components)
        if self.values:
            result["values"] = [v.document(canonical) for v in self.values]
        return result


@dataclass(frozen=True, slots=True)
class Piece:
    """A line segment or a minor circular arc, traversed with the domain on
    its left; ``trimmed=False`` uses a line's whole supporting line."""

    kind: str
    start: tuple[Expression, Expression]
    end: tuple[Expression, Expression]
    center: tuple[Expression, Expression] | None = None
    trimmed: bool = True

    def __post_init__(self) -> None:
        if self.kind not in PIECE_KINDS:
            raise ProblemSpecError("lifting", f"a piece is one of {PIECE_KINDS}")
        if (self.kind == "arc") != (self.center is not None):
            raise ProblemSpecError("lifting", "an arc, and only an arc, has a center")
        if self.kind == "arc" and not self.trimmed:
            raise ProblemSpecError("lifting", "an arc is always trimmed")
        if not isinstance(self.trimmed, bool):
            raise ProblemSpecError("type", "trimmed must be true or false")

    @property
    def expressions(self) -> tuple[Expression, ...]:
        return (*self.start, *self.end, *(self.center or ()))

    def document(self, canonical: bool = False) -> dict[str, Any]:
        pair = _pair_canonical if canonical else _pair_document
        body: dict[str, Any] = {"start": pair(self.start), "end": pair(self.end)}
        if self.center is not None:
            body["center"] = pair(self.center)
        result: dict[str, Any] = {self.kind: body}
        if not self.trimmed:
            result["trimmed"] = False
        return result


@dataclass(frozen=True, slots=True)
class Lifting:
    """The exact Dirichlet lifting: ``quarter`` or ``adf`` (ADR 0106).

    ``displacement_scale`` is the length that the operator's correction is
    measured in. ``pieces`` (``adf`` only) gives, for every Dirichlet or
    symmetry condition, the segments and arcs its region consists of.
    """

    kind: str
    displacement_scale: Expression
    pieces: Mapping[str, tuple[Piece, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in LIFTING_KINDS:
            raise ProblemSpecError(
                "lifting", f"lifting kind must be one of {LIFTING_KINDS}"
            )
        if self.kind == "quarter" and self.pieces:
            raise ProblemSpecError("lifting", "the quarter lifting takes no pieces")
        for name, pieces in self.pieces.items():
            if not pieces:
                raise ProblemSpecError("lifting", f"{name}: give at least one piece")

    def document(self, canonical: bool = False) -> dict[str, Any]:
        scale = (
            self.displacement_scale.canonical()
            if canonical
            else self.displacement_scale.source
        )
        result: dict[str, Any] = {"kind": self.kind, "displacement_scale": scale}
        if self.pieces:
            result["pieces"] = {
                name: [piece.document(canonical) for piece in pieces]
                for name, pieces in sorted(self.pieces.items())
            }
        return result


@dataclass(frozen=True, slots=True)
class Qoi:
    """The largest Frobenius norm of a field over a region's vertices."""

    name: str
    field: str
    reduction: str
    region: str
    exclude_regions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.name, "QoI name")
        if self.field not in QOI_FIELDS:
            raise ProblemSpecError(
                "qoi", f"{self.name}: field must be one of {QOI_FIELDS}"
            )
        if self.reduction not in QOI_REDUCTIONS:
            raise ProblemSpecError(
                "qoi", f"{self.name}: reduction must be one of {QOI_REDUCTIONS}"
            )

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "field": self.field,
            "reduction": self.reduction,
            "region": self.region,
            "exclude_regions": list(self.exclude_regions),
        }


@dataclass(frozen=True, slots=True)
class Verification:
    """The slice quantile and the tolerance of each verified quantity."""

    quantile: float
    tolerances: Mapping[str, float]

    def __post_init__(self) -> None:
        if not 0.0 < self.quantile < 1.0:
            raise ProblemSpecError("range", "the quantile must lie in (0, 1)")
        if set(self.tolerances) != set(VERIFIED_QUANTITIES):
            raise ProblemSpecError(
                "verification", f"give tolerances for exactly {VERIFIED_QUANTITIES}"
            )
        if any(value <= 0.0 for value in self.tolerances.values()):
            raise ProblemSpecError("range", "tolerances must be positive")

    def document(self) -> dict[str, Any]:
        return {
            "quantile": self.quantile,
            "tolerances": dict(sorted(self.tolerances.items())),
        }


@dataclass(frozen=True, slots=True)
class Split:
    """The DOE budget per evidence role."""

    counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if set(self.counts) != set(SPLIT_ROLES):
            raise ProblemSpecError("split", f"give counts for exactly {SPLIT_ROLES}")
        for role in ("train", "development", "holdout"):
            if self.counts[role] < 1:
                raise ProblemSpecError("range", f"the {role} budget must be positive")

    def document(self) -> dict[str, Any]:
        return {role: self.counts[role] for role in SPLIT_ROLES}


@dataclass(frozen=True, slots=True)
class FemSettings:
    """The reference solver settings and the FEM material name."""

    material_name: str
    element_order: int
    load_steps: int
    max_dofs: int
    max_newton_iterations: int
    plane_stress_max_iterations: int
    solver_tolerance: float

    def __post_init__(self) -> None:
        _text(self.material_name, "fem.material_name")
        if self.element_order not in (1, 2):
            raise ProblemSpecError("range", "fem.element_order must be 1 or 2")
        if self.solver_tolerance <= 0.0:
            raise ProblemSpecError("range", "fem.solver_tolerance must be positive")

    def document(self) -> dict[str, Any]:
        return {
            "material_name": self.material_name,
            "element_order": self.element_order,
            "load_steps": self.load_steps,
            "max_dofs": self.max_dofs,
            "max_newton_iterations": self.max_newton_iterations,
            "plane_stress_max_iterations": self.plane_stress_max_iterations,
            "solver_tolerance": self.solver_tolerance,
        }


# --------------------------------------------------------------- the spec


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """A validated problem description; see the module docstring."""

    name: str
    physics: str
    parameters: tuple[ParameterRange, ...]
    constraints: tuple[Constraint, ...]
    geometry: Geometry
    material: Material
    boundary_conditions: tuple[BoundaryCondition, ...]
    lifting: Lifting
    qois: tuple[Qoi, ...]
    verification: Verification
    split: Split
    fem: FemSettings
    constants: tuple[Constant, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ProblemSpecError(
                "name", "name must be lowercase letters, digits, '.', '_' or '-'"
            )
        if self.physics not in PHYSICS:
            raise ProblemSpecError("physics", f"physics must be one of {PHYSICS}")
        if not self.parameters:
            raise ProblemSpecError("missing-key", "declare at least one parameter")
        _unique((p.name for p in self.parameters), "parameter")
        _unique(
            (
                *(p.name for p in self.parameters),
                *(c.name for c in self.constants),
                *(c.name for c in self.constraints),
            ),
            "parameter, constant or constraint",
        )
        dimensions = self.dimensions
        for constraint in self.constraints:
            constraint.expression.check(
                dimensions, constraint.unit.dimension, f"constraint {constraint.name}"
            )
        self._check_geometry(dimensions)
        self._check_material()
        self._check_boundary_conditions(dimensions)
        self._check_material_symmetry()
        self._check_lifting(dimensions)
        regions = set(self.geometry.regions)
        _unique((q.name for q in self.qois), "QoI")
        for qoi in self.qois:
            for region in (qoi.region, *qoi.exclude_regions):
                if region not in regions:
                    raise ProblemSpecError(
                        "region", f"QoI {qoi.name}: region {region!r} is not declared"
                    )

    # ------------------------------------------------------------ checks

    @property
    def dimensions(self) -> dict[str, Dimension]:
        """The dimension of every name an expression may use."""

        names = {c.name: c.unit.dimension for c in self.constants}
        names.update({p.name: p.dimension for p in self.parameters})
        return names

    def parameter(self, name: str) -> ParameterRange:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(name)

    def _check_geometry(self, dimensions: Mapping[str, Dimension]) -> None:
        for axis, expression in enumerate(
            (*self.geometry.frame_origin, *self.geometry.frame_extent)
        ):
            expression.check(dimensions, Dimension.LENGTH, f"frame entry {axis}")

    def _check_material(self) -> None:
        required = self.material.dimensions
        for prop, name in sorted(self.material.parameters.items()):
            try:
                parameter = self.parameter(name)
            except KeyError:
                raise ProblemSpecError(
                    "material", f"{prop}: parameter {name!r} is not declared"
                ) from None
            if (
                parameter.role != "material"
                or parameter.dimension is not required[prop]
            ):
                raise ProblemSpecError(
                    "material",
                    f"{prop}: {name!r} must be a material parameter of dimension "
                    f"{required[prop].value}",
                )

    def _check_material_symmetry(self) -> None:
        """ADR 0108 decision 6: a symmetry condition needs a material symmetric
        under the reflection across its plane.

        Both in-plane reflections map a fibre angle to its negative, so every
        family needs a mirror family at the negative angle over the whole
        parameter box: the same angle parameter with the opposite sign, or
        fixed angles that sum to a multiple of 180 degrees (0 and 90 degrees
        mirror themselves).
        """

        formula = self.material.energy or BUILTIN_FORMULAS.get(self.material.model)
        if formula is None or not formula.fibres:
            return
        planes = sorted(
            {
                component
                for condition in self.boundary_conditions
                if condition.kind == "symmetry"
                for component in condition.components
            }
        )
        if not planes:
            return
        families = []
        for label, reference in formula.fibres.items():
            sign, prop = fibre_angle(reference)
            families.append(
                (label, sign, prop, self.parameter(self.material.parameters[prop]))
            )

        def mirrored(sign: int, prop: str, parameter: ParameterRange) -> bool:
            for _, other_sign, other_prop, other in families:
                if other_prop == prop and other_sign == -sign:
                    return True
                fixed = parameter.lo == parameter.hi and other.lo == other.hi
                total = other_sign * other.lo + sign * parameter.lo
                if fixed and abs(math.remainder(total, 180.0)) <= 1e-9:
                    return True
            return False

        for label, sign, prop, parameter in families:
            if not mirrored(sign, prop, parameter):
                raise ProblemSpecError(
                    "material-symmetry",
                    f"the symmetry condition(s) on {', '.join(planes)} need a "
                    "material symmetric about that plane; fibre family "
                    f"{label!r} has no mirror family at the negative angle",
                )

    def _check_boundary_conditions(self, dimensions: Mapping[str, Dimension]) -> None:
        conditions = self.boundary_conditions
        _unique((c.name for c in conditions), "boundary condition")
        regions = set(self.geometry.regions)
        prescribed: dict[tuple[str, str], str] = {}
        for condition in conditions:
            if condition.region not in regions:
                raise ProblemSpecError(
                    "region",
                    f"{condition.name}: region {condition.region!r} is not declared",
                )
            for component, value in zip(
                condition.components,
                condition.values or (None,) * len(condition.components),
                strict=True,
            ):
                key = (condition.region, component)
                kind = "neumann" if condition.kind == "neumann" else "dirichlet"
                previous = prescribed.get(key)
                if previous is not None:
                    rule = "dirichlet-neumann" if previous != kind else "duplicate"
                    raise ProblemSpecError(
                        rule,
                        f"{condition.name}: u_{component} on {condition.region!r} "
                        "is prescribed twice",
                    )
                prescribed[key] = kind
                if value is None:
                    continue
                if value.nominal_strain is not None:
                    name = value.nominal_strain
                    if name not in dimensions:
                        raise ProblemSpecError(
                            "expression",
                            f"{condition.name}: unknown parameter {name!r}",
                        )
                    if dimensions[name] is not Dimension.DIMENSIONLESS:
                        raise ProblemSpecError(
                            "expression",
                            f"{condition.name}: a nominal strain is dimensionless",
                        )
                    continue
                assert value.expression is not None and value.unit is not None
                expected = (
                    Dimension.STRESS
                    if condition.kind == "neumann"
                    else Dimension.LENGTH
                )
                if value.unit.dimension is not expected:
                    raise ProblemSpecError(
                        "unit",
                        f"{condition.name}: the value must be a {expected.value}",
                    )
                value.expression.check(dimensions, expected, condition.name)
        constrained = {
            component
            for (_, component), kind in prescribed.items()
            if kind == "dirichlet"
        }
        if constrained != set(COMPONENTS):
            raise ProblemSpecError(
                "rigid-modes",
                "both displacement components must be fixed somewhere",
            )

    def _check_lifting(self, dimensions: Mapping[str, Dimension]) -> None:
        lifting = self.lifting
        lifting.displacement_scale.check(
            dimensions, Dimension.LENGTH, "lifting.displacement_scale"
        )
        dirichlet = [c for c in self.boundary_conditions if c.is_dirichlet]
        if lifting.kind == "quarter":
            self._check_quarter(dirichlet)
            return
        names = {c.name for c in dirichlet}
        if set(lifting.pieces) != names:
            raise ProblemSpecError(
                "lifting-coverage",
                "adf pieces must be given for exactly the Dirichlet and symmetry "
                f"conditions {sorted(names)}",
            )
        for name, pieces in lifting.pieces.items():
            for piece in pieces:
                for expression in piece.expressions:
                    expression.check(dimensions, Dimension.LENGTH, f"{name} piece")

    def _check_quarter(self, dirichlet: Sequence[BoundaryCondition]) -> None:
        def on(component: str) -> list[BoundaryCondition]:
            return [c for c in dirichlet if component in c.components]

        x_conditions, y_conditions = on("x"), on("y")
        origin = [e.constant for e in self.geometry.frame_origin]
        valid = (
            origin == [0.0, 0.0]
            and len(x_conditions) == 2
            and sum(c.is_zero("x") for c in x_conditions) == 1
            and len(y_conditions) == 1
            and y_conditions[0].is_zero("y")
        )
        if not valid:
            raise ProblemSpecError(
                "lifting-coverage",
                "the quarter lifting needs a frame at the origin, u_x fixed on two "
                "regions (one at zero) and u_y fixed at zero on one region",
            )

    def check_mesh(self, mesh: CanonicalMesh) -> None:
        """The rules that need the reference mesh (ADR 0106 decision 2)."""

        if mesh.mesh_id != self.geometry.reference_mesh_id:
            raise ProblemSpecError(
                "mesh-id", "the mesh is not the spec's reference mesh"
            )
        tags = mesh.semantic_tags
        missing = [r for r in self.geometry.regions if r not in tags]
        if missing:
            raise ProblemSpecError(
                "region", f"region(s) absent from the mesh: {', '.join(missing)}"
            )
        xs = [node[0] for node in mesh.nodes]
        ys = [node[1] for node in mesh.nodes]
        size = max(max(xs) - min(xs), max(ys) - min(ys))
        fixed: dict[str, set[int]] = {component: set() for component in COMPONENTS}
        for condition in self.boundary_conditions:
            if not condition.is_dirichlet:
                continue
            nodes = region_nodes(mesh, condition.region)
            for component in condition.components:
                fixed[component] |= nodes
                value = condition.value(component)
                if value is None or value.nominal_strain is None:
                    continue
                axis = COMPONENTS.index(component)
                spread = [mesh.nodes[i][axis] for i in nodes]
                if max(spread) - min(spread) > GEOMETRY_TOLERANCE * size:
                    raise ProblemSpecError(
                        "nominal-strain-region",
                        f"{condition.name}: region {condition.region!r} is not "
                        f"perpendicular to {component}",
                    )
        if not _rigid_modes_fixed(mesh, fixed, size):
            raise ProblemSpecError(
                "rigid-modes", "the Dirichlet conditions leave a rigid rotation free"
            )

    # --------------------------------------------------------- identity

    def document(self) -> dict[str, Any]:
        """The JSON document, expressions as written."""

        return self._document(canonical=False)

    def canonical_form(self) -> dict[str, Any]:
        """The hashed form: the document with every expression as its tree."""

        return self._document(canonical=True)

    def _document(self, *, canonical: bool) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "name": self.name,
            "physics": self.physics,
            "parameters": [p.document() for p in self.parameters],
            "constants": [c.document() for c in self.constants],
            "constraints": [c.document(canonical) for c in self.constraints],
            "geometry": self.geometry.document(canonical),
            "material": self.material.document(canonical),
            "boundary_conditions": [
                c.document(canonical) for c in self.boundary_conditions
            ],
            "lifting": self.lifting.document(canonical),
            "qois": [q.document() for q in self.qois],
            "verification": self.verification.document(),
            "split": self.split.document(),
            "fem": self.fem.document(),
        }

    @property
    def problem_hash(self) -> str:
        """BLAKE2b-256 of the canonical form in the ``problem`` domain."""

        return hash_spec(self.canonical_form(), "problem")

    def values(self, inputs: Mapping[str, object]) -> dict[str, float]:
        """The constants and every parameter's float value from request inputs.

        Inputs may be numbers or objects with a ``value`` attribute
        (``CanonicalQuantity``); other inputs are ignored.
        """

        result = {constant.name: constant.value for constant in self.constants}
        for parameter in self.parameters:
            raw = inputs[parameter.name]
            value = getattr(raw, "value", raw)
            result[parameter.name] = float(cast(float, value))
        return result

    def scope(self, values: Mapping[str, float]) -> dict[str, float]:
        """``values`` with the constants added: what expressions evaluate in."""

        result = {constant.name: constant.value for constant in self.constants}
        result.update(values)
        return result

    # ------------------------------------------------------------ reading

    @classmethod
    def from_document(cls, document: object) -> ProblemSpec:
        body = _fields(
            document,
            "problem spec",
            (
                "schema",
                "name",
                "physics",
                "parameters",
                "geometry",
                "material",
                "boundary_conditions",
                "lifting",
                "qois",
                "verification",
                "split",
                "fem",
            ),
            ("constants", "constraints"),
        )
        if body["schema"] != SCHEMA:
            raise ProblemSpecError("schema", f"schema must be {SCHEMA!r}")
        return cls(
            name=_text(body["name"], "name"),
            physics=_text(body["physics"], "physics"),
            parameters=tuple(
                _parameter(item, f"parameters[{i}]")
                for i, item in enumerate(_list(body["parameters"], "parameters"))
            ),
            constants=tuple(
                _constant(item, f"constants[{i}]")
                for i, item in enumerate(_list(body.get("constants", []), "constants"))
            ),
            constraints=tuple(
                _constraint(item, f"constraints[{i}]")
                for i, item in enumerate(
                    _list(body.get("constraints", []), "constraints")
                )
            ),
            geometry=_geometry(body["geometry"]),
            material=_material(body["material"]),
            boundary_conditions=tuple(
                _boundary_condition(item, f"boundary_conditions[{i}]")
                for i, item in enumerate(
                    _list(body["boundary_conditions"], "boundary_conditions")
                )
            ),
            lifting=_lifting(body["lifting"]),
            qois=tuple(
                _qoi(item, f"qois[{i}]")
                for i, item in enumerate(_list(body["qois"], "qois"))
            ),
            verification=_verification(body["verification"]),
            split=_split(body["split"]),
            fem=_fem(body["fem"]),
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> ProblemSpec:
        """Read a spec from a JSON file's bytes (UTF-8, -16 or -32)."""

        try:
            document = strict_json_loads(payload)
        except ValueError as exc:
            raise ProblemSpecError(
                "json", f"not a strict JSON document: {exc}"
            ) from exc
        return cls.from_document(document)

    @classmethod
    def load(cls, path: Path | str) -> ProblemSpec:
        return cls.from_bytes(Path(path).read_bytes())


def region_nodes(mesh: CanonicalMesh, region: str) -> set[int]:
    """The node indices of a region's boundary facets."""

    return {
        index
        for facet in mesh.semantic_tags[region]
        for index in mesh.boundary_facets[facet]
    }


def _rigid_modes_fixed(
    mesh: CanonicalMesh, fixed: Mapping[str, set[int]], size: float
) -> bool:
    """Whether rows ``[1, 0, -y]`` (u_x fixed) and ``[0, 1, x]`` (u_y fixed)
    have rank 3.

    With at least one row of each kind, the rank is 3 exactly when the Schur
    complement on the rotation, ``n_x Var(y | u_x fixed) + n_y Var(x | u_y
    fixed)``, is positive.
    """

    def variance(values: list[float]) -> float:
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    x_nodes, y_nodes = fixed["x"], fixed["y"]
    if not x_nodes or not y_nodes:
        return False
    complement = len(x_nodes) * variance(
        [mesh.nodes[i][1] / size for i in x_nodes]
    ) + len(y_nodes) * variance([mesh.nodes[i][0] / size for i in y_nodes])
    return complement > GEOMETRY_TOLERANCE


# ------------------------------------------------------------ section readers


def _parameter(value: object, label: str) -> ParameterRange:
    body = _fields(value, label, ("name", "role", "lo", "hi", "unit", "distribution"))
    return ParameterRange(
        name=_identifier(body["name"], f"{label}.name"),
        role=_text(body["role"], f"{label}.role"),
        lo=_number(body["lo"], f"{label}.lo"),
        hi=_number(body["hi"], f"{label}.hi"),
        unit=_unit(body["unit"], label),
        distribution=_text(body["distribution"], f"{label}.distribution"),
    )


def _constant(value: object, label: str) -> Constant:
    body = _fields(value, label, ("name", "value", "unit"))
    return Constant(
        name=_identifier(body["name"], f"{label}.name"),
        value=_number(body["value"], f"{label}.value"),
        unit=_unit(body["unit"], label),
    )


def _constraint(value: object, label: str) -> Constraint:
    body = _fields(value, label, ("name", "expression", "lo", "hi", "unit"))
    return Constraint(
        name=_identifier(body["name"], f"{label}.name"),
        expression=Expression.parse(body["expression"], f"{label}.expression"),
        lo=_number(body["lo"], f"{label}.lo"),
        hi=_number(body["hi"], f"{label}.hi"),
        unit=_unit(body["unit"], label),
    )


def _geometry(value: object) -> Geometry:
    body = _fields(
        value,
        "geometry",
        ("reference_mesh_id", "morph_map_sha256", "regions", "frame"),
    )
    frame = _fields(body["frame"], "geometry.frame", ("origin", "extent"))
    return Geometry(
        reference_mesh_id=_text(body["reference_mesh_id"], "reference_mesh_id"),
        morph_map_sha256=_text(body["morph_map_sha256"], "morph_map_sha256"),
        regions=tuple(
            _text(item, "region") for item in _list(body["regions"], "regions")
        ),
        frame_origin=_pair(frame["origin"], "geometry.frame.origin"),
        frame_extent=_pair(frame["extent"], "geometry.frame.extent"),
    )


def _material(value: object) -> Material:
    body = _fields(
        value, "material", ("parameters",), ("model", "energy", "stretch_range")
    )
    parameters = body["parameters"]
    if not isinstance(parameters, Mapping):
        raise ProblemSpecError("type", "material.parameters must be an object")
    energy = None
    if "energy" in body:
        try:
            energy = EnergyFormula.from_document(body["energy"])
        except EnergyFormulaError as exc:
            raise ProblemSpecError("energy", f"material.energy: {exc}") from exc
    if "model" in body:
        model = _text(body["model"], "material.model")
    elif energy is not None:
        model = energy.model_name
    else:
        raise ProblemSpecError("missing-key", "material: give a model or an energy")
    stretch = body.get("stretch_range", list(DEFAULT_STRETCH_RANGE))
    items = _list(stretch, "material.stretch_range")
    if len(items) != 2:
        raise ProblemSpecError("type", "material.stretch_range holds two numbers")
    return Material(
        model=model,
        parameters={
            _text(k, "material property"): _identifier(v, f"material.{k}")
            for k, v in parameters.items()
        },
        energy=energy,
        stretch_range=(
            _number(items[0], "material.stretch_range"),
            _number(items[1], "material.stretch_range"),
        ),
    )


def _value(value: object, label: str) -> Value:
    if isinstance(value, Mapping) and "nominal_strain" in value:
        body = _fields(value, label, ("nominal_strain",))
        return Value(nominal_strain=_identifier(body["nominal_strain"], label))
    body = _fields(value, label, ("expression", "unit"))
    return Value(
        expression=Expression.parse(body["expression"], f"{label}.expression"),
        unit=_unit(body["unit"], label),
    )


def _boundary_condition(value: object, label: str) -> BoundaryCondition:
    body = _fields(
        value,
        label,
        ("name", "region", "kind"),
        ("components", "values", "load_path"),
    )
    return BoundaryCondition(
        name=_text(body["name"], f"{label}.name"),
        region=_text(body["region"], f"{label}.region"),
        kind=_text(body["kind"], f"{label}.kind"),
        components=tuple(
            _text(item, f"{label}.components")
            for item in _list(body.get("components", []), f"{label}.components")
        ),
        values=tuple(
            _value(item, f"{label}.values[{i}]")
            for i, item in enumerate(_list(body.get("values", []), f"{label}.values"))
        ),
        load_path=body.get("load_path"),
    )


def _piece(value: object, label: str) -> Piece:
    if not isinstance(value, Mapping):
        raise ProblemSpecError("type", f"{label} must be an object")
    kinds = [kind for kind in PIECE_KINDS if kind in value]
    if len(kinds) != 1:
        raise ProblemSpecError(
            "lifting", f"{label} must be exactly one of {PIECE_KINDS}"
        )
    kind = kinds[0]
    outer = _fields(value, label, (kind,), ("trimmed",))
    trimmed = outer.get("trimmed", True)
    if not isinstance(trimmed, bool):
        raise ProblemSpecError("type", f"{label}.trimmed must be true or false")
    if kind == "line":
        body = _fields(outer[kind], f"{label}.line", ("start", "end"))
        return Piece(
            kind,
            _pair(body["start"], f"{label}.start"),
            _pair(body["end"], f"{label}.end"),
            trimmed=trimmed,
        )
    body = _fields(outer[kind], f"{label}.arc", ("start", "end", "center"))
    return Piece(
        kind,
        _pair(body["start"], f"{label}.start"),
        _pair(body["end"], f"{label}.end"),
        _pair(body["center"], f"{label}.center"),
        trimmed=trimmed,
    )


def _lifting(value: object) -> Lifting:
    body = _fields(value, "lifting", ("kind", "displacement_scale"), ("pieces",))
    pieces = body.get("pieces", {})
    if not isinstance(pieces, Mapping):
        raise ProblemSpecError("type", "lifting.pieces must be an object")
    return Lifting(
        kind=_text(body["kind"], "lifting.kind"),
        displacement_scale=Expression.parse(
            body["displacement_scale"], "lifting.displacement_scale"
        ),
        pieces={
            _text(name, "lifting.pieces"): tuple(
                _piece(item, f"lifting.pieces.{name}[{i}]")
                for i, item in enumerate(_list(items, f"lifting.pieces.{name}"))
            )
            for name, items in pieces.items()
        },
    )


def _qoi(value: object, label: str) -> Qoi:
    body = _fields(
        value, label, ("name", "field", "reduction", "region"), ("exclude_regions",)
    )
    return Qoi(
        name=_identifier(body["name"], f"{label}.name"),
        field=_text(body["field"], f"{label}.field"),
        reduction=_text(body["reduction"], f"{label}.reduction"),
        region=_text(body["region"], f"{label}.region"),
        exclude_regions=tuple(
            _text(item, f"{label}.exclude_regions")
            for item in _list(
                body.get("exclude_regions", []), f"{label}.exclude_regions"
            )
        ),
    )


def _verification(value: object) -> Verification:
    body = _fields(value, "verification", ("quantile", "tolerances"))
    tolerances = body["tolerances"]
    if not isinstance(tolerances, Mapping):
        raise ProblemSpecError("type", "verification.tolerances must be an object")
    return Verification(
        quantile=_number(body["quantile"], "verification.quantile"),
        tolerances={
            _text(k, "tolerance"): _number(v, f"tolerance {k}")
            for k, v in tolerances.items()
        },
    )


def _split(value: object) -> Split:
    body = _fields(value, "split", SPLIT_ROLES)
    return Split({role: _count(body[role], f"split.{role}", 0) for role in SPLIT_ROLES})


def _fem(value: object) -> FemSettings:
    body = _fields(
        value,
        "fem",
        (
            "material_name",
            "element_order",
            "load_steps",
            "max_dofs",
            "max_newton_iterations",
            "plane_stress_max_iterations",
            "solver_tolerance",
        ),
    )
    return FemSettings(
        material_name=_text(body["material_name"], "fem.material_name"),
        element_order=_count(body["element_order"], "fem.element_order", 1),
        load_steps=_count(body["load_steps"], "fem.load_steps", 1),
        max_dofs=_count(body["max_dofs"], "fem.max_dofs", 1),
        max_newton_iterations=_count(
            body["max_newton_iterations"], "fem.max_newton_iterations", 1
        ),
        plane_stress_max_iterations=_count(
            body["plane_stress_max_iterations"], "fem.plane_stress_max_iterations", 1
        ),
        solver_tolerance=_number(body["solver_tolerance"], "fem.solver_tolerance"),
    )


__all__ = (
    "BOUNDARY_KINDS",
    "COMPONENTS",
    "LIFTING_KINDS",
    "MATERIAL_MODELS",
    "SCHEMA",
    "BoundaryCondition",
    "Constant",
    "Constraint",
    "Expression",
    "FemSettings",
    "Geometry",
    "Lifting",
    "Material",
    "ParameterRange",
    "Piece",
    "ProblemSpec",
    "ProblemSpecError",
    "Qoi",
    "Split",
    "Value",
    "Verification",
    "region_nodes",
)
