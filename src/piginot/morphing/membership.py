"""Executable, serialisable family-membership predicates for S6."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, cast

from piginot.contracts import (
    CanonicalMesh,
    CanonicalQuantity,
    DerivedExpression,
    Dimension,
    GeometryFamilySpec,
    Unit,
    canonical_json_bytes,
    hash_spec,
    strict_json_loads,
)
from piginot.contracts.expressions import ExpressionTree
from piginot.morphing.models import (
    MembershipDecision,
    MorphingError,
    MorphReasonCode,
    MorphValidityCriteria,
)

_SCHEMA_VERSION = "1.0"
_CANONICAL_NAMESPACE = "piginot.morphing"
_QUANTITY_TYPE = "piginot.contracts.CanonicalQuantity"
_CRITERIA_TYPE = "piginot.morphing.models.MorphValidityCriteria"


def _invalid_serialized(message: str) -> MorphingError:
    return MorphingError(MorphReasonCode.INVALID_SERIALIZED_PREDICATE, message)


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _invalid_serialized(f"{label} must be a JSON object")
    keys = set(value)
    if any(not isinstance(key, str) for key in keys):
        raise _invalid_serialized(f"{label} object names must be strings")
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing:
        raise _invalid_serialized(f"{label} is missing field(s): {', '.join(missing)}")
    if unknown:
        raise _invalid_serialized(
            f"{label} has unknown field(s): {', '.join(cast(list[str], unknown))}"
        )
    return cast(Mapping[str, object], value)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid_serialized(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise _invalid_serialized(f"{label} must be a finite number")
    return number


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid_serialized(f"{label} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class PrimitiveBound:
    """One compiled primitive-parameter interval in canonical units."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    name: str
    lower_bound: CanonicalQuantity
    upper_bound: CanonicalQuantity

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("primitive bound name must be a valid identifier")
        if not isinstance(self.lower_bound, CanonicalQuantity) or not isinstance(
            self.upper_bound, CanonicalQuantity
        ):
            raise ValueError("primitive bounds must be CanonicalQuantity values")
        if (
            self.lower_bound.unit is not self.upper_bound.unit
            or self.lower_bound.dimension is not self.upper_bound.dimension
        ):
            raise ValueError(
                "primitive bounds must have identical units and dimensions"
            )
        if self.lower_bound.value > self.upper_bound.value:
            raise ValueError("primitive lower bound must not exceed upper bound")

    @property
    def unit(self) -> Unit:
        return self.lower_bound.unit

    @property
    def dimension(self) -> Dimension:
        dimension = self.lower_bound.dimension
        if not isinstance(dimension, Dimension):  # pragma: no cover - S2 invariant
            raise ValueError("primitive bound dimension is unresolved")
        return dimension


@dataclass(frozen=True, slots=True)
class DerivedBound:
    """One compiled derived-expression interval in canonical units."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    name: str
    canonical_ast: ExpressionTree
    lower_bound: CanonicalQuantity
    upper_bound: CanonicalQuantity

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("derived bound name must be a valid identifier")
        if not isinstance(self.lower_bound, CanonicalQuantity) or not isinstance(
            self.upper_bound, CanonicalQuantity
        ):
            raise ValueError("derived bounds must be CanonicalQuantity values")
        if (
            self.lower_bound.unit is not self.upper_bound.unit
            or self.lower_bound.dimension is not self.upper_bound.dimension
        ):
            raise ValueError("derived bounds must have identical units and dimensions")
        if self.lower_bound.value > self.upper_bound.value:
            raise ValueError("derived lower bound must not exceed upper bound")

    @property
    def unit(self) -> Unit:
        return self.lower_bound.unit

    @property
    def dimension(self) -> Dimension:
        dimension = self.lower_bound.dimension
        if not isinstance(dimension, Dimension):  # pragma: no cover - S2 invariant
            raise ValueError("derived bound dimension is unresolved")
        return dimension


def _canonical_tree(value: object, declared_names: frozenset[str]) -> ExpressionTree:
    """Validate and freeze an S2 canonical expression tree."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _invalid_serialized("canonical expression nodes must be arrays")
    node = tuple(value)
    if not node or not isinstance(node[0], str):
        raise _invalid_serialized("canonical expression node kind must be text")
    kind = node[0]
    if kind == "number":
        if len(node) != 2:
            raise _invalid_serialized("number nodes must contain exactly two items")
        number = _finite_number(node[1], "expression number")
        normalized: int | float = int(number) if number.is_integer() else number
        return ("number", normalized)
    if kind == "name":
        if len(node) != 2 or not isinstance(node[1], str):
            raise _invalid_serialized("name nodes must contain one text symbol")
        if node[1] not in declared_names:
            raise _invalid_serialized(
                f"expression references undeclared primitive {node[1]!r}"
            )
        return ("name", node[1])
    if kind == "unary":
        if len(node) != 3 or node[1] not in {"+", "-"}:
            raise _invalid_serialized("unary nodes require '+' or '-'")
        return ("unary", node[1], _canonical_tree(node[2], declared_names))
    if kind == "binary":
        if len(node) != 4 or node[1] not in {"+", "-", "*", "/"}:
            raise _invalid_serialized("binary nodes require '+', '-', '*', or '/'")
        return (
            "binary",
            node[1],
            _canonical_tree(node[2], declared_names),
            _canonical_tree(node[3], declared_names),
        )
    raise _invalid_serialized(f"unknown canonical expression node kind {kind!r}")


def _evaluate_tree(tree: ExpressionTree, values: Mapping[str, float]) -> float:
    kind = tree[0]
    if kind == "number":
        result = float(cast(int | float, tree[1]))
    elif kind == "name":
        result = values[cast(str, tree[1])]
    elif kind == "unary":
        operand = _evaluate_tree(cast(ExpressionTree, tree[2]), values)
        result = operand if tree[1] == "+" else -operand
    elif kind == "binary":
        left = _evaluate_tree(cast(ExpressionTree, tree[2]), values)
        right = _evaluate_tree(cast(ExpressionTree, tree[3]), values)
        operator = tree[1]
        if operator == "+":
            result = left + right
        elif operator == "-":
            result = left - right
        elif operator == "*":
            result = left * right
        elif operator == "/":
            if right == 0.0:
                raise ZeroDivisionError("derived expression divided by zero")
            result = left / right
        else:  # pragma: no cover - trees are validated at construction
            raise ValueError("invalid canonical expression operator")
    else:  # pragma: no cover - trees are validated at construction
        raise ValueError("invalid canonical expression node")
    if not math.isfinite(result):
        raise ArithmeticError("derived expression produced a non-finite value")
    return result


def _quantity_from_payload(value: object, label: str) -> CanonicalQuantity:
    payload = _require_exact_keys(
        value,
        frozenset({"$type", "value", "unit", "dimension"}),
        label,
    )
    if payload["$type"] != _QUANTITY_TYPE:
        raise _invalid_serialized(f"{label} has an invalid type discriminator")
    try:
        unit = Unit(_text(payload["unit"], f"{label}.unit"))
        dimension = Dimension(_text(payload["dimension"], f"{label}.dimension"))
        return CanonicalQuantity(
            _finite_number(payload["value"], f"{label}.value"),
            unit,
            dimension,
        )
    except ValueError as exc:
        if isinstance(exc, MorphingError):
            raise
        raise _invalid_serialized(f"invalid {label}: {exc}") from exc


def _bound_from_payload(
    value: object, *, derived: bool
) -> PrimitiveBound | DerivedBound:
    label = "derived bound" if derived else "primitive bound"
    expected = {"$type", "name", "lower_bound", "upper_bound"}
    if derived:
        expected.add("canonical_ast")
    payload = _require_exact_keys(value, frozenset(expected), label)
    expected_type = (
        f"{_CANONICAL_NAMESPACE}.{'DerivedBound' if derived else 'PrimitiveBound'}"
    )
    if payload["$type"] != expected_type:
        raise _invalid_serialized(f"{label} has an invalid type discriminator")
    name = _text(payload["name"], f"{label}.name")
    lower = _quantity_from_payload(payload["lower_bound"], f"{label}.lower_bound")
    upper = _quantity_from_payload(payload["upper_bound"], f"{label}.upper_bound")
    if derived:
        return DerivedBound(
            name=name,
            canonical_ast=cast(ExpressionTree, payload["canonical_ast"]),
            lower_bound=lower,
            upper_bound=upper,
        )
    return PrimitiveBound(name=name, lower_bound=lower, upper_bound=upper)


def _criteria_from_payload(value: object) -> MorphValidityCriteria:
    fields = frozenset(
        {
            "$type",
            "jacobian_margin",
            "minimum_feature_size_mm",
            "minimum_angle_degrees",
            "maximum_aspect_ratio",
        }
    )
    payload = _require_exact_keys(value, fields, "validity criteria")
    if payload["$type"] != _CRITERIA_TYPE:
        raise _invalid_serialized("validity criteria has an invalid type discriminator")
    try:
        return MorphValidityCriteria(
            jacobian_margin=_finite_number(
                payload["jacobian_margin"], "validity criteria.jacobian_margin"
            ),
            minimum_feature_size_mm=_finite_number(
                payload["minimum_feature_size_mm"],
                "validity criteria.minimum_feature_size_mm",
            ),
            minimum_angle_degrees=_finite_number(
                payload["minimum_angle_degrees"],
                "validity criteria.minimum_angle_degrees",
            ),
            maximum_aspect_ratio=_finite_number(
                payload["maximum_aspect_ratio"],
                "validity criteria.maximum_aspect_ratio",
            ),
        )
    except ValueError as exc:
        if isinstance(exc, MorphingError):
            raise _invalid_serialized(str(exc)) from exc
        raise


@dataclass(frozen=True, slots=True)
class FamilyMembershipPredicate:
    """Compiled primitive and coupled bounds bound to one reference mesh."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    family_name: str
    primitive_bounds: tuple[PrimitiveBound, ...]
    derived_bounds: tuple[DerivedBound, ...]
    reference_mesh_id: str
    reference_topology_signature: str
    semantic_tag_counts: Mapping[str, int]
    criteria: MorphValidityCriteria
    validity_policy_hash: str
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported membership schema {self.schema_version!r}")
        if not isinstance(self.family_name, str) or not self.family_name:
            raise ValueError("family_name must be non-empty text")
        primitive = tuple(self.primitive_bounds)
        derived = tuple(self.derived_bounds)
        if any(not isinstance(bound, PrimitiveBound) for bound in primitive):
            raise ValueError("primitive_bounds must contain PrimitiveBound values")
        if any(not isinstance(bound, DerivedBound) for bound in derived):
            raise ValueError("derived_bounds must contain DerivedBound values")
        names = tuple(bound.name for bound in primitive) + tuple(
            bound.name for bound in derived
        )
        if len(names) != len(set(names)):
            raise ValueError("compiled membership bound names must be unique")
        if not isinstance(self.reference_mesh_id, str) or not self.reference_mesh_id:
            raise ValueError("reference_mesh_id must be non-empty text")
        if (
            not isinstance(self.reference_topology_signature, str)
            or not self.reference_topology_signature
        ):
            raise ValueError("reference_topology_signature must be non-empty text")
        if not isinstance(self.semantic_tag_counts, Mapping):
            raise ValueError("semantic_tag_counts must be a mapping")
        tag_counts = dict(sorted(self.semantic_tag_counts.items()))
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for name, count in tag_counts.items()
        ):
            raise ValueError(
                "semantic tag counts require names and non-negative integers"
            )
        if not isinstance(self.criteria, MorphValidityCriteria):
            raise ValueError("criteria must be MorphValidityCriteria")
        expected_hash = hash_spec(self.criteria, "content")
        if self.validity_policy_hash != expected_hash:
            raise ValueError("validity_policy_hash does not match criteria")

        declared_names = frozenset(bound.name for bound in primitive)
        dimensions = {bound.name: bound.dimension for bound in primitive}
        checked_derived: list[DerivedBound] = []
        for bound in derived:
            tree = _canonical_tree(bound.canonical_ast, declared_names)
            DerivedExpression(source="", canonical_ast=tree).validate(
                parameter_dimensions=dimensions,
                expected_dimension=bound.dimension,
            )
            checked_derived.append(
                DerivedBound(
                    name=bound.name,
                    canonical_ast=tree,
                    lower_bound=bound.lower_bound,
                    upper_bound=bound.upper_bound,
                )
            )

        object.__setattr__(self, "primitive_bounds", primitive)
        object.__setattr__(self, "derived_bounds", tuple(checked_derived))
        object.__setattr__(
            self,
            "semantic_tag_counts",
            MappingProxyType(tag_counts),
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the exact primitive parameter vocabulary in declaration order."""

        return tuple(bound.name for bound in self.primitive_bounds)

    @property
    def validity(self) -> MorphValidityCriteria:
        """Compatibility name used by the morph-validation orchestrator."""

        return self.criteria

    @classmethod
    def compile(
        cls,
        family: GeometryFamilySpec,
        reference: CanonicalMesh,
        criteria: MorphValidityCriteria,
    ) -> FamilyMembershipPredicate:
        if not isinstance(family, GeometryFamilySpec):
            raise TypeError("family must be GeometryFamilySpec")
        if not isinstance(reference, CanonicalMesh):
            raise TypeError("reference must be CanonicalMesh")
        if not isinstance(criteria, MorphValidityCriteria):
            raise TypeError("criteria must be MorphValidityCriteria")
        if (
            family.topology_signature is not None
            and family.topology_signature != reference.topology_signature
        ):
            raise MorphingError(
                MorphReasonCode.VALIDATION_CONTEXT_MISMATCH,
                "family topology does not match the reference mesh",
            )
        reference_tags = tuple(sorted(reference.semantic_tags))
        if (
            family.semantic_tags
            and tuple(sorted(family.semantic_tags)) != reference_tags
        ):
            raise MorphingError(
                MorphReasonCode.VALIDATION_CONTEXT_MISMATCH,
                "family semantic tags do not match the reference mesh",
            )

        primitive = tuple(
            PrimitiveBound(
                name=parameter.name,
                lower_bound=parameter.lower_bound,
                upper_bound=parameter.upper_bound,
            )
            for parameter in family.parameters
        )
        derived = tuple(
            DerivedBound(
                name=quantity.name,
                canonical_ast=quantity.expression.canonical_ast,
                lower_bound=quantity.lower_bound,
                upper_bound=quantity.upper_bound,
            )
            for quantity in family.derived_quantities
        )
        return cls(
            family_name=family.name,
            primitive_bounds=primitive,
            derived_bounds=derived,
            reference_mesh_id=reference.mesh_id,
            reference_topology_signature=reference.topology_signature,
            semantic_tag_counts={
                name: len(facet_ids)
                for name, facet_ids in sorted(reference.semantic_tags.items())
            },
            criteria=criteria,
            validity_policy_hash=hash_spec(criteria, "content"),
        )

    def evaluate(self, theta: Mapping[str, object]) -> MembershipDecision:
        if not isinstance(theta, Mapping):
            return MembershipDecision(
                accepted=False,
                reason_codes=(MorphReasonCode.INVALID_PARAMETER_VALUE,),
            )
        supplied_names = tuple(theta)
        if any(not isinstance(name, str) for name in supplied_names):
            return MembershipDecision(
                accepted=False,
                reason_codes=(MorphReasonCode.INVALID_PARAMETER_VALUE,),
            )
        expected_names = frozenset(bound.name for bound in self.primitive_bounds)
        supplied_set = frozenset(supplied_names)
        if expected_names - supplied_set:
            return MembershipDecision(
                accepted=False,
                reason_codes=(MorphReasonCode.MISSING_PARAMETER,),
            )
        if supplied_set - expected_names:
            return MembershipDecision(
                accepted=False,
                reason_codes=(MorphReasonCode.UNKNOWN_PARAMETER,),
            )

        values: dict[str, float] = {}
        for primitive_bound in self.primitive_bounds:
            value = theta[primitive_bound.name]
            if not isinstance(value, CanonicalQuantity) or not math.isfinite(
                value.value
            ):
                return MembershipDecision(
                    accepted=False,
                    reason_codes=(MorphReasonCode.INVALID_PARAMETER_VALUE,),
                )
            if (
                value.unit is not primitive_bound.unit
                or value.dimension is not primitive_bound.dimension
            ):
                return MembershipDecision(
                    accepted=False,
                    reason_codes=(MorphReasonCode.PARAMETER_UNIT_MISMATCH,),
                )
            if not (
                primitive_bound.lower_bound.value
                <= value.value
                <= primitive_bound.upper_bound.value
            ):
                return MembershipDecision(
                    accepted=False,
                    reason_codes=(MorphReasonCode.PARAMETER_OUT_OF_BOUNDS,),
                )
            values[primitive_bound.name] = value.value

        derived_values: dict[str, CanonicalQuantity] = {}
        for derived_bound in self.derived_bounds:
            try:
                result = _evaluate_tree(derived_bound.canonical_ast, values)
                quantity = CanonicalQuantity(
                    result, derived_bound.unit, derived_bound.dimension
                )
            except (ArithmeticError, KeyError, OverflowError, ValueError):
                return MembershipDecision(
                    accepted=False,
                    reason_codes=(MorphReasonCode.DERIVED_EVALUATION_ERROR,),
                    derived_values=derived_values,
                )
            derived_values[derived_bound.name] = quantity
            if not (
                derived_bound.lower_bound.value
                <= result
                <= derived_bound.upper_bound.value
            ):
                return MembershipDecision(
                    accepted=False,
                    reason_codes=(MorphReasonCode.DERIVED_QUANTITY_OUT_OF_BOUNDS,),
                    derived_values=derived_values,
                )
        return MembershipDecision(
            accepted=True,
            derived_values=derived_values,
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    @classmethod
    def from_bytes(cls, document: str | bytes) -> FamilyMembershipPredicate:
        root = _require_exact_keys(
            strict_json_loads(document),
            frozenset(
                {
                    "$type",
                    "family_name",
                    "primitive_bounds",
                    "derived_bounds",
                    "reference_mesh_id",
                    "reference_topology_signature",
                    "semantic_tag_counts",
                    "criteria",
                    "validity_policy_hash",
                    "schema_version",
                }
            ),
            "membership predicate",
        )
        if root["$type"] != f"{_CANONICAL_NAMESPACE}.FamilyMembershipPredicate":
            raise _invalid_serialized(
                "membership predicate has an invalid type discriminator"
            )
        if root["schema_version"] != _SCHEMA_VERSION:
            raise _invalid_serialized(
                f"unsupported membership schema {root['schema_version']!r}"
            )

        primitive_payload = root["primitive_bounds"]
        derived_payload = root["derived_bounds"]
        if not isinstance(primitive_payload, list):
            raise _invalid_serialized("primitive_bounds must be an array")
        if not isinstance(derived_payload, list):
            raise _invalid_serialized("derived_bounds must be an array")
        primitive = tuple(
            cast(PrimitiveBound, _bound_from_payload(item, derived=False))
            for item in primitive_payload
        )
        declared_names = frozenset(bound.name for bound in primitive)
        derived: list[DerivedBound] = []
        for item in derived_payload:
            bound = cast(DerivedBound, _bound_from_payload(item, derived=True))
            derived.append(
                DerivedBound(
                    name=bound.name,
                    canonical_ast=_canonical_tree(bound.canonical_ast, declared_names),
                    lower_bound=bound.lower_bound,
                    upper_bound=bound.upper_bound,
                )
            )

        tag_payload = root["semantic_tag_counts"]
        if not isinstance(tag_payload, Mapping):
            raise _invalid_serialized("semantic_tag_counts must be an object")
        tag_counts: dict[str, int] = {}
        for name, count in tag_payload.items():
            if (
                not isinstance(name, str)
                or not name
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise _invalid_serialized(
                    "semantic tag counts require names and non-negative integers"
                )
            tag_counts[name] = count

        try:
            return cls(
                family_name=_text(root["family_name"], "family_name"),
                primitive_bounds=primitive,
                derived_bounds=tuple(derived),
                reference_mesh_id=_text(root["reference_mesh_id"], "reference_mesh_id"),
                reference_topology_signature=_text(
                    root["reference_topology_signature"],
                    "reference_topology_signature",
                ),
                semantic_tag_counts=tag_counts,
                criteria=_criteria_from_payload(root["criteria"]),
                validity_policy_hash=_text(
                    root["validity_policy_hash"], "validity_policy_hash"
                ),
                schema_version=_SCHEMA_VERSION,
            )
        except MorphingError:
            raise
        except ValueError as exc:
            raise _invalid_serialized(str(exc)) from exc
