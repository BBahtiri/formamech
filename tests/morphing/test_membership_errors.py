from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any, cast

import pytest

from piginot.contracts import (
    CanonicalQuantity,
    DerivedExpression,
    DerivedQuantitySpec,
    GeometryFamilySpec,
    ParameterSpec,
    Quantity,
    Unit,
)
from piginot.geometry import canonicalize_mesh
from piginot.morphing import (
    DerivedBound,
    FamilyMembershipPredicate,
    MorphingError,
    MorphReasonCode,
    MorphValidityCriteria,
    PrimitiveBound,
)
from tests.morphing.helpers import rectangle_payload

JsonObject = dict[str, Any]
PathItem = str | int
Mutation = Callable[[JsonObject], None]


def _canonical(value: float, unit: Unit = Unit.ONE) -> CanonicalQuantity:
    return Quantity(value, unit).to_canonical()


def _parameter(
    name: str,
    lower: float,
    upper: float,
    unit: Unit = Unit.ONE,
) -> ParameterSpec:
    return ParameterSpec(name, _canonical(lower, unit), _canonical(upper, unit))


def _predicate() -> FamilyMembershipPredicate:
    reference = canonicalize_mesh(rectangle_payload())
    family = GeometryFamilySpec(
        name="membership_edge_cases",
        parameters=(
            _parameter("width", 16.0, 30.0, Unit.MM),
            _parameter("notch_depth", 0.5, 5.0, Unit.MM),
        ),
        derived_quantities=(
            DerivedQuantitySpec(
                name="net_section_factor",
                expression=DerivedExpression.parse("width / (width - 2 * notch_depth)"),
                lower_bound=_canonical(1.0),
                upper_bound=_canonical(2.6),
                unit=Unit.ONE,
            ),
        ),
        topology_signature=reference.topology_signature,
        semantic_tags=tuple(reference.semantic_tags),
    )
    return FamilyMembershipPredicate.compile(
        family,
        reference,
        MorphValidityCriteria(),
    )


def _payload() -> JsonObject:
    decoded = json.loads(_predicate().to_bytes())
    return cast(JsonObject, decoded)


def _replace(path: tuple[PathItem, ...], value: object) -> Mutation:
    def mutate(payload: JsonObject) -> None:
        target: Any = payload
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value

    return mutate


def _remove(path: tuple[PathItem, ...]) -> Mutation:
    def mutate(payload: JsonObject) -> None:
        target: Any = payload
        for part in path[:-1]:
            target = target[part]
        del target[path[-1]]

    return mutate


def _add(path: tuple[PathItem, ...], name: str, value: object) -> Mutation:
    def mutate(payload: JsonObject) -> None:
        target: Any = payload
        for part in path:
            target = target[part]
        target[name] = value

    return mutate


def _assert_invalid_serialized(mutation: Mutation) -> None:
    payload = copy.deepcopy(_payload())
    mutation(payload)

    with pytest.raises(MorphingError) as error:
        FamilyMembershipPredicate.from_bytes(json.dumps(payload))

    assert error.value.reason_code is MorphReasonCode.INVALID_SERIALIZED_PREDICATE


@pytest.mark.parametrize(
    "mutation",
    [
        _remove(("family_name",)),
        _add((), "unexpected", True),
        _replace(("$type",), "wrong.Predicate"),
        _replace(("schema_version",), "2.0"),
        _replace(("primitive_bounds",), {}),
        _replace(("derived_bounds",), {}),
        _replace(("semantic_tag_counts",), []),
        _replace(("family_name",), ""),
        _replace(("reference_mesh_id",), ""),
        _replace(("reference_topology_signature",), ""),
        _replace(("validity_policy_hash",), "not-the-policy-hash"),
    ],
    ids=(
        "missing-root-field",
        "unknown-root-field",
        "root-type",
        "schema-version",
        "primitive-array",
        "derived-array",
        "tag-map",
        "family-name",
        "reference-id",
        "topology-signature",
        "policy-hash",
    ),
)
def test_serialized_predicate_rejects_invalid_root_contract(
    mutation: Mutation,
) -> None:
    _assert_invalid_serialized(mutation)


def test_serialized_predicate_requires_an_object_root() -> None:
    with pytest.raises(MorphingError) as error:
        FamilyMembershipPredicate.from_bytes("[]")

    assert error.value.reason_code is MorphReasonCode.INVALID_SERIALIZED_PREDICATE


@pytest.mark.parametrize(
    "mutation",
    [
        _replace(("primitive_bounds", 0), []),
        _remove(("primitive_bounds", 0, "upper_bound")),
        _add(("primitive_bounds", 0), "unexpected", True),
        _replace(("primitive_bounds", 0, "$type"), "wrong.PrimitiveBound"),
        _replace(("primitive_bounds", 0, "name"), ""),
        _replace(("derived_bounds", 0, "$type"), "wrong.DerivedBound"),
        _replace(("derived_bounds", 0, "name"), ""),
        _replace(("primitive_bounds", 0, "lower_bound"), []),
        _replace(
            ("primitive_bounds", 0, "lower_bound", "$type"),
            "wrong.Quantity",
        ),
        _replace(("primitive_bounds", 0, "lower_bound", "value"), True),
        _replace(("primitive_bounds", 0, "lower_bound", "unit"), "parsec"),
        _replace(("primitive_bounds", 0, "lower_bound", "unit"), "cm"),
        _replace(("primitive_bounds", 0, "lower_bound", "dimension"), "force"),
    ],
    ids=(
        "bound-object",
        "missing-bound-field",
        "unknown-bound-field",
        "primitive-type",
        "primitive-name",
        "derived-type",
        "derived-name",
        "quantity-object",
        "quantity-type",
        "quantity-number",
        "unknown-unit",
        "noncanonical-unit",
        "unit-dimension",
    ),
)
def test_serialized_predicate_rejects_invalid_bound_and_quantity_contracts(
    mutation: Mutation,
) -> None:
    _assert_invalid_serialized(mutation)


@pytest.mark.parametrize(
    "tree",
    [
        "width",
        [],
        [1, "width"],
        ["number"],
        ["number", True],
        ["name", 1],
        ["name", "undeclared"],
        ["unary", "!", ["name", "width"]],
        ["binary", "%", ["name", "width"], ["number", 2]],
        ["call", "abs", ["name", "width"]],
    ],
    ids=(
        "not-array",
        "empty",
        "nontext-kind",
        "number-arity",
        "number-type",
        "name-type",
        "undeclared-name",
        "unary-operator",
        "binary-operator",
        "unknown-node",
    ),
)
def test_serialized_predicate_rejects_invalid_expression_trees(
    tree: object,
) -> None:
    _assert_invalid_serialized(_replace(("derived_bounds", 0, "canonical_ast"), tree))


@pytest.mark.parametrize(
    "mutation",
    [
        _replace(("criteria",), []),
        _replace(("criteria", "$type"), "wrong.Criteria"),
        _replace(("criteria", "jacobian_margin"), True),
        _replace(("criteria", "jacobian_margin"), -0.1),
        _replace(("criteria", "minimum_angle_degrees"), 61.0),
        _replace(("criteria", "maximum_aspect_ratio"), 0.9),
        _replace(("semantic_tag_counts", "left"), True),
        _replace(("semantic_tag_counts", "left"), -1),
    ],
    ids=(
        "criteria-object",
        "criteria-type",
        "criteria-number",
        "negative-jacobian-margin",
        "invalid-angle",
        "invalid-aspect-ratio",
        "boolean-tag-count",
        "negative-tag-count",
    ),
)
def test_serialized_predicate_rejects_invalid_policy_and_tag_contracts(
    mutation: Mutation,
) -> None:
    _assert_invalid_serialized(mutation)


def test_compile_rejects_reference_topology_and_tag_mismatches() -> None:
    reference = canonicalize_mesh(rectangle_payload())
    topology_mismatch = GeometryFamilySpec(
        name="wrong_topology",
        topology_signature="planar-cell-complex-v1:components=2;holes=0",
    )
    tag_mismatch = GeometryFamilySpec(
        name="wrong_tags",
        semantic_tags=("not-a-reference-tag",),
    )

    for family in (topology_mismatch, tag_mismatch):
        with pytest.raises(MorphingError) as error:
            FamilyMembershipPredicate.compile(
                family,
                reference,
                MorphValidityCriteria(),
            )
        assert error.value.reason_code is MorphReasonCode.VALIDATION_CONTEXT_MISMATCH


@pytest.mark.parametrize("argument", ["family", "reference", "criteria"])
def test_compile_rejects_wrong_contract_types(argument: str) -> None:
    reference = canonicalize_mesh(rectangle_payload())
    family = GeometryFamilySpec(name="typed")
    criteria = MorphValidityCriteria()
    arguments: dict[str, object] = {
        "family": family,
        "reference": reference,
        "criteria": criteria,
    }
    arguments[argument] = object()

    with pytest.raises(TypeError):
        FamilyMembershipPredicate.compile(
            cast(GeometryFamilySpec, arguments["family"]),
            cast(type(reference), arguments["reference"]),
            cast(MorphValidityCriteria, arguments["criteria"]),
        )


def test_evaluate_rejects_nonmapping_and_nontext_parameter_names() -> None:
    predicate = _predicate()

    nonmapping = predicate.evaluate(cast(Any, []))
    nontext_name = predicate.evaluate(
        cast(
            dict[str, object],
            {1: _canonical(20.0, Unit.MM), "notch_depth": _canonical(1.0, Unit.MM)},
        )
    )

    assert nonmapping.reason_code is MorphReasonCode.INVALID_PARAMETER_VALUE
    assert nontext_name.reason_code is MorphReasonCode.INVALID_PARAMETER_VALUE


def test_derived_bounds_are_inclusive_and_cover_unary_arithmetic() -> None:
    reference = canonicalize_mesh(rectangle_payload())
    family = GeometryFamilySpec(
        name="inclusive_derived_bounds",
        parameters=(_parameter("factor", 1.0, 3.0),),
        derived_quantities=(
            DerivedQuantitySpec(
                name="positive",
                expression=DerivedExpression.parse("+factor * 2"),
                lower_bound=_canonical(2.0),
                upper_bound=_canonical(6.0),
                unit=Unit.ONE,
            ),
            DerivedQuantitySpec(
                name="negative",
                expression=DerivedExpression.parse("-factor + 4"),
                lower_bound=_canonical(1.0),
                upper_bound=_canonical(3.0),
                unit=Unit.ONE,
            ),
        ),
    )
    predicate = FamilyMembershipPredicate.compile(
        family,
        reference,
        MorphValidityCriteria(),
    )

    lower = predicate.evaluate({"factor": _canonical(1.0)})
    upper = predicate.evaluate({"factor": _canonical(3.0)})

    assert lower.accepted
    assert lower.derived_values["positive"].value == 2.0
    assert lower.derived_values["negative"].value == 3.0
    assert upper.accepted
    assert upper.derived_values["positive"].value == 6.0
    assert upper.derived_values["negative"].value == 1.0


def test_derived_division_by_zero_fails_closed_and_preserves_prior_values() -> None:
    reference = canonicalize_mesh(rectangle_payload())
    family = GeometryFamilySpec(
        name="derived_failure",
        parameters=(
            _parameter("numerator", 0.0, 10.0),
            _parameter("divisor", 0.0, 10.0),
        ),
        derived_quantities=(
            DerivedQuantitySpec(
                name="prior_value",
                expression=DerivedExpression.parse("numerator + 1"),
                lower_bound=_canonical(0.0),
                upper_bound=_canonical(11.0),
                unit=Unit.ONE,
            ),
            DerivedQuantitySpec(
                name="ratio",
                expression=DerivedExpression.parse("numerator / divisor"),
                lower_bound=_canonical(0.0),
                upper_bound=_canonical(10.0),
                unit=Unit.ONE,
            ),
        ),
    )
    predicate = FamilyMembershipPredicate.compile(
        family,
        reference,
        MorphValidityCriteria(),
    )

    decision = predicate.evaluate(
        {"numerator": _canonical(5.0), "divisor": _canonical(0.0)}
    )

    assert not decision.accepted
    assert decision.reason_code is MorphReasonCode.DERIVED_EVALUATION_ERROR
    assert decision.derived_values["prior_value"].value == 6.0
    assert "ratio" not in decision.derived_values


def test_nonfinite_derived_result_fails_closed() -> None:
    reference = canonicalize_mesh(rectangle_payload())
    family = GeometryFamilySpec(
        name="derived_overflow",
        parameters=(_parameter("factor", 0.0, 1.0e308),),
        derived_quantities=(
            DerivedQuantitySpec(
                name="squared",
                expression=DerivedExpression.parse("factor * factor"),
                lower_bound=_canonical(0.0),
                upper_bound=_canonical(1.0e308),
                unit=Unit.ONE,
            ),
        ),
    )
    predicate = FamilyMembershipPredicate.compile(
        family,
        reference,
        MorphValidityCriteria(),
    )

    decision = predicate.evaluate({"factor": _canonical(1.0e308)})

    assert not decision.accepted
    assert decision.reason_code is MorphReasonCode.DERIVED_EVALUATION_ERROR


@pytest.mark.parametrize(
    ("bound_type", "name", "lower", "upper", "message"),
    [
        (PrimitiveBound, "not valid", _canonical(0.0), _canonical(1.0), "name"),
        (PrimitiveBound, "x", _canonical(0.0), _canonical(-1.0), "lower"),
        (
            PrimitiveBound,
            "x",
            _canonical(0.0, Unit.MM),
            _canonical(1.0, Unit.N),
            "identical",
        ),
        (DerivedBound, "not valid", _canonical(0.0), _canonical(1.0), "name"),
        (DerivedBound, "x", _canonical(0.0), _canonical(-1.0), "lower"),
        (
            DerivedBound,
            "x",
            _canonical(0.0, Unit.MM),
            _canonical(1.0, Unit.N),
            "identical",
        ),
    ],
)
def test_compiled_bound_invariants_reject_invalid_intervals(
    bound_type: type[PrimitiveBound] | type[DerivedBound],
    name: str,
    lower: CanonicalQuantity,
    upper: CanonicalQuantity,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        if bound_type is PrimitiveBound:
            PrimitiveBound(name, lower, upper)
        else:
            DerivedBound(name, ("number", 0), lower, upper)
