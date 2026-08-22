from __future__ import annotations

import json

import pytest

from piginot.contracts import (
    DerivedExpression,
    DerivedQuantitySpec,
    GeometryFamilySpec,
    ParameterSpec,
    Quantity,
    Unit,
)
from piginot.geometry import canonicalize_mesh
from piginot.morphing import (
    FamilyMembershipPredicate,
    MembershipDecision,
    MorphReasonCode,
    MorphValidityCriteria,
)
from tests.morphing.helpers import rectangle_payload


def _length_parameter(name: str, lower: float, upper: float) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        lower_bound=Quantity(lower, Unit.MM).to_canonical(),
        upper_bound=Quantity(upper, Unit.MM).to_canonical(),
    )


def _predicate() -> FamilyMembershipPredicate:
    reference = canonicalize_mesh(rectangle_payload())
    family = GeometryFamilySpec(
        name="notched_plate",
        parameters=(
            _length_parameter("width", 16.0, 30.0),
            _length_parameter("notch_depth", 0.5, 5.0),
        ),
        derived_quantities=(
            DerivedQuantitySpec(
                name="net_section_factor",
                expression=DerivedExpression.parse("width / (width - 2 * notch_depth)"),
                lower_bound=Quantity(1.0, Unit.ONE).to_canonical(),
                upper_bound=Quantity(2.6, Unit.ONE).to_canonical(),
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


def _theta(width: float, notch_depth: float) -> dict[str, object]:
    return {
        "width": Quantity(width, Unit.MM).to_canonical(),
        "notch_depth": Quantity(notch_depth, Unit.MM).to_canonical(),
    }


def test_membership_predicate_has_a_byte_stable_strict_roundtrip() -> None:
    predicate = _predicate()

    restored = FamilyMembershipPredicate.from_bytes(predicate.to_bytes())

    assert restored == predicate
    assert restored.to_bytes() == predicate.to_bytes()


@pytest.mark.parametrize(
    ("theta", "expected_reason"),
    [
        (_theta(20.0, 2.0), None),
        (_theta(16.0, 0.5), None),
        (_theta(30.0, 5.0), None),
        (_theta(16.0 - 1.0e-12, 1.0), MorphReasonCode.PARAMETER_OUT_OF_BOUNDS),
        (_theta(20.0, 5.0 + 1.0e-12), MorphReasonCode.PARAMETER_OUT_OF_BOUNDS),
        (_theta(16.0, 4.99), MorphReasonCode.DERIVED_QUANTITY_OUT_OF_BOUNDS),
    ],
)
def test_membership_enforces_primitive_and_coupled_derived_bounds(
    theta: dict[str, object], expected_reason: MorphReasonCode | None
) -> None:
    decision = _predicate().evaluate(theta)

    assert isinstance(decision, MembershipDecision)
    assert decision.accepted is (expected_reason is None)
    assert decision.reason_code == expected_reason


def test_membership_fixture_has_100_percent_accuracy_on_204_labelled_cases() -> None:
    predicate = _predicate()
    labelled: list[tuple[dict[str, object], bool]] = []
    for index in range(50):
        width = 16.0 + index * 14.0 / 49.0
        labelled.append((_theta(width, 0.5), True))
        labelled.append((_theta(width, 5.0), width / (width - 10.0) <= 2.6))
    for index in range(50):
        depth = 0.5 + index * 4.5 / 49.0
        labelled.append((_theta(16.0 - 1.0e-9, depth), False))
        labelled.append((_theta(30.0 + 1.0e-9, depth), False))
    labelled.extend(
        [
            (_theta(16.0, 0.5), True),
            (_theta(16.0, 0.5 + 1.0e-9), True),
            (_theta(30.0, 5.0), True),
            (_theta(30.0, 5.0 + 1.0e-9), False),
        ]
    )

    predicted = tuple(predicate.evaluate(theta).accepted for theta, _ in labelled)
    expected = tuple(label for _, label in labelled)

    assert len(labelled) == 204
    assert predicted == expected


def test_membership_rejects_missing_unknown_bare_and_wrong_unit_values() -> None:
    predicate = _predicate()

    assert (
        predicate.evaluate(
            {"width": Quantity(20.0, Unit.MM).to_canonical()}
        ).reason_code
        is MorphReasonCode.MISSING_PARAMETER
    )
    assert (
        predicate.evaluate(
            {**_theta(20.0, 1.0), "extra": Quantity(1.0, Unit.MM).to_canonical()}
        ).reason_code
        is MorphReasonCode.UNKNOWN_PARAMETER
    )
    assert (
        predicate.evaluate(
            {"width": 20.0, "notch_depth": Quantity(1.0, Unit.MM).to_canonical()}
        ).reason_code
        is MorphReasonCode.INVALID_PARAMETER_VALUE
    )
    assert (
        predicate.evaluate(
            {
                "width": Quantity(20.0, Unit.N).to_canonical(),
                "notch_depth": Quantity(1.0, Unit.MM).to_canonical(),
            }
        ).reason_code
        is MorphReasonCode.PARAMETER_UNIT_MISMATCH
    )


def test_membership_rejects_non_mapping_and_non_text_parameter_names() -> None:
    predicate = _predicate()

    assert (
        predicate.evaluate(
            (("width", Quantity(20.0, Unit.MM).to_canonical()),)
        ).reason_code
        is MorphReasonCode.INVALID_PARAMETER_VALUE
    )
    assert (
        predicate.evaluate(
            {
                1: Quantity(20.0, Unit.MM).to_canonical(),
                "notch_depth": _theta(20.0, 1.0)["notch_depth"],
            }
        ).reason_code
        is MorphReasonCode.INVALID_PARAMETER_VALUE
    )


def test_derived_expression_division_by_zero_rejects_without_eval() -> None:
    reference = canonicalize_mesh(rectangle_payload())
    family = GeometryFamilySpec(
        name="singular_section",
        parameters=(
            _length_parameter("width", 10.0, 10.0),
            _length_parameter("notch_depth", 5.0, 5.0),
        ),
        derived_quantities=(
            DerivedQuantitySpec(
                name="net_section_factor",
                expression=DerivedExpression.parse("width / (width - 2 * notch_depth)"),
                lower_bound=Quantity(0.0, Unit.ONE).to_canonical(),
                upper_bound=Quantity(100.0, Unit.ONE).to_canonical(),
                unit=Unit.ONE,
            ),
        ),
        topology_signature=reference.topology_signature,
        semantic_tags=tuple(reference.semantic_tags),
    )
    predicate = FamilyMembershipPredicate.compile(
        family, reference, MorphValidityCriteria()
    )

    decision = predicate.evaluate(_theta(10.0, 5.0))

    assert decision.reason_code is MorphReasonCode.DERIVED_EVALUATION_ERROR


def test_predicate_deserialization_rejects_malformed_contract_documents() -> None:
    predicate = _predicate()
    payload = json.loads(predicate.to_bytes())

    with pytest.raises(ValueError, match="JSON object"):
        FamilyMembershipPredicate.from_bytes(b"[]")

    with pytest.raises(ValueError, match="invalid type discriminator"):
        broken = dict(payload)
        broken["$type"] = "wrong.Type"
        FamilyMembershipPredicate.from_bytes(json.dumps(broken).encode())

    with pytest.raises(ValueError, match="missing field"):
        broken = dict(payload)
        del broken["family_name"]
        FamilyMembershipPredicate.from_bytes(json.dumps(broken).encode())

    with pytest.raises(ValueError, match="schema"):
        broken = dict(payload)
        broken["schema_version"] = "0.0"
        FamilyMembershipPredicate.from_bytes(json.dumps(broken).encode())

    with pytest.raises(ValueError, match="primitive_bounds"):
        broken = dict(payload)
        broken["primitive_bounds"] = {"not": "an array"}
        FamilyMembershipPredicate.from_bytes(json.dumps(broken).encode())

    with pytest.raises(ValueError, match="semantic tag counts"):
        broken = dict(payload)
        broken["semantic_tag_counts"] = {"left": -1}
        FamilyMembershipPredicate.from_bytes(json.dumps(broken).encode())

    with pytest.raises(ValueError, match="validity_policy_hash"):
        broken = dict(payload)
        broken["validity_policy_hash"] = "sha256:not-the-policy"
        FamilyMembershipPredicate.from_bytes(json.dumps(broken).encode())

    with pytest.raises(ValueError, match="primitive bound"):
        broken = dict(payload)
        broken["primitive_bounds"] = [
            {**payload["primitive_bounds"][0], "$type": "wrong.Type"}
        ]
        FamilyMembershipPredicate.from_bytes(json.dumps(broken).encode())

    with pytest.raises(ValueError, match="finite number"):
        broken = dict(payload)
        first = dict(payload["primitive_bounds"][0])
        lower = dict(first["lower_bound"])
        lower["value"] = True
        first["lower_bound"] = lower
        broken["primitive_bounds"] = [first, *payload["primitive_bounds"][1:]]
        FamilyMembershipPredicate.from_bytes(json.dumps(broken).encode())

    with pytest.raises(ValueError, match="undeclared primitive"):
        broken = dict(payload)
        first = dict(payload["derived_bounds"][0])
        first["canonical_ast"] = ["name", "missing"]
        broken["derived_bounds"] = [first]
        FamilyMembershipPredicate.from_bytes(json.dumps(broken).encode())

    with pytest.raises(ValueError, match="unary nodes"):
        broken = dict(payload)
        first = dict(payload["derived_bounds"][0])
        first["canonical_ast"] = ["unary", "~", ["number", 1]]
        broken["derived_bounds"] = [first]
        FamilyMembershipPredicate.from_bytes(json.dumps(broken).encode())

    with pytest.raises(ValueError, match="binary nodes"):
        broken = dict(payload)
        first = dict(payload["derived_bounds"][0])
        first["canonical_ast"] = ["binary", "%", ["number", 1], ["number", 2]]
        broken["derived_bounds"] = [first]
        FamilyMembershipPredicate.from_bytes(json.dumps(broken).encode())
