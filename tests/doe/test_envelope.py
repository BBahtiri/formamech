from __future__ import annotations

from dataclasses import replace

import pytest

from piginot.contracts import (
    CanonicalQuantity,
    DerivedExpression,
    Dimension,
    Unit,
    hash_spec,
)
from piginot.doe.design import (
    AxisSource,
    AxisSpec,
    CoupledDesignSpec,
    generate_coupled_design,
)
from piginot.doe.envelope import (
    EnvelopeDecision,
    EnvelopeReasonCode,
    ExecutableEnvelope,
)
from piginot.doe.leakage import LeakagePolicy
from piginot.morphing.membership import (
    DerivedBound,
    FamilyMembershipPredicate,
    PrimitiveBound,
)
from piginot.morphing.models import MorphReasonCode, MorphValidityCriteria


def _quantity(value: float, unit: Unit = Unit.ONE) -> CanonicalQuantity:
    dimension = Dimension.DIMENSIONLESS if unit is Unit.ONE else Dimension.STRESS
    return CanonicalQuantity(value, unit, dimension)


def _primitive(name: str, lower: float, upper: float) -> PrimitiveBound:
    return PrimitiveBound(name, _quantity(lower), _quantity(upper))


def _derived(name: str, expression: str, lower: float, upper: float) -> DerivedBound:
    return DerivedBound(
        name=name,
        canonical_ast=DerivedExpression.parse(expression).canonical_ast,
        lower_bound=_quantity(lower),
        upper_bound=_quantity(upper),
    )


def _geometry_predicate() -> FamilyMembershipPredicate:
    criteria = MorphValidityCriteria(
        jacobian_margin=0.01,
        minimum_feature_size_mm=0.05,
        minimum_angle_degrees=5.0,
        maximum_aspect_ratio=20.0,
    )
    return FamilyMembershipPredicate(
        family_name="unit-family",
        primitive_bounds=(_primitive("width", 1.0, 10.0),),
        derived_bounds=(_derived("net_section_ratio", "width / 10", 0.1, 1.0),),
        reference_mesh_id="mesh",
        reference_topology_signature="topology",
        semantic_tag_counts={"loaded": 1},
        criteria=criteria,
        validity_policy_hash=hash_spec(criteria, "content"),
    )


def _stress_bound(name: str, lower: float, upper: float) -> PrimitiveBound:
    return PrimitiveBound(
        name,
        CanonicalQuantity(lower, Unit.MPA, Dimension.STRESS),
        CanonicalQuantity(upper, Unit.MPA, Dimension.STRESS),
    )


def _length_bound(name: str, lower: float, upper: float) -> PrimitiveBound:
    return PrimitiveBound(
        name,
        CanonicalQuantity(lower, Unit.MM, Dimension.LENGTH),
        CanonicalQuantity(upper, Unit.MM, Dimension.LENGTH),
    )


def _envelope() -> ExecutableEnvelope:
    geometry = _geometry_predicate()
    return ExecutableEnvelope(
        envelope_id="unit-envelope-v1",
        compiled_family_hash="a" * 64,
        geometry_membership=geometry,
        geometry_membership_hash=hash_spec(geometry, "content"),
        material_bounds=(
            _stress_bound("elastic_modulus", 760.0, 760.0),
            PrimitiveBound(
                "poisson_ratio",
                _quantity(0.23),
                _quantity(0.23),
            ),
        ),
        load_bounds=(_length_bound("prescribed_displacement", 1.0, 1.0),),
    )


def test_executable_envelope_extends_geometry_membership_with_material_axes() -> None:
    envelope = _envelope()

    assert tuple(bound.name for bound in envelope.primitive_bounds) == (
        "width",
        "elastic_modulus",
        "poisson_ratio",
        "prescribed_displacement",
    )
    assert tuple(bound.name for bound in envelope.derived_bounds) == (
        "net_section_ratio",
    )
    assert envelope.geometry_membership_hash == hash_spec(
        envelope.geometry_membership,
        "content",
    )


def test_envelope_evaluates_geometry_and_additional_unitized_bounds() -> None:
    accepted = _envelope().evaluate(
        {
            "width": _quantity(5.0),
            "elastic_modulus": CanonicalQuantity(760.0, Unit.MPA, Dimension.STRESS),
            "poisson_ratio": _quantity(0.23),
            "prescribed_displacement": CanonicalQuantity(
                1.0, Unit.MM, Dimension.LENGTH
            ),
        }
    )

    assert accepted.accepted is True
    assert accepted.derived_values["net_section_ratio"].value == 0.5


@pytest.mark.parametrize(
    ("theta", "reason"),
    (
        (
            {"width": _quantity(5.0), "poisson_ratio": _quantity(0.23)},
            EnvelopeReasonCode.MISSING_PARAMETER,
        ),
        (
            {
                "width": _quantity(5.0),
                "elastic_modulus": _quantity(760.0),
                "poisson_ratio": _quantity(0.23),
                "prescribed_displacement": CanonicalQuantity(
                    1.0, Unit.MM, Dimension.LENGTH
                ),
            },
            EnvelopeReasonCode.PARAMETER_UNIT_MISMATCH,
        ),
        (
            {
                "width": _quantity(5.0),
                "elastic_modulus": CanonicalQuantity(700.0, Unit.MPA, Dimension.STRESS),
                "poisson_ratio": _quantity(0.23),
                "prescribed_displacement": CanonicalQuantity(
                    1.0, Unit.MM, Dimension.LENGTH
                ),
            },
            EnvelopeReasonCode.PARAMETER_OUT_OF_BOUNDS,
        ),
    ),
)
def test_envelope_rejects_invalid_additional_domain_values(
    theta: dict[str, CanonicalQuantity],
    reason: EnvelopeReasonCode,
) -> None:
    decision = _envelope().evaluate(theta)

    assert decision.accepted is False
    assert decision.reason_codes == (reason,)


def test_envelope_preserves_geometry_rejection_reason() -> None:
    decision = _envelope().evaluate(
        {
            "width": _quantity(11.0),
            "elastic_modulus": CanonicalQuantity(760.0, Unit.MPA, Dimension.STRESS),
            "poisson_ratio": _quantity(0.23),
            "prescribed_displacement": CanonicalQuantity(
                1.0, Unit.MM, Dimension.LENGTH
            ),
        }
    )

    assert decision.accepted is False
    assert decision.reason_codes == (MorphReasonCode.PARAMETER_OUT_OF_BOUNDS,)


def test_envelope_rejects_identity_or_bound_drift() -> None:
    envelope = _envelope()

    with pytest.raises(ValueError, match="geometry_membership_hash"):
        replace(envelope, geometry_membership_hash="b" * 64)
    with pytest.raises(ValueError, match="unique"):
        replace(
            envelope,
            material_bounds=(
                *envelope.material_bounds,
                _primitive("width", 1.0, 2.0),
            ),
        )


def test_envelope_requires_explicit_material_and_load_domains() -> None:
    envelope = _envelope()

    with pytest.raises(ValueError, match="material_bounds"):
        replace(envelope, material_bounds=())
    with pytest.raises(ValueError, match="load_bounds"):
        replace(envelope, load_bounds=())


def test_envelope_binds_its_fail_closed_refusal_policy() -> None:
    envelope = _envelope()

    assert envelope.refusal_policy_id == "reject-any-bound-or-derived-violation-v1"
    with pytest.raises(ValueError, match="refusal_policy_id"):
        replace(envelope, refusal_policy_id="accept-and-warn")


def test_rejected_envelope_decisions_require_a_stable_reason() -> None:
    with pytest.raises(ValueError, match="rejected decisions"):
        EnvelopeDecision(accepted=False)


def test_fixed_material_and_load_coordinates_are_bound_into_the_doe() -> None:
    envelope = _envelope()
    spec = CoupledDesignSpec(
        seed=b"public-envelope-test-seed",
        sample_count=2,
        candidate_count=4,
        validation_count=1,
        validation_candidate_count=2,
        axes=(
            AxisSpec("width", AxisSource.PRIMITIVE, strata=2),
            AxisSpec("elastic_modulus", AxisSource.PRIMITIVE, strata=1),
            AxisSpec("poisson_ratio", AxisSource.PRIMITIVE, strata=1),
            AxisSpec("prescribed_displacement", AxisSource.PRIMITIVE, strata=1),
            AxisSpec("net_section_ratio", AxisSource.DERIVED, strata=2),
        ),
        leakage_policy=LeakagePolicy(near_duplicate_tolerance=0.0),
    )

    design = generate_coupled_design(envelope, spec)

    assert all(
        point.coordinates["elastic_modulus"] == 760.0
        and point.coordinates["poisson_ratio"] == 0.23
        and point.coordinates["prescribed_displacement"] == 1.0
        for point in (*design.selected_points, *design.validation_points)
    )

    incomplete = replace(
        spec,
        axes=tuple(
            axis for axis in spec.axes if axis.name != "prescribed_displacement"
        ),
    )
    with pytest.raises(ValueError, match="primitive axes"):
        generate_coupled_design(envelope, incomplete)
