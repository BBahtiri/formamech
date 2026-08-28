from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from piginot.contracts import (
    CanonicalQuantity,
    DerivedExpression,
    Dimension,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.contracts.errors import ContractValidationError
from piginot.doe.allocation import (
    ProductionRoleBudget,
    allocate_evidence_roles,
)
from piginot.doe.checkpoint import CampaignBudget, EvidenceReference
from piginot.doe.design import (
    AxisSource,
    AxisSpec,
    CoupledDesign,
    CoupledDesignSpec,
    DesignPoint,
    generate_coupled_design,
)
from piginot.doe.envelope import ExecutableEnvelope
from piginot.doe.leakage import (
    LeakagePolicy,
    bind_leakage_groups,
)
from piginot.doe.preregistration import (
    LegacyCaseLedger,
    PreregistrationError,
    build_preregistration,
)
from piginot.doe.publication import PublishedCaseStatus
from piginot.doe.replacement import OrderedReservePolicy, ReserveCandidate
from piginot.doe.roles import CaseRoleRegistry, EvidenceRole
from piginot.doe.secret_material import (
    CAMPAIGN_SECRET_GENERATOR_ID,
    CampaignSecrets,
    generate_campaign_secrets,
)
from piginot.morphing.membership import (
    DerivedBound,
    FamilyMembershipPredicate,
    PrimitiveBound,
)
from piginot.morphing.models import MorphValidityCriteria

_DOE_SEED = bytes(range(32))
_HOLDOUT_SALT = bytes(range(32, 64))
_ALLOCATION_SEED = bytes(range(64, 96))


def _campaign_secrets(
    *,
    doe_seed: bytes = _DOE_SEED,
    allocation_seed: bytes = _ALLOCATION_SEED,
    holdout_salt: bytes = _HOLDOUT_SALT,
) -> CampaignSecrets:
    with patch(
        "piginot.doe.secret_material.secrets.token_bytes",
        side_effect=(doe_seed, allocation_seed, holdout_salt),
    ):
        return generate_campaign_secrets()


def _quantity(
    value: float,
    unit: Unit = Unit.ONE,
    dimension: Dimension = Dimension.DIMENSIONLESS,
) -> CanonicalQuantity:
    return CanonicalQuantity(value, unit, dimension)


def _predicate(*, upper_width: float = 256.0) -> FamilyMembershipPredicate:
    criteria = MorphValidityCriteria(
        jacobian_margin=0.01,
        minimum_feature_size_mm=0.05,
        minimum_angle_degrees=5.0,
        maximum_aspect_ratio=20.0,
    )
    return FamilyMembershipPredicate(
        family_name="coupled-family-v1",
        primitive_bounds=(
            PrimitiveBound("width", _quantity(1.0), _quantity(upper_width)),
        ),
        derived_bounds=(
            DerivedBound(
                name="net_section_ratio",
                canonical_ast=DerivedExpression.parse("width / 100").canonical_ast,
                lower_bound=_quantity(0.01),
                upper_bound=_quantity(upper_width / 100.0),
            ),
        ),
        reference_mesh_id="reference-mesh",
        reference_topology_signature="topology",
        semantic_tag_counts={"loaded": 1},
        criteria=criteria,
        validity_policy_hash=hash_spec(criteria, "content"),
    )


@dataclass(frozen=True, slots=True)
class _CompiledFamily:
    __canonical_namespace__: ClassVar[str] = "piginot.tests"

    family_id: str
    membership: FamilyMembershipPredicate


def _compiled_family(*, upper_width: float = 256.0) -> _CompiledFamily:
    return _CompiledFamily(
        family_id="coupled-family-v1",
        membership=_predicate(upper_width=upper_width),
    )


def _envelope(
    compiled_family: _CompiledFamily,
    *,
    compiled_family_hash: str | None = None,
) -> ExecutableEnvelope:
    predicate = compiled_family.membership
    return ExecutableEnvelope(
        envelope_id="accepted-envelope-v1",
        compiled_family_hash=compiled_family_hash
        or hash_spec(compiled_family, "content"),
        geometry_membership=predicate,
        geometry_membership_hash=hash_spec(predicate, "content"),
        material_bounds=(
            PrimitiveBound(
                "elastic_modulus",
                _quantity(760.0, Unit.MPA, Dimension.STRESS),
                _quantity(760.0, Unit.MPA, Dimension.STRESS),
            ),
            PrimitiveBound(
                "poisson_ratio",
                _quantity(0.23),
                _quantity(0.23),
            ),
        ),
        load_bounds=(
            PrimitiveBound(
                "prescribed_displacement",
                _quantity(1.0, Unit.MM, Dimension.LENGTH),
                _quantity(1.0, Unit.MM, Dimension.LENGTH),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class _Identity:
    case_id: str
    leakage_group_id: str


def _inputs() -> dict[str, Any]:
    campaign_secrets = _campaign_secrets()
    compiled_family = _compiled_family()
    envelope = _envelope(compiled_family)
    leakage_policy = LeakagePolicy(near_duplicate_tolerance=0.0)
    spec = CoupledDesignSpec(
        seed=campaign_secrets.doe_seed,
        sample_count=67,
        reserve_count=4,
        candidate_count=80,
        validation_count=1,
        validation_candidate_count=8,
        axes=(
            AxisSpec("width", AxisSource.PRIMITIVE),
            AxisSpec("elastic_modulus", AxisSource.PRIMITIVE, strata=1),
            AxisSpec("poisson_ratio", AxisSource.PRIMITIVE, strata=1),
            AxisSpec("prescribed_displacement", AxisSource.PRIMITIVE, strata=1),
            AxisSpec("net_section_ratio", AxisSource.DERIVED),
        ),
        leakage_policy=leakage_policy,
    )
    design = generate_coupled_design(envelope, spec)
    selected = design.selected_points
    reserves = design.reserve_points
    reserve_roles = (
        EvidenceRole.PILOT,
        EvidenceRole.TRAIN,
        EvidenceRole.TRAIN,
        EvidenceRole.DEVELOPMENT,
    )
    allocation = allocate_evidence_roles(
        selected,
        ProductionRoleBudget(
            pilot=1,
            train=1,
            development=1,
            holdout=64,
        ),
        seed=campaign_secrets.allocation_seed,
    )
    policy = OrderedReservePolicy.preregister(
        rule_id="role-local-design-order-v1",
        reserves_by_role={
            role: tuple(
                point
                for point, assigned_role in zip(reserves, reserve_roles, strict=True)
                if assigned_role is role
            )
            for role in (
                EvidenceRole.PILOT,
                EvidenceRole.TRAIN,
                EvidenceRole.DEVELOPMENT,
            )
        },
    )
    return {
        "compiled_family": compiled_family,
        "executable_envelope": envelope,
        "pilot_evidence": (
            EvidenceReference(
                "legacy-closeout",
                "a" * 64,
                EvidenceRole.LEGACY,
            ),
        ),
        "spec": spec,
        "design": design,
        "role_allocation": allocation,
        "campaign_secrets": campaign_secrets,
        "legacy_ledger": LegacyCaseLedger(("legacy-0", "legacy-1")),
        "replacement_policy": policy,
        "budget": CampaignBudget(
            pilot=1,
            train=1,
            development=1,
            holdout=64,
            legacy=2,
            reserve=4,
        ),
        "created_utc": "2026-08-24T12:00:00Z",
        "manifest_id": "s9-public-cases",
    }


def _build(**changes: object):  # type: ignore[no-untyped-def]
    values = _inputs()
    values.update(changes)
    return build_preregistration(**values)


def test_builder_closes_the_campaign_and_publishes_only_nonsealed_cases() -> None:
    values = _inputs()

    result = build_preregistration(**values)

    assert result.checkpoint.binding == result.binding
    assert result.checkpoint.campaign_hash == result.campaign_hash
    assert result.holdout_commitment == result.checkpoint.holdout_commitment
    assert result.reveal_verified is True
    assert result.binding.compiled_family_hash == hash_spec(
        values["compiled_family"], "content"
    )
    assert result.binding.executable_envelope_hash == hash_spec(
        values["executable_envelope"], "content"
    )
    assert result.binding.private_doe_plan_hash == hash_spec(
        {"design": values["design"], "spec": values["spec"]}, "content"
    )
    assert result.binding.role_ledger_hash == hash_spec(
        {
            "legacy_ledger": values["legacy_ledger"],
            "production_allocation": values["role_allocation"],
        },
        "content",
    )
    assert result.binding.replacement_policy_hash == hash_spec(
        values["replacement_policy"], "content"
    )
    assert (
        result.binding.public_case_manifest_hash == result.public_manifest.content_hash
    )
    assert result.binding.doe_seed_commitment == hash_spec(
        values["spec"].seed, "content"
    )
    assert result.binding.secret_generator_id == CAMPAIGN_SECRET_GENERATOR_ID
    assert result.checkpoint.secret_generator_id == CAMPAIGN_SECRET_GENERATOR_ID
    assert len(result.public_manifest.cases) == 7
    assert {case.role for case in result.public_manifest.cases} == {
        EvidenceRole.PILOT,
        EvidenceRole.TRAIN,
        EvidenceRole.DEVELOPMENT,
    }
    assert (
        sum(
            case.status is PublishedCaseStatus.ACTIVE
            for case in result.public_manifest.cases
        )
        == 3
    )
    assert (
        sum(
            case.status is PublishedCaseStatus.RESERVE
            for case in result.public_manifest.cases
        )
        == 4
    )
    holdout_ids = {
        case_id
        for case_id, assignment in values[
            "role_allocation"
        ].registry.assignments.items()
        if assignment.role is EvidenceRole.HOLDOUT
    }
    assert holdout_ids.isdisjoint(case.case_id for case in result.public_manifest.cases)


def test_result_is_immutable_and_contains_no_seed_salt_or_holdout_coordinates() -> None:
    values = _inputs()
    result = build_preregistration(**values)
    encoded = canonical_json_bytes(result)
    seed_payload = values["spec"].seed.payload.encode()
    holdout_coordinate = canonical_json_bytes(
        values["design"].selected_points[3].all_coordinates
    )

    assert seed_payload not in encoded
    assert _HOLDOUT_SALT.hex().encode() not in encoded
    assert _ALLOCATION_SEED.hex().encode() not in encoded
    assert holdout_coordinate not in result.public_manifest.to_bytes()
    assert b'"seed"' not in encoded
    assert b'"salt"' not in encoded
    assert b'"holdout_coordinates"' not in encoded
    assert not hasattr(result, "seed")
    assert not hasattr(result, "salt")
    assert not hasattr(result, "holdout_coordinates")
    with pytest.raises(FrozenInstanceError):
        result.reveal_verified = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("pilot", 0, "PILOT"),
        ("train", 0, "TRAIN"),
        ("development", 0, "DEVELOPMENT"),
        ("holdout", 63, "64"),
        ("legacy", 1, "LEGACY"),
        ("reserve", 3, "reserve"),
    ),
)
def test_every_budget_dimension_is_checked(
    field: str, value: int, message: str
) -> None:
    values = _inputs()
    budget = replace(values["budget"], **{field: value})

    with pytest.raises(PreregistrationError, match=message):
        _build(budget=budget)


def test_selected_cases_must_exactly_match_the_production_allocation() -> None:
    values = _inputs()
    selected = values["design"].selected_points
    allocation = values["role_allocation"]
    registry = allocation.registry
    assignments = dict(registry.assignments)
    first = selected[0]
    removed_role = assignments[first.case_id].role
    assignments.pop(first.case_id)
    group_roles = dict(registry.leakage_group_roles)
    group_roles.pop(first.leakage_group_id)
    substitute = _Identity("substitute", "substitute-group")
    missing_registry = CaseRoleRegistry(assignments, group_roles).assign(
        substitute, removed_role
    )
    changed_allocation = replace(allocation, registry=missing_registry)
    with pytest.raises(PreregistrationError, match="registry"):
        _build(role_allocation=changed_allocation)


def test_registry_cannot_contain_an_extra_nonlegacy_campaign_case() -> None:
    values = _inputs()
    allocation = values["role_allocation"]
    selected = values["design"].selected_points
    removed = selected[1]
    assignments = dict(allocation.registry.assignments)
    removed_role = assignments[removed.case_id].role
    assignments.pop(removed.case_id)
    groups = dict(allocation.registry.leakage_group_roles)
    groups.pop(removed.leakage_group_id)
    registry = CaseRoleRegistry(assignments, groups).assign(
        _Identity("extra", "extra-group"), removed_role
    )
    changed_allocation = replace(allocation, registry=registry)

    with pytest.raises(PreregistrationError, match="unexpected active or reserve"):
        _build(role_allocation=changed_allocation)


def test_whole_point_leakage_identity_must_match_registry() -> None:
    values = _inputs()
    design = values["design"]
    first_point = design.selected_points[0]
    changed_descriptors = tuple(
        replace(
            descriptor,
            geometry_lineage_id=f"{descriptor.geometry_lineage_id}-changed",
        )
        if descriptor.case_id == first_point.case_id
        else descriptor
        for descriptor in design.leakage_descriptors
    )
    changed_groups = bind_leakage_groups(
        changed_descriptors,
        policy=design.leakage_policy,
    )
    changed = replace(
        first_point,
        leakage_group_id=changed_groups[first_point.case_id],
    )
    changed_design = replace(
        design,
        selected_points=(changed, *design.selected_points[1:]),
        leakage_descriptors=changed_descriptors,
    )

    with pytest.raises(PreregistrationError, match="leakage_group_id"):
        _build(design=changed_design)


def test_supplied_allocation_must_replay_from_the_private_seed() -> None:
    values = _inputs()
    allocation = values["role_allocation"]
    assignments = dict(allocation.registry.assignments)
    pilot_id = next(
        case_id
        for case_id, assignment in assignments.items()
        if assignment.role is EvidenceRole.PILOT
    )
    train_id = next(
        case_id
        for case_id, assignment in assignments.items()
        if assignment.role is EvidenceRole.TRAIN
    )
    pilot = assignments[pilot_id]
    train = assignments[train_id]
    assignments[pilot_id] = replace(pilot, role=EvidenceRole.TRAIN)
    assignments[train_id] = replace(train, role=EvidenceRole.PILOT)
    group_roles = dict(allocation.registry.leakage_group_roles)
    group_roles[pilot.leakage_group_id] = EvidenceRole.TRAIN
    group_roles[train.leakage_group_id] = EvidenceRole.PILOT
    forged = replace(
        allocation,
        registry=CaseRoleRegistry(assignments, group_roles),
    )

    with pytest.raises(PreregistrationError, match="allocation replay"):
        _build(role_allocation=forged)


def test_campaign_secrets_must_come_from_csprng_bundle() -> None:
    with pytest.raises(PreregistrationError, match="campaign_secrets"):
        _build(campaign_secrets=object())


def test_preregistration_no_longer_accepts_raw_secret_keywords() -> None:
    values = _inputs()
    values.pop("campaign_secrets")

    with pytest.raises(TypeError, match="unexpected keyword"):
        build_preregistration(
            **values,
            allocation_seed=_ALLOCATION_SEED,
            holdout_salt=_HOLDOUT_SALT,
        )


def test_spec_seed_must_match_private_doe_secret() -> None:
    with pytest.raises(PreregistrationError, match="DOE seed"):
        _build(campaign_secrets=_campaign_secrets(doe_seed=bytes(range(96, 128))))


def test_allocation_seed_must_replay_from_private_secret() -> None:
    with pytest.raises(PreregistrationError, match="allocation replay"):
        _build(
            campaign_secrets=_campaign_secrets(allocation_seed=bytes(range(96, 128)))
        )


def test_reserves_must_exactly_match_policy_role_distribution_and_ranks() -> None:
    values = _inputs()
    policy = values["replacement_policy"]
    missing_policy = OrderedReservePolicy(
        rule_id=policy.rule_id,
        candidates=policy.candidates[:-1],
    )
    with pytest.raises(PreregistrationError, match="reserve policy membership"):
        _build(replacement_policy=missing_policy)

    train = policy.candidates_for_role(EvidenceRole.TRAIN)
    swapped = tuple(
        ReserveCandidate(
            case_id=item.case_id,
            leakage_group_id=item.leakage_group_id,
            role=item.role,
            rank=3 - item.rank,
        )
        for item in train
    )
    other = tuple(
        item for item in policy.candidates if item.role is not EvidenceRole.TRAIN
    )
    rank_mismatch = OrderedReservePolicy(
        rule_id=policy.rule_id,
        candidates=(*other, *swapped),
    )
    with pytest.raises(PreregistrationError, match="reserve policy rank"):
        _build(replacement_policy=rank_mismatch)

    wrong_role = tuple(
        ReserveCandidate(
            case_id=item.case_id,
            leakage_group_id=item.leakage_group_id,
            role=(EvidenceRole.TRAIN if item.role is EvidenceRole.PILOT else item.role),
            rank=item.rank,
        )
        for item in policy.candidates
    )
    with pytest.raises(ValueError, match="rank"):
        OrderedReservePolicy(rule_id=policy.rule_id, candidates=wrong_role)


def test_public_reserve_cannot_share_a_hidden_holdout_leakage_group() -> None:
    values = _inputs()
    design = values["design"]
    descriptor_by_id = {
        descriptor.case_id: descriptor for descriptor in design.leakage_descriptors
    }
    first_reserve = design.reserve_points[0]
    for active in design.selected_points:
        active_seed = descriptor_by_id[active.case_id].morph_seed
        changed_descriptors = tuple(
            replace(descriptor, morph_seed=active_seed)
            if descriptor.case_id == first_reserve.case_id
            else descriptor
            for descriptor in design.leakage_descriptors
        )
        changed_groups = bind_leakage_groups(
            changed_descriptors,
            policy=design.leakage_policy,
        )
        changed_selected = tuple(
            replace(point, leakage_group_id=changed_groups[point.case_id])
            for point in design.selected_points
        )
        changed_reserves = tuple(
            replace(point, leakage_group_id=changed_groups[point.case_id])
            for point in design.reserve_points
        )
        changed_design = replace(
            design,
            selected_points=changed_selected,
            reserve_points=changed_reserves,
            leakage_descriptors=changed_descriptors,
        )
        changed_allocation = allocate_evidence_roles(
            changed_design.selected_points,
            values["role_allocation"].budget,
            seed=values["campaign_secrets"].allocation_seed,
        )
        if (
            changed_allocation.registry.assignments[active.case_id].role
            is EvidenceRole.HOLDOUT
        ):
            break
    else:  # pragma: no cover - exact 64/67 HOLDOUT budget makes this unreachable
        raise AssertionError("no deterministic HOLDOUT case found")

    changed_policy = OrderedReservePolicy.preregister(
        rule_id="role-local-design-order-v1",
        reserves_by_role={
            EvidenceRole.PILOT: (changed_design.reserve_points[0],),
            EvidenceRole.TRAIN: changed_design.reserve_points[1:3],
            EvidenceRole.DEVELOPMENT: (changed_design.reserve_points[3],),
        },
    )

    with pytest.raises(PreregistrationError, match="HOLDOUT|leakage group"):
        _build(
            design=changed_design,
            role_allocation=changed_allocation,
            replacement_policy=changed_policy,
        )


def test_preregistration_replays_approved_leakage_policy_before_closure() -> None:
    values = _inputs()
    design = values["design"]
    first, second, *remaining = design.leakage_descriptors
    split_descriptors = (
        replace(first, morph_seed=777),
        replace(second, morph_seed=777),
        *remaining,
    )
    object.__setattr__(design, "leakage_descriptors", split_descriptors)

    with pytest.raises(PreregistrationError, match="leakage grouping"):
        _build(design=design)


def test_leakage_policy_is_bound_even_when_group_partition_is_unchanged() -> None:
    values = _inputs()
    baseline = build_preregistration(**values)
    changed_policy = LeakagePolicy(near_duplicate_tolerance=1e-6)
    changed_spec = replace(values["spec"], leakage_policy=changed_policy)
    changed_design = replace(values["design"], leakage_policy=changed_policy)

    changed = build_preregistration(
        **{
            **values,
            "spec": changed_spec,
            "design": changed_design,
        }
    )

    assert changed.binding.role_ledger_hash == baseline.binding.role_ledger_hash
    assert (
        changed.binding.private_doe_plan_hash != baseline.binding.private_doe_plan_hash
    )
    assert changed.campaign_hash != baseline.campaign_hash


def test_legacy_ledger_is_exact_unique_and_disjoint() -> None:
    with pytest.raises(PreregistrationError, match="LEGACY"):
        _build(legacy_ledger=LegacyCaseLedger(("legacy-0",)))
    with pytest.raises(ValueError, match="unique"):
        LegacyCaseLedger(("legacy", "legacy"))
    active_case_id = _inputs()["design"].selected_points[0].case_id
    with pytest.raises(PreregistrationError, match="overlap"):
        _build(legacy_ledger=LegacyCaseLedger((active_case_id, "legacy")))


def test_production_allocation_budget_must_match_campaign_budget() -> None:
    values = _inputs()
    allocation = values["role_allocation"]
    changed_budget = replace(values["budget"], pilot=0, train=2)

    with pytest.raises(PreregistrationError, match="allocation budget"):
        _build(budget=changed_budget, role_allocation=allocation)


@pytest.mark.parametrize(
    "seed",
    (
        "0123456789abcdefghijklmnopqrstuv",
        b"short",
        b"x" * 32,
        bytes(range(8)) * 4,
    ),
)
def test_seed_must_have_strong_byte_origin(seed: bytes | str) -> None:
    spec = replace(_inputs()["spec"], seed=seed)

    with pytest.raises(PreregistrationError, match="seed"):
        _build(spec=spec)


def test_exact64_holdout_coordinates_must_be_unique() -> None:
    values = _inputs()
    design = values["design"]
    holdout_ids = {
        case_id
        for case_id, assignment in values[
            "role_allocation"
        ].registry.assignments.items()
        if assignment.role is EvidenceRole.HOLDOUT
    }
    holdout_indices = [
        index
        for index, point in enumerate(design.selected_points)
        if point.case_id in holdout_ids
    ]
    source_index, target_index = holdout_indices[:2]
    duplicate = replace(
        design.selected_points[target_index],
        coordinates=design.selected_points[source_index].coordinates,
        derived_coordinates=design.selected_points[source_index].derived_coordinates,
    )
    changed_points = list(design.selected_points)
    changed_points[target_index] = duplicate
    duplicate_design = replace(
        design,
        selected_points=tuple(changed_points),
    )

    with pytest.raises(ValueError, match="duplicate holdout coordinates"):
        _build(design=duplicate_design)


@pytest.mark.parametrize(
    "point_change",
    (
        {"coordinates": {"other": 1.0}},
        {"derived_coordinates": {"other_ratio": 0.5}},
        {"strata": {"width": 0}},
    ),
)
def test_design_point_projection_must_match_declared_axes(
    point_change: dict[str, object],
) -> None:
    values = _inputs()
    design = values["design"]
    changed = replace(design.selected_points[0], **point_change)
    changed_design = replace(
        design,
        selected_points=(changed, *design.selected_points[1:]),
    )

    with pytest.raises(PreregistrationError, match="projection"):
        _build(design=changed_design)


def test_spec_and_design_population_counts_must_agree() -> None:
    values = _inputs()
    design = values["design"]
    short_design = replace(design, validation_points=())

    with pytest.raises(PreregistrationError, match="validation_count"):
        _build(design=short_design)


def test_family_must_be_canonically_hashable() -> None:
    with pytest.raises(ContractValidationError, match="unsupported canonical"):
        _build(compiled_family=object())


def test_preregistration_requires_a_typed_executable_envelope() -> None:
    with pytest.raises(PreregistrationError, match="ExecutableEnvelope"):
        _build(executable_envelope={"envelope_id": "hashable-but-not-executable"})


def test_envelope_must_bind_the_same_compiled_family_as_checkpoint() -> None:
    values = _inputs()

    with pytest.raises(PreregistrationError, match="compiled_family_hash"):
        _build(
            executable_envelope=_envelope(
                values["compiled_family"],
                compiled_family_hash="b" * 64,
            )
        )


def test_envelope_must_bind_the_compiled_family_membership() -> None:
    values = _inputs()
    other_family = _compiled_family(upper_width=255.0)
    forged = replace(
        values["executable_envelope"],
        compiled_family_hash=hash_spec(other_family, "content"),
    )

    with pytest.raises(PreregistrationError, match="geometry membership"):
        _build(compiled_family=other_family, executable_envelope=forged)


def test_preregistration_replays_positive_points_against_the_envelope() -> None:
    narrow_family = _compiled_family(upper_width=200.0)

    with pytest.raises(PreregistrationError, match="(selected|reserve|validation) point"):
        _build(
            compiled_family=narrow_family,
            executable_envelope=_envelope(narrow_family),
        )


def test_preregistration_rejects_material_and_load_omitted_from_the_doe() -> None:
    values = _inputs()
    omitted = frozenset({"elastic_modulus", "poisson_ratio", "prescribed_displacement"})

    def geometry_only(point: DesignPoint) -> DesignPoint:
        return replace(
            point,
            coordinates={
                name: value
                for name, value in point.coordinates.items()
                if name not in omitted
            },
            strata={
                name: value
                for name, value in point.strata.items()
                if name not in omitted
            },
        )

    design = values["design"]
    geometry_only_design = replace(
        design,
        selected_points=tuple(geometry_only(point) for point in design.selected_points),
        reserve_points=tuple(geometry_only(point) for point in design.reserve_points),
        validation_points=tuple(
            geometry_only(point) for point in design.validation_points
        ),
    )
    geometry_only_spec = replace(
        values["spec"],
        axes=tuple(axis for axis in values["spec"].axes if axis.name not in omitted),
    )

    with pytest.raises(PreregistrationError, match="primitive axes"):
        _build(spec=geometry_only_spec, design=geometry_only_design)


def _guarded_design(
    values: dict[str, Any],
    *,
    reason_codes: tuple[str, ...] = ("PARAMETER_OUT_OF_BOUNDS",),
) -> tuple[CoupledDesignSpec, CoupledDesign]:
    spec = replace(values["spec"], guard_band_count=1)
    design = generate_coupled_design(values["executable_envelope"], spec)
    if reason_codes != design.guard_band_rejections[0].reason_codes:
        changed_rejection = replace(
            design.guard_band_rejections[0],
            reason_codes=reason_codes,
        )
        design = replace(design, guard_band_rejections=(changed_rejection,))
    return spec, design


def test_preregistration_accepts_a_guard_with_partial_derived_coordinates() -> None:
    values = _inputs()
    spec, design = _guarded_design(values)

    result = _build(spec=spec, design=design)

    assert result.reveal_verified is True


def test_preregistration_replays_the_recorded_guard_rejection_reason() -> None:
    values = _inputs()
    spec, design = _guarded_design(values, reason_codes=("FORGED_REASON",))

    with pytest.raises(PreregistrationError, match="guard-band.*reason"):
        _build(spec=spec, design=design)


def test_preregistration_binds_guard_coordinates_to_rejection_evidence() -> None:
    values = _inputs()
    spec, design = _guarded_design(values)
    rejection = design.guard_band_rejections[0]
    changed_coordinates = {
        **rejection.coordinates,
        "elastic_modulus": 759.94,
    }
    forged = replace(rejection, coordinates=changed_coordinates)
    changed_design = replace(design, guard_band_rejections=(forged,))

    with pytest.raises(PreregistrationError, match="guard-band.*coordinates"):
        _build(spec=spec, design=changed_design)


def test_preregistration_replays_the_complete_doe_from_envelope_and_seed() -> None:
    values = _inputs()
    forged = replace(
        values["design"],
        selection_trace=("caller-substituted-selection",),
    )

    with pytest.raises(PreregistrationError, match="complete DOE replay"):
        _build(design=forged)


def test_recomputation_is_byte_deterministic() -> None:
    first = _build()
    second = _build()

    assert first == second
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first.campaign_hash == second.campaign_hash
    assert first.holdout_commitment.commitment == second.holdout_commitment.commitment
