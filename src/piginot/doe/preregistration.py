"""Pure, fail-closed construction of the secret-free S9 preregistration."""

from __future__ import annotations

import base64
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from piginot.contracts import CanonicalQuantity, canonical_json_bytes, hash_spec
from piginot.doe.allocation import (
    ProductionRoleBudget,
    RoleAllocation,
    RoleAllocationError,
    allocate_evidence_roles,
)
from piginot.doe.checkpoint import (
    CampaignBinding,
    CampaignBudget,
    CPEnvelope,
    EvidenceReference,
    compute_campaign_hash,
)
from piginot.doe.commitment import (
    HoldoutCommitment,
    compute_holdout_commitment,
    verify_holdout_reveal,
)
from piginot.doe.design import (
    AxisSource,
    CoupledDesign,
    CoupledDesignSpec,
    DesignPoint,
    DoeSeed,
    RejectedProposal,
    SeedEncoding,
    generate_coupled_design,
)
from piginot.doe.envelope import ExecutableEnvelope
from piginot.doe.leakage import LeakageDescriptor, bind_leakage_groups
from piginot.doe.publication import (
    PublishedCase,
    PublishedCaseManifest,
    PublishedCaseStatus,
)
from piginot.doe.replacement import OrderedReservePolicy, ReserveCandidate
from piginot.doe.roles import EvidenceRole
from piginot.doe.secret_material import (
    CampaignSecrets,
    SecretMaterialError,
    validate_secret_bytes,
)
from piginot.morphing.membership import FamilyMembershipPredicate

_CANONICAL_NAMESPACE = "piginot.doe"
_REPLACEABLE_ROLES = (
    EvidenceRole.PILOT,
    EvidenceRole.TRAIN,
    EvidenceRole.DEVELOPMENT,
)
_PRODUCTION_ROLES = (*_REPLACEABLE_ROLES, EvidenceRole.HOLDOUT)


class PreregistrationError(ValueError):
    """A supplied S9 campaign cannot be closed without changing its meaning."""


def _case_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("LEGACY case_id must be non-empty text")
    if ":" in value:
        raise ValueError("LEGACY case_id must identify a case-level sample")
    return value


@dataclass(frozen=True, slots=True)
class LegacyCaseLedger:
    """Coordinate-free identities for the already-sealed LEGACY population."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        case_ids = tuple(_case_id(case_id) for case_id in self.case_ids)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("LEGACY case IDs must be unique")
        object.__setattr__(self, "case_ids", tuple(sorted(case_ids)))


@dataclass(frozen=True, slots=True)
class PreregistrationResult:
    """Immutable public closure; it contains commitments but no reveal material."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    production_allocation: RoleAllocation
    legacy_ledger: LegacyCaseLedger
    public_manifest: PublishedCaseManifest
    binding: CampaignBinding
    campaign_hash: str
    holdout_commitment: HoldoutCommitment
    checkpoint: CPEnvelope
    reveal_verified: bool

    def __post_init__(self) -> None:
        if not isinstance(self.production_allocation, RoleAllocation):
            raise ValueError("production_allocation must be a RoleAllocation")
        if not isinstance(self.legacy_ledger, LegacyCaseLedger):
            raise ValueError("legacy_ledger must be a LegacyCaseLedger")
        if not isinstance(self.public_manifest, PublishedCaseManifest):
            raise ValueError("public_manifest must be a PublishedCaseManifest")
        if not isinstance(self.binding, CampaignBinding):
            raise ValueError("binding must be a CampaignBinding")
        if not isinstance(self.holdout_commitment, HoldoutCommitment):
            raise ValueError("holdout_commitment must be a HoldoutCommitment")
        if not isinstance(self.checkpoint, CPEnvelope):
            raise ValueError("checkpoint must be a CPEnvelope")
        if self.reveal_verified is not True:
            raise ValueError("the private holdout reveal must verify before closure")
        if self.binding.role_ledger_hash != _role_ledger_hash(
            self.production_allocation,
            self.legacy_ledger,
        ):
            raise ValueError("binding does not match the public role ledgers")
        if self.binding.public_case_manifest_hash != self.public_manifest.content_hash:
            raise ValueError("binding does not match the public case manifest")
        if self.campaign_hash != compute_campaign_hash(self.binding):
            raise ValueError("campaign_hash does not match the campaign binding")
        if self.checkpoint.binding != self.binding:
            raise ValueError("checkpoint does not match the campaign binding")
        if self.checkpoint.campaign_hash != self.campaign_hash:
            raise ValueError("checkpoint does not match the campaign hash")
        if self.checkpoint.holdout_commitment != self.holdout_commitment:
            raise ValueError("checkpoint does not match the holdout commitment")


def _decode_and_validate_seed(seed: DoeSeed) -> bytes:
    if seed.encoding is not SeedEncoding.BYTES_BASE64URL_V1:
        raise PreregistrationError("DOE seed must have byte origin, not text origin")
    raw = base64.urlsafe_b64decode(seed.payload + "=" * (-len(seed.payload) % 4))
    try:
        return validate_secret_bytes(raw, "DOE seed")
    except SecretMaterialError as exc:
        raise PreregistrationError(str(exc)) from exc


def _axis_schemas(
    spec: CoupledDesignSpec,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    primitive = tuple(
        axis.name for axis in spec.axes if axis.source is AxisSource.PRIMITIVE
    )
    derived = tuple(
        axis.name for axis in spec.axes if axis.source is AxisSource.DERIVED
    )
    strata = tuple(axis.name for axis in spec.axes)
    return primitive, derived, strata


def _validate_projection(
    point: DesignPoint,
    *,
    primitive_axes: tuple[str, ...],
    derived_axes: tuple[str, ...],
    stratum_axes: tuple[str, ...],
    collection: str,
    allow_partial_derived: bool = False,
) -> None:
    if set(point.coordinates) != set(primitive_axes):
        raise PreregistrationError(
            f"{collection} point {point.case_id!r} primitive projection mismatch"
        )
    observed_derived = set(point.derived_coordinates)
    expected_derived = set(derived_axes)
    derived_mismatch = (
        not observed_derived <= expected_derived
        if allow_partial_derived
        else observed_derived != expected_derived
    )
    if derived_mismatch:
        raise PreregistrationError(
            f"{collection} point {point.case_id!r} derived projection mismatch"
        )
    if set(point.strata) != set(stratum_axes):
        raise PreregistrationError(
            f"{collection} point {point.case_id!r} strata projection mismatch"
        )


def _validate_design_shape(
    spec: CoupledDesignSpec,
    design: CoupledDesign,
    budget: CampaignBudget,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    expected_counts = (
        ("sample_count", len(design.selected_points), spec.sample_count),
        ("reserve_count", len(design.reserve_points), spec.reserve_count),
        ("validation_count", len(design.validation_points), spec.validation_count),
        ("guard_band_count", len(design.guard_band_points), spec.guard_band_count),
    )
    for label, observed, expected in expected_counts:
        if observed != expected:
            raise PreregistrationError(
                f"design {label} mismatch: observed {observed}, expected {expected}"
            )
    if spec.sample_count != budget.production_cases:
        raise PreregistrationError(
            "DOE sample_count must equal the fixed production case budget"
        )
    if spec.reserve_count != budget.reserve:
        raise PreregistrationError(
            "DOE reserve_count must equal the fixed reserve budget"
        )
    _validate_leakage_grouping(spec, design)

    primitive, derived, strata = _axis_schemas(spec)
    for collection, points, expected_guard in (
        ("selected", design.selected_points, False),
        ("reserve", design.reserve_points, False),
        ("validation", design.validation_points, False),
        ("guard-band", design.guard_band_points, True),
    ):
        for point in points:
            if point.guard_band is not expected_guard:
                raise PreregistrationError(
                    f"{collection} point {point.case_id!r} has an invalid guard flag"
                )
            _validate_projection(
                point,
                primitive_axes=primitive,
                derived_axes=derived,
                stratum_axes=strata,
                collection=collection,
                allow_partial_derived=expected_guard,
            )

    campaign_points = (*design.selected_points, *design.reserve_points)
    case_ids = tuple(point.case_id for point in campaign_points)
    if len(case_ids) != len(set(case_ids)):
        raise PreregistrationError(
            "selected and reserve case IDs must be globally unique"
        )
    return primitive, derived, strata


def _theta_from_point(
    envelope: ExecutableEnvelope,
    point: DesignPoint,
) -> dict[str, CanonicalQuantity]:
    quantities: dict[str, CanonicalQuantity] = {}
    for bound in envelope.primitive_bounds:
        if bound.name not in point.coordinates:
            raise PreregistrationError(
                f"point {point.case_id!r} omits envelope coordinate {bound.name!r}"
            )
        quantities[bound.name] = CanonicalQuantity(
            point.coordinates[bound.name],
            bound.unit,
            bound.dimension,
        )
    return quantities


def _validate_positive_point_against_envelope(
    envelope: ExecutableEnvelope,
    point: DesignPoint,
    *,
    collection: str,
) -> None:
    decision = envelope.evaluate(_theta_from_point(envelope, point))
    if not decision.accepted:
        raise PreregistrationError(
            f"{collection} point {point.case_id!r} is not accepted by ExecutableEnvelope"
        )
    observed = {
        name: quantity.value for name, quantity in decision.derived_values.items()
    }
    if observed != dict(point.derived_coordinates):
        raise PreregistrationError(
            f"{collection} point {point.case_id!r} derived coordinates do not replay"
        )


def _validate_guard_point_against_envelope(
    envelope: ExecutableEnvelope,
    point: DesignPoint,
    recorded_rejection: RejectedProposal,
    *,
    collection: str,
) -> None:
    if dict(recorded_rejection.coordinates) != dict(point.coordinates):
        raise PreregistrationError(
            f"{collection} point {point.case_id!r} coordinates differ from rejection evidence"
        )
    decision = envelope.evaluate(_theta_from_point(envelope, point))
    if decision.accepted:
        raise PreregistrationError(
            f"{collection} point {point.case_id!r} is accepted by ExecutableEnvelope"
        )
    replayed_reasons = tuple(
        getattr(reason, "value", reason) for reason in decision.reason_codes
    )
    if replayed_reasons != recorded_rejection.reason_codes:
        raise PreregistrationError(
            f"{collection} point {point.case_id!r} rejection reasons do not replay"
        )
    replayed_derived = {
        name: quantity.value for name, quantity in decision.derived_values.items()
    }
    if replayed_derived != dict(point.derived_coordinates):
        raise PreregistrationError(
            f"{collection} point {point.case_id!r} derived coordinates do not replay"
        )


def _validate_envelope_replay(
    envelope: ExecutableEnvelope,
    spec: CoupledDesignSpec,
    design: CoupledDesign,
) -> None:
    primitive_names = {bound.name for bound in envelope.primitive_bounds}
    derived_names = {bound.name for bound in envelope.derived_bounds}
    spec_primitive_names = {
        axis.name for axis in spec.axes if axis.source is AxisSource.PRIMITIVE
    }
    spec_derived_names = {
        axis.name for axis in spec.axes if axis.source is AxisSource.DERIVED
    }
    if primitive_names != spec_primitive_names:
        raise PreregistrationError("DOE primitive axes do not match ExecutableEnvelope")
    if derived_names != spec_derived_names:
        raise PreregistrationError("DOE derived axes do not match ExecutableEnvelope")
    for collection, points in (
        ("selected", design.selected_points),
        ("reserve", design.reserve_points),
        ("validation", design.validation_points),
    ):
        for point in points:
            _validate_positive_point_against_envelope(
                envelope,
                point,
                collection=collection,
            )
    for point, rejection in zip(
        design.guard_band_points,
        design.guard_band_rejections,
        strict=True,
    ):
        _validate_guard_point_against_envelope(
            envelope,
            point,
            rejection,
            collection="guard-band",
        )


def _validate_complete_doe_replay(
    envelope: ExecutableEnvelope,
    spec: CoupledDesignSpec,
    design: CoupledDesign,
) -> None:
    try:
        replayed = generate_coupled_design(envelope, spec)
    except ValueError as exc:
        raise PreregistrationError(f"complete DOE replay failed: {exc}") from exc
    if canonical_json_bytes(replayed) != canonical_json_bytes(design):
        raise PreregistrationError(
            "complete DOE replay from ExecutableEnvelope and secret seed differs"
        )


def _validate_leakage_grouping(
    spec: CoupledDesignSpec,
    design: CoupledDesign,
) -> None:
    if design.leakage_policy != spec.leakage_policy:
        raise PreregistrationError("design leakage policy does not match DOE spec")
    campaign_points = (*design.selected_points, *design.reserve_points)
    if any(
        type(descriptor) is not LeakageDescriptor
        for descriptor in design.leakage_descriptors
    ):
        raise PreregistrationError(
            "leakage descriptors must contain LeakageDescriptor values"
        )
    expected_case_ids = {point.case_id for point in campaign_points}
    descriptor_case_ids = {
        descriptor.case_id for descriptor in design.leakage_descriptors
    }
    if descriptor_case_ids != expected_case_ids:
        raise PreregistrationError(
            "leakage descriptors must match selected and reserve cases"
        )
    try:
        groups = bind_leakage_groups(
            design.leakage_descriptors,
            policy=spec.leakage_policy,
        )
    except ValueError as exc:
        raise PreregistrationError(f"invalid leakage grouping: {exc}") from exc
    for point in campaign_points:
        if point.leakage_group_id != groups[point.case_id]:
            raise PreregistrationError(
                f"case {point.case_id!r} does not match the approved leakage grouping"
            )


def _validate_campaign_budget_shape(budget: CampaignBudget) -> None:
    if budget.pilot <= 0:
        raise PreregistrationError("production PILOT budget must be positive")
    if budget.train <= 0:
        raise PreregistrationError("production TRAIN budget must be positive")
    if budget.development <= 0:
        raise PreregistrationError("production DEVELOPMENT budget must be positive")
    if budget.holdout != 64:
        raise PreregistrationError("production HOLDOUT budget must equal exactly 64")


def _expected_production_budget(budget: CampaignBudget) -> ProductionRoleBudget:
    return ProductionRoleBudget(
        pilot=budget.pilot,
        train=budget.train,
        development=budget.development,
        holdout=budget.holdout,
    )


def _validate_allocation_budget_consistency(
    spec: CoupledDesignSpec,
    role_allocation: RoleAllocation,
    budget: CampaignBudget,
) -> None:
    if budget.production_cases != spec.sample_count:
        return
    expected = _expected_production_budget(budget)
    if role_allocation.budget != expected:
        differing = ", ".join(
            role.value
            for role in _PRODUCTION_ROLES
            if role_allocation.budget.count_for(role) != expected.count_for(role)
        )
        raise PreregistrationError(
            "production allocation budget does not match CampaignBudget"
            f" for {differing}"
        )


def _validate_active_allocation(
    design: CoupledDesign,
    role_allocation: RoleAllocation,
    budget: CampaignBudget,
) -> dict[str, EvidenceRole]:
    expected_budget = _expected_production_budget(budget)
    if role_allocation.budget != expected_budget:
        raise PreregistrationError(
            "production allocation budget does not match CampaignBudget"
        )
    registry = role_allocation.registry
    selected_ids = {point.case_id for point in design.selected_points}
    registry_ids = set(registry.assignments)
    if registry_ids != selected_ids:
        if registry_ids - selected_ids:
            raise PreregistrationError(
                "production registry contains an unexpected active or reserve case"
            )
        raise PreregistrationError(
            "selected case population does not match the immutable registry"
        )

    roles: dict[str, EvidenceRole] = {}
    for point in design.selected_points:
        assignment = registry.assignments[point.case_id]
        if assignment.leakage_group_id != point.leakage_group_id:
            raise PreregistrationError(
                f"case {point.case_id!r} leakage_group_id differs from its registry"
            )
        if assignment.role is EvidenceRole.LEGACY:
            raise PreregistrationError(
                "selected production cases cannot be active LEGACY"
            )
        roles[point.case_id] = assignment.role

    observed = Counter(roles.values())
    for role in _PRODUCTION_ROLES:
        if observed[role] != expected_budget.count_for(role):
            raise PreregistrationError(
                f"selected {role.value} count does not match CampaignBudget"
            )
    return roles


def _verify_allocation_replay(
    design: CoupledDesign,
    role_allocation: RoleAllocation,
    budget: CampaignBudget,
    allocation_seed: bytes,
) -> None:
    try:
        replay = allocate_evidence_roles(
            design.selected_points,
            _expected_production_budget(budget),
            seed=allocation_seed,
        )
    except RoleAllocationError as exc:
        raise PreregistrationError(f"allocation seed is invalid: {exc}") from exc
    if replay.to_bytes() != role_allocation.to_bytes():
        raise PreregistrationError(
            "secret-seeded allocation replay does not match the supplied allocation"
        )


def _validate_reserve_policy(
    design: CoupledDesign,
    replacement_policy: OrderedReservePolicy,
    budget: CampaignBudget,
) -> dict[str, ReserveCandidate]:
    reserve_ids = {point.case_id for point in design.reserve_points}
    candidates = {
        candidate.case_id: candidate for candidate in replacement_policy.candidates
    }
    if (
        len(replacement_policy.candidates) != budget.reserve
        or set(candidates) != reserve_ids
    ):
        raise PreregistrationError(
            "reserve policy membership must exactly match the reserve budget and design"
        )

    point_by_id = {point.case_id: point for point in design.reserve_points}
    for candidate in replacement_policy.candidates:
        point = point_by_id[candidate.case_id]
        if candidate.leakage_group_id != point.leakage_group_id:
            raise PreregistrationError(
                f"reserve {point.case_id!r} leakage_group_id differs from its policy"
            )
        if candidate.role not in _REPLACEABLE_ROLES:
            raise PreregistrationError(
                f"reserve {point.case_id!r} cannot use {candidate.role.value}"
            )

    for role in _REPLACEABLE_ROLES:
        points = tuple(
            point
            for point in design.reserve_points
            if candidates[point.case_id].role is role
        )
        observed = tuple(
            (candidate.case_id, candidate.rank)
            for candidate in replacement_policy.candidates_for_role(role)
        )
        expected = tuple(
            (point.case_id, rank) for rank, point in enumerate(points, start=1)
        )
        if observed != expected:
            raise PreregistrationError(
                f"reserve policy rank order differs from design order for {role.value}"
            )
    return candidates


def _validate_reserve_leakage(
    design: CoupledDesign,
    active_roles: dict[str, EvidenceRole],
    reserve_candidates: dict[str, ReserveCandidate],
) -> None:
    group_roles = {
        point.leakage_group_id: active_roles[point.case_id]
        for point in design.selected_points
    }
    for point in design.reserve_points:
        candidate = reserve_candidates[point.case_id]
        existing = group_roles.get(point.leakage_group_id)
        if existing is not None and existing is not candidate.role:
            if existing is EvidenceRole.HOLDOUT:
                raise PreregistrationError(
                    f"reserve {point.case_id!r} shares a hidden HOLDOUT leakage group"
                )
            raise PreregistrationError(
                f"reserve {point.case_id!r} crosses an incompatible leakage group"
            )
        group_roles[point.leakage_group_id] = candidate.role


def _validate_legacy(
    legacy_ledger: LegacyCaseLedger,
    budget: CampaignBudget,
    design: CoupledDesign,
) -> None:
    if len(legacy_ledger.case_ids) != budget.legacy:
        raise PreregistrationError(
            "LEGACY ledger count must exactly match CampaignBudget"
        )
    campaign_ids = {
        point.case_id for point in (*design.selected_points, *design.reserve_points)
    }
    if campaign_ids & set(legacy_ledger.case_ids):
        raise PreregistrationError(
            "LEGACY case IDs cannot overlap active or reserve case IDs"
        )


def _published_case(
    point: DesignPoint,
    *,
    role: EvidenceRole,
    status: PublishedCaseStatus,
) -> PublishedCase:
    return PublishedCase(
        case_id=point.case_id,
        leakage_group_id=point.leakage_group_id,
        role=role,
        primitive_coordinates=point.coordinates,
        derived_coordinates=point.derived_coordinates,
        strata=point.strata,
        status=status,
    )


def _public_manifest(
    *,
    manifest_id: str,
    design: CoupledDesign,
    active_roles: dict[str, EvidenceRole],
    reserve_candidates: dict[str, ReserveCandidate],
    primitive_axes: tuple[str, ...],
    derived_axes: tuple[str, ...],
    stratum_axes: tuple[str, ...],
) -> PublishedCaseManifest:
    active = tuple(
        _published_case(
            point,
            role=active_roles[point.case_id],
            status=PublishedCaseStatus.ACTIVE,
        )
        for point in design.selected_points
        if active_roles[point.case_id] is not EvidenceRole.HOLDOUT
    )
    reserves = tuple(
        _published_case(
            point,
            role=reserve_candidates[point.case_id].role,
            status=PublishedCaseStatus.RESERVE,
        )
        for point in design.reserve_points
    )
    return PublishedCaseManifest(
        manifest_id=manifest_id,
        primitive_axes=primitive_axes,
        derived_axes=derived_axes,
        stratum_axes=stratum_axes,
        cases=(*active, *reserves),
    )


def _role_ledger_hash(
    production_allocation: RoleAllocation,
    legacy_ledger: LegacyCaseLedger,
) -> str:
    return hash_spec(
        {
            "legacy_ledger": legacy_ledger,
            "production_allocation": production_allocation,
        },
        "content",
    )


def build_preregistration(
    *,
    compiled_family: object,
    executable_envelope: object,
    pilot_evidence: Sequence[EvidenceReference],
    spec: CoupledDesignSpec,
    design: CoupledDesign,
    role_allocation: RoleAllocation,
    legacy_ledger: LegacyCaseLedger,
    replacement_policy: OrderedReservePolicy,
    budget: CampaignBudget,
    campaign_secrets: CampaignSecrets,
    created_utc: str,
    manifest_id: str = "s9-public-cases",
) -> PreregistrationResult:
    """Validate and close one S9 campaign without retaining private preimages."""

    if not isinstance(spec, CoupledDesignSpec):
        raise PreregistrationError("spec must be a CoupledDesignSpec")
    if not isinstance(design, CoupledDesign):
        raise PreregistrationError("design must be a CoupledDesign")
    if type(executable_envelope) is not ExecutableEnvelope:
        raise PreregistrationError("executable_envelope must be an ExecutableEnvelope")
    if not isinstance(role_allocation, RoleAllocation):
        raise PreregistrationError("role_allocation must be a RoleAllocation")
    if not isinstance(legacy_ledger, LegacyCaseLedger):
        raise PreregistrationError("legacy_ledger must be a LegacyCaseLedger")
    if not isinstance(replacement_policy, OrderedReservePolicy):
        raise PreregistrationError("replacement_policy must be an OrderedReservePolicy")
    if not isinstance(budget, CampaignBudget):
        raise PreregistrationError("budget must be a CampaignBudget")
    if type(campaign_secrets) is not CampaignSecrets:
        raise PreregistrationError(
            "campaign_secrets must come from generate_campaign_secrets()"
        )

    compiled_family_hash = hash_spec(compiled_family, "content")
    if executable_envelope.compiled_family_hash != compiled_family_hash:
        raise PreregistrationError(
            "ExecutableEnvelope compiled_family_hash does not match compiled_family"
        )
    compiled_membership = getattr(compiled_family, "membership", None)
    if type(compiled_membership) is not FamilyMembershipPredicate:
        raise PreregistrationError(
            "compiled_family must expose an exact FamilyMembershipPredicate"
        )
    if (
        hash_spec(compiled_membership, "content")
        != executable_envelope.geometry_membership_hash
    ):
        raise PreregistrationError(
            "ExecutableEnvelope geometry membership does not match compiled_family"
        )

    seed = spec.seed
    if not isinstance(seed, DoeSeed):
        raise PreregistrationError("spec seed was not canonicalized")
    doe_seed = _decode_and_validate_seed(seed)
    if doe_seed != campaign_secrets.doe_seed:
        raise PreregistrationError(
            "DOE seed must match the generated campaign secret bundle"
        )
    if (
        len(
            {
                campaign_secrets.doe_seed,
                campaign_secrets.allocation_seed,
                campaign_secrets.holdout_salt,
            }
        )
        != 3
    ):
        raise PreregistrationError(
            "DOE seed, allocation seed, and holdout salt require independent secret material"
        )
    if budget.holdout != 64:
        raise PreregistrationError("production HOLDOUT budget must equal exactly 64")
    _validate_allocation_budget_consistency(spec, role_allocation, budget)
    _validate_campaign_budget_shape(budget)
    primitive_axes, derived_axes, stratum_axes = _validate_design_shape(
        spec,
        design,
        budget,
    )
    _validate_envelope_replay(executable_envelope, spec, design)
    active_roles = _validate_active_allocation(design, role_allocation, budget)
    _verify_allocation_replay(
        design,
        role_allocation,
        budget,
        campaign_secrets.allocation_seed,
    )
    reserve_candidates = _validate_reserve_policy(
        design,
        replacement_policy,
        budget,
    )
    _validate_reserve_leakage(design, active_roles, reserve_candidates)
    _validate_legacy(legacy_ledger, budget, design)
    public_manifest = _public_manifest(
        manifest_id=manifest_id,
        design=design,
        active_roles=active_roles,
        reserve_candidates=reserve_candidates,
        primitive_axes=primitive_axes,
        derived_axes=derived_axes,
        stratum_axes=stratum_axes,
    )

    binding = CampaignBinding(
        compiled_family_hash=compiled_family_hash,
        pilot_evidence=tuple(pilot_evidence),
        executable_envelope_hash=hash_spec(executable_envelope, "content"),
        private_doe_plan_hash=hash_spec(
            {"design": design, "spec": spec},
            "content",
        ),
        role_ledger_hash=_role_ledger_hash(role_allocation, legacy_ledger),
        replacement_policy_hash=hash_spec(replacement_policy, "content"),
        public_case_manifest_hash=public_manifest.content_hash,
        doe_seed_commitment=hash_spec(seed, "content"),
        secret_generator_id=campaign_secrets.generator_id,
        budget=budget,
        derived_axes=derived_axes,
    )
    campaign_hash = compute_campaign_hash(binding)
    holdout_coordinates = tuple(
        point.all_coordinates
        for point in design.selected_points
        if active_roles[point.case_id] is EvidenceRole.HOLDOUT
    )
    commitment = compute_holdout_commitment(
        campaign_hash=campaign_hash,
        campaign_secrets=campaign_secrets,
        holdout_coordinates=holdout_coordinates,
        created_utc=created_utc,
    )
    reveal_verified = verify_holdout_reveal(
        commitment,
        salt=campaign_secrets.holdout_salt,
        holdout_coordinates=holdout_coordinates,
    )
    if not reveal_verified:
        raise PreregistrationError("computed holdout reveal did not verify")
    _validate_complete_doe_replay(executable_envelope, spec, design)
    checkpoint = CPEnvelope(
        campaign_hash=campaign_hash,
        compiled_family_hash=binding.compiled_family_hash,
        pilot_evidence=binding.pilot_evidence,
        executable_envelope_hash=binding.executable_envelope_hash,
        private_doe_plan_hash=binding.private_doe_plan_hash,
        role_ledger_hash=binding.role_ledger_hash,
        replacement_policy_hash=binding.replacement_policy_hash,
        public_case_manifest_hash=binding.public_case_manifest_hash,
        doe_seed_commitment=binding.doe_seed_commitment,
        secret_generator_id=binding.secret_generator_id,
        holdout_commitment=commitment,
        budget=binding.budget,
        derived_axes=binding.derived_axes,
        created_utc=created_utc,
    )
    return PreregistrationResult(
        production_allocation=role_allocation,
        legacy_ledger=legacy_ledger,
        public_manifest=public_manifest,
        binding=binding,
        campaign_hash=campaign_hash,
        holdout_commitment=commitment,
        checkpoint=checkpoint,
        reveal_verified=reveal_verified,
    )


__all__ = (
    "LegacyCaseLedger",
    "PreregistrationError",
    "PreregistrationResult",
    "build_preregistration",
)
