"""Immutable S9 evidence-role assignment and node-split validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, TypeVar


class EvidenceRole(StrEnum):
    """The five mutually exclusive evidence roles fixed by S9."""

    PILOT = "PILOT"
    TRAIN = "TRAIN"
    DEVELOPMENT = "DEVELOPMENT"
    HOLDOUT = "HOLDOUT"
    LEGACY = "LEGACY"


Role = EvidenceRole


class RoleAssignmentError(ValueError):
    """Base class for invalid evidence-role operations."""


class ImmutableRoleAssignmentError(RoleAssignmentError):
    """A case-level assignment was written more than once."""


class LeakageGroupRoleError(RoleAssignmentError):
    """One correlated leakage group was split or misidentified."""


class MissingRoleAssignmentError(RoleAssignmentError):
    """A downstream operation preceded immutable role assignment."""


class RandomNodeSplitError(RoleAssignmentError):
    """Nodes belonging to one case were assigned different roles."""


class NodeRoleMismatchError(RoleAssignmentError):
    """A node-level role disagreed with its case-level ledger role."""


class DesignCase(Protocol):
    """Structural identity required from a DOE case."""

    @property
    def case_id(self) -> str: ...

    @property
    def leakage_group_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class RoleAssignment:
    """One immutable case-level evidence-role assignment."""

    case_id: str
    leakage_group_id: str
    role: EvidenceRole

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _case_id(self.case_id))
        object.__setattr__(
            self,
            "leakage_group_id",
            _identifier(self.leakage_group_id, "leakage_group_id"),
        )
        object.__setattr__(self, "role", EvidenceRole(self.role))


@dataclass(frozen=True, slots=True)
class ProductionLabelAuthorization:
    """Evidence that role assignment existed before label production."""

    case_id: str
    leakage_group_id: str
    role: EvidenceRole


@dataclass(frozen=True, slots=True)
class NodeObservation:
    """The role attached to one node-level observation."""

    case_id: str
    node_id: str
    role: EvidenceRole

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _case_id(self.case_id))
        object.__setattr__(self, "node_id", _identifier(self.node_id, "node_id"))
        object.__setattr__(self, "role", EvidenceRole(self.role))


_Value = TypeVar("_Value")


def _frozen_mapping(values: Mapping[str, _Value]) -> Mapping[str, _Value]:
    return MappingProxyType(dict(sorted(values.items())))


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _case_id(value: object) -> str:
    case_id = _identifier(value, "case_id")
    if ":" in case_id:
        raise ValueError("case_id must identify a case-level sample")
    return case_id


def _case_identity(case: DesignCase) -> tuple[str, str]:
    return (
        _case_id(case.case_id),
        _identifier(case.leakage_group_id, "leakage_group_id"),
    )


@dataclass(frozen=True, slots=True)
class CaseRoleRegistry:
    """Persistent registry enforcing one case and leakage group per role."""

    assignments: Mapping[str, RoleAssignment] = field(default_factory=dict)
    leakage_group_roles: Mapping[str, EvidenceRole] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.assignments, Mapping):
            raise ValueError("assignments must be a mapping")
        if not isinstance(self.leakage_group_roles, Mapping):
            raise ValueError("leakage_group_roles must be a mapping")

        assignments = dict(self.assignments)
        group_roles = {
            _identifier(group_id, "leakage_group_id"): EvidenceRole(role)
            for group_id, role in self.leakage_group_roles.items()
        }
        observed_groups: dict[str, EvidenceRole] = {}
        for key, assignment in assignments.items():
            if not isinstance(assignment, RoleAssignment):
                raise ValueError("assignments must contain RoleAssignment values")
            if key != assignment.case_id:
                raise ValueError("assignment key must equal its case_id")
            prior = observed_groups.setdefault(
                assignment.leakage_group_id,
                assignment.role,
            )
            if prior is not assignment.role:
                raise LeakageGroupRoleError(
                    "leakage_group_id cannot span multiple roles"
                )
            declared = group_roles.setdefault(
                assignment.leakage_group_id,
                assignment.role,
            )
            if declared is not assignment.role:
                raise LeakageGroupRoleError(
                    "leakage_group_id cannot span multiple roles"
                )
        if set(group_roles) != set(observed_groups):
            raise ValueError("leakage_group_roles contains an unassigned group")

        object.__setattr__(self, "assignments", _frozen_mapping(assignments))
        object.__setattr__(
            self,
            "leakage_group_roles",
            _frozen_mapping(group_roles),
        )

    @classmethod
    def empty(cls) -> CaseRoleRegistry:
        """Return a registry with no case or group assignments."""

        return cls()

    def assign(
        self,
        case: DesignCase,
        role: EvidenceRole,
    ) -> CaseRoleRegistry:
        """Return a new registry with one previously unseen case assigned."""

        case_id, group_id = _case_identity(case)
        resolved_role = EvidenceRole(role)
        if case_id in self.assignments:
            raise ImmutableRoleAssignmentError(
                f"role assignment for {case_id!r} is immutable"
            )
        group_role = self.leakage_group_roles.get(group_id)
        if group_role is not None and group_role is not resolved_role:
            raise LeakageGroupRoleError(
                f"leakage_group_id {group_id!r} cannot span multiple roles"
            )

        assignments = dict(self.assignments)
        assignments[case_id] = RoleAssignment(case_id, group_id, resolved_role)
        group_roles = dict(self.leakage_group_roles)
        group_roles[group_id] = resolved_role
        return CaseRoleRegistry(assignments, group_roles)

    def assign_group(
        self,
        cases: Sequence[DesignCase],
        role: EvidenceRole,
    ) -> CaseRoleRegistry:
        """Atomically assign every member of one correlated leakage group."""

        frozen_cases = tuple(cases)
        if not frozen_cases:
            raise LeakageGroupRoleError(
                "a leakage group must contain at least one case"
            )
        identities = tuple(_case_identity(case) for case in frozen_cases)
        group_ids = {group_id for _, group_id in identities}
        if len(group_ids) != 1:
            raise LeakageGroupRoleError(
                "assign_group accepts exactly one leakage_group_id"
            )
        case_ids = tuple(case_id for case_id, _ in identities)
        if len(case_ids) != len(set(case_ids)):
            raise ImmutableRoleAssignmentError(
                "a case appears more than once in the group assignment"
            )
        existing = tuple(case_id for case_id in case_ids if case_id in self.assignments)
        if existing:
            raise ImmutableRoleAssignmentError(
                f"role assignment for {existing[0]!r} is immutable"
            )

        resolved_role = EvidenceRole(role)
        group_id = next(iter(group_ids))
        group_role = self.leakage_group_roles.get(group_id)
        if group_role is not None and group_role is not resolved_role:
            raise LeakageGroupRoleError(
                f"leakage_group_id {group_id!r} cannot span multiple roles"
            )

        assignments = dict(self.assignments)
        assignments.update(
            {
                case_id: RoleAssignment(case_id, group_id, resolved_role)
                for case_id in case_ids
            }
        )
        group_roles = dict(self.leakage_group_roles)
        group_roles[group_id] = resolved_role
        return CaseRoleRegistry(assignments, group_roles)

    def role_for_case(self, case_id: str) -> EvidenceRole:
        """Return the immutable role of a case or fail closed if absent."""

        resolved_id = _case_id(case_id)
        try:
            return self.assignments[resolved_id].role
        except KeyError as exc:
            raise MissingRoleAssignmentError(
                f"case {resolved_id!r} has no role assignment"
            ) from exc

    def authorize_production_label(
        self,
        case: DesignCase,
    ) -> ProductionLabelAuthorization:
        """Authorize label production only after exact case assignment exists."""

        case_id, group_id = _case_identity(case)
        try:
            assignment = self.assignments[case_id]
        except KeyError as exc:
            raise MissingRoleAssignmentError(
                f"assign case {case_id!r} before production labels are created"
            ) from exc
        if assignment.leakage_group_id != group_id:
            raise LeakageGroupRoleError(
                f"case {case_id!r} leakage_group_id does not match its ledger"
            )
        return ProductionLabelAuthorization(
            case_id=assignment.case_id,
            leakage_group_id=assignment.leakage_group_id,
            role=assignment.role,
        )

    def validate_node_observations(
        self,
        observations: Sequence[NodeObservation],
    ) -> None:
        """Reject RANDOM_NODE_SPLIT and ledger-role mismatches."""

        roles_by_case: dict[str, set[EvidenceRole]] = {}
        for observation in tuple(observations):
            if not isinstance(observation, NodeObservation):
                raise ValueError("observations must contain NodeObservation values")
            roles_by_case.setdefault(observation.case_id, set()).add(observation.role)

        for case_id, observed_roles in roles_by_case.items():
            if len(observed_roles) != 1:
                raise RandomNodeSplitError(
                    f"RANDOM_NODE_SPLIT: case {case_id!r} spans node roles"
                )
            observed = next(iter(observed_roles))
            expected = self.role_for_case(case_id)
            if observed is not expected:
                raise NodeRoleMismatchError(
                    f"node role {observed.value} disagrees with ledger role "
                    f"{expected.value} for case {case_id!r}"
                )


def assign_group(
    registry: CaseRoleRegistry,
    cases: Sequence[DesignCase],
    role: EvidenceRole,
) -> CaseRoleRegistry:
    """Functional form of :meth:`CaseRoleRegistry.assign_group`."""

    return registry.assign_group(cases, role)


def authorize_production_label(
    registry: CaseRoleRegistry,
    case: DesignCase,
) -> ProductionLabelAuthorization:
    """Functional form of the production-label authorization boundary."""

    return registry.authorize_production_label(case)


def validate_node_observations(
    registry: CaseRoleRegistry,
    observations: Sequence[NodeObservation],
) -> None:
    """Functional form of the node-observation validation boundary."""

    registry.validate_node_observations(observations)
