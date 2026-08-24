from __future__ import annotations

import random
from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import pytest

from piginot.doe.roles import (
    CaseRoleRegistry,
    EvidenceRole,
    ImmutableRoleAssignmentError,
    LeakageGroupRoleError,
    MissingRoleAssignmentError,
    NodeObservation,
    NodeRoleMismatchError,
    RandomNodeSplitError,
    RoleAssignment,
    assign_group,
    authorize_production_label,
    validate_node_observations,
)


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    leakage_group_id: str


def test_evidence_roles_are_the_exact_five_preregistered_roles() -> None:
    assert tuple(role.value for role in EvidenceRole) == (
        "PILOT",
        "TRAIN",
        "DEVELOPMENT",
        "HOLDOUT",
        "LEGACY",
    )


def test_role_rewrite_raises_even_when_the_requested_role_is_unchanged() -> None:
    case = Case("case-001", "geometry-001")
    registry = CaseRoleRegistry.empty().assign(case, EvidenceRole.TRAIN)

    with pytest.raises(ImmutableRoleAssignmentError, match="immutable"):
        registry.assign(case, EvidenceRole.TRAIN)
    with pytest.raises(ImmutableRoleAssignmentError, match="immutable"):
        registry.assign(case, EvidenceRole.HOLDOUT)

    assert registry.role_for_case(case.case_id) is EvidenceRole.TRAIN


def test_group_assignment_is_atomic_when_one_case_conflicts() -> None:
    existing = Case("existing", "correlated-geometry")
    new = Case("new", "correlated-geometry")
    registry = CaseRoleRegistry.empty().assign(existing, EvidenceRole.TRAIN)

    with pytest.raises(ImmutableRoleAssignmentError, match="existing"):
        registry.assign_group((new, existing), EvidenceRole.TRAIN)

    assert tuple(registry.assignments) == ("existing",)
    with pytest.raises(MissingRoleAssignmentError):
        registry.role_for_case("new")


def test_assign_group_requires_one_atomic_leakage_group() -> None:
    registry = CaseRoleRegistry.empty()

    with pytest.raises(LeakageGroupRoleError, match="one leakage_group_id"):
        registry.assign_group(
            (Case("case-a", "group-a"), Case("case-b", "group-b")),
            EvidenceRole.TRAIN,
        )

    assert registry.assignments == {}


def test_one_role_per_case_holds_for_1000_random_assignments() -> None:
    rng = random.Random(0xC0FFEE)
    roles = tuple(EvidenceRole)

    for index in range(1000):
        case = Case(f"case-{index:04d}", f"group-{index:04d}")
        role = rng.choice(roles)
        registry = CaseRoleRegistry.empty().assign(case, role)

        assert registry.role_for_case(case.case_id) is role
        with pytest.raises(ImmutableRoleAssignmentError):
            registry.assign(case, rng.choice(roles))


def test_leakage_groups_never_split_across_roles_for_1000_random_trials() -> None:
    rng = random.Random(0x5EED)
    roles = tuple(EvidenceRole)

    for index in range(1000):
        group_id = f"geometry-{index:04d}"
        role = rng.choice(roles)
        cases = (
            Case(f"case-{index:04d}-a", group_id),
            Case(f"case-{index:04d}-b", group_id),
        )
        registry = CaseRoleRegistry.empty().assign_group(cases, role)

        assert {assignment.role for assignment in registry.assignments.values()} == {
            role
        }
        with pytest.raises(LeakageGroupRoleError, match="cannot span"):
            registry.assign(
                Case(f"case-{index:04d}-c", group_id),
                rng.choice(
                    tuple(candidate for candidate in roles if candidate is not role)
                ),
            )


def test_production_label_authorization_requires_prior_exact_assignment() -> None:
    assigned = Case("case-001", "geometry-001")
    registry = CaseRoleRegistry.empty().assign(assigned, EvidenceRole.TRAIN)

    authorization = registry.authorize_production_label(assigned)

    assert authorization.case_id == assigned.case_id
    assert authorization.role is EvidenceRole.TRAIN
    with pytest.raises(MissingRoleAssignmentError, match="before production labels"):
        CaseRoleRegistry.empty().authorize_production_label(assigned)
    with pytest.raises(LeakageGroupRoleError, match="does not match"):
        registry.authorize_production_label(Case("case-001", "different-geometry"))


def test_role_registry_and_assignments_are_immutable_values() -> None:
    case = Case("case-001", "geometry-001")
    registry = CaseRoleRegistry.empty().assign(case, EvidenceRole.DEVELOPMENT)

    with pytest.raises(TypeError):
        registry.assignments[case.case_id] = registry.assignments[case.case_id]  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        registry.assignments[case.case_id].role = EvidenceRole.HOLDOUT  # type: ignore[misc]


def test_node_validator_fires_on_random_node_split_mutant() -> None:
    case = Case("case-001", "geometry-001")
    registry = CaseRoleRegistry.empty().assign(case, EvidenceRole.TRAIN)
    mutant = (
        NodeObservation("case-001", "node-0", EvidenceRole.TRAIN),
        NodeObservation("case-001", "node-1", EvidenceRole.HOLDOUT),
    )

    with pytest.raises(RandomNodeSplitError, match="RANDOM_NODE_SPLIT"):
        registry.validate_node_observations(mutant)


def test_node_validator_rejects_uniform_role_mismatch_against_ledger() -> None:
    case = Case("case-001", "geometry-001")
    registry = CaseRoleRegistry.empty().assign(case, EvidenceRole.TRAIN)
    observations = (
        NodeObservation("case-001", "node-0", EvidenceRole.DEVELOPMENT),
        NodeObservation("case-001", "node-1", EvidenceRole.DEVELOPMENT),
    )

    with pytest.raises(NodeRoleMismatchError, match="ledger role TRAIN"):
        registry.validate_node_observations(observations)


def test_case_level_assignment_rejects_node_identifiers() -> None:
    with pytest.raises(ValueError, match="case-level"):
        CaseRoleRegistry.empty().assign(
            Case("case-001:node-4", "geometry-001"),
            EvidenceRole.TRAIN,
        )


def test_assignment_rejects_empty_case_and_group_identifiers() -> None:
    with pytest.raises(ValueError, match="case_id"):
        CaseRoleRegistry.empty().assign(Case("", "group"), EvidenceRole.TRAIN)
    with pytest.raises(ValueError, match="leakage_group_id"):
        CaseRoleRegistry.empty().assign(Case("case", ""), EvidenceRole.TRAIN)


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ({"assignments": []}, "assignments must be a mapping"),
        (
            {"leakage_group_roles": []},
            "leakage_group_roles must be a mapping",
        ),
        ({"assignments": {"case": object()}}, "RoleAssignment"),
        (
            {
                "assignments": {
                    "wrong-key": RoleAssignment(
                        "case",
                        "group",
                        EvidenceRole.TRAIN,
                    )
                }
            },
            "key must equal",
        ),
        (
            {
                "assignments": {
                    "case-a": RoleAssignment(
                        "case-a",
                        "group",
                        EvidenceRole.TRAIN,
                    ),
                    "case-b": RoleAssignment(
                        "case-b",
                        "group",
                        EvidenceRole.HOLDOUT,
                    ),
                }
            },
            "cannot span",
        ),
        (
            {
                "assignments": {
                    "case": RoleAssignment(
                        "case",
                        "group",
                        EvidenceRole.TRAIN,
                    )
                },
                "leakage_group_roles": {"group": EvidenceRole.HOLDOUT},
            },
            "cannot span",
        ),
        (
            {"leakage_group_roles": {"orphan": EvidenceRole.TRAIN}},
            "unassigned group",
        ),
    ),
)
def test_registry_constructor_fails_closed_on_corrupted_state(
    arguments: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CaseRoleRegistry(**arguments)


def test_group_assignment_rejects_empty_and_duplicate_populations() -> None:
    registry = CaseRoleRegistry.empty()
    case = Case("case", "group")

    with pytest.raises(LeakageGroupRoleError, match="at least one"):
        registry.assign_group((), EvidenceRole.TRAIN)
    with pytest.raises(ImmutableRoleAssignmentError, match="more than once"):
        registry.assign_group((case, case), EvidenceRole.TRAIN)


def test_group_assignment_rejects_existing_group_role_before_any_write() -> None:
    existing = Case("existing", "group")
    registry = CaseRoleRegistry.empty().assign(existing, EvidenceRole.TRAIN)

    with pytest.raises(LeakageGroupRoleError, match="cannot span"):
        registry.assign_group(
            (Case("new-a", "group"), Case("new-b", "group")),
            EvidenceRole.HOLDOUT,
        )

    assert tuple(registry.assignments) == ("existing",)


def test_node_validator_rejects_non_observation_values() -> None:
    with pytest.raises(ValueError, match="NodeObservation"):
        CaseRoleRegistry.empty().validate_node_observations([object()])  # type: ignore[list-item]


def test_functional_role_boundaries_delegate_without_losing_guarantees() -> None:
    cases = (Case("case-a", "group"), Case("case-b", "group"))
    registry = assign_group(
        CaseRoleRegistry.empty(),
        cases,
        EvidenceRole.DEVELOPMENT,
    )

    authorization = authorize_production_label(registry, cases[0])
    validate_node_observations(
        registry,
        (
            NodeObservation("case-a", "node-0", EvidenceRole.DEVELOPMENT),
            NodeObservation("case-b", "node-0", EvidenceRole.DEVELOPMENT),
        ),
    )

    assert authorization.role is EvidenceRole.DEVELOPMENT
