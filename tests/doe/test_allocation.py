from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from typing import Any

import pytest

from piginot.contracts import canonical_json_bytes
from piginot.doe.allocation import (
    ALLOCATION_ALGORITHM_ID,
    DuplicateDesignCaseError,
    ImpossibleRoleAllocationError,
    InvalidAllocationSeedError,
    InvalidProductionBudgetError,
    ProductionRoleBudget,
    RoleAllocation,
    allocate_evidence_roles,
)
from piginot.doe.leakage import LeakageDescriptor, bind_leakage_groups
from piginot.doe.roles import CaseRoleRegistry, EvidenceRole


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    leakage_group_id: str


def _seed(offset: int = 0) -> bytes:
    return bytes((index + offset) % 256 for index in range(32))


def _singletons(count: int) -> tuple[Case, ...]:
    return tuple(
        Case(f"case-{index:03d}", f"group-{index:03d}") for index in range(count)
    )


def _budget() -> ProductionRoleBudget:
    return ProductionRoleBudget(pilot=2, train=4, development=3, holdout=3)


def test_allocation_is_deterministic_and_exact() -> None:
    cases = _singletons(12)

    first = allocate_evidence_roles(cases, _budget(), seed=_seed())
    replay = allocate_evidence_roles(cases, _budget(), seed=_seed())

    assert first == replay
    assert first.to_bytes() == replay.to_bytes()
    assert first.algorithm_id == ALLOCATION_ALGORITHM_ID
    assert len(first.seed_commitment) == 64
    assert first.role_counts == {
        EvidenceRole.PILOT: 2,
        EvidenceRole.TRAIN: 4,
        EvidenceRole.DEVELOPMENT: 3,
        EvidenceRole.HOLDOUT: 3,
    }
    assert len(first.registry.assignments) == 12


def test_secret_changes_the_allocation_and_commitment() -> None:
    cases = _singletons(12)

    first = allocate_evidence_roles(cases, _budget(), seed=_seed())
    second = allocate_evidence_roles(cases, _budget(), seed=_seed(64))

    assert first.seed_commitment != second.seed_commitment
    assert first.registry != second.registry


def test_input_order_does_not_affect_allocation() -> None:
    cases = _singletons(12)

    forward = allocate_evidence_roles(cases, _budget(), seed=_seed())
    reversed_input = allocate_evidence_roles(
        tuple(reversed(cases)),
        _budget(),
        seed=_seed(),
    )

    assert forward == reversed_input


def test_whole_leakage_groups_are_allocated_atomically() -> None:
    cases = (
        Case("correlated-a", "correlated"),
        Case("correlated-b", "correlated"),
        Case("pilot", "pilot"),
        Case("development", "development"),
        Case("holdout", "holdout"),
    )
    budget = ProductionRoleBudget(pilot=1, train=2, development=1, holdout=1)

    allocation = allocate_evidence_roles(cases, budget, seed=_seed())

    assert allocation.registry.role_for_case("correlated-a") is EvidenceRole.TRAIN
    assert allocation.registry.role_for_case("correlated-b") is EvidenceRole.TRAIN


def test_transitively_bound_groups_stay_atomic() -> None:
    descriptors = (
        LeakageDescriptor("a", "lineage-a", 1, {"x": 0.0}),
        LeakageDescriptor("b", "lineage-a", 2, {"x": 0.5}),
        LeakageDescriptor("c", "lineage-c", 2, {"x": 1.0}),
        LeakageDescriptor("d", "lineage-d", 3, {"x": 0.2}),
        LeakageDescriptor("e", "lineage-e", 4, {"x": 0.4}),
        LeakageDescriptor("f", "lineage-f", 5, {"x": 0.8}),
    )
    group_ids = bind_leakage_groups(descriptors, near_duplicate_tolerance=0.0)
    cases = tuple(Case(item.case_id, group_ids[item.case_id]) for item in descriptors)
    budget = ProductionRoleBudget(pilot=3, train=1, development=1, holdout=1)

    allocation = allocate_evidence_roles(cases, budget, seed=_seed())

    correlated_roles = {allocation.registry.role_for_case(case_id) for case_id in "abc"}
    assert correlated_roles == {EvidenceRole.PILOT}


def test_impossible_exact_group_partition_fails_closed() -> None:
    cases = (
        Case("a-1", "a"),
        Case("a-2", "a"),
        Case("b-1", "b"),
        Case("b-2", "b"),
    )

    with pytest.raises(ImpossibleRoleAllocationError, match="exact"):
        allocate_evidence_roles(
            cases,
            ProductionRoleBudget(pilot=1, train=1, development=1, holdout=1),
            seed=_seed(),
        )


def test_budget_total_must_equal_case_population() -> None:
    with pytest.raises(InvalidProductionBudgetError, match="total"):
        allocate_evidence_roles(_singletons(3), _budget(), seed=_seed())


@pytest.mark.parametrize(
    "cases",
    (
        (Case("duplicate", "one"), Case("duplicate", "two")),
        (Case("duplicate", "one"), Case("duplicate", "one")),
    ),
)
def test_duplicate_case_ids_are_rejected(cases: tuple[Case, ...]) -> None:
    with pytest.raises(DuplicateDesignCaseError, match="duplicate"):
        allocate_evidence_roles(
            cases,
            ProductionRoleBudget(pilot=1, train=1, development=0, holdout=0),
            seed=_seed(),
        )


@pytest.mark.parametrize("invalid_count", (True, -1, 1.5, "1", None))
def test_budget_counts_must_be_nonnegative_integers(invalid_count: object) -> None:
    with pytest.raises(InvalidProductionBudgetError, match="non-negative integer"):
        ProductionRoleBudget(
            pilot=invalid_count,  # type: ignore[arg-type]
            train=1,
            development=1,
            holdout=1,
        )


def test_zero_total_budget_is_rejected() -> None:
    with pytest.raises(InvalidProductionBudgetError, match="at least one"):
        ProductionRoleBudget(pilot=0, train=0, development=0, holdout=0)


def test_budget_mapping_is_normalized_and_legacy_count_lookup_is_rejected() -> None:
    budget = ProductionRoleBudget.from_mapping(
        {
            "PILOT": 2,
            "TRAIN": 4,
            "DEVELOPMENT": 3,
            "HOLDOUT": 3,
        }
    )

    allocation = allocate_evidence_roles(_singletons(12), budget, seed=_seed())

    assert allocation.budget == _budget()
    with pytest.raises(InvalidProductionBudgetError, match="LEGACY"):
        budget.count_for(EvidenceRole.LEGACY)


def test_budget_normalizer_rejects_a_non_mapping() -> None:
    with pytest.raises(InvalidProductionBudgetError, match="mapping"):
        ProductionRoleBudget.from_mapping([])  # type: ignore[arg-type]


def test_allocator_rejects_a_non_budget_value() -> None:
    with pytest.raises(InvalidProductionBudgetError, match="role mapping"):
        allocate_evidence_roles(_singletons(12), object(), seed=_seed())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mapping", "message"),
    (
        (
            {
                EvidenceRole.PILOT: 1,
                EvidenceRole.TRAIN: 1,
                EvidenceRole.DEVELOPMENT: 1,
            },
            "exactly",
        ),
        (
            {
                EvidenceRole.PILOT: 1,
                EvidenceRole.TRAIN: 1,
                EvidenceRole.DEVELOPMENT: 1,
                EvidenceRole.HOLDOUT: 1,
                EvidenceRole.LEGACY: 1,
            },
            "LEGACY",
        ),
        (
            {
                EvidenceRole.PILOT: 1,
                EvidenceRole.TRAIN: 1,
                EvidenceRole.DEVELOPMENT: 1,
                EvidenceRole.HOLDOUT: 1,
                "UNKNOWN": 1,
            },
            "unknown",
        ),
    ),
)
def test_mapping_budgets_fail_closed(
    mapping: dict[object, int],
    message: str,
) -> None:
    with pytest.raises(InvalidProductionBudgetError, match=message):
        allocate_evidence_roles(
            _singletons(sum(mapping.values())),
            mapping,  # type: ignore[arg-type]
            seed=_seed(),
        )


def test_text_and_non_bytes_seed_types_are_rejected() -> None:
    for seed in (
        "0123456789abcdefghijklmnopqrstuv",
        bytearray(_seed()),
        memoryview(_seed()),
    ):
        with pytest.raises(InvalidAllocationSeedError, match="bytes"):
            allocate_evidence_roles(_singletons(12), _budget(), seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "seed",
    (b"short", b"x" * 32, bytes(range(8)) * 4, bytes(range(16)) * 2),
)
def test_short_or_obviously_low_entropy_seeds_are_rejected(seed: bytes) -> None:
    with pytest.raises(InvalidAllocationSeedError):
        allocate_evidence_roles(_singletons(12), _budget(), seed=seed)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        (Case("", "group"), "case_id"),
        (Case("case:node", "group"), "case-level"),
        (Case("case", ""), "leakage_group_id"),
        (object(), "DesignCase"),
    ),
)
def test_design_case_identity_is_validated(case: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        allocate_evidence_roles(
            (case,),  # type: ignore[arg-type]
            ProductionRoleBudget(pilot=1, train=0, development=0, holdout=0),
            seed=_seed(),
        )


def test_empty_population_cannot_satisfy_a_production_budget() -> None:
    with pytest.raises(InvalidProductionBudgetError, match="total"):
        allocate_evidence_roles(
            (),
            ProductionRoleBudget(pilot=1, train=0, development=0, holdout=0),
            seed=_seed(),
        )


def test_group_sizes_that_cannot_fill_a_smaller_role_fail_closed() -> None:
    cases = (
        Case("two-a", "two"),
        Case("two-b", "two"),
        Case("three-a", "three"),
        Case("three-b", "three"),
        Case("three-c", "three"),
    )

    with pytest.raises(ImpossibleRoleAllocationError, match="exact"):
        allocate_evidence_roles(
            cases,
            ProductionRoleBudget(pilot=1, train=4, development=0, holdout=0),
            seed=_seed(),
        )


def test_secret_never_appears_in_repr_or_canonical_bytes() -> None:
    secret = b"0123456789abcdefghijklmnopqrstuv"

    allocation = allocate_evidence_roles(_singletons(12), _budget(), seed=secret)
    encoded = canonical_json_bytes(allocation)

    assert secret not in repr(allocation).encode()
    assert secret not in encoded
    assert secret.hex().encode() not in encoded
    assert allocation.to_bytes() == encoded


def test_returned_allocation_and_registry_are_immutable() -> None:
    allocation = allocate_evidence_roles(_singletons(12), _budget(), seed=_seed())

    with pytest.raises(FrozenInstanceError):
        allocation.seed_commitment = "0" * 64  # type: ignore[misc]
    with pytest.raises(TypeError):
        allocation.registry.assignments["extra"] = next(  # type: ignore[index]
            iter(allocation.registry.assignments.values())
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"algorithm_id": "other"}, "algorithm"),
        ({"seed_commitment": "ABC"}, "commitment"),
        ({"budget": object()}, "ProductionRoleBudget"),
        ({"registry": object()}, "CaseRoleRegistry"),
        ({"registry": CaseRoleRegistry.empty()}, "total"),
        (
            ({"budget": ProductionRoleBudget(3, 3, 3, 3)}),
            "exact budget",
        ),
    ),
)
def test_allocation_contract_rejects_corrupt_state(
    changes: dict[str, Any],
    message: str,
) -> None:
    valid = allocate_evidence_roles(_singletons(12), _budget(), seed=_seed())

    with pytest.raises(ValueError, match=message):
        replace(valid, **changes)


def test_allocation_contract_rejects_legacy_assignment() -> None:
    case = Case("legacy", "legacy-group")
    registry = CaseRoleRegistry.empty().assign(case, EvidenceRole.LEGACY)

    with pytest.raises(ValueError, match="LEGACY"):
        RoleAllocation(
            algorithm_id=ALLOCATION_ALGORITHM_ID,
            seed_commitment="0" * 64,
            budget=ProductionRoleBudget(
                pilot=1,
                train=0,
                development=0,
                holdout=0,
            ),
            registry=registry,
        )
