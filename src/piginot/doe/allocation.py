"""Secret-seeded, leakage-group-atomic S9 evidence-role allocation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from piginot.contracts import canonical_json_bytes
from piginot.doe.roles import CaseRoleRegistry, DesignCase, EvidenceRole
from piginot.doe.secret_material import SecretMaterialError, validate_secret_bytes

ALLOCATION_ALGORITHM_ID = "s9-leakage-group-role-allocation-blake2b-256-v1"

_CANONICAL_NAMESPACE = "piginot.doe"
_COMMITMENT_PERSON = b"piginot-s9-seed"
_KEY_PERSON = b"piginot-s9-key"
_GROUP_PERSON = b"piginot-s9-grp"
_ROLE_PERSON = b"piginot-s9-role"
_PRODUCTION_ROLES = (
    EvidenceRole.PILOT,
    EvidenceRole.TRAIN,
    EvidenceRole.DEVELOPMENT,
    EvidenceRole.HOLDOUT,
)


class RoleAllocationError(ValueError):
    """Base class for fail-closed production role-allocation errors."""


class InvalidProductionBudgetError(RoleAllocationError):
    """The requested production case-count budget is malformed."""


class InvalidAllocationSeedError(RoleAllocationError):
    """The independent allocation seed does not meet the secret contract."""


class DuplicateDesignCaseError(RoleAllocationError):
    """A case identity appears more than once in the allocation population."""


class ImpossibleRoleAllocationError(RoleAllocationError):
    """Whole leakage groups cannot satisfy every exact case-count budget."""


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidProductionBudgetError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ProductionRoleBudget:
    """Exact case counts for the four production evidence roles."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    pilot: int
    train: int
    development: int
    holdout: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "pilot", _count(self.pilot, "pilot"))
        object.__setattr__(self, "train", _count(self.train, "train"))
        object.__setattr__(
            self,
            "development",
            _count(self.development, "development"),
        )
        object.__setattr__(self, "holdout", _count(self.holdout, "holdout"))
        if self.total == 0:
            raise InvalidProductionBudgetError(
                "a production budget must allocate at least one case"
            )

    @property
    def total(self) -> int:
        """Return the exact production population size."""

        return self.pilot + self.train + self.development + self.holdout

    def count_for(self, role: EvidenceRole) -> int:
        """Return one production role's exact case count."""

        resolved_role = EvidenceRole(role)
        if resolved_role is EvidenceRole.PILOT:
            return self.pilot
        if resolved_role is EvidenceRole.TRAIN:
            return self.train
        if resolved_role is EvidenceRole.DEVELOPMENT:
            return self.development
        if resolved_role is EvidenceRole.HOLDOUT:
            return self.holdout
        raise InvalidProductionBudgetError("LEGACY is not a production allocation role")

    @classmethod
    def from_mapping(cls, values: Mapping[object, object]) -> ProductionRoleBudget:
        """Normalize an exact four-role mapping and reject ``LEGACY``."""

        if not isinstance(values, Mapping):
            raise InvalidProductionBudgetError("budget must be a mapping")
        normalized: dict[EvidenceRole, object] = {}
        for raw_role, count in values.items():
            if isinstance(raw_role, EvidenceRole):
                role = raw_role
            elif isinstance(raw_role, str):
                try:
                    role = EvidenceRole(raw_role)
                except ValueError as exc:
                    raise InvalidProductionBudgetError(
                        f"budget contains unknown role {raw_role!r}"
                    ) from exc
            else:
                raise InvalidProductionBudgetError(
                    f"budget contains unknown role {raw_role!r}"
                )
            if role is EvidenceRole.LEGACY:
                raise InvalidProductionBudgetError(
                    "LEGACY is not a production allocation role"
                )
            normalized[role] = count
        if set(normalized) != set(_PRODUCTION_ROLES):
            raise InvalidProductionBudgetError(
                "budget must contain exactly PILOT, TRAIN, DEVELOPMENT, and HOLDOUT"
            )
        return cls(
            pilot=_count(normalized[EvidenceRole.PILOT], "pilot"),
            train=_count(normalized[EvidenceRole.TRAIN], "train"),
            development=_count(
                normalized[EvidenceRole.DEVELOPMENT],
                "development",
            ),
            holdout=_count(normalized[EvidenceRole.HOLDOUT], "holdout"),
        )


@dataclass(frozen=True, slots=True)
class RoleAllocation:
    """Immutable allocation exposing a seed commitment, never the secret seed."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    algorithm_id: str
    seed_commitment: str
    budget: ProductionRoleBudget
    registry: CaseRoleRegistry

    def __post_init__(self) -> None:
        if self.algorithm_id != ALLOCATION_ALGORITHM_ID:
            raise ValueError("unsupported role-allocation algorithm")
        if not _is_digest(self.seed_commitment):
            raise ValueError("seed commitment must be lowercase BLAKE2b-256 hex")
        if not isinstance(self.budget, ProductionRoleBudget):
            raise ValueError("budget must be a ProductionRoleBudget")
        if not isinstance(self.registry, CaseRoleRegistry):
            raise ValueError("registry must be a CaseRoleRegistry")

        observed = {role: 0 for role in _PRODUCTION_ROLES}
        for assignment in self.registry.assignments.values():
            if assignment.role is EvidenceRole.LEGACY:
                raise ValueError("LEGACY cannot appear in a production allocation")
            observed[assignment.role] += 1
        if len(self.registry.assignments) != self.budget.total:
            raise ValueError("registry total does not match the production budget")
        for role in _PRODUCTION_ROLES:
            if observed[role] != self.budget.count_for(role):
                raise ValueError(
                    f"registry count for {role.value} does not match its exact budget"
                )

    @property
    def role_counts(self) -> Mapping[EvidenceRole, int]:
        """Return immutable exact counts in the preregistered role order."""

        return MappingProxyType(
            {role: self.budget.count_for(role) for role in _PRODUCTION_ROLES}
        )

    def to_bytes(self) -> bytes:
        """Return the canonical public allocation record without secret material."""

        return canonical_json_bytes(self)


@dataclass(frozen=True, slots=True)
class _LeakageGroup:
    group_id: str
    cases: tuple[DesignCase, ...]

    @property
    def size(self) -> int:
        return len(self.cases)


@dataclass(frozen=True, slots=True)
class _SearchEntry:
    state: tuple[int, tuple[int, int, int, int]]
    parent: tuple[int, tuple[int, int, int, int]] | None
    role_index: int | None


def allocate_evidence_roles(
    cases: Sequence[DesignCase],
    budget: ProductionRoleBudget | Mapping[object, object],
    *,
    seed: bytes,
) -> RoleAllocation:
    """Allocate whole leakage groups to exact production roles deterministically.

    The allocation is replayable only with the independent secret ``seed``.  The
    returned value contains a domain-separated commitment to that seed, but no seed
    bytes or reversible seed-derived key material.
    """

    resolved_budget = _resolve_budget(budget)
    validated_seed = _validate_seed(seed)
    population = tuple(cases)
    groups = _leakage_groups(population)
    if len(population) != resolved_budget.total:
        raise InvalidProductionBudgetError(
            "production budget total must equal the case population"
        )

    allocation_key = hashlib.blake2b(
        validated_seed,
        digest_size=32,
        person=_KEY_PERSON,
    ).digest()
    ordered_groups = tuple(
        sorted(
            groups,
            key=lambda group: (
                _group_rank(allocation_key, group),
                group.group_id,
            ),
        )
    )
    capacities = (
        resolved_budget.count_for(EvidenceRole.PILOT),
        resolved_budget.count_for(EvidenceRole.TRAIN),
        resolved_budget.count_for(EvidenceRole.DEVELOPMENT),
        resolved_budget.count_for(EvidenceRole.HOLDOUT),
    )
    role_indices = _exact_group_partition(
        ordered_groups,
        capacities,
        allocation_key,
    )

    registry = CaseRoleRegistry.empty()
    for group, role_index in zip(ordered_groups, role_indices, strict=True):
        registry = registry.assign_group(group.cases, _PRODUCTION_ROLES[role_index])

    return RoleAllocation(
        algorithm_id=ALLOCATION_ALGORITHM_ID,
        seed_commitment=_seed_commitment(validated_seed),
        budget=resolved_budget,
        registry=registry,
    )


def _resolve_budget(
    budget: ProductionRoleBudget | Mapping[object, object],
) -> ProductionRoleBudget:
    if isinstance(budget, ProductionRoleBudget):
        return budget
    if isinstance(budget, Mapping):
        return ProductionRoleBudget.from_mapping(budget)
    raise InvalidProductionBudgetError(
        "budget must be a ProductionRoleBudget or role mapping"
    )


def _validate_seed(seed: object) -> bytes:
    try:
        return validate_secret_bytes(seed, "allocation seed")
    except SecretMaterialError as exc:
        raise InvalidAllocationSeedError(str(exc)) from exc


def _seed_commitment(seed: bytes) -> str:
    return hashlib.blake2b(
        seed,
        digest_size=32,
        person=_COMMITMENT_PERSON,
    ).hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _leakage_groups(cases: Sequence[DesignCase]) -> tuple[_LeakageGroup, ...]:
    grouped: dict[str, list[DesignCase]] = {}
    seen_case_ids: set[str] = set()
    for case in cases:
        case_id, group_id = _case_identity(case)
        if case_id in seen_case_ids:
            raise DuplicateDesignCaseError(f"duplicate case_id {case_id!r}")
        seen_case_ids.add(case_id)
        grouped.setdefault(group_id, []).append(case)
    return tuple(
        _LeakageGroup(
            group_id,
            tuple(sorted(group_cases, key=lambda item: item.case_id)),
        )
        for group_id, group_cases in sorted(grouped.items())
    )


def _case_identity(case: DesignCase) -> tuple[str, str]:
    try:
        case_id = case.case_id
        group_id = case.leakage_group_id
    except AttributeError as exc:
        raise ValueError("cases must contain DesignCase values") from exc
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be non-empty text")
    if ":" in case_id:
        raise ValueError("case_id must identify a case-level sample")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("leakage_group_id must be non-empty text")
    return case_id, group_id


def _group_rank(key: bytes, group: _LeakageGroup) -> bytes:
    payload = canonical_json_bytes(
        {
            "algorithm_id": ALLOCATION_ALGORITHM_ID,
            "case_ids": tuple(case.case_id for case in group.cases),
            "leakage_group_id": group.group_id,
        }
    )
    return hashlib.blake2b(
        payload,
        digest_size=32,
        key=key,
        person=_GROUP_PERSON,
    ).digest()


def _role_order(key: bytes, group_id: str) -> tuple[int, ...]:
    def rank(role_index: int) -> tuple[bytes, str]:
        role = _PRODUCTION_ROLES[role_index]
        payload = canonical_json_bytes(
            {
                "algorithm_id": ALLOCATION_ALGORITHM_ID,
                "leakage_group_id": group_id,
                "role": role.value,
            }
        )
        digest = hashlib.blake2b(
            payload,
            digest_size=32,
            key=key,
            person=_ROLE_PERSON,
        ).digest()
        return digest, role.value

    return tuple(sorted(range(len(_PRODUCTION_ROLES)), key=rank))


def _exact_group_partition(
    groups: Sequence[_LeakageGroup],
    capacities: tuple[int, int, int, int],
    key: bytes,
) -> tuple[int, ...]:
    sizes = tuple(group.size for group in groups)
    suffix_gcd, suffix_minimum, suffix_maximum, suffix_total = _suffix_bounds(sizes)
    start = (0, capacities)
    if not _state_is_feasible(
        0,
        capacities,
        suffix_gcd,
        suffix_minimum,
        suffix_maximum,
        suffix_total,
    ):
        raise ImpossibleRoleAllocationError(
            "no exact whole-leakage-group partition satisfies the production budget"
        )
    stack = [_SearchEntry(start, None, None)]
    visited: set[tuple[int, tuple[int, int, int, int]]] = set()
    parents: dict[
        tuple[int, tuple[int, int, int, int]],
        tuple[tuple[int, tuple[int, int, int, int]], int],
    ] = {}

    while stack:
        entry = stack.pop()
        index, remaining = entry.state
        if entry.state in visited:
            continue
        visited.add(entry.state)
        if entry.parent is not None and entry.role_index is not None:
            parents[entry.state] = (entry.parent, entry.role_index)
        if index == len(groups):
            return _reconstruct_roles(entry.state, start, parents)

        group = groups[index]
        choices = tuple(
            role_index
            for role_index in _role_order(key, group.group_id)
            if remaining[role_index] >= group.size
        )
        for role_index in reversed(choices):
            next_remaining = _subtract_capacity(
                remaining,
                role_index,
                group.size,
            )
            if _state_is_feasible(
                index + 1,
                next_remaining,
                suffix_gcd,
                suffix_minimum,
                suffix_maximum,
                suffix_total,
            ):
                stack.append(
                    _SearchEntry(
                        (index + 1, next_remaining),
                        entry.state,
                        role_index,
                    )
                )

    raise ImpossibleRoleAllocationError(
        "no exact whole-leakage-group partition satisfies the production budget"
    )


def _subtract_capacity(
    remaining: tuple[int, int, int, int],
    role_index: int,
    size: int,
) -> tuple[int, int, int, int]:
    return (
        remaining[0] - size if role_index == 0 else remaining[0],
        remaining[1] - size if role_index == 1 else remaining[1],
        remaining[2] - size if role_index == 2 else remaining[2],
        remaining[3] - size if role_index == 3 else remaining[3],
    )


def _suffix_bounds(
    sizes: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    count = len(sizes)
    gcds = [0] * (count + 1)
    minimums = [0] * (count + 1)
    maximums = [0] * (count + 1)
    totals = [0] * (count + 1)
    for index in range(count - 1, -1, -1):
        size = sizes[index]
        gcds[index] = math.gcd(size, gcds[index + 1])
        minimums[index] = (
            size
            if minimums[index + 1] == 0
            else min(
                size,
                minimums[index + 1],
            )
        )
        maximums[index] = max(size, maximums[index + 1])
        totals[index] = size + totals[index + 1]
    return tuple(gcds), tuple(minimums), tuple(maximums), tuple(totals)


def _state_is_feasible(
    index: int,
    remaining: tuple[int, int, int, int],
    suffix_gcd: Sequence[int],
    suffix_minimum: Sequence[int],
    suffix_maximum: Sequence[int],
    suffix_total: Sequence[int],
) -> bool:
    if suffix_total[index] == 0:
        return remaining == (0, 0, 0, 0)
    gcd = suffix_gcd[index]
    if any(capacity % gcd != 0 for capacity in remaining):
        return False
    minimum = suffix_minimum[index]
    if any(0 < capacity < minimum for capacity in remaining):
        return False
    return suffix_maximum[index] <= max(remaining)


def _reconstruct_roles(
    final: tuple[int, tuple[int, int, int, int]],
    start: tuple[int, tuple[int, int, int, int]],
    parents: Mapping[
        tuple[int, tuple[int, int, int, int]],
        tuple[tuple[int, tuple[int, int, int, int]], int],
    ],
) -> tuple[int, ...]:
    roles: list[int] = []
    state = final
    while state != start:
        parent, role_index = parents[state]
        roles.append(role_index)
        state = parent
    roles.reverse()
    return tuple(roles)


__all__ = (
    "ALLOCATION_ALGORITHM_ID",
    "DuplicateDesignCaseError",
    "ImpossibleRoleAllocationError",
    "InvalidAllocationSeedError",
    "InvalidProductionBudgetError",
    "ProductionRoleBudget",
    "RoleAllocation",
    "RoleAllocationError",
    "allocate_evidence_roles",
)
