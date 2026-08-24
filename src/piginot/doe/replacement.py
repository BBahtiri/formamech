"""Preregistered, role-specific replacement of failed S9 DOE cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from piginot.doe.roles import CaseRoleRegistry, DesignCase, EvidenceRole


class ReplacementPolicyError(ValueError):
    """A replacement policy or its role registration is inconsistent."""


class DuplicateFailureError(ValueError):
    """A failed case was presented to the append-only ledger twice."""


class ReplacementStatus(StrEnum):
    """Outcome of applying the preregistered reserve policy."""

    ASSIGNED = "ASSIGNED"
    EXHAUSTED = "EXHAUSTED"


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _case_id(value: object, label: str = "case_id") -> str:
    case_id = _identifier(value, label)
    if ":" in case_id:
        raise ValueError(f"{label} must identify a case-level sample")
    return case_id


def _case_identity(case: DesignCase) -> tuple[str, str]:
    return (
        _case_id(case.case_id),
        _identifier(case.leakage_group_id, "leakage_group_id"),
    )


@dataclass(frozen=True, slots=True)
class ReserveCandidate:
    """One reserve case at a fixed rank within one evidence role."""

    case_id: str
    leakage_group_id: str
    role: EvidenceRole
    rank: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _case_id(self.case_id))
        object.__setattr__(
            self,
            "leakage_group_id",
            _identifier(self.leakage_group_id, "leakage_group_id"),
        )
        object.__setattr__(self, "role", EvidenceRole(self.role))
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise ReplacementPolicyError("reserve rank must be an integer")
        if self.rank <= 0:
            raise ReplacementPolicyError("reserve rank must be positive")


@dataclass(frozen=True, slots=True)
class OrderedReservePolicy:
    """Immutable role-specific reserve order fixed before failures occur."""

    rule_id: str
    candidates: tuple[ReserveCandidate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _identifier(self.rule_id, "rule_id"))
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, ReserveCandidate) for candidate in candidates):
            raise ReplacementPolicyError(
                "candidates must contain ReserveCandidate values"
            )
        case_ids = tuple(candidate.case_id for candidate in candidates)
        if len(case_ids) != len(set(case_ids)):
            raise ReplacementPolicyError("candidate case_id values must be unique")
        role_ranks = tuple((candidate.role, candidate.rank) for candidate in candidates)
        if len(role_ranks) != len(set(role_ranks)):
            raise ReplacementPolicyError("reserve ranks must be unique within a role")

        for role in EvidenceRole:
            ranks = sorted(
                candidate.rank for candidate in candidates if candidate.role is role
            )
            if ranks and ranks != list(range(1, len(ranks) + 1)):
                raise ReplacementPolicyError(
                    "reserve ranks must be contiguous from one within a role"
                )
        object.__setattr__(
            self,
            "candidates",
            tuple(
                sorted(
                    candidates,
                    key=lambda candidate: (
                        candidate.role.value,
                        candidate.rank,
                        candidate.case_id,
                    ),
                )
            ),
        )

    @classmethod
    def preregister(
        cls,
        *,
        rule_id: str,
        reserves_by_role: Mapping[EvidenceRole, Sequence[DesignCase]],
    ) -> OrderedReservePolicy:
        """Freeze caller-declared reserve sequences into a canonical policy."""

        candidates: list[ReserveCandidate] = []
        normalized = {
            EvidenceRole(role): tuple(cases) for role, cases in reserves_by_role.items()
        }
        for role in sorted(normalized, key=lambda item: item.value):
            for rank, case in enumerate(normalized[role], start=1):
                case_id, group_id = _case_identity(case)
                candidates.append(
                    ReserveCandidate(
                        case_id=case_id,
                        leakage_group_id=group_id,
                        role=role,
                        rank=rank,
                    )
                )
        return cls(rule_id=rule_id, candidates=tuple(candidates))

    def candidates_for_role(
        self,
        role: EvidenceRole,
    ) -> tuple[ReserveCandidate, ...]:
        """Return the fixed rank order for one role."""

        resolved_role = EvidenceRole(role)
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.role is resolved_role
        )


@dataclass(frozen=True, slots=True)
class FailureReplacementEvent:
    """One visible failure and its assigned reserve or explicit exhaustion."""

    sequence: int
    failed_case_id: str
    failed_leakage_group_id: str
    role: EvidenceRole
    reason: str
    status: ReplacementStatus
    replacement_case_id: str | None = None
    replacement_leakage_group_id: str | None = None
    reserve_rank: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValueError("event sequence must be an integer")
        if self.sequence <= 0:
            raise ValueError("event sequence must be positive")
        object.__setattr__(
            self,
            "failed_case_id",
            _case_id(self.failed_case_id, "failed_case_id"),
        )
        object.__setattr__(
            self,
            "failed_leakage_group_id",
            _identifier(
                self.failed_leakage_group_id,
                "failed_leakage_group_id",
            ),
        )
        object.__setattr__(self, "role", EvidenceRole(self.role))
        object.__setattr__(self, "reason", _identifier(self.reason, "reason"))
        status = ReplacementStatus(self.status)
        object.__setattr__(self, "status", status)

        replacement_fields = (
            self.replacement_case_id,
            self.replacement_leakage_group_id,
            self.reserve_rank,
        )
        if status is ReplacementStatus.EXHAUSTED:
            if any(value is not None for value in replacement_fields):
                raise ValueError("exhausted events cannot name a replacement")
            return
        if any(value is None for value in replacement_fields):
            raise ValueError("assigned events require complete replacement identity")
        object.__setattr__(
            self,
            "replacement_case_id",
            _case_id(self.replacement_case_id, "replacement_case_id"),
        )
        object.__setattr__(
            self,
            "replacement_leakage_group_id",
            _identifier(
                self.replacement_leakage_group_id,
                "replacement_leakage_group_id",
            ),
        )
        if isinstance(self.reserve_rank, bool) or not isinstance(
            self.reserve_rank, int
        ):
            raise ValueError("reserve_rank must be an integer")
        if self.reserve_rank <= 0:
            raise ValueError("reserve_rank must be positive")


@dataclass(frozen=True, slots=True)
class FailureReplacementLedger:
    """Immutable append-only account of every failed DOE case."""

    policy: OrderedReservePolicy
    events: tuple[FailureReplacementEvent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, OrderedReservePolicy):
            raise ValueError("policy must be an OrderedReservePolicy")
        events = tuple(self.events)
        if any(not isinstance(event, FailureReplacementEvent) for event in events):
            raise ValueError("events must contain FailureReplacementEvent values")
        expected_sequences = tuple(range(1, len(events) + 1))
        if tuple(event.sequence for event in events) != expected_sequences:
            raise ValueError("event sequence must be contiguous and append-only")
        failed_ids = tuple(event.failed_case_id for event in events)
        if len(failed_ids) != len(set(failed_ids)):
            raise DuplicateFailureError("a failed case may be recorded only once")

        unavailable_case_ids: set[str] = set()
        for event in events:
            unavailable_case_ids.add(event.failed_case_id)
            expected = self._next_candidate(
                event.role,
                unavailable_case_ids=unavailable_case_ids,
            )
            if expected is None:
                if event.status is not ReplacementStatus.EXHAUSTED:
                    raise ReplacementPolicyError(
                        "assigned replacement violates the preregistered policy"
                    )
                continue
            if event.status is not ReplacementStatus.ASSIGNED:
                raise ReplacementPolicyError(
                    "exhaustion recorded while a preregistered reserve was available"
                )
            observed = (
                event.replacement_case_id,
                event.replacement_leakage_group_id,
                event.reserve_rank,
            )
            declared = (
                expected.case_id,
                expected.leakage_group_id,
                expected.rank,
            )
            if observed != declared:
                raise ReplacementPolicyError(
                    "replacement violates the preregistered role-specific order"
                )
            unavailable_case_ids.add(expected.case_id)
        object.__setattr__(self, "events", events)

    @classmethod
    def empty(cls, policy: OrderedReservePolicy) -> FailureReplacementLedger:
        """Create an empty ledger bound to an immutable policy."""

        return cls(policy=policy)

    def _next_candidate(
        self,
        role: EvidenceRole,
        *,
        unavailable_case_ids: set[str],
    ) -> ReserveCandidate | None:
        for candidate in self.policy.candidates_for_role(role):
            if candidate.case_id not in unavailable_case_ids:
                return candidate
        return None

    def record_failure(
        self,
        failed: DesignCase,
        *,
        reason: str,
        roles: CaseRoleRegistry,
    ) -> FailureReplacementLedger:
        """Append one failure and apply the fixed policy without caller choice."""

        failed_case_id, failed_group_id = _case_identity(failed)
        if failed_case_id in self.failed_case_ids:
            raise DuplicateFailureError(
                f"failure for {failed_case_id!r} is already recorded"
            )
        failure_reason = _identifier(reason, "reason")
        authorization = roles.authorize_production_label(failed)
        unavailable_case_ids = set(self.failed_case_ids) | {
            event.replacement_case_id
            for event in self.events
            if event.replacement_case_id is not None
        }
        unavailable_case_ids.add(failed_case_id)
        candidate = self._next_candidate(
            authorization.role,
            unavailable_case_ids=unavailable_case_ids,
        )
        sequence = len(self.events) + 1
        if candidate is None:
            event = FailureReplacementEvent(
                sequence=sequence,
                failed_case_id=failed_case_id,
                failed_leakage_group_id=failed_group_id,
                role=authorization.role,
                reason=failure_reason,
                status=ReplacementStatus.EXHAUSTED,
            )
        else:
            reserve_authorization = roles.authorize_production_label(candidate)
            if reserve_authorization.role is not authorization.role:
                raise ReplacementPolicyError(
                    f"reserve {candidate.case_id!r} has registered role "
                    f"{reserve_authorization.role.value}, expected "
                    f"{authorization.role.value}"
                )
            event = FailureReplacementEvent(
                sequence=sequence,
                failed_case_id=failed_case_id,
                failed_leakage_group_id=failed_group_id,
                role=authorization.role,
                reason=failure_reason,
                status=ReplacementStatus.ASSIGNED,
                replacement_case_id=candidate.case_id,
                replacement_leakage_group_id=candidate.leakage_group_id,
                reserve_rank=candidate.rank,
            )
        return FailureReplacementLedger(
            policy=self.policy,
            events=(*self.events, event),
        )

    @property
    def failed_case_ids(self) -> tuple[str, ...]:
        """Return every failed case, including explicitly exhausted ones."""

        return tuple(event.failed_case_id for event in self.events)

    @property
    def replacement_case_ids(self) -> tuple[str, ...]:
        """Return used reserves in ledger order."""

        return tuple(
            event.replacement_case_id
            for event in self.events
            if event.replacement_case_id is not None
        )
