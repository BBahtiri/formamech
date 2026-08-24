from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import pytest

from piginot.doe.replacement import (
    DuplicateFailureError,
    FailureReplacementEvent,
    FailureReplacementLedger,
    OrderedReservePolicy,
    ReplacementPolicyError,
    ReplacementStatus,
    ReserveCandidate,
)
from piginot.doe.roles import CaseRoleRegistry, EvidenceRole


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    leakage_group_id: str


def _roles(*assignments: tuple[Case, EvidenceRole]) -> CaseRoleRegistry:
    registry = CaseRoleRegistry.empty()
    for case, role in assignments:
        registry = registry.assign(case, role)
    return registry


def test_role_specific_reserve_order_is_preregistered_and_deterministic() -> None:
    failed_train_1 = Case("failed-train-1", "train-primary-1")
    failed_train_2 = Case("failed-train-2", "train-primary-2")
    failed_development = Case("failed-dev", "dev-primary")
    train_reserve_1 = Case("train-reserve-1", "train-reserve-group-1")
    train_reserve_2 = Case("train-reserve-2", "train-reserve-group-2")
    development_reserve = Case("dev-reserve-1", "dev-reserve-group-1")
    roles = _roles(
        (failed_train_1, EvidenceRole.TRAIN),
        (failed_train_2, EvidenceRole.TRAIN),
        (failed_development, EvidenceRole.DEVELOPMENT),
        (train_reserve_1, EvidenceRole.TRAIN),
        (train_reserve_2, EvidenceRole.TRAIN),
        (development_reserve, EvidenceRole.DEVELOPMENT),
    )
    policy = OrderedReservePolicy.preregister(
        rule_id="role-ordered-reserve-v1",
        reserves_by_role={
            EvidenceRole.DEVELOPMENT: (development_reserve,),
            EvidenceRole.TRAIN: (train_reserve_1, train_reserve_2),
        },
    )
    ledger = FailureReplacementLedger.empty(policy)

    after_train_1 = ledger.record_failure(
        failed_train_1,
        reason="FEM_NONCONVERGENCE",
        roles=roles,
    )
    after_development = after_train_1.record_failure(
        failed_development,
        reason="INVALID_GEOMETRY",
        roles=roles,
    )
    after_train_2 = after_development.record_failure(
        failed_train_2,
        reason="FEM_NONCONVERGENCE",
        roles=roles,
    )

    assert ledger.events == ()
    assert tuple(event.replacement_case_id for event in after_train_2.events) == (
        "train-reserve-1",
        "dev-reserve-1",
        "train-reserve-2",
    )
    assert tuple(event.role for event in after_train_2.events) == (
        EvidenceRole.TRAIN,
        EvidenceRole.DEVELOPMENT,
        EvidenceRole.TRAIN,
    )


def test_caller_cannot_choose_the_replacement() -> None:
    failed = Case("failed", "failed-group")
    reserve = Case("reserve", "reserve-group")
    roles = _roles(
        (failed, EvidenceRole.TRAIN),
        (reserve, EvidenceRole.TRAIN),
    )
    ledger = FailureReplacementLedger.empty(
        OrderedReservePolicy.preregister(
            rule_id="ordered-v1",
            reserves_by_role={EvidenceRole.TRAIN: (reserve,)},
        )
    )

    with pytest.raises(TypeError, match="replacement"):
        ledger.record_failure(  # type: ignore[call-arg]
            failed,
            reason="FAILED",
            roles=roles,
            replacement=reserve,
        )


def test_duplicate_failure_recording_raises_and_history_is_unchanged() -> None:
    failed = Case("failed", "failed-group")
    reserve = Case("reserve", "reserve-group")
    roles = _roles(
        (failed, EvidenceRole.TRAIN),
        (reserve, EvidenceRole.TRAIN),
    )
    initial = FailureReplacementLedger.empty(
        OrderedReservePolicy.preregister(
            rule_id="ordered-v1",
            reserves_by_role={EvidenceRole.TRAIN: (reserve,)},
        )
    )
    updated = initial.record_failure(failed, reason="FAILED", roles=roles)

    with pytest.raises(DuplicateFailureError, match="failed"):
        updated.record_failure(failed, reason="FAILED_AGAIN", roles=roles)

    assert len(updated.events) == 1
    assert updated.events[0].reason == "FAILED"


def test_replacement_candidates_are_never_reused() -> None:
    failures = tuple(
        Case(f"failed-{index}", f"failed-group-{index}") for index in range(3)
    )
    reserves = tuple(
        Case(f"reserve-{index}", f"reserve-group-{index}") for index in range(2)
    )
    roles = _roles(*((case, EvidenceRole.TRAIN) for case in (*failures, *reserves)))
    ledger = FailureReplacementLedger.empty(
        OrderedReservePolicy.preregister(
            rule_id="ordered-v1",
            reserves_by_role={EvidenceRole.TRAIN: reserves},
        )
    )

    for failed in failures:
        ledger = ledger.record_failure(failed, reason="FAILED", roles=roles)

    replacements = tuple(
        event.replacement_case_id
        for event in ledger.events
        if event.replacement_case_id is not None
    )
    assert replacements == ("reserve-0", "reserve-1")
    assert len(replacements) == len(set(replacements))


def test_a_failed_reserve_candidate_is_never_selected_later() -> None:
    failed_reserve = Case("reserve-1", "reserve-group-1")
    later_failure = Case("failed-later", "failed-group-later")
    healthy_reserve = Case("reserve-2", "reserve-group-2")
    roles = _roles(
        (failed_reserve, EvidenceRole.TRAIN),
        (later_failure, EvidenceRole.TRAIN),
        (healthy_reserve, EvidenceRole.TRAIN),
    )
    ledger = FailureReplacementLedger.empty(
        OrderedReservePolicy.preregister(
            rule_id="ordered-v1",
            reserves_by_role={EvidenceRole.TRAIN: (failed_reserve, healthy_reserve)},
        )
    )

    ledger = ledger.record_failure(failed_reserve, reason="FAILED", roles=roles)
    ledger = ledger.record_failure(later_failure, reason="FAILED", roles=roles)

    assert ledger.events[0].replacement_case_id == "reserve-2"
    assert ledger.events[1].status is ReplacementStatus.EXHAUSTED
    assert ledger.events[1].replacement_case_id is None


def test_reserve_exhaustion_is_an_explicit_persisted_failure() -> None:
    failed_1 = Case("failed-1", "failed-group-1")
    failed_2 = Case("failed-2", "failed-group-2")
    reserve = Case("reserve", "reserve-group")
    roles = _roles(
        (failed_1, EvidenceRole.TRAIN),
        (failed_2, EvidenceRole.TRAIN),
        (reserve, EvidenceRole.TRAIN),
    )
    ledger = FailureReplacementLedger.empty(
        OrderedReservePolicy.preregister(
            rule_id="ordered-v1",
            reserves_by_role={EvidenceRole.TRAIN: (reserve,)},
        )
    )
    ledger = ledger.record_failure(failed_1, reason="FAILED", roles=roles)
    ledger = ledger.record_failure(failed_2, reason="FAILED", roles=roles)

    exhausted = ledger.events[-1]
    assert exhausted.status is ReplacementStatus.EXHAUSTED
    assert exhausted.failed_case_id == "failed-2"
    assert exhausted.replacement_case_id is None
    assert ledger.failed_case_ids == ("failed-1", "failed-2")


def test_replacement_must_have_the_failed_cases_preregistered_role() -> None:
    failed = Case("failed", "failed-group")
    reserve = Case("reserve", "reserve-group")
    roles = _roles(
        (failed, EvidenceRole.TRAIN),
        (reserve, EvidenceRole.HOLDOUT),
    )
    ledger = FailureReplacementLedger.empty(
        OrderedReservePolicy.preregister(
            rule_id="ordered-v1",
            reserves_by_role={EvidenceRole.TRAIN: (reserve,)},
        )
    )

    with pytest.raises(ReplacementPolicyError, match="registered role HOLDOUT"):
        ledger.record_failure(failed, reason="FAILED", roles=roles)

    assert ledger.events == ()


def test_policy_rejects_duplicate_candidates_and_duplicate_role_ranks() -> None:
    reserve = Case("reserve", "reserve-group")

    with pytest.raises(ReplacementPolicyError, match="candidate case_id"):
        OrderedReservePolicy.preregister(
            rule_id="ordered-v1",
            reserves_by_role={
                EvidenceRole.TRAIN: (reserve,),
                EvidenceRole.DEVELOPMENT: (reserve,),
            },
        )


def test_replacement_history_is_a_frozen_append_only_value() -> None:
    failed = Case("failed", "failed-group")
    reserve = Case("reserve", "reserve-group")
    roles = _roles(
        (failed, EvidenceRole.TRAIN),
        (reserve, EvidenceRole.TRAIN),
    )
    ledger = FailureReplacementLedger.empty(
        OrderedReservePolicy.preregister(
            rule_id="ordered-v1",
            reserves_by_role={EvidenceRole.TRAIN: (reserve,)},
        )
    ).record_failure(failed, reason="FAILED", roles=roles)

    with pytest.raises(FrozenInstanceError):
        ledger.events[0].reason = "HIDDEN"  # type: ignore[misc]

    assert ledger.replacement_case_ids == ("reserve",)


@pytest.mark.parametrize(
    ("candidate", "message"),
    (
        (
            {
                "case_id": "reserve:node-1",
                "leakage_group_id": "group",
                "role": EvidenceRole.TRAIN,
                "rank": 1,
            },
            "case-level",
        ),
        (
            {
                "case_id": "reserve",
                "leakage_group_id": "group",
                "role": EvidenceRole.TRAIN,
                "rank": True,
            },
            "rank must be an integer",
        ),
        (
            {
                "case_id": "reserve",
                "leakage_group_id": "group",
                "role": EvidenceRole.TRAIN,
                "rank": 0,
            },
            "rank must be positive",
        ),
    ),
)
def test_reserve_candidate_rejects_node_identity_and_invalid_rank(
    candidate: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ReserveCandidate(**candidate)


def test_policy_rejects_empty_rule_and_non_candidate_values() -> None:
    with pytest.raises(ValueError, match="rule_id"):
        OrderedReservePolicy(rule_id="")
    with pytest.raises(ReplacementPolicyError, match="ReserveCandidate"):
        OrderedReservePolicy(
            rule_id="ordered-v1",
            candidates=(object(),),  # type: ignore[arg-type]
        )


def test_policy_rejects_duplicate_or_noncontiguous_role_ranks() -> None:
    first = ReserveCandidate("reserve-1", "group-1", EvidenceRole.TRAIN, 1)
    duplicate_rank = ReserveCandidate(
        "reserve-2",
        "group-2",
        EvidenceRole.TRAIN,
        1,
    )
    skipped_rank = ReserveCandidate(
        "reserve-3",
        "group-3",
        EvidenceRole.HOLDOUT,
        2,
    )

    with pytest.raises(ReplacementPolicyError, match="unique within a role"):
        OrderedReservePolicy("ordered-v1", (first, duplicate_rank))
    with pytest.raises(ReplacementPolicyError, match="contiguous"):
        OrderedReservePolicy("ordered-v1", (skipped_rank,))


def _event(**overrides: Any) -> FailureReplacementEvent:
    values: dict[str, Any] = {
        "sequence": 1,
        "failed_case_id": "failed",
        "failed_leakage_group_id": "failed-group",
        "role": EvidenceRole.TRAIN,
        "reason": "FAILED",
        "status": ReplacementStatus.EXHAUSTED,
    }
    values.update(overrides)
    return FailureReplacementEvent(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"sequence": True}, "sequence must be an integer"),
        ({"sequence": 0}, "sequence must be positive"),
        (
            {
                "replacement_case_id": "reserve",
                "replacement_leakage_group_id": "reserve-group",
                "reserve_rank": 1,
            },
            "cannot name a replacement",
        ),
        (
            {"status": ReplacementStatus.ASSIGNED},
            "require complete replacement identity",
        ),
        (
            {
                "status": ReplacementStatus.ASSIGNED,
                "replacement_case_id": "reserve",
                "replacement_leakage_group_id": "reserve-group",
                "reserve_rank": True,
            },
            "reserve_rank must be an integer",
        ),
        (
            {
                "status": ReplacementStatus.ASSIGNED,
                "replacement_case_id": "reserve",
                "replacement_leakage_group_id": "reserve-group",
                "reserve_rank": 0,
            },
            "reserve_rank must be positive",
        ),
    ),
)
def test_failure_event_rejects_ambiguous_or_invalid_accounting(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _event(**overrides)


def test_ledger_rejects_invalid_policy_event_and_sequence_types() -> None:
    policy = OrderedReservePolicy("ordered-v1")

    with pytest.raises(ValueError, match="OrderedReservePolicy"):
        FailureReplacementLedger(policy=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="FailureReplacementEvent"):
        FailureReplacementLedger(policy=policy, events=(object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="contiguous"):
        FailureReplacementLedger(policy=policy, events=(_event(sequence=2),))


def test_ledger_constructor_rejects_duplicate_failure_history() -> None:
    policy = OrderedReservePolicy("ordered-v1")
    duplicate = _event(sequence=2)

    with pytest.raises(DuplicateFailureError, match="only once"):
        FailureReplacementLedger(
            policy=policy,
            events=(_event(), duplicate),
        )


def test_ledger_rejects_assigned_event_when_role_reserves_are_exhausted() -> None:
    event = _event(
        status=ReplacementStatus.ASSIGNED,
        replacement_case_id="reserve",
        replacement_leakage_group_id="reserve-group",
        reserve_rank=1,
    )

    with pytest.raises(ReplacementPolicyError, match="violates"):
        FailureReplacementLedger(
            policy=OrderedReservePolicy("ordered-v1"),
            events=(event,),
        )


def test_ledger_rejects_exhaustion_while_a_role_reserve_is_available() -> None:
    policy = OrderedReservePolicy(
        "ordered-v1",
        (ReserveCandidate("reserve", "reserve-group", EvidenceRole.TRAIN, 1),),
    )

    with pytest.raises(ReplacementPolicyError, match="reserve was available"):
        FailureReplacementLedger(policy=policy, events=(_event(),))


def test_ledger_rejects_out_of_order_replacement_history() -> None:
    policy = OrderedReservePolicy(
        "ordered-v1",
        (
            ReserveCandidate("reserve-1", "group-1", EvidenceRole.TRAIN, 1),
            ReserveCandidate("reserve-2", "group-2", EvidenceRole.TRAIN, 2),
        ),
    )
    out_of_order = _event(
        status=ReplacementStatus.ASSIGNED,
        replacement_case_id="reserve-2",
        replacement_leakage_group_id="group-2",
        reserve_rank=2,
    )

    with pytest.raises(ReplacementPolicyError, match="role-specific order"):
        FailureReplacementLedger(policy=policy, events=(out_of_order,))
