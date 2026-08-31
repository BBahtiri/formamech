from __future__ import annotations

import json
from pathlib import Path

from piginot.doe import (
    CaseRoleRegistry,
    EvidenceRole,
    FailureReplacementLedger,
    OrderedReservePolicy,
    PublishedCaseManifest,
    PublishedCaseStatus,
    ReplacementStatus,
    ReserveCandidate,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_ROOT = _REPOSITORY_ROOT / "artifacts" / "campaigns" / "dogbone-v1"
_PUBLIC_MANIFEST_PATH = _ARTIFACT_ROOT / "public-case-manifest.json"
_REPLACEMENT_POLICY_PATH = _ARTIFACT_ROOT / "replacement-policy.json"
_REPLACEABLE_ROLES = (
    EvidenceRole.PILOT,
    EvidenceRole.TRAIN,
    EvidenceRole.DEVELOPMENT,
)


def _published_reserve_policy() -> OrderedReservePolicy:
    payload = json.loads(_REPLACEMENT_POLICY_PATH.read_bytes())
    assert payload["$type"] == "piginot.doe.replacement.OrderedReservePolicy"

    candidates = tuple(
        ReserveCandidate(
            case_id=candidate["case_id"],
            leakage_group_id=candidate["leakage_group_id"],
            role=EvidenceRole(candidate["role"]),
            rank=candidate["rank"],
        )
        for candidate in payload["candidates"]
    )
    return OrderedReservePolicy(
        rule_id=payload["rule_id"],
        candidates=candidates,
    )


def test_sealed_campaign_excludes_holdout_and_exercises_each_replacement_role() -> None:
    manifest_payload = _PUBLIC_MANIFEST_PATH.read_bytes()
    manifest = PublishedCaseManifest.from_bytes(manifest_payload)
    policy = _published_reserve_policy()

    assert b'"role":"HOLDOUT"' not in manifest_payload
    assert len(manifest.cases) == 376
    assert (
        sum(case.status is PublishedCaseStatus.ACTIVE for case in manifest.cases) == 344
    )
    assert (
        sum(case.status is PublishedCaseStatus.RESERVE for case in manifest.cases) == 32
    )

    registry = CaseRoleRegistry.empty()
    for case in manifest.cases:
        registry = registry.assign(case, case.role)

    ledger = FailureReplacementLedger.empty(policy)
    expected_replacements: list[str] = []
    for role in _REPLACEABLE_ROLES:
        failed = next(
            case
            for case in manifest.cases
            if case.role is role and case.status is PublishedCaseStatus.ACTIVE
        )
        expected = policy.candidates_for_role(role)[0]
        expected_replacements.append(expected.case_id)
        ledger = ledger.record_failure(
            failed,
            reason="G9_INJECTED_SAMPLE_FAILURE",
            roles=registry,
        )

    assert len(ledger.events) == 3
    assert all(event.status is ReplacementStatus.ASSIGNED for event in ledger.events)
    assert ledger.replacement_case_ids == tuple(expected_replacements)
    assert tuple(event.reserve_rank for event in ledger.events) == (1, 1, 1)
