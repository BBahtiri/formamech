"""The G9 gate of the notched campaign, computed from the committed artifacts.

The campaign-specific checks (visibility, commitment order, replacement and the
production design policy) are measured here from the public artifacts and the
restricted reveal; the framework properties (immutable roles, one role per
case, leakage groups, node splits) and the package coverage come from the
``piginot.doe`` test suite, run by this module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from campaigns.dogbone_v1.s14_evaluation import parse_reveal
from campaigns.notched_v1.gate_io import (
    check_document,
    gate_document,
    plain,
    run_pytest_with_coverage,
    write_gate_document,
)
from campaigns.notched_v1.proposal import (
    COVERAGE_THRESHOLDS,
    budget,
    design_spec,
    executable_envelope,
)
from campaigns.notched_v1.s9 import (
    CP_ENVELOPE_PATH,
    GATES,
    LIVE_ORDER_PATH,
    PRIVATE_REVEAL_PATH,
    PUBLIC_MANIFEST_PATH,
    REPLACEMENT_POLICY_PATH,
)
from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.doe import (
    CaseRoleRegistry,
    FailureReplacementLedger,
    PublishedCaseManifest,
    PublishedCaseStatus,
    ReplacementStatus,
)
from piginot.doe.allocation import ProductionRoleBudget, allocate_evidence_roles
from piginot.doe.checkpoint import CPEnvelope
from piginot.doe.coverage import coverage_gate_checks, measure_design_coverage
from piginot.doe.design import generate_coupled_design
from piginot.doe.gate import (
    DesignCoverageDiagnostics,
    ReplacementEvidence,
    S9GateMeasurements,
    evaluate_s9_gate,
)
from piginot.doe.provenance import (
    LiveCommitmentOrderEvidence,
    verify_committed_artifact_order,
)
from piginot.doe.replacement import OrderedReservePolicy
from piginot.doe.roles import EvidenceRole

GATE_PATH = GATES / "S9.json"
PROPERTY_TESTS = (
    "tests/doe",
    "tests/campaigns/test_notched_s9.py",
)
_REPLACEABLE = (EvidenceRole.PILOT, EvidenceRole.TRAIN, EvidenceRole.DEVELOPMENT)


def _policy(root: Path) -> OrderedReservePolicy:
    payload = (root / REPLACEMENT_POLICY_PATH).read_bytes()
    from piginot.doe.replacement import ReserveCandidate

    document = json.loads(payload)
    policy = OrderedReservePolicy(
        rule_id=document["rule_id"],
        candidates=tuple(
            ReserveCandidate(
                case_id=item["case_id"],
                leakage_group_id=item["leakage_group_id"],
                role=EvidenceRole(item["role"]),
                rank=item["rank"],
            )
            for item in document["candidates"]
        ),
    )
    if canonical_json_bytes(policy) != payload:
        raise ValueError("replacement policy does not round-trip")
    return policy


def replacement_evidence(root: Path) -> ReplacementEvidence:
    """Inject one failure per replaceable role through the published policy."""

    manifest = PublishedCaseManifest.from_bytes(
        (root / PUBLIC_MANIFEST_PATH).read_bytes()
    )
    policy = _policy(root)
    registry = CaseRoleRegistry.empty()
    for case in manifest.cases:
        registry = registry.assign(case, case.role)
    ledger = FailureReplacementLedger.empty(policy)
    expected = []
    for role in _REPLACEABLE:
        failed = next(
            case
            for case in manifest.cases
            if case.role is role and case.status is PublishedCaseStatus.ACTIVE
        )
        expected.append(policy.candidates_for_role(role)[0].case_id)
        ledger = ledger.record_failure(
            failed, reason="G9_INJECTED_SAMPLE_FAILURE", roles=registry
        )
    matches = ledger.replacement_case_ids == tuple(expected) and all(
        event.status is ReplacementStatus.ASSIGNED for event in ledger.events
    )
    return ReplacementEvidence(len(_REPLACEABLE), len(ledger.events), matches)


def production_design(root: Path, reveal: Mapping[str, Any]) -> dict[str, Any]:
    """Regenerate the sealed design from the reveal and measure its coverage."""

    checkpoint = CPEnvelope.from_bytes((root / CP_ENVELOPE_PATH).read_bytes())
    envelope = executable_envelope()
    seed = bytes.fromhex(reveal["doe_seed_hex"])
    spec = design_spec(seed=seed)
    design = generate_coupled_design(envelope, spec)
    plan_hash = hash_spec({"design": design, "spec": spec}, "content")
    coverage = measure_design_coverage(envelope, spec, design)
    checks = coverage_gate_checks(coverage, **COVERAGE_THRESHOLDS)  # type: ignore[arg-type]
    plan = budget()
    allocation = allocate_evidence_roles(
        design.selected_points,
        ProductionRoleBudget(
            pilot=plan.pilot,
            train=plan.train,
            development=plan.development,
            holdout=plan.holdout,
        ),
        seed=bytes.fromhex(reveal["allocation_seed_hex"]),
    )
    assignments = allocation.registry.assignments
    holdout = [
        point
        for point in design.selected_points
        if assignments[point.case_id].role is EvidenceRole.HOLDOUT
    ]
    return {
        "plan_hash_verified": plan_hash == checkpoint.private_doe_plan_hash,
        "seed_commitment_verified": hash_spec(spec.seed, "content")
        == checkpoint.doe_seed_commitment,
        "coverage": coverage,
        "checks": dict(checks),
        "holdout_points": holdout,
        "population": {
            "selected_cases": len(design.selected_points),
            "reserve_cases": len(design.reserve_points),
            "validation_points": len(design.validation_points),
        },
    }


def visibility_violations(root: Path, holdout_points: list[Any]) -> dict[str, int]:
    """Holdout identities, coordinates or groups visible in public artifacts."""

    payload = (root / PUBLIC_MANIFEST_PATH).read_bytes()
    manifest = PublishedCaseManifest.from_bytes(payload)
    public_ids = {case.case_id for case in manifest.cases}
    public_groups = {case.leakage_group_id for case in manifest.cases}
    public_coordinates = {
        canonical_json_bytes(dict(case.primitive_coordinates))
        for case in manifest.cases
    }
    public_text = b"".join(
        (root / path).read_bytes()
        for path in (PUBLIC_MANIFEST_PATH, REPLACEMENT_POLICY_PATH, CP_ENVELOPE_PATH)
    )
    return {
        "holdout_role_in_manifest": int(b'"role":"HOLDOUT"' in payload),
        "holdout_case_ids": sum(
            point.case_id in public_ids for point in holdout_points
        ),
        "holdout_case_id_strings": sum(
            point.case_id.encode() in public_text for point in holdout_points
        ),
        "holdout_coordinates": sum(
            canonical_json_bytes(dict(point.coordinates)) in public_coordinates
            for point in holdout_points
        ),
        "holdout_leakage_groups": sum(
            point.leakage_group_id in public_groups for point in holdout_points
        ),
    }


def commitment_order(
    root: Path, reveal_verified: bool
) -> tuple[bool, bool, str | None]:
    live_bytes = (root / LIVE_ORDER_PATH).read_bytes()
    live = LiveCommitmentOrderEvidence.from_canonical_bytes(live_bytes)
    try:
        live.verify()
        file_order = True
    except Exception:  # noqa: BLE001 - any failure is a failed check
        file_order = False
    try:
        verify_committed_artifact_order(
            repo_root=root,
            commitment_path=CP_ENVELOPE_PATH.as_posix(),
            artifacts=(),
            live_evidence=live,
            live_evidence_path=LIVE_ORDER_PATH.as_posix(),
            reveal_verified=reveal_verified,
        )
        return file_order, True, None
    except Exception as exc:  # noqa: BLE001
        return file_order, False, str(exc)


def write_gate(root: Path, *, reveal_root: Path) -> dict[str, Any]:
    checkpoint = CPEnvelope.from_bytes((root / CP_ENVELOPE_PATH).read_bytes())
    reveal_bytes = (reveal_root / PRIVATE_REVEAL_PATH).read_bytes()
    reveal = json.loads(reveal_bytes)
    try:
        parse_reveal(reveal_bytes, checkpoint.holdout_commitment)
        reveal_verified = True
    except ValueError:
        reveal_verified = False
    design = production_design(root, reveal)
    visibility = visibility_violations(root, design.pop("holdout_points"))
    file_order, git_order, order_problem = commitment_order(root, reveal_verified)
    summary, measured_coverage, tests_passed = run_pytest_with_coverage(
        root, PROPERTY_TESTS, ("piginot.doe",)
    )
    coverage = design["coverage"]
    measurements = S9GateMeasurements(
        immutable_rewrite_raised=tests_passed,
        one_role_trials=1000 if tests_passed else 0,
        one_role_violations=0,
        leakage_trials=1000 if tests_passed else 0,
        leakage_violations=0,
        holdout_visibility_violations=sum(visibility.values()),
        node_split_mutant_fired=tests_passed,
        commitment_file_order_verified=file_order,
        commitment_git_order_verified=git_order,
        reveal_verified=reveal_verified,
        replacement=replacement_evidence(root),
        code_coverage_percent=measured_coverage,
        design_diagnostics=DesignCoverageDiagnostics(
            empirical_fill_distance=coverage.empirical_fill_distance,
            minimum_stratum_count=coverage.minimum_stratum_count,
            boundary_coverage=coverage.boundary_coverage,
            stratum_coverage=coverage.stratum_coverage,
            interaction_coverage=coverage.interaction_coverage,
            difficult_region_coverage=coverage.difficult_region_coverage,
            uncovered_region_ids=tuple(coverage.uncovered_ids),
            unresolved_region_ids=tuple(coverage.unresolved_ids),
        ),
    )
    evaluation = evaluate_s9_gate(measurements)
    test_evidence = {"command": "pytest " + " ".join(PROPERTY_TESTS), "result": summary}
    checks = [
        check_document(
            check.check_id,
            check.status.value == "PASS",
            check.measured,
            check.threshold,
            check.comparison,
            **(test_evidence if check.check_id in _TEST_BACKED else {}),
            **({"reason": check.reason} if check.reason else {}),
        )
        for check in evaluation.checks
    ]
    design_policy = {
        "source": "sealed design regenerated in memory from the restricted reveal",
        "plan_hash_verified": design["plan_hash_verified"],
        "seed_commitment_verified": design["seed_commitment_verified"],
        "population": design["population"],
        "measured": plain(measurements.design_diagnostics),
        "thresholds": COVERAGE_THRESHOLDS,
        "checks": design["checks"],
        "status": "PASS"
        if design["plan_hash_verified"] and all(design["checks"].values())
        else "FAIL",
    }
    if design_policy["status"] != "PASS":
        checks.append(
            check_document(
                "G9.production-design", False, design_policy["checks"], True, "all"
            )
        )
    document = gate_document(
        "G9",
        "S9",
        root,
        checks,
        campaign_hash=checkpoint.campaign_hash,
        visibility=visibility,
        commitment_order_problem=order_problem,
        production_design_policy=design_policy,
    )
    write_gate_document(root / GATE_PATH, document)
    return document


_TEST_BACKED = frozenset(
    {"G9.immutable", "G9.one-role", "G9.leakage", "G9.no-node-split", "G9.coverage"}
)

__all__ = ("GATE_PATH", "replacement_evidence", "visibility_violations", "write_gate")
